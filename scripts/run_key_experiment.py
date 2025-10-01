#!/usr/bin/env python3
"""
The Key Experiment: Neural Field vs Baseline Transformer

Trains both models on the same data and compares:
- Validation perplexity
- Generation coherence
- Parameter efficiency

Phase 3: Curriculum training with compression benchmarking
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import math
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import ScaledNeuralFieldLM, BPETokenizer
from virgo.baseline_transformer import BaselineTransformerLM
from scripts.train_scaled_nflm import StreamingTextDataset, collate_fn, train_epoch, evaluate, test_generation_quality


def run_comparison_experiment(num_samples, epochs, batch_size, device, save_dir):
    """
    Run the key experiment: train both models and compare results.
    
    Args:
        num_samples: Number of training samples
        epochs: Number of epochs
        batch_size: Batch size
        device: Device to use
        save_dir: Where to save results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("THE KEY EXPERIMENT")
    print("Neural Field LM vs Baseline Transformer")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Device: {device}")
    
    # Initialize tokenizer
    tokenizer = BPETokenizer()
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    
    # Create datasets
    print(f"\nLoading dataset...")
    train_dataset = StreamingTextDataset(
        "fineweb-edu",
        tokenizer,
        max_seq_len=512,
        num_samples=num_samples,
        split="train"
    )
    
    val_dataset = StreamingTextDataset(
        "fineweb-edu",
        tokenizer,
        max_seq_len=512,
        num_samples=min(1000, num_samples // 10),
        split="validation"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    results = {
        "config": {
            "num_samples": num_samples,
            "epochs": epochs,
            "batch_size": batch_size,
            "vocab_size": tokenizer.vocab_size
        },
        "models": {}
    }
    
    # Train Neural Field LM
    print("\n" + "="*70)
    print("Training Neural Field Language Model")
    print("="*70)
    
    nf_model = ScaledNeuralFieldLM(vocab_size=tokenizer.vocab_size)
    nf_model = nf_model.to(device)
    nf_params = nf_model.count_parameters()
    print(f"Parameters: {nf_params:,}")
    
    nf_optimizer = torch.optim.AdamW(nf_model.parameters(), lr=1e-4)
    nf_scaler = GradScaler() if device.type == "cuda" else None
    
    nf_history = []
    best_nf_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_loss, train_ppl = train_epoch(
            nf_model, train_loader, nf_optimizer, nf_scaler, device, epoch, 
            use_amp=(device.type == "cuda")
        )
        val_loss, val_ppl = evaluate(
            nf_model, val_loader, device, 
            use_amp=(device.type == "cuda")
        )
        
        if val_loss < best_nf_val_loss:
            best_nf_val_loss = val_loss
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
        
        nf_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl
        })
    
    # Save Neural Field model
    nf_model_path = save_dir / "neural_field_model.pt"
    torch.save({
        "model_state_dict": nf_model.state_dict(),
        "history": nf_history,
        "config": {"vocab_size": tokenizer.vocab_size}
    }, nf_model_path)
    
    # Test generation
    nf_generations = test_generation_quality(nf_model, tokenizer, device)
    
    results["models"]["neural_field"] = {
        "parameters": nf_params,
        "best_val_loss": best_nf_val_loss,
        "best_val_perplexity": math.exp(best_nf_val_loss),
        "final_val_loss": nf_history[-1]["val_loss"],
        "final_val_perplexity": nf_history[-1]["val_ppl"],
        "history": nf_history,
        "generations": nf_generations
    }
    
    # Train Baseline Transformer
    print("\n" + "="*70)
    print("Training Baseline Transformer")
    print("="*70)
    
    baseline_model = BaselineTransformerLM(vocab_size=tokenizer.vocab_size)
    baseline_model = baseline_model.to(device)
    baseline_params = baseline_model.count_parameters()
    print(f"Parameters: {baseline_params:,}")
    
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4)
    baseline_scaler = GradScaler() if device.type == "cuda" else None
    
    baseline_history = []
    best_baseline_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_loss, train_ppl = train_epoch(
            baseline_model, train_loader, baseline_optimizer, baseline_scaler, device, epoch,
            use_amp=(device.type == "cuda")
        )
        val_loss, val_ppl = evaluate(
            baseline_model, val_loader, device,
            use_amp=(device.type == "cuda")
        )
        
        if val_loss < best_baseline_val_loss:
            best_baseline_val_loss = val_loss
        
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f})")
        
        baseline_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl
        })
    
    # Save Baseline model
    baseline_model_path = save_dir / "baseline_model.pt"
    torch.save({
        "model_state_dict": baseline_model.state_dict(),
        "history": baseline_history,
        "config": {"vocab_size": tokenizer.vocab_size}
    }, baseline_model_path)
    
    # Test generation
    baseline_generations = test_generation_quality(baseline_model, tokenizer, device)
    
    results["models"]["baseline_transformer"] = {
        "parameters": baseline_params,
        "best_val_loss": best_baseline_val_loss,
        "best_val_perplexity": math.exp(best_baseline_val_loss),
        "final_val_loss": baseline_history[-1]["val_loss"],
        "final_val_perplexity": baseline_history[-1]["val_ppl"],
        "history": baseline_history,
        "generations": baseline_generations
    }
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\nParameter Count:")
    print(f"  Neural Field: {nf_params:,}")
    print(f"  Baseline: {baseline_params:,}")
    print(f"  Difference: {abs(nf_params - baseline_params):,}")
    
    print(f"\nBest Validation Loss:")
    print(f"  Neural Field: {best_nf_val_loss:.4f}")
    print(f"  Baseline: {best_baseline_val_loss:.4f}")
    print(f"  Winner: {'Neural Field' if best_nf_val_loss < best_baseline_val_loss else 'Baseline'}")
    
    print(f"\nBest Validation Perplexity:")
    nf_ppl = math.exp(best_nf_val_loss)
    baseline_ppl = math.exp(best_baseline_val_loss)
    print(f"  Neural Field: {nf_ppl:.2f}")
    print(f"  Baseline: {baseline_ppl:.2f}")
    print(f"  Winner: {'Neural Field' if nf_ppl < baseline_ppl else 'Baseline'}")
    
    improvement = ((baseline_ppl - nf_ppl) / baseline_ppl) * 100
    print(f"\nPerplexity Improvement: {improvement:+.2f}%")
    
    results["comparison"] = {
        "parameter_difference": abs(nf_params - baseline_params),
        "nf_better_loss": best_nf_val_loss < best_baseline_val_loss,
        "perplexity_improvement_percent": improvement,
        "winner": "neural_field" if best_nf_val_loss < best_baseline_val_loss else "baseline",
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    results_path = save_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Neural Field vs Baseline comparison")
    parser.add_argument("--num-samples", type=int, default=50000,
                       help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="./experiments/key_experiment",
                       help="Save directory")
    
    args = parser.parse_args()
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    
    # Load HF token
    from scripts.train_scaled_nflm import load_hf_token
    load_hf_token()
    
    # Run experiment
    results = run_comparison_experiment(
        num_samples=args.num_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
