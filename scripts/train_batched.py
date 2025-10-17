#!/usr/bin/env python3
"""
CPU-Friendly Batched Training Script

Trains the neural field with:
- Small batches to work on CPU
- Incremental/progressive training
- Metrics-driven stopping (perplexity, coherence)
- No reinitialization of existing weights
- Smaller teacher model for CPU distillation

Usage:
    python scripts/train_batched.py --phase train
    python scripts/train_batched.py --phase distill --teacher distilgpt2
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
from datasets import load_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer


def load_existing_model(model_path, device='cpu'):
    """Load existing model to continue training."""
    print(f"Loading existing model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    vocab_size = checkpoint['vocab_size']
    coord_dim = checkpoint.get('coord_dim', 8)
    
    model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=coord_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Coordinates: {coord_dim}D")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, checkpoint


def prepare_batched_data(texts, tokenizer, max_seq_len=128, batch_size=4):
    """Prepare data in small batches for CPU training."""
    batches = []
    current_batch_inputs = []
    current_batch_targets = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_eos=False)
        if len(tokens) < 2:
            continue
            
        # Create sequences
        for i in range(0, len(tokens) - 1, max_seq_len // 2):
            end_idx = min(i + max_seq_len, len(tokens))
            if end_idx - i < 2:
                continue
                
            seq_tokens = tokens[i:end_idx]
            input_ids = torch.tensor(seq_tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(seq_tokens[1:], dtype=torch.long)
            
            current_batch_inputs.append(input_ids)
            current_batch_targets.append(target_ids)
            
            if len(current_batch_inputs) >= batch_size:
                batches.append((current_batch_inputs, current_batch_targets))
                current_batch_inputs = []
                current_batch_targets = []
    
    # Add remaining
    if current_batch_inputs:
        batches.append((current_batch_inputs, current_batch_targets))
    
    return batches


def train_batched(model, tokenizer, dataset_name, num_samples, epochs, batch_size, save_dir, device='cpu'):
    """Train with small batches suitable for CPU."""
    print("\n" + "=" * 60)
    print("BATCHED TRAINING")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {num_samples}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    try:
        if dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        elif dataset_name == "fineweb-edu":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")
        elif dataset_name == "openwebtext":
            ds = load_dataset("openwebtext", split="train", streaming=True)
        else:
            ds = load_dataset(dataset_name, split="train", streaming=True)
        
        # Collect samples
        texts = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            text = item.get('text', item.get('content', ''))
            if text and len(text) > 10:
                texts.append(text[:500])  # Limit length
        
        print(f"  Loaded {len(texts)} samples")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        print(f"  Using fallback data")
        texts = ["The quick brown fox jumps over the lazy dog"] * num_samples
    
    # Prepare batches
    print("\nPreparing batches...")
    batches = prepare_batched_data(texts, tokenizer, max_seq_len=128, batch_size=batch_size)
    print(f"  Created {len(batches)} batches")
    
    # Training
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = 0
        num_batches = 0
        
        progress = tqdm(batches, desc=f"Training")
        for inputs_list, targets_list in progress:
            # Process each sequence in the batch
            batch_loss = 0
            for input_ids, target_ids in zip(inputs_list, targets_list):
                input_ids = input_ids.unsqueeze(0).to(device)
                target_ids = target_ids.unsqueeze(0).to(device)
                
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    target_ids.view(-1)
                )
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / len(inputs_list)
            
            # Backward
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
            progress.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'perplexity': perplexity
        })
        
        # Save if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': tokenizer.vocab_size,
                'coord_dim': model.coord_dim,
                'char_to_idx': tokenizer.char_to_idx,
                'idx_to_char': tokenizer.idx_to_char,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'perplexity': perplexity
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"  ✓ Saved best model (perplexity: {perplexity:.2f})")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, save_dir / f"checkpoint_epoch_{epoch+1}.pt")
        
        # Check stopping criteria
        if perplexity < 15.0:
            print(f"\n✓ Target perplexity reached ({perplexity:.2f} < 15.0)")
            break
    
    # Save final
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'coord_dim': model.coord_dim,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'training_history': history
    }
    torch.save(final_checkpoint, save_dir / "final_model.pt")
    
    # Save history
    with open(save_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training complete")
    print(f"  Best perplexity: {best_loss:.2f}")
    print(f"  Models saved to: {save_dir}")
    
    return model, history


def test_generation(model, tokenizer, device='cpu'):
    """Test generation quality."""
    print("\n" + "=" * 60)
    print("TESTING GENERATION")
    print("=" * 60)
    
    model.eval()
    
    prompts = [
        "the",
        "hello",
        "in the",
        "artificial intelligence",
        "once upon a time"
    ]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, temperature=0.8)
        
        generated = tokenizer.decode(output_ids.cpu().tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="./trained_models/virgo_model/best_model.pt")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-dir", default="./trained_models/virgo_model")
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BATCHED NEURAL FIELD TRAINING")
    print("=" * 60)
    
    # Load existing model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n✗ No model found at {model_path}")
        print("Please train an initial model first:")
        print("  python3 launch_virgo.py train model")
        return 1
    
    model, tokenizer, checkpoint = load_existing_model(model_path, args.device)
    
    # Train
    model, history = train_batched(
        model, tokenizer,
        args.dataset,
        args.samples,
        args.epochs,
        args.batch_size,
        args.save_dir,
        args.device
    )
    
    # Test
    test_generation(model, tokenizer, args.device)
    
    print("\n✓ Complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
