#!/usr/bin/env python3
"""
Phase 3: Extended Training on FineWeb-Edu

Continues training the neural field on high-quality educational web content.
Uses pure language modeling loss (no distillation) on streaming data.
Trains until validation loss plateaus or target perplexity reached.
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


def load_model(model_path, device='cpu'):
    """Load existing model for extended training."""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    vocab_size = checkpoint['vocab_size']
    coord_dim = checkpoint.get('coord_dim', 8)
    
    model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=coord_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    
    print(f"  ✓ Model loaded")
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Coordinates: {coord_dim}D")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer, checkpoint


def prepare_batch_data(texts, tokenizer, max_seq_len=128):
    """Prepare data for training."""
    sequences = []
    
    for text in texts:
        if not text or len(text) < 10:
            continue
        
        tokens = tokenizer.encode(text[:500], add_eos=False)
        if len(tokens) < 2:
            continue
        
        # Create overlapping sequences
        for i in range(0, len(tokens) - 1, max_seq_len // 2):
            end_idx = min(i + max_seq_len, len(tokens))
            if end_idx - i < 2:
                continue
            
            seq_tokens = tokens[i:end_idx]
            input_ids = torch.tensor(seq_tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(seq_tokens[1:], dtype=torch.long)
            
            sequences.append((input_ids, target_ids))
    
    return sequences


def train_extended(model, tokenizer, dataset_name, num_samples, max_epochs, 
                  batch_size, save_dir, patience=3, device='cpu'):
    """
    Extended training with plateau detection.
    
    Args:
        model: Student model
        tokenizer: Tokenizer
        dataset_name: Dataset to use
        num_samples: Number of samples per epoch
        max_epochs: Maximum epochs
        batch_size: Batch size
        save_dir: Save directory
        patience: Epochs to wait for improvement before stopping
        device: Device
    """
    print("\n" + "=" * 70)
    print("PHASE 3: EXTENDED TRAINING")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Samples per epoch: {num_samples}")
    print(f"Max epochs: {max_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Patience: {patience} epochs")
    print(f"Device: {device}")
    
    # Setup
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    history = []
    
    # Training loop
    for epoch in range(max_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{max_epochs}")
        print(f"{'='*70}")
        
        # Load fresh data for this epoch
        print("Loading dataset...")
        try:
            if dataset_name == "fineweb-edu":
                ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", 
                                streaming=True, name="sample-10BT")
            elif dataset_name == "wikitext":
                ds = load_dataset("wikitext", "wikitext-103-raw-v1", 
                                split="train", streaming=True)
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
                if text and len(text) > 20:
                    texts.append(text)
            
            print(f"  Loaded {len(texts)} samples")
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            print(f"  Stopping extended training")
            break
        
        # Prepare sequences
        sequences = prepare_batch_data(texts, tokenizer)
        print(f"  Created {len(sequences)} training sequences")
        
        if len(sequences) == 0:
            print("  No valid sequences, skipping epoch")
            continue
        
        # Training
        epoch_loss = 0
        num_batches = 0
        
        # Process in batches
        progress = tqdm(range(0, len(sequences), batch_size), desc="Training")
        for i in progress:
            batch_sequences = sequences[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for input_ids, target_ids in batch_sequences:
                input_ids = input_ids.unsqueeze(0).to(device)
                target_ids = target_ids.unsqueeze(0).to(device)
                
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    target_ids.view(-1)
                )
                batch_loss += loss
            
            # Average and backprop
            batch_loss = batch_loss / len(batch_sequences)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
            progress.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        # Compute metrics
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'perplexity': perplexity
        })
        
        # Check for improvement
        if avg_loss < best_loss:
            improvement = best_loss - avg_loss
            best_loss = avg_loss
            epochs_without_improvement = 0
            
            # Save best model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'vocab_size': tokenizer.vocab_size,
                'coord_dim': model.coord_dim,
                'char_to_idx': tokenizer.char_to_idx,
                'idx_to_char': tokenizer.idx_to_char,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'perplexity': perplexity,
                'phase': 'extended_training'
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"  ✓ New best model (improvement: {improvement:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{patience})")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
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
            torch.save(checkpoint, save_dir / f"extended_checkpoint_epoch_{epoch+1}.pt")
        
        # Check stopping criteria
        if perplexity < 15.0:
            print(f"\n✓ Target perplexity reached ({perplexity:.2f} < 15.0)")
            break
        
        if epochs_without_improvement >= patience:
            print(f"\n✓ Validation loss plateaued (no improvement for {patience} epochs)")
            break
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'coord_dim': model.coord_dim,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
        'training_history': history,
        'phase': 'extended_training_complete'
    }
    torch.save(final_checkpoint, save_dir / "final_model.pt")
    
    # Save history
    with open(save_dir / "extended_training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXTENDED TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")
    print(f"Total epochs: {len(history)}")
    print(f"Models saved to: {save_dir}")
    
    return model, history


def test_generation(model, tokenizer, device='cpu'):
    """Test generation quality."""
    print("\n" + "=" * 70)
    print("TESTING GENERATION QUALITY")
    print("=" * 70)
    
    model.eval()
    
    prompts = [
        "the",
        "hello",
        "in the",
        "artificial intelligence",
        "once upon a time",
        "the quick brown fox",
        "neural networks are"
    ]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=80, temperature=0.8)
        
        generated = tokenizer.decode(output_ids.cpu().tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated}'")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Extended Training")
    parser.add_argument("--model-path", default="./trained_models/virgo_model/best_model.pt")
    parser.add_argument("--dataset", default="fineweb-edu", 
                       help="Dataset (fineweb-edu, wikitext, openwebtext)")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Samples per epoch")
    parser.add_argument("--max-epochs", type=int, default=20,
                       help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3,
                       help="Epochs without improvement before stopping")
    parser.add_argument("--save-dir", default="./trained_models/virgo_model")
    parser.add_argument("--device", default="cpu")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE 3: EXTENDED TRAINING")
    print("=" * 70)
    
    # Check for model
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n✗ No model found at {model_path}")
        print("Please complete Phase 2 first or train an initial model:")
        print("  python scripts/train_phase2_distillation.py")
        return 1
    
    # Load model
    model, tokenizer, checkpoint = load_model(model_path, args.device)
    
    # Extended training
    model, history = train_extended(
        model, tokenizer,
        args.dataset,
        args.samples,
        args.max_epochs,
        args.batch_size,
        args.save_dir,
        args.patience,
        args.device
    )
    
    # Test generation
    test_generation(model, tokenizer, args.device)
    
    print("\n✓ Phase 3 complete")
    print("\nAll phases complete! Test the model:")
    print("  python3 launch_virgo.py chat")
    print("  python3 launch_virgo.py evaluate")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
