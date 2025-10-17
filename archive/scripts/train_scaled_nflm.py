#!/usr/bin/env python3
"""
Production Training Script for Scaled Neural Field Language Model

Implements Phase 2: Production training pipeline with:
- Streaming large datasets (FineWeb-Edu, Dolma, C4)
- Mixed precision training (fp16/bf16)
- Automatic checkpoint saving
- Comprehensive metrics tracking
- Early stopping
- Generation quality tests

Usage:
    # Start Phase 2 training
    python scripts/train_scaled_nflm.py --dataset fineweb-edu --num-samples 50000 --epochs 50
    
    # Resume training
    python scripts/train_scaled_nflm.py --resume ./trained_models/scaled/checkpoint_latest.pt
    
    # Continue on Dolma after FineWeb
    python scripts/train_scaled_nflm.py --dataset dolma --resume ./trained_models/scaled/checkpoint_epoch_50.pt --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
from datetime import datetime
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import ScaledNeuralFieldLM, BPETokenizer


def load_hf_token():
    """Load HuggingFace token"""
    token_file = Path(__file__).parent.parent / ".hf_token"
    if token_file.exists():
        token = token_file.read_text().strip()
        try:
            from huggingface_hub import login
            login(token=token)
            return token
        except Exception as e:
            print(f"Note: Could not authenticate: {e}")
            return None
    return None


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for large text corpora"""
    
    def __init__(self, dataset_name, tokenizer, max_seq_len=512, num_samples=None, split="train"):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_samples = num_samples
        self.split = split
        
    def __iter__(self):
        from datasets import load_dataset
        
        # Load streaming dataset
        if self.dataset_name == "fineweb-edu":
            dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                                  name="sample-10BT",
                                  split=self.split,
                                  streaming=True)
        elif self.dataset_name == "dolma":
            dataset = load_dataset("allenai/dolma",
                                  split=self.split,
                                  streaming=True)
        elif self.dataset_name == "c4":
            dataset = load_dataset("allenai/c4",
                                  "en",
                                  split=self.split,
                                  streaming=True)
        elif self.dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=self.split)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        count = 0
        for item in dataset:
            if self.num_samples and count >= self.num_samples:
                break
            
            text = item.get("text", "").strip()
            if len(text) < 50:  # Skip very short texts
                continue
            
            # Tokenize
            tokens = self.tokenizer.encode(text, add_eos=True, max_length=self.max_seq_len + 1)
            
            if len(tokens) > 1:
                # Create input-target pairs
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                target_ids = torch.tensor(tokens[1:], dtype=torch.long)
                
                # Pad to max_seq_len if needed
                if len(input_ids) < self.max_seq_len:
                    pad_len = self.max_seq_len - len(input_ids)
                    input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
                    target_ids = torch.cat([target_ids, torch.full((pad_len,), -100)])  # -100 for ignore_index
                
                yield input_ids, target_ids
                count += 1


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([x[0] for x in batch])
    target_ids = torch.stack([x[1] for x in batch])
    return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, use_amp=False):
    """Train for one epoch with mixed precision support"""
    model.train()
    total_loss = 0
    total_tokens = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Count non-padding tokens
        non_pad_mask = (target_ids != -100)
        tokens_in_batch = non_pad_mask.sum().item()
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast(dtype=torch.float16):
                _, loss = model(input_ids, target_ids)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(input_ids, target_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item() * tokens_in_batch
        total_tokens += tokens_in_batch
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'ppl': f'{perplexity:.2f}'
        })
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return avg_loss, perplexity


def evaluate(model, dataloader, device, use_amp=False):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Count non-padding tokens
            non_pad_mask = (target_ids != -100)
            tokens_in_batch = non_pad_mask.sum().item()
            
            if use_amp:
                with autocast(dtype=torch.float16):
                    _, loss = model(input_ids, target_ids)
            else:
                _, loss = model(input_ids, target_ids)
            
            total_loss += loss.item() * tokens_in_batch
            total_tokens += tokens_in_batch
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
    
    return avg_loss, perplexity


def test_generation_quality(model, tokenizer, device, prompts=None):
    """Test generation quality with various prompts"""
    if prompts is None:
        prompts = [
            "The meaning of life is",
            "In the future,",
            "Artificial intelligence will",
            "The most important discovery was",
            "Once upon a time,"
        ]
    
    print("\n" + "="*70)
    print("Generation Quality Tests")
    print("="*70)
    
    model.eval()
    results = []
    
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt, add_eos=False)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_length=50, temperature=0.8, top_k=50, top_p=0.95)
        
        generated_text = tokenizer.decode(generated.cpu().tolist())
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
        
        results.append({
            "prompt": prompt,
            "generated": generated_text
        })
    
    print("="*70)
    return results


def save_checkpoint(model, tokenizer, optimizer, scaler, epoch, metrics, save_dir, is_best=False):
    """Save training checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'model_config': {
            'vocab_size': model.vocab_size,
            'coord_dim': model.coord_dim,
        }
    }
    
    # Save latest checkpoint
    latest_path = save_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save epoch checkpoint
    if epoch % 5 == 0:
        epoch_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, epoch_path)
        print(f"Saved checkpoint: {epoch_path}")
    
    # Save best model
    if is_best:
        best_path = save_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Scaled Neural Field Language Model")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="fineweb-edu",
                       choices=["fineweb-edu", "dolma", "c4", "wikitext"],
                       help="Dataset to train on")
    parser.add_argument("--num-samples", type=int, default=50000,
                       help="Number of samples to train on")
    parser.add_argument("--max-seq-len", type=int, default=512,
                       help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--coord-dim", type=int, default=8,
                       help="Coordinate dimension")
    parser.add_argument("--encoder-hidden-dim", type=int, default=512,
                       help="Encoder hidden dimension")
    parser.add_argument("--encoder-layers", type=int, default=4,
                       help="Number of encoder layers")
    parser.add_argument("--field-hidden-dim", type=int, default=1024,
                       help="Field hidden dimension")
    parser.add_argument("--field-layers", type=int, default=8,
                       help="Number of field layers")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--use-amp", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    # Checkpoint arguments
    parser.add_argument("--save-dir", type=str, default="./trained_models/scaled",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    # Validation
    parser.add_argument("--val-samples", type=int, default=1000,
                       help="Number of validation samples")
    parser.add_argument("--patience", type=int, default=5,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Load HuggingFace token
    load_hf_token()
    
    # Initialize tokenizer
    print("\nInitializing BPE tokenizer...")
    tokenizer = BPETokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create or load model
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        model = ScaledNeuralFieldLM(
            vocab_size=tokenizer.vocab_size,
            coord_dim=checkpoint['model_config']['coord_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['metrics'].get('best_val_loss', float('inf'))
        
        scaler = GradScaler() if args.use_amp else None
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        print("\nCreating model...")
        model = ScaledNeuralFieldLM(
            vocab_size=tokenizer.vocab_size,
            coord_dim=args.coord_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_layers=args.encoder_layers,
            field_hidden_dim=args.field_hidden_dim,
            field_layers=args.field_layers
        )
        model = model.to(device)
        
        total_params = model.count_parameters()
        print(f"Model parameters: {total_params:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scaler = GradScaler() if args.use_amp else None
        start_epoch = 0
        best_val_loss = float('inf')
    
    # Create datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = StreamingTextDataset(
        args.dataset,
        tokenizer,
        max_seq_len=args.max_seq_len,
        num_samples=args.num_samples,
        split="train"
    )
    
    val_dataset = StreamingTextDataset(
        args.dataset,
        tokenizer,
        max_seq_len=args.max_seq_len,
        num_samples=args.val_samples,
        split="validation" if args.dataset != "dolma" else "train"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    training_history = []
    patience_counter = 0
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        train_loss, train_ppl = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, use_amp=args.use_amp
        )
        
        # Validate
        val_loss, val_ppl = evaluate(model, val_loader, device, use_amp=args.use_amp)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        
        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print("  ✓ New best model!")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
        
        # Save metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_perplexity': train_ppl,
            'val_loss': val_loss,
            'val_perplexity': val_ppl,
            'best_val_loss': best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        training_history.append(metrics)
        
        # Save checkpoint
        save_checkpoint(model, tokenizer, optimizer, scaler, epoch, metrics, args.save_dir, is_best)
        
        # Save training history
        history_path = Path(args.save_dir) / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Test generation every 10 epochs
        if epoch % 10 == 0:
            gen_results = test_generation_quality(model, tokenizer, device)
            
            # Save generation results
            gen_path = Path(args.save_dir) / f"generation_epoch_{epoch}.json"
            with open(gen_path, 'w') as f:
                json.dump(gen_results, f, indent=2)
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(best_val_loss):.2f}")
    
    # Final generation test
    test_generation_quality(model, tokenizer, device)


if __name__ == "__main__":
    main()
