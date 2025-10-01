#!/usr/bin/env python3
"""
Train Neural Field Language Model on Real Text Corpora

This script trains the neural field language model on high-quality text datasets
from HuggingFace, following the curriculum approach recommended for neural fields.

Usage:
    python scripts/train_nflm.py --dataset wikitext --epochs 50 --save-dir ./trained_models
    python scripts/train_nflm.py --dataset fineweb-edu --sample-size 10000 --epochs 30
    python scripts/train_nflm.py --resume ./checkpoints/checkpoint_epoch_10.pt
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer


def load_hf_token():
    """Load HuggingFace token from .hf_token file (if needed for gated datasets)"""
    token_file = Path(__file__).parent.parent / ".hf_token"
    if token_file.exists():
        token = token_file.read_text().strip()
        try:
            from huggingface_hub import login
            login(token=token)
            return token
        except Exception as e:
            print(f"Note: Could not authenticate with HuggingFace token: {e}")
            print("Continuing without authentication (public datasets will still work)")
            return None
    return None


def load_dataset_texts(dataset_name, split="train", sample_size=None, max_length=512):
    """
    Load text data from HuggingFace datasets.
    
    Args:
        dataset_name: Name of the dataset (wikitext, fineweb-edu, etc.)
        split: Dataset split to use
        sample_size: Optional limit on number of samples
        max_length: Maximum text length to consider
        
    Returns:
        List of text strings
    """
    from datasets import load_dataset
    
    print(f"\nLoading dataset: {dataset_name}")
    print(f"  Split: {split}")
    if sample_size:
        print(f"  Sample size: {sample_size}")
    
    texts = []
    
    if dataset_name == "wikitext":
        # WikiText-103: High-quality Wikipedia articles
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        
        for item in tqdm(dataset, desc="Processing WikiText"):
            text = item["text"].strip()
            # Filter out empty lines and headers
            if text and len(text) > 50 and not text.startswith("="):
                if len(text) <= max_length:
                    texts.append(text)
                else:
                    # Chunk long texts
                    for i in range(0, len(text), max_length):
                        chunk = text[i:i+max_length]
                        if len(chunk) > 50:
                            texts.append(chunk)
                
                if sample_size and len(texts) >= sample_size:
                    break
    
    elif dataset_name == "fineweb-edu":
        # FineWeb-Edu: High-quality educational web text
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                              name="sample-10BT",  # Use 10B token sample
                              split=split, 
                              streaming=True)
        
        for item in tqdm(dataset, desc="Processing FineWeb-Edu", total=sample_size):
            text = item["text"].strip()
            if text and len(text) > 50:
                if len(text) <= max_length:
                    texts.append(text)
                else:
                    # Chunk long texts
                    for i in range(0, len(text), max_length):
                        chunk = text[i:i+max_length]
                        if len(chunk) > 50:
                            texts.append(chunk)
                
                if sample_size and len(texts) >= sample_size:
                    break
    
    elif dataset_name == "openwebtext":
        # OpenWebText: Reddit-curated quality web text
        dataset = load_dataset("Skylion007/openwebtext", split=split, streaming=True)
        
        for item in tqdm(dataset, desc="Processing OpenWebText", total=sample_size):
            text = item["text"].strip()
            if text and len(text) > 50:
                if len(text) <= max_length:
                    texts.append(text)
                else:
                    for i in range(0, len(text), max_length):
                        chunk = text[i:i+max_length]
                        if len(chunk) > 50:
                            texts.append(chunk)
                
                if sample_size and len(texts) >= sample_size:
                    break
    
    elif dataset_name == "c4":
        # C4: Colossal Clean Crawled Corpus
        dataset = load_dataset("allenai/c4", "en", split=split, streaming=True)
        
        for item in tqdm(dataset, desc="Processing C4", total=sample_size):
            text = item["text"].strip()
            if text and len(text) > 50:
                if len(text) <= max_length:
                    texts.append(text)
                else:
                    for i in range(0, len(text), max_length):
                        chunk = text[i:i+max_length]
                        if len(chunk) > 50:
                            texts.append(chunk)
                
                if sample_size and len(texts) >= sample_size:
                    break
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"\nLoaded {len(texts)} text samples")
    return texts


class TextDataset(Dataset):
    """Dataset for text sequences with next-token prediction"""
    
    def __init__(self, texts, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        
        print(f"\nTokenizing {len(texts)} texts...")
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text, add_eos=False)
            
            # Create overlapping sequences
            if len(tokens) > max_seq_len:
                # Chunk with overlap
                stride = max_seq_len // 2
                for i in range(0, len(tokens) - max_seq_len, stride):
                    seq = tokens[i:i+max_seq_len]
                    if len(seq) == max_seq_len:
                        self.samples.append(seq)
            elif len(tokens) > 1:
                # Pad short sequences
                padded = tokens + [tokenizer.char_to_idx[tokenizer.pad_token]] * (max_seq_len - len(tokens))
                self.samples.append(padded[:max_seq_len])
        
        print(f"Created {len(self.samples)} training sequences")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        # Input: all tokens except last, Target: all tokens except first
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        _, loss = model(input_ids, target_ids)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in tqdm(dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            _, loss = model(input_ids, target_ids)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def save_checkpoint(model, tokenizer, optimizer, epoch, loss, save_dir, is_best=False):
    """Save training checkpoint"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'vocab_size': tokenizer.vocab_size,
        'coord_dim': model.coord_dim,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
    }
    
    # Save regular checkpoint
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    if is_best:
        best_path = save_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    # Save final model (just the model weights)
    model_path = save_dir / f"model_epoch_{epoch}.pt"
    torch.save(model.state_dict(), model_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate tokenizer
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    tokenizer.vocab_size = checkpoint['vocab_size']
    
    # Recreate model
    model = NeuralFieldLM(
        vocab_size=checkpoint['vocab_size'],
        coord_dim=checkpoint['coord_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Recreate optimizer
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, tokenizer, optimizer, checkpoint['epoch'], checkpoint['loss']


def test_generation(model, tokenizer, device, prompts=None):
    """Test model generation"""
    if prompts is None:
        prompts = ["the", "hello", "in the"]
    
    print("\n" + "="*70)
    print("Testing Generation")
    print("="*70)
    
    model.eval()
    for prompt in prompts:
        prompt_tokens = tokenizer.encode(prompt, add_eos=False)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_length=50, temperature=0.8)
        
        generated_text = tokenizer.decode(generated.cpu().tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{generated_text}'")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Train Neural Field Language Model")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext",
                       choices=["wikitext", "fineweb-edu", "openwebtext", "c4"],
                       help="Dataset to train on")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Limit number of samples (useful for testing)")
    parser.add_argument("--max-text-length", type=int, default=512,
                       help="Maximum text length to process")
    parser.add_argument("--max-seq-len", type=int, default=128,
                       help="Maximum sequence length for training")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--coord-dim", type=int, default=8,
                       help="Coordinate dimension")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, auto-detect if not specified)")
    
    # Checkpoint arguments
    parser.add_argument("--save-dir", type=str, default="./trained_models",
                       help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=5,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    
    # Validation
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split ratio")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Load HuggingFace token (for gated datasets)
    token = load_hf_token()
    if token:
        print("✓ HuggingFace token loaded successfully")
    else:
        print("Note: No HuggingFace authentication (public datasets will still work)")
    
    # Load or resume training
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model, tokenizer, optimizer, start_epoch, prev_loss = load_checkpoint(args.resume, device)
        print(f"Resumed from epoch {start_epoch}, loss: {prev_loss:.4f}")
    else:
        # Load dataset
        all_texts = load_dataset_texts(
            args.dataset,
            sample_size=args.sample_size,
            max_length=args.max_text_length
        )
        
        if len(all_texts) == 0:
            print("Error: No texts loaded!")
            return
        
        # Split into train/val
        val_size = int(len(all_texts) * args.val_split)
        train_texts = all_texts[val_size:]
        val_texts = all_texts[:val_size] if val_size > 0 else []
        
        print(f"\nTrain samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Build tokenizer
        print("\nBuilding tokenizer...")
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(all_texts)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        
        # Create model
        print("\nCreating model...")
        model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=args.coord_dim)
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Create datasets
        train_dataset = TextDataset(train_texts, tokenizer, max_seq_len=args.max_seq_len)
        val_dataset = TextDataset(val_texts, tokenizer, max_seq_len=args.max_seq_len) if val_texts else None
    
    # If resuming, we need to reload the datasets (simplified: just skip this for resume)
    if not args.resume:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == "cuda" else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if device.type == "cuda" else False
            )
    else:
        print("\nNote: When resuming, you need to recreate the dataset.")
        print("For now, we'll continue without validation.")
        train_loader = None
        val_loader = None
    
    if not args.resume and train_loader is None:
        print("Error: No training data!")
        return
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        if train_loader:
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
            print(f"\nEpoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = evaluate(model, val_loader, device)
                print(f"Epoch {epoch}/{args.epochs} - Val Loss: {val_loss:.4f}")
                
                # Check if best model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    print("✓ New best model!")
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % args.save_every == 0 or epoch == args.epochs:
                save_checkpoint(model, tokenizer, optimizer, epoch, train_loss, args.save_dir, is_best)
            
            # Save history
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save history to file
            history_dir = Path(args.save_dir)
            history_dir.mkdir(parents=True, exist_ok=True)
            history_path = history_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(training_history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    
    # Final test
    if train_loader:
        test_generation(model, tokenizer, device)
    
    # Save final model
    final_path = Path(args.save_dir) / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'coord_dim': model.coord_dim,
        'char_to_idx': tokenizer.char_to_idx,
        'idx_to_char': tokenizer.idx_to_char,
    }, final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    
    print(f"\nAll checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
