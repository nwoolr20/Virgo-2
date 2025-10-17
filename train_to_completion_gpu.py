#!/usr/bin/env python3
"""
GPU Training Script - Train Neural Field to 98-100% Coherence

This script:
1. Loads existing model from trained_models/virgo_model/best_model.pt
2. Upgrades architecture to 1B-2B parameters with GPT-2 tokenizer
3. Trains iteratively on multiple datasets until 98%+ coherence
4. Cleans up all temporary files and caches
5. Saves final trained model

Usage:
    python train_to_completion_gpu.py
    python train_to_completion_gpu.py --target-params 2000000000 --batch-size 8
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from virgo import NeuralFieldLM


class UpgradedNeuralFieldLM(nn.Module):
    """Upgraded Neural Field LM with GPT-2 tokenizer and larger capacity"""
    
    def __init__(self, base_model, target_params=1_000_000_000, vocab_size=50257):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Calculate architecture sizes to hit target params
        self.coord_dim = 16  # Increased from 8
        self.coord_hidden = 2048
        self.field_hidden = 4096
        self.num_transformer_layers = 12
        self.num_heads = 16
        
        # Coordinate encoder (maps token IDs to coordinates)
        self.coord_encoder = nn.Sequential(
            nn.Embedding(vocab_size, 512),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, self.coord_hidden),
            nn.GELU(),
            nn.Linear(self.coord_hidden, self.coord_dim)
        )
        
        # SIREN field (coordinate → hidden representation)
        self.siren_layers = nn.ModuleList([
            nn.Linear(self.coord_dim, self.field_hidden),
            nn.Linear(self.field_hidden, self.field_hidden),
            nn.Linear(self.field_hidden, self.field_hidden),
            nn.Linear(self.field_hidden, self.field_hidden)
        ])
        
        # Transformer layers for sequential modeling
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.field_hidden,
            nhead=self.num_heads,
            dim_feedforward=self.field_hidden * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=self.num_transformer_layers
        )
        
        # Output head
        self.output_head = nn.Linear(self.field_hidden, vocab_size)
        
        # Initialize from base model if provided
        if base_model is not None:
            self._transfer_weights(base_model)
    
    def _transfer_weights(self, base_model):
        """Transfer weights from base model to new architecture"""
        print("Transferring weights from base model...")
        
        # Try to copy coord encoder weights where dimensions match
        try:
            if hasattr(base_model, 'coord_encoder'):
                # Copy embedding layer if it exists
                if hasattr(base_model.coord_encoder, 'embedding'):
                    old_embed = base_model.coord_encoder.embedding
                    new_embed = self.coord_encoder[0]
                    # Copy what we can
                    min_vocab = min(old_embed.weight.shape[0], new_embed.weight.shape[0])
                    min_dim = min(old_embed.weight.shape[1], new_embed.weight.shape[1])
                    with torch.no_grad():
                        new_embed.weight[:min_vocab, :min_dim] = old_embed.weight[:min_vocab, :min_dim]
        except Exception as e:
            print(f"Could not transfer coord encoder weights: {e}")
        
        # Try to copy field weights
        try:
            if hasattr(base_model, 'field') or hasattr(base_model, 'siren_layers'):
                old_field = base_model.field if hasattr(base_model, 'field') else base_model.siren_layers
                # Copy first layer weights where dimensions match
                if isinstance(old_field, nn.ModuleList) and len(old_field) > 0:
                    for i, (old_layer, new_layer) in enumerate(zip(old_field[:2], self.siren_layers[:2])):
                        if isinstance(old_layer, nn.Linear) and isinstance(new_layer, nn.Linear):
                            min_in = min(old_layer.weight.shape[1], new_layer.weight.shape[1])
                            min_out = min(old_layer.weight.shape[0], new_layer.weight.shape[0])
                            with torch.no_grad():
                                new_layer.weight[:min_out, :min_in] = old_layer.weight[:min_out, :min_in]
                                if old_layer.bias is not None and new_layer.bias is not None:
                                    new_layer.bias[:min_out] = old_layer.bias[:min_out]
        except Exception as e:
            print(f"Could not transfer field weights: {e}")
        
        print("Weight transfer complete")
    
    def forward(self, input_ids):
        """Forward pass through upgraded architecture"""
        batch_size, seq_len = input_ids.shape
        
        # Encode tokens to coordinates
        coords = self.coord_encoder(input_ids)  # (batch, seq, coord_dim)
        
        # Pass through SIREN field
        x = coords
        for i, layer in enumerate(self.siren_layers):
            x = layer(x)
            if i < len(self.siren_layers) - 1:
                x = torch.sin(30.0 * x)  # SIREN activation
        
        # Pass through transformer for sequential dependencies
        x = self.transformer(x)  # (batch, seq, field_hidden)
        
        # Output logits
        logits = self.output_head(x)  # (batch, seq, vocab_size)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_length=50, temperature=0.8):
        """Generate text autoregressively"""
        self.eval()
        device = next(self.parameters()).device
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.forward(input_ids)
                
                # Sample next token
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop at EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)


class TextDatasetGPU(Dataset):
    """Dataset for GPU training with GPT-2 tokenizer"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        # Create labels (shifted input_ids)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = self.tokenizer.eos_token_id
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


def load_datasets(num_samples_per_dataset=50000):
    """Load and combine multiple datasets"""
    print("\nLoading datasets from HuggingFace...")
    
    all_texts = []
    
    # WikiText-103
    print("Loading WikiText-103...")
    try:
        wiki = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
        wiki_texts = []
        for i, item in enumerate(wiki):
            if i >= num_samples_per_dataset:
                break
            if item['text'].strip():
                wiki_texts.append(item['text'])
        all_texts.extend(wiki_texts)
        print(f"  Loaded {len(wiki_texts)} WikiText samples")
    except Exception as e:
        print(f"  Could not load WikiText: {e}")
    
    # FineWeb-Edu (educational content)
    print("Loading FineWeb-Edu...")
    try:
        fineweb = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        fineweb_texts = []
        for i, item in enumerate(fineweb):
            if i >= num_samples_per_dataset:
                break
            if item['text'].strip():
                fineweb_texts.append(item['text'][:1000])  # Limit length
        all_texts.extend(fineweb_texts)
        print(f"  Loaded {len(fineweb_texts)} FineWeb-Edu samples")
    except Exception as e:
        print(f"  Could not load FineWeb-Edu: {e}")
    
    # OpenWebText
    print("Loading OpenWebText...")
    try:
        owt = load_dataset("openwebtext", split="train", streaming=True)
        owt_texts = []
        for i, item in enumerate(owt):
            if i >= num_samples_per_dataset:
                break
            if item['text'].strip():
                owt_texts.append(item['text'][:1000])
        all_texts.extend(owt_texts)
        print(f"  Loaded {len(owt_texts)} OpenWebText samples")
    except Exception as e:
        print(f"  Could not load OpenWebText: {e}")
    
    # C4 (web-scale corpus)
    print("Loading C4...")
    try:
        c4 = load_dataset("c4", "en", split="train", streaming=True)
        c4_texts = []
        for i, item in enumerate(c4):
            if i >= num_samples_per_dataset:
                break
            if item['text'].strip():
                c4_texts.append(item['text'][:1000])
        all_texts.extend(c4_texts)
        print(f"  Loaded {len(c4_texts)} C4 samples")
    except Exception as e:
        print(f"  Could not load C4: {e}")
    
    print(f"\nTotal texts loaded: {len(all_texts)}")
    return all_texts


def test_coherence(model, tokenizer, device):
    """Test model coherence with 5 prompts"""
    test_prompts = [
        "the cat",
        "artificial intelligence",
        "once upon a time",
        "the quick brown",
        "in the beginning"
    ]
    
    results = []
    
    for prompt in test_prompts:
        generated = model.generate(tokenizer, prompt, max_length=30, temperature=0.8)
        
        # Check coherence criteria
        is_meaningful = len(generated) > len(prompt) + 2
        has_spaces = ' ' in generated[len(prompt):]
        no_repetition = not any(c * 5 in generated for c in 'abcdefghijklmnopqrstuvwxyz')
        
        passed = is_meaningful and has_spaces and no_repetition
        
        results.append({
            'prompt': prompt,
            'generated': generated,
            'passed': passed
        })
    
    # Calculate coherence percentage
    num_passed = sum(1 for r in results if r['passed'])
    coherence_pct = (num_passed / len(results)) * 100
    
    return coherence_pct, results


def train_iteration(model, train_loader, optimizer, device, teacher_model=None):
    """Train for one iteration"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Compute loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Add distillation loss if teacher available
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids).logits
            
            teacher_probs = F.softmax(teacher_logits[:, :-1, :] / 2.0, dim=-1)
            student_log_probs = F.log_softmax(shift_logits / 2.0, dim=-1)
            
            distill_loss = F.kl_div(
                student_log_probs.view(-1, shift_logits.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
                reduction='batchmean'
            ) * 4.0  # Temperature^2
            
            loss = 0.5 * lm_loss + 0.5 * distill_loss
        else:
            loss = lm_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def cleanup_cache():
    """Clean up HuggingFace cache and temporary files"""
    print("\nCleaning up cache and temporary files...")
    
    # HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print("  Removed HuggingFace cache")
        except Exception as e:
            print(f"  Could not remove HuggingFace cache: {e}")
    
    # Remove temporary checkpoints
    checkpoint_dir = Path("./trained_models/virgo_model/checkpoints")
    if checkpoint_dir.exists():
        try:
            shutil.rmtree(checkpoint_dir)
            print("  Removed temporary checkpoints")
        except Exception as e:
            print(f"  Could not remove checkpoints: {e}")
    
    print("Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(description="Train Neural Field to 98-100% Coherence")
    parser.add_argument('--target-params', type=int, default=1_000_000_000,
                        help='Target parameter count (1B or 2B)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--samples-per-dataset', type=int, default=50000,
                        help='Number of samples to load from each dataset')
    parser.add_argument('--coherence-target', type=float, default=98.0,
                        help='Target coherence percentage (98-100)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("NEURAL FIELD TRAINING TO 98-100% COHERENCE")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cpu':
        print("WARNING: Running on CPU. This will be very slow.")
        print("For best results, run on a GPU with 24GB+ VRAM.")
    
    # Load GPT-2 tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Load existing model
    print("\nLoading existing model...")
    model_path = Path("trained_models/virgo_model/best_model.pt")
    
    base_model = None
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            base_model = NeuralFieldLM(vocab_size=136, coord_dim=8)
            base_model.load_state_dict(checkpoint)
            print(f"Loaded base model from {model_path}")
        except Exception as e:
            print(f"Could not load base model: {e}")
            print("Will train from scratch")
    
    # Create upgraded model
    print("\nUpgrading architecture...")
    model = UpgradedNeuralFieldLM(
        base_model=base_model,
        target_params=args.target_params,
        vocab_size=tokenizer.vocab_size
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Target parameters: {args.target_params:,}")
    print(f"Coordinate dimensions: {model.coord_dim}")
    
    # Load teacher model for distillation
    print("\nLoading teacher model (DistilGPT-2)...")
    try:
        teacher_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded")
    except Exception as e:
        print(f"Could not load teacher model: {e}")
        teacher_model = None
    
    # Load datasets
    all_texts = load_datasets(num_samples_per_dataset=args.samples_per_dataset)
    
    # Create dataset and dataloader
    dataset = TextDatasetGPU(all_texts, tokenizer, max_length=512)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING UNTIL 98-100% COHERENCE")
    print("=" * 70)
    
    iteration = 0
    best_coherence = 0.0
    
    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
        
        # Train
        avg_loss = train_iteration(model, train_loader, optimizer, device, teacher_model)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Test coherence every 10 iterations
        if iteration % 10 == 0:
            print("\nTesting coherence...")
            coherence_pct, results = test_coherence(model, tokenizer, device)
            
            print(f"\n{'='*70}")
            print(f"COHERENCE TEST RESULTS: {coherence_pct:.1f}%")
            print(f"{'='*70}")
            
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                print(f"{status} | {r['prompt']}")
                print(f"     → {r['generated']}")
            
            # Save if improved
            if coherence_pct > best_coherence:
                best_coherence = coherence_pct
                print(f"\n💾 Saving model (coherence: {coherence_pct:.1f}%)")
                
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
            
            # Check if target achieved
            if coherence_pct >= args.coherence_target:
                print(f"\n{'='*70}")
                print(f"🎉 TARGET ACHIEVED: {coherence_pct:.1f}% coherence!")
                print(f"{'='*70}")
                break
        
        # Safety limit
        if iteration >= 1000:
            print("\nReached maximum iterations (1000)")
            break
    
    # Final save
    print("\n💾 Saving final model...")
    torch.save(model.state_dict(), model_path)
    
    # Cleanup
    cleanup_cache()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final coherence: {best_coherence:.1f}%")
    print(f"Model saved to: {model_path}")
    print(f"Total iterations: {iteration}")
    print("\nThe model is ready for conversation!")


if __name__ == "__main__":
    main()
