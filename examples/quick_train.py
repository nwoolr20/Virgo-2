#!/usr/bin/env python3
"""
Quick Training Example - Train a tiny neural field language model

This example trains a very small model in just a few minutes to demonstrate
the complete training pipeline.

Usage:
    python examples/quick_train.py
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer
from scripts.train_nflm import TextDataset, train_epoch, save_checkpoint


def main():
    print("=" * 70)
    print("Quick Training Example - Neural Field Language Model")
    print("=" * 70)
    
    # Simple training corpus
    print("\n1. Preparing training data...")
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew in the sky",
        "the fish swam in the sea",
        "quick brown foxes are fast",
        "lazy dogs like to sleep",
        "cats are cute animals",
        "birds can fly high",
        "fish live in water",
        "the sun shines bright",
        "the moon glows at night",
        "stars twinkle in the sky",
        "clouds float above us",
        "rain falls from clouds",
        "snow is white and cold",
        "wind blows through trees",
        "flowers bloom in spring",
        "leaves fall in autumn",
        "winter brings the snow"
    ]
    
    print(f"   Training corpus: {len(texts)} sentences")
    
    # Build tokenizer
    print("\n2. Building character-level tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset
    print("\n3. Creating training dataset...")
    dataset = TextDataset(texts, tokenizer, max_seq_len=64)
    print(f"   Training sequences: {len(dataset)}")
    
    # Create model
    print("\n4. Creating Neural Field Language Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=8)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Create data loader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train
    print("\n5. Training the model...")
    print("-" * 70)
    
    epochs = 15
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")
    
    print("-" * 70)
    
    # Test generation
    print("\n6. Testing text generation...")
    print("-" * 70)
    
    model.eval()
    test_prompts = ["the ", "quick ", "cat "]
    
    for prompt_text in test_prompts:
        prompt_tokens = tokenizer.encode(prompt_text, add_eos=False)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_length=30, temperature=0.8)
        
        generated_text = tokenizer.decode(generated.cpu().tolist())
        print(f"\nPrompt: '{prompt_text}'")
        print(f"Generated: '{generated_text}'")
    
    print("-" * 70)
    
    # Test interpolation
    print("\n7. Testing sequence interpolation...")
    print("-" * 70)
    
    seq1_text = "the cat"
    seq2_text = "the dog"
    
    seq1_tokens = torch.tensor(tokenizer.encode(seq1_text, add_eos=False), dtype=torch.long).to(device)
    seq2_tokens = torch.tensor(tokenizer.encode(seq2_text, add_eos=False), dtype=torch.long).to(device)
    
    print(f"\nInterpolating between:")
    print(f"  Sequence 1: '{seq1_text}'")
    print(f"  Sequence 2: '{seq2_text}'")
    print()
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with torch.no_grad():
            interp_tokens = model.interpolate_sequences(seq1_tokens, seq2_tokens, alpha=alpha)
        interp_text = tokenizer.decode(interp_tokens.cpu().tolist())
        print(f"  α={alpha:.2f}: '{interp_text}'")
    
    print("-" * 70)
    
    # Save model
    print("\n8. Saving model...")
    save_dir = Path("./trained_models/quick_example")
    save_checkpoint(model, tokenizer, optimizer, epochs, avg_loss, save_dir)
    print(f"✓ Model saved to: {save_dir}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Character-level tokenization")
    print("  ✓ Training on simple text corpus")
    print("  ✓ Loss reduction during training")
    print("  ✓ Autoregressive text generation")
    print("  ✓ Smooth interpolation in coordinate space")
    print("  ✓ Model persistence")
    print("\nNext Steps:")
    print("  • Train on larger datasets with scripts/train_nflm.py")
    print("  • Use WikiText-103, FineWeb-Edu, or other HuggingFace datasets")
    print("  • Experiment with hyperparameters")
    print("  • Test interpolation quality")
    print("\nSee TRAINING.md for full documentation.")
    print("=" * 70)


if __name__ == "__main__":
    main()
