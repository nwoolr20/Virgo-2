#!/usr/bin/env python3
"""
Demonstration of Neural Field Language Model

This script shows how to:
1. Create and train a neural field LM
2. Generate text autoregressively
3. Interpolate between sequences in coordinate space
"""

import torch
from virgo import NeuralFieldLM, CharTokenizer, train_neural_field_lm


def main():
    print("=" * 70)
    print("Neural Field Language Model Demo")
    print("=" * 70)
    
    # Create simple training corpus
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
        "fish live in water"
    ]
    
    print(f"   Training corpus: {len(texts)} sentences")
    
    # Build tokenizer
    print("\n2. Building character-level tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Prepare training data (next-token prediction)
    print("\n3. Preparing training batches...")
    train_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_eos=False)
        if len(tokens) > 1:
            # Input: all tokens except last, Target: all tokens except first
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
            train_data.append((input_ids, target_ids))
    
    print(f"   Training batches: {len(train_data)}")
    
    # Create model
    print("\n4. Creating Neural Field Language Model...")
    model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=8)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Coordinate dimensions: 8")
    
    # Train the model
    print("\n5. Training the model...")
    print("-" * 70)
    train_neural_field_lm(model, train_data, epochs=20, lr=1e-3)
    print("-" * 70)
    
    # Test 1: Autoregressive generation
    print("\n6. Testing autoregressive generation...")
    print("-" * 70)
    
    # Generate from different prompts
    prompts = ["the ", "quick ", "cat "]
    for prompt_text in prompts:
        prompt_tokens = tokenizer.encode(prompt_text, add_eos=False)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long)
        
        generated = model.generate(prompt_tensor, max_length=30, temperature=0.8)
        generated_text = tokenizer.decode(generated.tolist())
        
        print(f"\nPrompt: '{prompt_text}'")
        print(f"Generated: '{generated_text}'")
    
    print("-" * 70)
    
    # Test 2: Sequence interpolation
    print("\n7. Testing sequence interpolation in coordinate space...")
    print("-" * 70)
    
    # Choose two sequences to interpolate between
    seq1_text = "the cat"
    seq2_text = "the dog"
    
    seq1_tokens = torch.tensor(tokenizer.encode(seq1_text, add_eos=False), dtype=torch.long)
    seq2_tokens = torch.tensor(tokenizer.encode(seq2_text, add_eos=False), dtype=torch.long)
    
    print(f"\nInterpolating between:")
    print(f"  Sequence 1: '{seq1_text}'")
    print(f"  Sequence 2: '{seq2_text}'")
    print()
    
    # Interpolate at different alphas
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interp_tokens = model.interpolate_sequences(seq1_tokens, seq2_tokens, alpha=alpha)
        interp_text = tokenizer.decode(interp_tokens.tolist())
        print(f"  α={alpha:.2f}: '{interp_text}'")
    
    print("-" * 70)
    
    # Test 3: Coordinate space visualization
    print("\n8. Analyzing coordinate space...")
    print("-" * 70)
    
    test_phrases = ["the cat", "the dog", "the bird"]
    print("\nCoordinate embeddings for different phrases:")
    
    with torch.no_grad():
        for phrase in test_phrases:
            tokens = torch.tensor(tokenizer.encode(phrase, add_eos=False), dtype=torch.long)
            coords = model.coord_encoder(tokens.unsqueeze(0))[0]  # [seq_len, coord_dim]
            
            # Average coordinates over sequence
            avg_coord = coords.mean(dim=0)
            
            print(f"\n  '{phrase}':")
            print(f"    Avg coordinate: {avg_coord.numpy()}")
    
    print("-" * 70)
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Learned coordinate encoding from text")
    print("  ✓ Neural field generates token distributions")
    print("  ✓ Autoregressive text generation")
    print("  ✓ Smooth interpolation in coordinate space")
    print("  ✓ Continuous semantic representations")
    print("\nThis is a TRUE neural field language model!")
    print("=" * 70)


if __name__ == "__main__":
    main()
