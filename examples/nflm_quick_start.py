#!/usr/bin/env python3
"""
Quick example of using the Neural Field Language Model
"""

import torch
from virgo import NeuralFieldLM, CharTokenizer, train_neural_field_lm


def quick_example():
    """Simple example showing NFLM in action"""
    
    print("\n" + "="*60)
    print("NEURAL FIELD LANGUAGE MODEL - Quick Example")
    print("="*60)
    
    # 1. Create a small training dataset
    print("\n1. Creating training data...")
    texts = [
        "hello world",
        "hello there",
        "hi world",
        "hi there"
    ]
    print(f"   Texts: {texts}")
    
    # 2. Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # 3. Prepare training pairs
    print("\n3. Preparing training data...")
    train_data = []
    for text in texts:
        tokens = tokenizer.encode(text, add_eos=False)
        if len(tokens) > 1:
            # Next-token prediction
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
            target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
            train_data.append((input_ids, target_ids))
    print(f"   Training batches: {len(train_data)}")
    
    # 4. Create model
    print("\n4. Creating model...")
    model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=6)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {params:,}")
    
    # 5. Train
    print("\n5. Training (10 epochs)...")
    train_neural_field_lm(model, train_data, epochs=10, lr=1e-3)
    
    # 6. Generate
    print("\n6. Generating text...")
    prompt = "h"
    prompt_tokens = torch.tensor(tokenizer.encode(prompt, add_eos=False), dtype=torch.long)
    generated = model.generate(prompt_tokens, max_length=10, temperature=1.0)
    result = tokenizer.decode(generated.tolist())
    print(f"   Prompt: '{prompt}'")
    print(f"   Generated: '{result}'")
    
    # 7. Interpolate
    print("\n7. Interpolating between sequences...")
    seq1 = torch.tensor(tokenizer.encode("hello", add_eos=False), dtype=torch.long)
    seq2 = torch.tensor(tokenizer.encode("hi wo", add_eos=False), dtype=torch.long)
    
    interp = model.interpolate_sequences(seq1, seq2, alpha=0.5)
    interp_text = tokenizer.decode(interp.tolist())
    
    print(f"   Seq1: 'hello'")
    print(f"   Seq2: 'hi wo'")
    print(f"   Interpolated (α=0.5): '{interp_text}'")
    
    print("\n" + "="*60)
    print("DONE! This is a TRUE generative neural field LM.")
    print("="*60 + "\n")


if __name__ == "__main__":
    quick_example()
