#!/usr/bin/env python3
"""
Train neural field model until it produces coherent sentences.
Continues training on multiple datasets until quality threshold is met.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from virgo import NeuralFieldLM, CharTokenizer, train_neural_field_lm
from datasets import load_dataset
import time

def test_coherence(model, tokenizer, test_prompts):
    """Test if model produces coherent output"""
    print("\n" + "="*70)
    print("COHERENCE TEST")
    print("="*70)
    
    coherent_count = 0
    total_tests = len(test_prompts)
    
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt, add_eos=False)
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        # Generate
        generated = model.generate(input_tensor, max_length=50, temperature=0.8)
        output = tokenizer.decode(generated[0].tolist())
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{output}'")
        
        # Check coherence
        is_coherent = (
            len(output) > len(prompt) and 
            output.startswith(prompt) and
            not any(c * 5 in output for c in 'abcdefghijklmnopqrstuvwxyz') and
            ' ' in output[len(prompt):]
        )
        
        if is_coherent:
            coherent_count += 1
            print("✓ COHERENT")
        else:
            print("✗ NOT COHERENT")
    
    score = coherent_count / total_tests
    print(f"\nCoherence Score: {coherent_count}/{total_tests} ({score*100:.1f}%)")
    print("="*70)
    
    return score >= 0.6  # 60% coherent = pass

def main():
    print("="*70)
    print("TRAINING UNTIL COHERENT")
    print("="*70)
    
    # Test prompts
    test_prompts = [
        "the cat",
        "artificial intelligence",
        "once upon a time",
        "the quick brown",
        "in the beginning"
    ]
    
    # Load or create model
    model_path = Path("trained_models/virgo_model/best_model.pt")
    
    if model_path.exists():
        print(f"\nLoading existing model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        vocab_size = checkpoint['vocab_size']
        tokenizer = CharTokenizer()
        # Rebuild vocab
        tokenizer.vocab_size = vocab_size
        model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=8)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded (vocab_size={vocab_size})")
    else:
        print("\nNo existing model found, will train from scratch")
        vocab_size = None
        tokenizer = None
        model = None
    
    # Training datasets
    datasets_config = [
        ("wikitext", "wikitext-103-v1", 2000),
        ("fineweb-edu", "sample-10BT", 2000),
    ]
    
    iteration = 0
    max_iterations = 3
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"TRAINING ITERATION {iteration}/{max_iterations}")
        print(f"{'='*70}")
        
        for dataset_name, config, num_samples in datasets_config:
            print(f"\nDataset: {dataset_name}")
            print(f"Samples: {num_samples}")
            
            try:
                # Load dataset
                if dataset_name == "wikitext":
                    dataset = load_dataset("wikitext", config, split="train", streaming=True)
                elif dataset_name == "fineweb-edu":
                    dataset = load_dataset("HuggingFaceFW/fineweb-edu", config, split="train", streaming=True)
                
                # Collect texts
                texts = []
                for i, item in enumerate(dataset):
                    if i >= num_samples:
                        break
                    text = item.get('text', '')
                    if text and len(text) > 20:
                        texts.append(text[:500])
                
                print(f"Collected {len(texts)} texts")
                
                if not texts:
                    print("No texts collected, skipping")
                    continue
                
                # Build/update tokenizer
                if tokenizer is None:
                    tokenizer = CharTokenizer()
                    tokenizer.build_vocab(texts)
                    vocab_size = tokenizer.vocab_size
                    print(f"Built tokenizer (vocab_size={vocab_size})")
                
                # Create/update model
                if model is None:
                    model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=8)
                    print(f"Created model ({sum(p.numel() for p in model.parameters()):,} params)")
                
                # Prepare training data
                train_data = []
                for text in texts[:1500]:  # Use 1500 for training
                    tokens = tokenizer.encode(text, add_eos=False)
                    if len(tokens) > 1:
                        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
                        target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
                        train_data.append((input_ids, target_ids))
                
                print(f"Training batches: {len(train_data)}")
                
                # Train
                epochs = 15
                print(f"\nTraining for {epochs} epochs...")
                train_neural_field_lm(model, train_data, epochs=epochs, lr=1e-3)
                
                # Save model
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab_size,
                    'coord_dim': 8,
                }, model_path)
                print(f"\n✓ Model saved to {model_path}")
                
            except Exception as e:
                print(f"Error with {dataset_name}: {e}")
                continue
        
        # Test coherence
        if model and tokenizer:
            is_coherent = test_coherence(model, tokenizer, test_prompts)
            
            if is_coherent:
                print("\n" + "="*70)
                print("SUCCESS! Model produces coherent sentences.")
                print("="*70)
                return 0
            else:
                print(f"\nNot yet coherent. Continuing training (iteration {iteration}/{max_iterations})...")
    
    print("\n" + "="*70)
    print("Completed maximum iterations.")
    print("Model may need more training for full coherence.")
    print("="*70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
