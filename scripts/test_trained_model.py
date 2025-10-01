#!/usr/bin/env python3
"""
Load and test a trained Neural Field Language Model

Usage:
    python scripts/test_trained_model.py ./trained_models/final_model.pt
    python scripts/test_trained_model.py ./trained_models/best_model.pt --prompts "the" "hello" "in"
"""

import argparse
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
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
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Coordinate dimension: {checkpoint['coord_dim']}")
    
    return model, tokenizer


def test_generation(model, tokenizer, prompts, device, max_length=100, temperature=0.8):
    """Test text generation"""
    print("\n" + "="*70)
    print("Testing Text Generation")
    print("="*70)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt, add_eos=False)
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(prompt_tensor, max_length=max_length, temperature=temperature)
        
        # Decode
        generated_text = tokenizer.decode(generated.cpu().tolist())
        print(f"Generated: '{generated_text}'")
        print(f"Length: {len(generated)} tokens")
    
    print("="*70)


def test_interpolation(model, tokenizer, device):
    """Test sequence interpolation"""
    print("\n" + "="*70)
    print("Testing Sequence Interpolation")
    print("="*70)
    
    # Test pairs
    test_pairs = [
        ("the cat", "the dog"),
        ("hello world", "hi there"),
        ("in the", "on the"),
    ]
    
    for seq1_text, seq2_text in test_pairs:
        print(f"\nInterpolating between:")
        print(f"  A: '{seq1_text}'")
        print(f"  B: '{seq2_text}'")
        print()
        
        seq1_tokens = torch.tensor(tokenizer.encode(seq1_text, add_eos=False), dtype=torch.long)
        seq2_tokens = torch.tensor(tokenizer.encode(seq2_text, add_eos=False), dtype=torch.long)
        
        # Make sequences same length
        if len(seq1_tokens) < len(seq2_tokens):
            pad_len = len(seq2_tokens) - len(seq1_tokens)
            seq1_tokens = torch.cat([seq1_tokens, torch.full((pad_len,), tokenizer.char_to_idx[tokenizer.pad_token])])
        elif len(seq2_tokens) < len(seq1_tokens):
            pad_len = len(seq1_tokens) - len(seq2_tokens)
            seq2_tokens = torch.cat([seq2_tokens, torch.full((pad_len,), tokenizer.char_to_idx[tokenizer.pad_token])])
        
        seq1_tokens = seq1_tokens.to(device)
        seq2_tokens = seq2_tokens.to(device)
        
        # Interpolate
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            with torch.no_grad():
                interp_tokens = model.interpolate_sequences(seq1_tokens, seq2_tokens, alpha=alpha)
            interp_text = tokenizer.decode(interp_tokens.cpu().tolist())
            print(f"  α={alpha:.2f}: '{interp_text}'")
    
    print("="*70)


def analyze_coordinates(model, tokenizer, device):
    """Analyze coordinate space"""
    print("\n" + "="*70)
    print("Analyzing Coordinate Space")
    print("="*70)
    
    test_phrases = [
        "the cat",
        "the dog", 
        "the bird",
        "hello world",
        "hi there",
        "good morning"
    ]
    
    print("\nCoordinate embeddings for different phrases:")
    
    with torch.no_grad():
        for phrase in test_phrases:
            tokens = torch.tensor(tokenizer.encode(phrase, add_eos=False), dtype=torch.long).to(device)
            coords = model.coord_encoder(tokens.unsqueeze(0))[0]  # [seq_len, coord_dim]
            
            # Average coordinates over sequence
            avg_coord = coords.mean(dim=0).cpu().numpy()
            
            print(f"\n  '{phrase}':")
            print(f"    Shape: {coords.shape}")
            print(f"    Avg: [{', '.join([f'{x:.3f}' for x in avg_coord])}]")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Test trained Neural Field Language Model")
    parser.add_argument("model_path", type=str, help="Path to trained model checkpoint")
    parser.add_argument("--prompts", nargs="+", default=["the", "hello", "in the"],
                       help="Prompts for text generation")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--skip-interpolation", action="store_true",
                       help="Skip interpolation tests")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip coordinate analysis")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # Load model
    model, tokenizer = load_model(args.model_path, device)
    
    # Test generation
    test_generation(model, tokenizer, args.prompts, device, args.max_length, args.temperature)
    
    # Test interpolation
    if not args.skip_interpolation:
        test_interpolation(model, tokenizer, device)
    
    # Analyze coordinates
    if not args.skip_analysis:
        analyze_coordinates(model, tokenizer, device)
    
    print("\n✓ All tests complete!")


if __name__ == "__main__":
    main()
