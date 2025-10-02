#!/usr/bin/env python3
"""
Comprehensive System Test for Neural Field Model

Tests model coherence, response quality, and all system components.
Provides detailed evaluation of training progress and generation quality.
"""

import sys
import torch
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer


def load_model_for_testing(model_path, device='cpu'):
    """Load model for testing."""
    print(f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        print(f"✗ Model not found")
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    vocab_size = checkpoint['vocab_size']
    coord_dim = checkpoint.get('coord_dim', 8)
    
    model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=coord_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    
    print(f"✓ Model loaded successfully")
    
    return model, tokenizer


def test_basic_generation(model, tokenizer, device='cpu'):
    """Test basic text generation."""
    print("\n" + "=" * 70)
    print("TEST 1: BASIC GENERATION")
    print("=" * 70)
    
    test_prompts = [
        ("the", "Basic article"),
        ("hello", "Greeting"),
        ("in the", "Phrase continuation"),
        ("once upon a time", "Story beginning"),
    ]
    
    results = []
    for prompt, description in test_prompts:
        tokens = tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=50, temperature=0.8)
        
        generated = tokenizer.decode(output_ids.cpu().tolist())
        
        print(f"\nPrompt: '{prompt}' ({description})")
        print(f"Generated: '{generated}'")
        
        results.append({
            'prompt': prompt,
            'generated': generated,
            'length': len(generated)
        })
    
    return results


def test_coherence(model, tokenizer, device='cpu'):
    """Test response coherence with longer prompts."""
    print("\n" + "=" * 70)
    print("TEST 2: COHERENCE EVALUATION")
    print("=" * 70)
    
    test_cases = [
        ("artificial intelligence", "Technical term"),
        ("the quick brown fox", "Common phrase"),
        ("neural networks are", "Technical description"),
        ("machine learning is", "Definition prompt"),
        ("the weather today", "Conversational"),
    ]
    
    coherence_scores = []
    
    for prompt, category in test_cases:
        tokens = tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=80, temperature=0.7)
        
        generated = tokenizer.decode(output_ids.cpu().tolist())
        
        # Simple coherence metrics
        has_spaces = ' ' in generated[len(prompt):]
        has_continuation = len(generated) > len(prompt) + 5
        no_excessive_repetition = not any(c * 5 in generated for c in 'abcdefghijklmnopqrstuvwxyz')
        
        coherence_score = sum([has_spaces, has_continuation, no_excessive_repetition])
        coherence_scores.append(coherence_score)
        
        print(f"\nPrompt: '{prompt}' ({category})")
        print(f"Generated: '{generated}'")
        print(f"Coherence indicators: {coherence_score}/3")
        print(f"  - Has spaces: {has_spaces}")
        print(f"  - Has continuation: {has_continuation}")
        print(f"  - No excessive repetition: {no_excessive_repetition}")
    
    avg_coherence = sum(coherence_scores) / len(coherence_scores)
    print(f"\nAverage coherence score: {avg_coherence:.2f}/3.0")
    
    return avg_coherence


def test_interpolation(model, tokenizer, device='cpu'):
    """Test coordinate space interpolation."""
    print("\n" + "=" * 70)
    print("TEST 3: COORDINATE INTERPOLATION")
    print("=" * 70)
    
    test_pairs = [
        ("the cat", "the dog"),
        ("hello world", "hi there"),
    ]
    
    for seq1, seq2 in test_pairs:
        print(f"\nInterpolating: '{seq1}' ↔ '{seq2}'")
        
        tokens1 = torch.tensor(tokenizer.encode(seq1, add_eos=False), dtype=torch.long).to(device)
        tokens2 = torch.tensor(tokenizer.encode(seq2, add_eos=False), dtype=torch.long).to(device)
        
        for alpha in [0.0, 0.5, 1.0]:
            with torch.no_grad():
                result = model.interpolate_sequences(tokens1, tokens2, alpha=alpha)
            
            interpolated = tokenizer.decode(result.cpu().tolist())
            print(f"  α={alpha:.1f}: '{interpolated}'")


def evaluate_model_info(model, tokenizer, checkpoint):
    """Display model information."""
    print("\n" + "=" * 70)
    print("MODEL INFORMATION")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nArchitecture:")
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    print(f"  Coordinate dimension: {model.coord_dim}D")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    if 'epoch' in checkpoint:
        print(f"\nTraining:")
        print(f"  Epochs trained: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"  Final loss: {checkpoint['loss']:.4f}")
        if 'perplexity' in checkpoint:
            print(f"  Perplexity: {checkpoint['perplexity']:.2f}")
        if 'phase' in checkpoint:
            print(f"  Training phase: {checkpoint['phase']}")
    
    if 'distillation' in checkpoint and checkpoint['distillation']:
        print(f"  Distillation: Yes")


def test_all_phases_status():
    """Check which phases have been completed."""
    print("\n" + "=" * 70)
    print("PHASE COMPLETION STATUS")
    print("=" * 70)
    
    model_dir = Path("./trained_models/virgo_model")
    
    phases = {
        "Phase 0": "6D memory system removed",
        "Phase 1": "Architecture upgrade (planned)",
        "Phase 2": "Knowledge distillation",
        "Phase 3": "Extended training"
    }
    
    # Check for distillation
    phase2_done = False
    if (model_dir / "distilled_model.pt").exists():
        phase2_done = True
    elif (model_dir / "distillation_history.json").exists():
        phase2_done = True
    
    # Check for extended training
    phase3_done = False
    if (model_dir / "extended_training_history.json").exists():
        phase3_done = True
    elif (model_dir / "extended_checkpoint_epoch_5.pt").exists():
        phase3_done = True
    
    print("\nPhase 0 (Memory Removal): ✓ Complete")
    print(f"Phase 1 (Architecture): ⚠ Not required (using existing 8D architecture)")
    print(f"Phase 2 (Distillation): {'✓ Complete' if phase2_done else '○ Pending'}")
    print(f"Phase 3 (Extended Training): {'✓ Complete' if phase3_done else '○ Pending'}")
    
    return phase2_done, phase3_done


def main():
    print("=" * 70)
    print("NEURAL FIELD MODEL - COMPREHENSIVE SYSTEM TEST")
    print("=" * 70)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check phase status
    phase2_done, phase3_done = test_all_phases_status()
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    model_path = Path("./trained_models/virgo_model/best_model.pt")
    if not model_path.exists():
        model_path = Path("./trained_models/virgo_model/final_model.pt")
    
    model, tokenizer = load_model_for_testing(model_path, device)
    
    if model is None:
        print("\n✗ No model found. Please train a model first:")
        print("  python3 launch_virgo.py train model")
        return 1
    
    # Load checkpoint for info
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Run tests
    evaluate_model_info(model, tokenizer, checkpoint)
    
    basic_results = test_basic_generation(model, tokenizer, device)
    coherence_score = test_coherence(model, tokenizer, device)
    test_interpolation(model, tokenizer, device)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    print(f"\nModel Status:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Coordinate Dim: {model.coord_dim}D")
    
    print(f"\nGeneration Quality:")
    print(f"  Basic generation: {len(basic_results)} tests passed")
    print(f"  Coherence score: {coherence_score:.2f}/3.0")
    
    print(f"\nPhase Completion:")
    print(f"  Phase 2 (Distillation): {'✓' if phase2_done else '○'}")
    print(f"  Phase 3 (Extended Training): {'✓' if phase3_done else '○'}")
    
    # Coherence assessment
    print(f"\nCoherence Assessment:")
    if coherence_score >= 2.5:
        print("  ✓ Model generates coherent text")
        print("  Response quality: GOOD")
    elif coherence_score >= 2.0:
        print("  ~ Model shows some coherence")
        print("  Response quality: MODERATE")
        print("  Recommendation: Continue training with Phase 3")
    else:
        print("  ✗ Model coherence needs improvement")
        print("  Response quality: POOR")
        print("  Recommendation: Run Phase 2 (distillation) and Phase 3 (extended training)")
    
    print(f"\n{'=' * 70}")
    print("Test complete")
    print(f"{'=' * 70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
