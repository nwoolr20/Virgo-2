"""Comprehensive evaluation of neural field language model."""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer


def evaluate_generation_quality(model, tokenizer, prompts, device='cpu'):
    """Evaluate text generation quality."""
    print("\n" + "=" * 60)
    print("GENERATION QUALITY EVALUATION")
    print("=" * 60)
    
    model.eval()
    results = []
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode
        tokens = tokenizer.encode(prompt, add_eos=False)
        input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=100, temperature=0.8)
        
        generated = tokenizer.decode(output_ids.cpu().tolist())
        print(f"Generated: '{generated}'")
        
        results.append({
            "prompt": prompt,
            "generated": generated,
            "length": len(output_ids)
        })
    
    return results


def evaluate_interpolation(model, tokenizer, device='cpu'):
    """Test coordinate space interpolation."""
    print("\n" + "=" * 60)
    print("INTERPOLATION EVALUATION")
    print("=" * 60)
    
    model.eval()
    
    test_pairs = [
        ("the cat", "the dog"),
        ("hello world", "hi there"),
        ("good morning", "good evening"),
    ]
    
    for text1, text2 in test_pairs:
        print(f"\nInterpolating: '{text1}' → '{text2}'")
        
        tokens1 = torch.tensor(tokenizer.encode(text1, add_eos=False), dtype=torch.long).to(device)
        tokens2 = torch.tensor(tokenizer.encode(text2, add_eos=False), dtype=torch.long).to(device)
        
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            with torch.no_grad():
                result = model.interpolate_sequences(tokens1, tokens2, alpha=alpha)
            interpolated = tokenizer.decode(result.cpu().tolist())
            print(f"  α={alpha:.2f}: '{interpolated}'")


def main():
    print("=" * 60)
    print("NEURAL FIELD SYSTEM EVALUATION")
    print("=" * 60)
    
    # Check for trained model
    model_path = Path("./trained_models/virgo_model/best_model.pt")
    if not model_path.exists():
        print(f"\n✗ No trained model found at {model_path}")
        print("Please train a model first:")
        print("  python3 launch_virgo.py train model")
        return
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nLoading model from: {model_path}")
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    vocab_size = checkpoint['vocab_size']
    coord_dim = checkpoint.get('coord_dim', 8)
    
    model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=coord_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    
    print(f"✓ Model loaded")
    print(f"  Vocabulary: {vocab_size}")
    print(f"  Coordinates: {coord_dim}D")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate generation
    test_prompts = [
        "the",
        "hello",
        "artificial intelligence",
        "in the",
        "once upon a time",
    ]
    
    generation_results = evaluate_generation_quality(model, tokenizer, test_prompts, device)
    
    # Evaluate interpolation
    evaluate_interpolation(model, tokenizer, device)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

            ("I live in Seattle", 0),
        ]
        
        for text, speaker in test_data:
            system1.store(text, speaker_id=speaker)
        
        system1.fit_field(num_steps=1000, verbose=False)
        system1.save(temp_dir)
        print("✓ System saved")
        
        # Phase 2: Load and verify
        print("\nPhase 2: Loading in new instance...")
        system2 = MemorySystem()
        system2.load(temp_dir)
        print("✓ System loaded")
        
        # Verify data
        assert len(system2.memories) == 3, "Memory count mismatch"
        assert system2.is_fitted, "Field not fitted after load"
        print(f"✓ Verified {len(system2.memories)} memories")
        print(f"✓ Field fitted: {system2.is_fitted}")
        
        # Test retrieval
        print("\nTesting retrieval after load...")
        test_queries = [
            ("What color do I like?", "blue"),
            ("Where do I live?", "Seattle"),
        ]
        
        for query, expected in test_queries:
            results = system2.retrieve(query, k=2)
            texts = [m.text for m, _ in results]
            found = any(expected in text for text in texts)
            status = "✓" if found else "✗"
            print(f"{status} Query: '{query}' - Expected: '{expected}'")
        
        print("\n✓ Persistence test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Persistence test FAILED: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all evaluations."""
    print("\n" + "=" * 60)
    print("NEURAL FIELD SYSTEM EVALUATION")
    print("=" * 60)
    
    # Run evaluations
    compression_results = evaluate_compression()
    persistence_passed = evaluate_persistence()
    
    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\nCompression Results:")
    print(f"  vs JSON:  {compression_results['compression_vs_json']:.2f}x")
    print(f"  vs Gzip:  {compression_results['compression_vs_gzip']:.2f}x")
    print(f"  Retrieval: {compression_results['retrieval_accuracy'] * 100:.1f}%")
    
    print("\nPersistence:")
    print(f"  {'✓ PASSED' if persistence_passed else '✗ FAILED'}")
    
    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    print("=" * 60)
    
    gzip_pass = compression_results['compression_vs_gzip'] > 1.5
    accuracy_pass = compression_results['retrieval_accuracy'] > 0.75
    
    print(f"  Compression vs Gzip > 1.5x: {'✓ PASS' if gzip_pass else '✗ FAIL'}")
    print(f"  Retrieval Accuracy > 75%:   {'✓ PASS' if accuracy_pass else '✗ FAIL'}")
    print(f"  Persistence:                {'✓ PASS' if persistence_passed else '✗ FAIL'}")
    
    overall_pass = gzip_pass and accuracy_pass and persistence_passed
    
    print("\n" + "=" * 60)
    if overall_pass:
        print("✓ OVERALL: SYSTEM VIABLE")
    else:
        print("✗ OVERALL: SYSTEM NEEDS IMPROVEMENT")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
