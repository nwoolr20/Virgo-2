"""Comprehensive evaluation of neural field system."""

import json
import gzip
import tempfile
import shutil
from pathlib import Path
from virgo.memory import MemorySystem


def evaluate_compression():
    """Compare storage efficiency."""
    print("\n" + "=" * 60)
    print("COMPRESSION EVALUATION")
    print("=" * 60)
    
    # Create test conversation
    system = MemorySystem()
    
    conversations = [
        ("My name is Alice", 0),
        ("I work as a software engineer", 0),
        ("I have two cats named Whiskers and Mittens", 0),
        ("Nice to meet you, Alice!", 1),
        ("What do you do for work?", 1),
        ("Tell me about your cats", 1),
    ] * 50  # Repeat to simulate longer conversation
    
    for text, speaker in conversations:
        system.store(text, speaker)
    
    system.fit_field(num_steps=3000, verbose=True)
    
    # Save neural field system
    temp_dir = Path(tempfile.mkdtemp())
    try:
        system.save(temp_dir)
        
        # Calculate sizes
        nf_size = sum(f.stat().st_size for f in temp_dir.rglob("*") if f.is_file())
        
        # Compare to JSON
        json_data = [{"text": m.text, "speaker": m.speaker_id} for m in system.memories]
        json_bytes = json.dumps(json_data).encode()
        json_size = len(json_bytes)
        json_gzip_size = len(gzip.compress(json_bytes))
        
        print("\n=== Storage Comparison ===")
        print(f"Memories stored: {len(system.memories)}")
        print(f"Raw JSON: {json_size:,} bytes")
        print(f"Gzipped JSON: {json_gzip_size:,} bytes")
        print(f"Neural Field: {nf_size:,} bytes")
        print(f"\nCompression ratio (vs JSON): {json_size / nf_size:.2f}x")
        print(f"Compression ratio (vs gzip): {json_gzip_size / nf_size:.2f}x")
        
        # Test retrieval accuracy
        test_queries = [
            ("What is my name?", "Alice"),
            ("What are my cats called?", "Whiskers"),
            ("What is my job?", "engineer"),
        ]
        
        print("\n=== Retrieval Tests ===")
        correct = 0
        for query, expected_word in test_queries:
            results = system.retrieve(query, k=3)
            found = any(expected_word.lower() in m.text.lower() for m, _ in results)
            status = "✓" if found else "✗"
            print(f"{status} Query: '{query}' - Expected: '{expected_word}'")
            if found:
                correct += 1
        
        accuracy = correct / len(test_queries)
        print(f"\nRetrieval accuracy: {accuracy * 100:.1f}%")
        
        return {
            "compression_vs_json": json_size / nf_size,
            "compression_vs_gzip": json_gzip_size / nf_size,
            "retrieval_accuracy": accuracy
        }
    finally:
        shutil.rmtree(temp_dir)


def evaluate_persistence():
    """Test save/load functionality."""
    print("\n" + "=" * 60)
    print("PERSISTENCE EVALUATION")
    print("=" * 60)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Phase 1: Create and save
        print("\nPhase 1: Creating and saving system...")
        system1 = MemorySystem()
        
        test_data = [
            ("My favorite color is blue", 0),
            ("I like pizza", 0),
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
