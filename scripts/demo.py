"""Interactive demo of neural field system."""

from virgo.memory import MemorySystem


def demo():
    """Run interactive demo."""
    print("=" * 60)
    print("Neural Field Memory System - Demo")
    print("=" * 60)
    
    system = MemorySystem()
    
    # Add some initial memories
    print("\nAdding initial memories...")
    conversations = [
        ("My name is Alice", 0),
        ("I work as a software engineer at Google", 0),
        ("I have two cats named Whiskers and Mittens", 0),
        ("My favorite programming language is Python", 0),
        ("I live in San Francisco", 0),
    ]
    
    for text, speaker in conversations:
        system.store(text, speaker_id=speaker)
        print(f"  Stored: {text}")
    
    # Train the field
    print("\nTraining neural field...")
    system.fit_field(num_steps=1000, verbose=False)
    print("✓ Field trained")
    
    # Test queries
    print("\n" + "=" * 60)
    print("Testing Retrieval")
    print("=" * 60)
    
    queries = [
        "What is my name?",
        "Where do I work?",
        "Tell me about my pets",
        "What programming language do I like?",
        "Where do I live?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = system.retrieve(query, k=2)
        print("Top results:")
        for i, (memory, distance) in enumerate(results, 1):
            print(f"  {i}. [dist={distance:.2f}] {memory.text}")
    
    # Show stats
    print("\n" + "=" * 60)
    print("System Statistics")
    print("=" * 60)
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    demo()
