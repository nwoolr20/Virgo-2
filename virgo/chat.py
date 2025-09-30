"""Simple chat interface for testing."""

from pathlib import Path
from typing import Optional
from .memory import MemorySystem


class SimpleChat:
    """Minimal chat interface with neural field memory."""
    
    def __init__(self, memory_path: Optional[Path] = None):
        """
        Args:
            memory_path: Path to persist memory (optional)
        """
        self.memory = MemorySystem()
        self.memory_path = memory_path or Path("./memory_store")
        
        if self.memory_path.exists():
            print("Loading existing memories...")
            try:
                self.memory.load(self.memory_path)
                stats = self.memory.get_stats()
                print(f"Loaded {stats['total_memories']} memories")
                print(f"Field fitted: {stats['is_fitted']}")
            except Exception as e:
                print(f"Failed to load memories: {e}")
                print("Starting fresh")
    
    def respond(self, user_input: str) -> str:
        """
        Generate response with retrieved context.
        
        Args:
            user_input: User message
            
        Returns:
            Assistant response
        """
        # Store user message
        self.memory.store(user_input, speaker_id=0)
        
        # Retrieve relevant context
        if len(self.memory.memories) > 1:
            relevant = self.memory.retrieve(user_input, k=3)
            
            # Build context
            context_lines = []
            for memory, distance in relevant:
                if memory.text != user_input:  # Don't include current message
                    context_lines.append(
                        f"[Turn {memory.turn_id}, dist={distance:.2f}] {memory.text}"
                    )
            
            context = "\n".join(context_lines) if context_lines else "No relevant history"
        else:
            context = "First message"
        
        # Simple template response (replace with real LM later)
        response = f"I understand you said: '{user_input}'\n\nRelevant context:\n{context}"
        
        # Store assistant response
        self.memory.store(response, speaker_id=1)
        
        return response
    
    def save_memory(self):
        """Fit and save memory system."""
        if len(self.memory.memories) < 2:
            print("Not enough memories to train field")
            return
            
        print("\nFitting neural field...")
        self.memory.fit_field(num_steps=2000, verbose=True)
        
        print(f"\nSaving to {self.memory_path}...")
        self.memory.save(self.memory_path)
        print("Saved!")
    
    def run(self):
        """Interactive chat loop."""
        print("=" * 60)
        print("Neural Field Chat Interface")
        print("=" * 60)
        print("Commands:")
        print("  'quit' - Exit")
        print("  'save' - Fit field and persist memory")
        print("  'stats' - Show memory statistics")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                    
                elif user_input.lower() == 'save':
                    self.save_memory()
                    continue
                    
                elif user_input.lower() == 'stats':
                    stats = self.memory.get_stats()
                    print("\n--- Memory Statistics ---")
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                    continue
                
                response = self.respond(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
            except Exception as e:
                print(f"\nError: {e}")
                print("Continuing...")


def main():
    """Entry point."""
    import sys
    
    memory_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    chat = SimpleChat(memory_path)
    chat.run()


if __name__ == "__main__":
    main()
