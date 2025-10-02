#!/usr/bin/env python3
"""
Interactive chat interface with trained neural field language model.
Allows conversational interaction with the trained model.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from virgo import NeuralFieldLM, CharTokenizer


def load_model(model_path, device='cpu'):
    """Load trained model and tokenizer."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract model config
    vocab_size = checkpoint['vocab_size']
    coord_dim = checkpoint.get('coord_dim', 8)
    
    # Create model
    model = NeuralFieldLM(vocab_size=vocab_size, coord_dim=coord_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create tokenizer
    tokenizer = CharTokenizer()
    tokenizer.char_to_idx = checkpoint['char_to_idx']
    tokenizer.idx_to_char = checkpoint['idx_to_char']
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=100, temperature=0.8, device='cpu'):
    """Generate a response from the model."""
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt, add_eos=False)
    input_ids = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=temperature)
    
    # Decode
    response = tokenizer.decode(output_ids.cpu().tolist())
    return response


def main():
    if len(sys.argv) < 2:
        print("Usage: python chat_with_model.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading model...")
    try:
        model, tokenizer = load_model(model_path, device)
        print(f"✓ Model loaded successfully!")
        print(f"  Vocabulary size: {tokenizer.vocab_size}")
        print(f"  Coordinate dimension: {model.coord_dim}D")
        print(f"  Device: {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("Virgo Neural Field Conversational Interface")
    print("=" * 70)
    print()
    print("Type your messages and the neural field will respond.")
    print("Type 'quit' to exit.")
    print()
    print("=" * 70)
    print()
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            # Add to conversation history
            conversation_history += f"User: {user_input}\nAssistant: "
            
            # Generate response using the conversation history as prompt
            response = generate_response(
                model, 
                tokenizer, 
                conversation_history[-500:],  # Use last 500 chars as context
                max_length=150,
                temperature=0.8,
                device=device
            )
            
            # Extract just the new response part (after the prompt)
            prompt_len = len(tokenizer.encode(conversation_history[-500:], add_eos=False))
            response_tokens = tokenizer.encode(response, add_eos=False)
            new_response_tokens = response_tokens[prompt_len:]
            new_response = tokenizer.decode(new_response_tokens)
            
            # Clean up response (take first sentence or line)
            if '\n' in new_response:
                new_response = new_response.split('\n')[0]
            new_response = new_response.strip()
            
            if not new_response:
                new_response = response[:50]  # Fallback to first 50 chars
            
            print(f"Assistant: {new_response}")
            print()
            
            # Update conversation history
            conversation_history += new_response + "\n"
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Continuing...")


if __name__ == "__main__":
    main()
