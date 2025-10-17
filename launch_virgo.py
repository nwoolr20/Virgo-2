#!/usr/bin/env python3
"""Virgo Neural Field Language Model - Launch Script"""

import sys
import os
import subprocess
from pathlib import Path
import torch
import signal
import threading


def print_header(text):
    """Print a formatted header."""
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()


def check_installation():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'sentence_transformers',
        'faiss',
        'sklearn',
        'textblob',
        'nltk',
        'pytest'
    ]
    
    try:
        for package in required_packages:
            __import__(package)
        return True
    except ImportError:
        print("Error: Required packages are not installed")
        print("Please run 'python3 install.py' first")
        return False


def load_model(model_path, device='cpu'):
    """Load trained model and tokenizer."""
    sys.path.insert(0, str(Path(__file__).parent.absolute()))
    from virgo import NeuralFieldLM, CharTokenizer
    
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


def show_usage():
    """Display usage information."""
    print_header("Virgo Neural Field Launch Script")
    print("Usage: python3 launch_virgo.py [command] [options]")
    print()
    print("Commands:")
    print("  (no command)      Launch interactive chat interface (default)")
    print()
    print("  train model       Interactive guided training setup")
    print()
    print("  chat [model_path] Start interactive chat with trained neural field")
    print("                    Optional: Specify model path (default: ./trained_models/virgo_model/best_model.pt)")
    print()
    print("  demo              Run the demo script")
    print()
    print("  evaluate          Run the evaluation script")
    print()
    print("  test              Run the test suite")
    print()
    print("  help              Show this help message")
    print()
    print("Examples:")
    print("  python3 launch_virgo.py                    # Launch chat (default)")
    print("  python3 launch_virgo.py train model        # Interactive training")
    print("  python3 launch_virgo.py chat")
    print("  python3 launch_virgo.py chat ./trained_models/my_model.pt")
    print("  python3 launch_virgo.py demo")
    print("  python3 launch_virgo.py test")
    print()


def launch_train():
    """Launch interactive guided training setup."""
    print_header("Virgo Neural Field Training Setup")
    
    if not check_installation():
        sys.exit(1)
    
    print("Welcome to the interactive training setup!")
    print()
    
    # Check for existing model
    default_model_path = Path("./trained_models/virgo_model")
    existing_model = None
    resume_training = False
    
    if default_model_path.exists():
        best_model = default_model_path / "best_model.pt"
        final_model = default_model_path / "final_model.pt"
        
        if best_model.exists() or final_model.exists():
            print("✓ Existing model found!")
            print()
            print("Options:")
            print("  1. Continue training existing model (resume)")
            print("  2. Start fresh training (will save to new directory)")
            print()
            
            while True:
                choice = input("Enter choice (1 or 2): ").strip()
                if choice == "1":
                    resume_training = True
                    existing_model = str(best_model if best_model.exists() else final_model)
                    print(f"\n✓ Will resume training from: {existing_model}")
                    break
                elif choice == "2":
                    resume_training = False
                    print("\n✓ Will start fresh training")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
    
    print()
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print()
    
    # Dataset selection
    print("Available datasets:")
    print("  1. WikiText-103 (high-quality Wikipedia articles)")
    print("  2. FineWeb-Edu (educational web content)")
    print("  3. OpenWebText (diverse web text)")
    print("  4. C4 (Colossal Clean Crawled Corpus)")
    print()
    
    dataset_options = {
        "1": "wikitext",
        "2": "fineweb-edu",
        "3": "openwebtext",
        "4": "c4"
    }
    
    while True:
        dataset_choice = input("Select dataset (1-4): ").strip()
        if dataset_choice in dataset_options:
            dataset = dataset_options[dataset_choice]
            break
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    print()
    
    # Sample size
    print("Training sample size:")
    print("  1. Quick test (100 samples, ~2 minutes)")
    print("  2. Small (1,000 samples, ~15 minutes)")
    print("  3. Medium (10,000 samples, ~2 hours)")
    print("  4. Large (50,000 samples, ~10 hours)")
    print("  5. Custom")
    print()
    
    sample_options = {
        "1": 100,
        "2": 1000,
        "3": 10000,
        "4": 50000
    }
    
    while True:
        sample_choice = input("Select sample size (1-5): ").strip()
        if sample_choice in sample_options:
            sample_size = sample_options[sample_choice]
            break
        elif sample_choice == "5":
            try:
                sample_size = int(input("Enter custom sample size: ").strip())
                if sample_size > 0:
                    break
            except ValueError:
                pass
            print("Invalid input. Please enter a positive number.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
    
    print()
    
    # Epochs
    print("Number of training epochs:")
    print("  1. Quick (5 epochs)")
    print("  2. Standard (20 epochs)")
    print("  3. Extended (50 epochs)")
    print("  4. Custom")
    print()
    
    epoch_options = {
        "1": 5,
        "2": 20,
        "3": 50
    }
    
    while True:
        epoch_choice = input("Select epochs (1-4): ").strip()
        if epoch_choice in epoch_options:
            epochs = epoch_options[epoch_choice]
            break
        elif epoch_choice == "4":
            try:
                epochs = int(input("Enter custom epoch count: ").strip())
                if epochs > 0:
                    break
            except ValueError:
                pass
            print("Invalid input. Please enter a positive number.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Dataset: {dataset}")
    print(f"  Samples: {sample_size:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Resume: {'Yes' if resume_training else 'No'}")
    print(f"  Model will be saved to: ./trained_models/virgo_model/")
    print()
    print("Controls during training:")
    print("  - Press Ctrl+C to pause/stop training")
    print("=" * 60)
    print()
    
    confirm = input("Start training? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Training cancelled.")
        return
    
    print()
    print("Starting training...")
    print()
    
    # Build command
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    train_script = script_dir / "scripts" / "train_nflm.py"
    cmd = [
        sys.executable, str(train_script),
        "--dataset", dataset,
        "--sample-size", str(sample_size),
        "--epochs", str(epochs),
        "--batch-size", "16",
        "--coord-dim", "8",
        "--save-dir", "./trained_models/virgo_model"
    ]
    
    if resume_training and existing_model:
        cmd.extend(["--resume", existing_model])
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Progress has been saved. You can resume training later.")
    
    print()
    print("Training session ended.")
    print("To resume training, run 'python3 launch_virgo.py train model' again.")


def launch_chat(model_path=None):
    """Launch the conversational chat interface with trained neural field."""
    print_header("Virgo Neural Field Chat Interface")
    
    if not check_installation():
        sys.exit(1)
    
    model_path = model_path or "./trained_models/virgo_model/best_model.pt"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"✗ Model not found at {model_path}")
        print()
        print("No trained model available. Please train a model first:")
        print("  python3 launch_virgo.py train model")
        print()
        print("Or specify a valid model path:")
        print("  python3 launch_virgo.py chat /path/to/model.pt")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from: {model_path}")
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
    print("Type your messages and the neural field will respond.")
    print("Type 'quit' to exit.")
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


def launch_demo():
    """Launch the demo script."""
    print_header("Running Virgo Demo")
    
    if not check_installation():
        sys.exit(1)
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    demo_script = script_dir / "scripts" / "demo_nflm.py"
    subprocess.run([sys.executable, str(demo_script)], env=env)


def launch_evaluate():
    """Launch the evaluation script."""
    print_header("Running Virgo Evaluation")
    
    if not check_installation():
        sys.exit(1)
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    eval_script = script_dir / "scripts" / "evaluate.py"
    subprocess.run([sys.executable, str(eval_script)], env=env)


def launch_test():
    """Launch the test suite."""
    print_header("Running Virgo Test Suite")
    
    if not check_installation():
        sys.exit(1)
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    tests_dir = script_dir / "tests"
    subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-v"],
        env=env
    )


def main():
    """Main entry point."""
    # Default to chat if no arguments provided
    if len(sys.argv) < 2:
        launch_chat()
        return
    
    command = sys.argv[1].lower()
    
    # Handle "train model" command (two words)
    if command == "train" and len(sys.argv) > 2 and sys.argv[2].lower() == "model":
        launch_train()
    elif command == "chat":
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        launch_chat(model_path)
    elif command == "demo":
        launch_demo()
    elif command == "evaluate":
        launch_evaluate()
    elif command == "test":
        launch_test()
    elif command in ("help", "--help", "-h"):
        show_usage()
    else:
        print(f"Error: Unknown command '{command}'")
        print()
        show_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
