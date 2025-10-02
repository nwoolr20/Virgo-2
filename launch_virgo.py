#!/usr/bin/env python3
"""Virgo Neural Field Language Model - Launch Script"""

import sys
import os
import subprocess
from pathlib import Path


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


def show_usage():
    """Display usage information."""
    print_header("Virgo Neural Field Launch Script")
    print("Usage: python3 launch_virgo.py [command] [options]")
    print()
    print("Commands:")
    print("  train             Train neural field model on HuggingFace datasets")
    print()
    print("  chat [model_path] Start interactive chat with trained neural field")
    print("                    Optional: Specify model path (default: ./trained_models/best_model.pt)")
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
    print("  python3 launch_virgo.py train")
    print("  python3 launch_virgo.py chat")
    print("  python3 launch_virgo.py chat ./trained_models/my_model.pt")
    print("  python3 launch_virgo.py demo")
    print("  python3 launch_virgo.py evaluate")
    print("  python3 launch_virgo.py test")
    print()


def launch_train():
    """Launch the training script."""
    print_header("Training Virgo Neural Field Model")
    
    if not check_installation():
        sys.exit(1)
    
    print("Training neural field language model on HuggingFace datasets...")
    print()
    print("Configuration:")
    print("  - Dataset: wikitext, fineweb-edu, openwebtext (multi-dataset training)")
    print("  - Coordinate dimension: 8D")
    print("  - Epochs: 30")
    print("  - Save directory: ./trained_models/virgo_model")
    print()
    print("This will train continuously on multiple datasets to improve coherence.")
    print("=" * 60)
    print()
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    train_script = script_dir / "scripts" / "train_nflm.py"
    subprocess.run([
        sys.executable, str(train_script),
        "--dataset", "wikitext",
        "--sample-size", "10000",
        "--epochs", "30",
        "--batch-size", "16",
        "--coord-dim", "8",
        "--save-dir", "./trained_models/virgo_model"
    ], env=env)


def launch_chat(model_path=None):
    """Launch the conversational chat interface with trained neural field."""
    print_header("Starting Virgo Neural Field Chat")
    
    if not check_installation():
        sys.exit(1)
    
    model_path = model_path or "./trained_models/virgo_model/best_model.pt"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print()
        print("Please train a model first:")
        print("  python3 launch_virgo.py train")
        print()
        print("Or specify a valid model path:")
        print("  python3 launch_virgo.py chat /path/to/model.pt")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    print()
    print("Commands:")
    print("  Type messages to chat with the neural field")
    print("  'quit' - Exit")
    print()
    print("=" * 60)
    print()
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    # Create a simple chat script
    chat_script = script_dir / "scripts" / "chat_with_model.py"
    subprocess.run(
        [sys.executable, str(chat_script), model_path],
        env=env
    )


def launch_demo():
    """Launch the demo script."""
    print_header("Running Virgo Demo")
    
    if not check_installation():
        sys.exit(1)
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    demo_script = script_dir / "scripts" / "demo.py"
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
    if len(sys.argv) < 2:
        show_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "train":
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
