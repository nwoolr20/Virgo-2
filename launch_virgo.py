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
    print("  chat [path]       Start interactive chat interface")
    print("                    Optional: Specify memory storage path (default: ./memory_store)")
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
    print("  python3 launch_virgo.py chat")
    print("  python3 launch_virgo.py chat ./my_memory")
    print("  python3 launch_virgo.py demo")
    print("  python3 launch_virgo.py evaluate")
    print("  python3 launch_virgo.py test")
    print()


def launch_chat(memory_path=None):
    """Launch the chat interface."""
    print_header("Starting Virgo Chat Interface")
    
    if not check_installation():
        sys.exit(1)
    
    memory_path = memory_path or "./memory_store"
    
    print(f"Memory storage path: {memory_path}")
    print()
    print("Commands:")
    print("  Type messages to chat")
    print("  'save'  - Train field and save memory")
    print("  'stats' - Show memory statistics")
    print("  'quit'  - Exit")
    print()
    print("=" * 60)
    print()
    
    # Add current directory to PYTHONPATH
    script_dir = Path(__file__).parent.absolute()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{script_dir}:{env.get('PYTHONPATH', '')}"
    
    subprocess.run(
        [sys.executable, "-m", "virgo.chat", memory_path],
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
    
    if command == "chat":
        memory_path = sys.argv[2] if len(sys.argv) > 2 else None
        launch_chat(memory_path)
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
