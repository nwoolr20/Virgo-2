#!/usr/bin/env python3
"""Virgo Neural Field Language Model - Installation Script"""

import sys
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("=" * 60)
    print(text)
    print("=" * 60)
    print()


def print_step(step_num, total_steps, text):
    """Print a step indicator."""
    print(f"[{step_num}/{total_steps}] {text}")


def check_python_version():
    """Check Python version."""
    print_step(1, 5, "Checking Python version...")
    version = sys.version.split()[0]
    print(f"✓ Found Python {version}")
    print()
    return True


def check_pip():
    """Check if pip is available."""
    print_step(2, 5, "Checking pip...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ pip is available")
        print()
        return True
    except subprocess.CalledProcessError:
        print("Installing pip...")
        try:
            subprocess.run(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                check=True
            )
            print("✓ pip installed successfully")
            print()
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing pip: {e}")
            return False


def install_dependencies():
    """Install Python dependencies."""
    print_step(3, 5, "Installing Python dependencies...")
    print("This may take a few minutes...")
    
    packages = [
        "torch",
        "sentence-transformers",
        "faiss-cpu",
        "scikit-learn",
        "textblob",
        "pytest"
    ]
    
    try:
        # Upgrade pip first
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True
        )
        
        # Install packages
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + packages,
            check=True,
            capture_output=True
        )
        
        print("✓ Python dependencies installed successfully")
        print()
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing Python dependencies")
        print(f"Error: {e}")
        return False


def download_nltk_data():
    """Download required NLTK data."""
    print_step(4, 5, "Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('brown', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded successfully")
        print()
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False


def download_model():
    """Download sentence-transformers model."""
    print_step(5, 5, "Downloading sentence-transformers model...")
    print("This will download ~90MB and requires internet connection...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully")
        print()
        return True, 0
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['connection', 'internet', 'resolve']):
            print("⚠ Warning: Cannot download model (no internet connection)")
            print("  The model may already be cached, or you will need internet")
            print("  access when first running the system.")
            print()
            print("Note: Installation completed but model download was skipped.")
            print("      The system will attempt to download on first use if needed.")
            print()
            return True, 2  # Warning code
        else:
            print(f"✗ Error: {e}")
            return False, 1


def verify_installation():
    """Verify that all packages can be imported."""
    print_header("Verifying Installation")
    
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
        print("✓ All required packages are importable")
        print()
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print()
        return False


def main():
    """Main installation function."""
    print_header("Virgo Neural Field Installation")
    
    # Run installation steps
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not download_nltk_data():
        sys.exit(1)
    
    model_success, model_code = download_model()
    if not model_success:
        sys.exit(1)
    
    if not verify_installation():
        print_header("✗ Installation Failed")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    
    # Success!
    print_header("✓ Installation Complete!")
    print("You can now run the system using:")
    print("  python3 launch_virgo.py chat          # Start interactive chat")
    print("  python3 launch_virgo.py demo          # Run demo")
    print("  python3 launch_virgo.py evaluate      # Run evaluation")
    print("  python3 launch_virgo.py test          # Run tests")
    print()


if __name__ == "__main__":
    main()
