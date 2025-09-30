#!/bin/bash
# Virgo Neural Field Language Model - Installation Script

set -e  # Exit on error

echo "======================================"
echo "Virgo Neural Field Installation"
echo "======================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Install pip if not available
echo "[2/5] Checking pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Installing pip..."
    $PYTHON_CMD -m ensurepip --upgrade
fi
echo "✓ pip is available"
echo ""

# Install dependencies
echo "[3/5] Installing Python dependencies..."
echo "This may take a few minutes..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install torch sentence-transformers faiss-cpu scikit-learn textblob pytest

if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed successfully"
else
    echo "✗ Error installing Python dependencies"
    exit 1
fi
echo ""

# Download NLTK data
echo "[4/5] Downloading NLTK data..."
$PYTHON_CMD -c "import nltk; nltk.download('brown', quiet=True); nltk.download('punkt', quiet=True)"

if [ $? -eq 0 ]; then
    echo "✓ NLTK data downloaded successfully"
else
    echo "✗ Error downloading NLTK data"
    exit 1
fi
echo ""

# Download sentence-transformers model
echo "[5/5] Downloading sentence-transformers model..."
echo "This will download ~90MB and requires internet connection..."
$PYTHON_CMD -c "
from sentence_transformers import SentenceTransformer
import sys
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print('✓ Model loaded successfully')
except Exception as e:
    error_msg = str(e).lower()
    if 'connection' in error_msg or 'internet' in error_msg or 'resolve' in error_msg:
        print('⚠ Warning: Cannot download model (no internet connection)')
        print('  The model may already be cached, or you will need internet')
        print('  access when first running the system.')
        sys.exit(2)  # Special exit code for offline warning
    else:
        print(f'✗ Error: {e}')
        sys.exit(1)
" 2>&1

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
elif [ $EXIT_CODE -eq 2 ]; then
    echo ""
    echo "Note: Installation completed but model download was skipped."
    echo "      The system will attempt to download on first use if needed."
else
    echo "✗ Error downloading sentence-transformers model"
    exit 1
fi
echo ""

# Verify installation
echo "======================================"
echo "Verifying Installation"
echo "======================================"
echo ""

$PYTHON_CMD -c "
import sys
try:
    import torch
    import sentence_transformers
    import faiss
    import sklearn
    import textblob
    import nltk
    import pytest
    print('✓ All required packages are importable')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✓ Installation Complete!"
    echo "======================================"
    echo ""
    echo "You can now run the system using:"
    echo "  ./launch.sh chat          # Start interactive chat"
    echo "  ./launch.sh demo          # Run demo"
    echo "  ./launch.sh evaluate      # Run evaluation"
    echo "  ./launch.sh test          # Run tests"
    echo ""
else
    echo ""
    echo "======================================"
    echo "✗ Installation Failed"
    echo "======================================"
    echo ""
    echo "Please check the error messages above and try again."
    exit 1
fi
