# Virgo Quick Start Guide

This guide will get your Virgo installation working properly.

## Step 1: Install the Package

From the project root directory (`/workspaces/Virgo-2`):

```bash
# Install in editable mode (recommended for development)
pip install -e .

# This installs the virgo module so it can be imported from anywhere
```

## Step 2: Download Required Data

```bash
# Download NLTK data
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"

# Download sentence-transformers model (requires internet)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Step 3: Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Should see all tests passing
```

## Step 4: Run the Demo

```bash
# Run the interactive demo
python scripts/demo.py

# This will:
# - Add 5 sample memories
# - Train the neural field
# - Test retrieval with queries
# - Show system statistics
```

## Step 5: Run Evaluation

```bash
# Run comprehensive evaluation
python scripts/evaluate.py

# This tests:
# - Compression ratios
# - Retrieval accuracy
# - Persistence (save/load)
```

## Step 6: Try the Chat Interface

```bash
# Start the chat interface
python virgo/chat.py ./memory_store

# Commands:
# - Type to chat
# - 'save' to train and save
# - 'stats' to see memory statistics
# - 'quit' to exit
```

## Alternative: Using Launch Scripts

If you prefer the launch scripts:

```bash
# Using Python launcher
python launch_virgo.py demo
python launch_virgo.py evaluate
python launch_virgo.py test
python launch_virgo.py chat

# Using Bash launcher (if on Unix/Linux/Mac)
./launch.sh demo
./launch.sh evaluate
./launch.sh test
./launch.sh chat
```

## Troubleshooting

### "No module named 'virgo'" Error

If you still see this error, make sure you ran:
```bash
pip install -e .
```

You can verify the installation with:
```bash
python -c "import virgo; print(virgo.__version__)"
```

### "No module named 'joblib'" Error

Install joblib for pickle persistence:
```bash
pip install joblib
```

### Internet Connection Required

The first run requires internet to download:
- Sentence-transformers model (~90MB)
- NLTK data (~5MB)

After first download, these are cached and the system can run offline.

### Permission Errors on Launch Scripts

Make the scripts executable:
```bash
chmod +x launch.sh launch_virgo.py scripts/*.py virgo/*.py
```

## Quick Test

To verify everything works:

```bash
# Quick Python test
python -c "
from virgo import MemorySystem
m = MemorySystem()
m.store('test', speaker_id=0)
print('✓ Virgo is working!')
"
```

## Project Structure

```
Virgo-2/
├── virgo/              # Main package
│   ├── __init__.py
│   ├── coordinates.py  # 6D coordinate system
│   ├── field.py        # SIREN neural field
│   ├── memory.py       # Memory system
│   └── chat.py         # Chat interface
├── scripts/            # Utility scripts
│   ├── demo.py         # Interactive demo
│   └── evaluate.py     # Evaluation script
├── tests/              # Test suite
│   ├── test_coordinates.py
│   ├── test_field.py
│   ├── test_memory.py
│   └── test_integration.py
├── setup.py            # Package installation
└── launch_virgo.py     # Launch utility
```

## Next Steps

1. ✅ Install package: `pip install -e .`
2. ✅ Run tests: `pytest tests/ -v`
3. ✅ Run demo: `python scripts/demo.py`
4. ✅ Try evaluation: `python scripts/evaluate.py`
5. ✅ Start chatting: `python virgo/chat.py`

You're now ready to use Virgo!
