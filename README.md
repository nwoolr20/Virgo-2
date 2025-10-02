# Virgo Neural Field Language Model

A neural field-based system providing **two distinct capabilities**:
1. **Retrieval-Augmented Memory**: Conversation memory with SIREN compression and FAISS retrieval
2. **Generative Neural Field LM**: True generative language model using learned coordinates and continuous fields

## Two Systems in One

### 1. Memory System (Original)
A retrieval-based conversation memory that compresses context using neural fields.
- **Purpose**: Store and retrieve conversation history
- **Coordinates**: Hand-crafted 6D space (temporal, turn_id, semantic, importance, speaker, sentiment)
- **Output**: Retrieved memories for context
- **Use case**: Chatbots with memory, context-aware systems

### 2. Neural Field Language Model (NEW)
A true generative language model that learns continuous representations.
- **Purpose**: Generate new text autoregressively
- **Coordinates**: Learned 8D space (trained end-to-end)
- **Output**: Token distributions for text generation
- **Use case**: Text generation, interpolation, continuous semantic spaces

See [NFLM.md](NFLM.md) for detailed documentation on the generative system.

## Features

### Memory System
- **6D Coordinate System**: Maps conversations into a continuous coordinate space
- **SIREN Neural Field**: Uses sinusoidal representation networks for efficient memory compression
- **FAISS Integration**: Fast retrieval using vector similarity search
- **Persistent Storage**: Save and load trained models across sessions

### Neural Field Language Model (NEW)
- **Learned Coordinates**: 8D coordinate space learned from text
- **Generative Field**: SIREN-based field that outputs token logits
- **Autoregressive Generation**: Generate text token-by-token
- **Sequence Interpolation**: Interpolate between texts in continuous space
- **End-to-End Training**: Train on language modeling objective

## Installation

### Quick Install (Recommended)

Use the provided install script:

```bash
python3 install.py
```

This script will:
- Check Python version
- Install all required dependencies
- Download NLTK data
- Download sentence-transformers model (~90MB, requires internet)
- Verify installation

### Manual Install

```bash
pip install torch sentence-transformers faiss-cpu scikit-learn textblob pytest
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

**Note:** The system requires internet access on first run to download the sentence-transformers model (`all-MiniLM-L6-v2`).

## Quick Start

### Training Neural Field Language Model

The easiest way to train and use the model:

```bash
# Train the model on HuggingFace datasets
python3 launch_virgo.py train

# Chat with the trained model
python3 launch_virgo.py chat
```

For advanced training options:

```bash
# Quick test (5 minutes)
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 500 \
    --epochs 10 \
    --save-dir ./trained_models/my_model

# Test the trained model
python scripts/test_trained_model.py ./trained_models/my_model/best_model.pt

# Full training on larger dataset
python scripts/train_nflm.py \
    --dataset wikitext \
    --epochs 50 \
    --save-dir ./trained_models/wikitext_full
```

See [TRAINING.md](TRAINING.md) for complete training documentation.

### Neural Field Language Model (Generative)

```python
from virgo import NeuralFieldLM, CharTokenizer, train_neural_field_lm
import torch

# Prepare training data
texts = ["hello world", "hi there", "good morning"]
tokenizer = CharTokenizer()
tokenizer.build_vocab(texts)

# Create training pairs (next-token prediction)
train_data = []
for text in texts:
    tokens = tokenizer.encode(text, add_eos=False)
    if len(tokens) > 1:
        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long)
        target_ids = torch.tensor([tokens[1:]], dtype=torch.long)
        train_data.append((input_ids, target_ids))

# Create and train model
model = NeuralFieldLM(vocab_size=tokenizer.vocab_size, coord_dim=8)
train_neural_field_lm(model, train_data, epochs=10, lr=1e-3)

# Generate text
prompt = torch.tensor(tokenizer.encode("hello", add_eos=False), dtype=torch.long)
generated = model.generate(prompt, max_length=20)
print(tokenizer.decode(generated.tolist()))

# Interpolate between sequences
seq1 = torch.tensor(tokenizer.encode("hello", add_eos=False), dtype=torch.long)
seq2 = torch.tensor(tokenizer.encode("hi wo", add_eos=False), dtype=torch.long)
interp = model.interpolate_sequences(seq1, seq2, alpha=0.5)
print(tokenizer.decode(interp.tolist()))
```

**Try the demos:**
```bash
# Quick example
python examples/nflm_quick_start.py

# Full demo
python scripts/demo_nflm.py
```

### Memory System (Retrieval)

### Using the Launch Script (Recommended)

```bash
# Train the neural field model
python3 launch_virgo.py train

# Start interactive chat with trained model
python3 launch_virgo.py chat

# Run demo
python3 launch_virgo.py demo

# Run evaluation
python3 launch_virgo.py evaluate

# Run tests
python3 launch_virgo.py test

# Show help
python3 launch_virgo.py help
```

### Using Python Directly

```python
from virgo import MemorySystem

# Create memory system
system = MemorySystem()

# Store conversations
system.store("My name is Alice", speaker_id=0)
system.store("Nice to meet you!", speaker_id=1)

# Train neural field
system.fit_field(num_steps=1000)

# Retrieve relevant memories
results = system.retrieve("What is my name?", k=3)
for memory, distance in results:
    print(f"{memory.text} (distance: {distance:.2f})")

# Save system
from pathlib import Path
system.save(Path("./my_memory"))
```

## Chat Interface

### Using Launch Script

```bash
python3 launch_virgo.py chat [path]
```

Optional: Specify a custom memory storage path (default: `./memory_store`)

### Using Python Module

Run the interactive chat interface:

```bash
python -m virgo.chat ./memory_store
```

Commands:
- Type messages to chat
- `save` - Train field and save memory
- `stats` - Show memory statistics
- `quit` - Exit

## Testing

### Using Launch Script

```bash
python3 launch_virgo.py test
```

### Using pytest directly

Run the test suite:

```bash
pytest tests/ -v
```

## Evaluation

### Using Launch Script

```bash
python3 launch_virgo.py evaluate
```

### Using Python directly

Run comprehensive evaluation:

```bash
python scripts/evaluate.py
```

## Demo

### Using Launch Script

```bash
python3 launch_virgo.py demo
```

### Using Python directly

Run the demo script:

```bash
python scripts/demo.py
```

## Architecture

For detailed information about the simplified repository structure, see [STRUCTURE.md](STRUCTURE.md).

### Memory System (Retrieval)
- **Coordinates** (`virgo/coordinates.py`): 6D coordinate system mapping conversations to [temporal, turn_id, semantic, importance, speaker, sentiment]
- **Field** (`virgo/field.py`): SIREN-based neural field for continuous function representation
- **Memory** (`virgo/memory.py`): Complete memory storage and retrieval system
- **Chat** (`virgo/chat.py`): Interactive chat interface

### Neural Field Language Model (Generative)
- **CoordinateEncoder** (`virgo/neural_field_lm.py`): Learns 8D coordinates from token sequences
- **GenerativeField** (`virgo/neural_field_lm.py`): SIREN-based field that outputs token logits
- **NeuralFieldLM** (`virgo/neural_field_lm.py`): Complete generative language model
- **CharTokenizer** (`virgo/tokenizer.py`): Character-level tokenization

See [NFLM.md](NFLM.md) for detailed architecture documentation.

## License

MIT

## System Testing

To run a comprehensive system test following all steps in the QUICKSTART guide:

```bash
python run_system_test.py
```

This will:
1. Install the package and dependencies
2. Download required data (NLTK, sentence-transformers model)
3. Run the test suite
4. Execute the interactive demo
5. Run comprehensive evaluation
6. Test the chat interface

All results will be saved to `V2 System Test.md` with detailed output from each step.

