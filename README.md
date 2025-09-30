# Virgo Neural Field Language Model

A neural field-based conversation memory system that uses SIREN networks to compress and retrieve conversational context.

## Features

- **6D Coordinate System**: Maps conversations into a continuous coordinate space
- **SIREN Neural Field**: Uses sinusoidal representation networks for efficient memory compression
- **FAISS Integration**: Fast retrieval using vector similarity search
- **Persistent Storage**: Save and load trained models across sessions

## Installation

```bash
pip install torch sentence-transformers faiss-cpu scikit-learn textblob pytest
python -c "import nltk; nltk.download('brown'); nltk.download('punkt')"
```

## Quick Start

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

Run the test suite:

```bash
pytest tests/ -v
```

## Evaluation

Run comprehensive evaluation:

```bash
python scripts/evaluate.py
```

## Demo

Run the demo script:

```bash
python scripts/demo.py
```

## Architecture

- **Coordinates** (`virgo/coordinates.py`): 6D coordinate system mapping conversations to [temporal, turn_id, semantic, importance, speaker, sentiment]
- **Field** (`virgo/field.py`): SIREN-based neural field for continuous function representation
- **Memory** (`virgo/memory.py`): Complete memory storage and retrieval system
- **Chat** (`virgo/chat.py`): Interactive chat interface

## License

MIT
