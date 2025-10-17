# Virgo-2 Quick Reference

## One-Line Commands

```bash
# Train a neural field model
python3 launch_virgo.py train

# Chat with the trained model
python3 launch_virgo.py chat

# Run demo
python3 launch_virgo.py demo

# Run tests
python3 launch_virgo.py test

# Run evaluation
python3 launch_virgo.py evaluate

# Show help
python3 launch_virgo.py help
```

## Advanced Training

```bash
# Train on specific dataset
python scripts/train_nflm.py --dataset wikitext --epochs 30

# Train with custom settings
python scripts/train_nflm.py \
    --dataset fineweb-edu \
    --sample-size 10000 \
    --epochs 50 \
    --batch-size 32 \
    --coord-dim 8 \
    --save-dir ./trained_models/my_model

# Resume training from checkpoint
python scripts/train_nflm.py \
    --resume ./trained_models/my_model/checkpoint_epoch_10.pt \
    --epochs 50
```

## Testing Models

```bash
# Test trained model
python scripts/test_trained_model.py ./trained_models/virgo_model/best_model.pt

# Test with custom prompts
python scripts/test_trained_model.py \
    ./trained_models/virgo_model/best_model.pt \
    --prompts "the" "hello" "artificial intelligence" \
    --max-length 100
```

## Key Files

| File | Purpose |
|------|---------|
| `launch_virgo.py` | Main entry point |
| `scripts/train_nflm.py` | Training script |
| `scripts/chat_with_model.py` | Chat interface |
| `scripts/demo_nflm.py` | Quick demo |
| `scripts/test_trained_model.py` | Model testing |
| `virgo/neural_field_lm.py` | Neural field LM (8D) |
| `virgo/coordinates.py` | Memory coordinates (6D) |

## Pre-trained Model

Location: `trained_models/virgo_model/best_model.pt`

**Specs:**
- Parameters: 2.3M
- Coordinate Dimension: 8D
- Vocabulary: 162 characters
- Training: 20 epochs on WikiText-103
- Validation Loss: 1.88

## Two Systems

### 1. Memory System (6D)
**Purpose:** Store/retrieve conversations
```python
from virgo import MemorySystem
system = MemorySystem()
system.store("Hello", speaker_id=0)
results = system.retrieve("Hello", k=3)
```

### 2. Neural Field LM (8D)
**Purpose:** Generate text
```python
from virgo import NeuralFieldLM, CharTokenizer
model = NeuralFieldLM(vocab_size=100, coord_dim=8)
# Or load trained model:
# model, tokenizer = load_model("path/to/model.pt")
```

## Documentation

- **README.md** - Overview and quick start
- **STRUCTURE.md** - Repository structure guide
- **COMPLETION_SUMMARY.md** - All changes made
- **TRAINING.md** - Training documentation
- **NFLM.md** - Neural field LM details
- **THE_KEY_EXPERIMENT.md** - Experiments framework
- **WIN_LOSE_ANALYSIS.md** - Performance analysis

## Common Tasks

### Train and Chat
```bash
python3 launch_virgo.py train
python3 launch_virgo.py chat
```

### Quick Test
```bash
python3 launch_virgo.py demo
```

### Development
```bash
python3 launch_virgo.py test
```

## Need Help?

```bash
python3 launch_virgo.py help
```

See `STRUCTURE.md` for detailed repository organization.
See `COMPLETION_SUMMARY.md` for complete list of changes.
