# Neural Field Training Quick Reference

## Quick Commands

### Train on WikiText-103 (Recommended First Run)
```bash
python scripts/train_nflm.py --dataset wikitext --sample-size 1000 --epochs 20
```

### Test Trained Model
```bash
python scripts/test_trained_model.py ./trained_models/wikitext_demo/best_model.pt
```

### Quick Training Example
```bash
python examples/quick_train.py
```

## Datasets

| Dataset | Best For | Command |
|---------|----------|---------|
| WikiText-103 | Clean baseline | `--dataset wikitext` |
| FineWeb-Edu | Educational quality | `--dataset fineweb-edu` |
| OpenWebText | Reddit-curated | `--dataset openwebtext` |
| C4 | Web diversity | `--dataset c4` |

## Common Options

```bash
# Sample size (limit data)
--sample-size 1000

# Training epochs
--epochs 30

# Batch size
--batch-size 32

# Learning rate
--lr 1e-4

# Coordinate dimensions
--coord-dim 8

# Save directory
--save-dir ./trained_models/my_model

# Save frequency
--save-every 5

# Resume training
--resume ./trained_models/checkpoint_epoch_10.pt

# Use GPU
--device cuda
```

## Typical Workflows

### Quick Test (5 minutes)
```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 500 \
    --epochs 10 \
    --batch-size 16
```

### Standard Training (30-60 minutes)
```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 5000 \
    --epochs 30 \
    --batch-size 32
```

### Full Training (2+ hours)
```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --epochs 50 \
    --batch-size 64 \
    --coord-dim 16
```

### High-Quality Educational Data
```bash
python scripts/train_nflm.py \
    --dataset fineweb-edu \
    --sample-size 10000 \
    --epochs 50 \
    --batch-size 32
```

## Output Files

After training, you'll have:
- `checkpoint_epoch_N.pt` - Full checkpoints (model + optimizer + vocab)
- `best_model.pt` - Best model (lowest validation loss)
- `final_model.pt` - Final model after all epochs
- `model_epoch_N.pt` - Model weights only
- `training_history.json` - Training metrics

## Testing

### Basic Test
```bash
python scripts/test_trained_model.py ./trained_models/wikitext_demo/best_model.pt
```

### Custom Prompts
```bash
python scripts/test_trained_model.py ./trained_models/wikitext_demo/best_model.pt \
    --prompts "the cat" "hello world" "in the beginning"
```

### Longer Generation
```bash
python scripts/test_trained_model.py ./trained_models/wikitext_demo/best_model.pt \
    --max-length 150 \
    --temperature 0.9
```

### Skip Tests
```bash
python scripts/test_trained_model.py ./trained_models/wikitext_demo/best_model.pt \
    --skip-interpolation \
    --skip-analysis
```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 8 or 16)
- Reduce `--max-seq-len` (try 64)
- Use `--sample-size` to limit data

### Slow Training
- Use `--device cuda` if GPU available
- Reduce `--sample-size` for testing
- Reduce `--epochs`

### Poor Quality
- Train longer (more epochs)
- Use cleaner dataset (wikitext)
- Increase `--coord-dim` (try 16)
- Lower temperature during generation

## Expected Performance

### Training Time (CPU)
- 500 samples, 10 epochs: ~5 minutes
- 1000 samples, 20 epochs: ~15 minutes
- 5000 samples, 30 epochs: ~60 minutes

### Training Time (GPU)
- 500 samples, 10 epochs: ~1 minute
- 1000 samples, 20 epochs: ~3 minutes
- 5000 samples, 30 epochs: ~10 minutes

### Loss Reduction
- Typical starting loss: 3.0-4.0
- Good final loss: 1.5-2.0
- Excellent final loss: < 1.5

## File Sizes

- Checkpoint file: ~27 MB
- Model-only file: ~9 MB
- Training history: ~1-2 KB

## See Also

- `TRAINING.md` - Complete training documentation
- `README.md` - Project overview
- `NFLM.md` - Neural field architecture
- `trained_models/README.md` - Model storage info
