# Neural Field Language Model Training Guide

This guide explains how to train the Neural Field Language Model on real text corpora.

## Quick Start

### 1. Install Dependencies

```bash
# Install all required packages
pip install -e .

# Or install manually
pip install torch datasets huggingface-hub tqdm
```

### 2. Train on WikiText-103 (Recommended for First Run)

WikiText-103 is a high-quality, curated dataset that's perfect for learning how the system works:

```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --epochs 20 \
    --batch-size 32 \
    --save-dir ./trained_models/wikitext \
    --sample-size 5000
```

This will:
- Download WikiText-103 from HuggingFace
- Train for 20 epochs
- Save checkpoints every 5 epochs
- Complete in ~15-30 minutes on CPU (faster on GPU)

### 3. Test the Trained Model

```bash
python scripts/test_trained_model.py ./trained_models/wikitext/final_model.pt \
    --prompts "the" "hello" "in the"
```

## Supported Datasets

### WikiText-103 (Recommended for Starting)
High-quality Wikipedia articles, clean and coherent.

```bash
python scripts/train_nflm.py --dataset wikitext --epochs 50
```

**Characteristics:**
- ~103M tokens
- Very clean, well-structured text
- Good for learning smooth semantic spaces
- Fast to download and process

### FineWeb-Edu (Premium Quality)
Educational web text filtered by LLMs for quality.

```bash
python scripts/train_nflm.py --dataset fineweb-edu --sample-size 10000 --epochs 30
```

**Characteristics:**
- Educational content with clear semantic hierarchies
- Excellent for neural field training
- Use `sample-size` to limit download (dataset is very large)

### OpenWebText (Reddit-Curated)
Web text curated by Reddit community quality signals.

```bash
python scripts/train_nflm.py --dataset openwebtext --sample-size 10000 --epochs 30
```

**Characteristics:**
- Broad topical coverage
- Mix of formal and informal text
- Good for diverse semantic spaces

### C4 (Colossal Clean Crawled Corpus)
Strictly filtered web text from Common Crawl.

```bash
python scripts/train_nflm.py --dataset c4 --sample-size 10000 --epochs 30
```

**Characteristics:**
- Very large and diverse
- Strong filtering removes noise
- Use streaming to avoid downloading entire dataset

## Training Options

### Basic Training

```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --save-dir ./trained_models
```

### Advanced Training

```bash
python scripts/train_nflm.py \
    --dataset fineweb-edu \
    --sample-size 20000 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --coord-dim 16 \
    --max-seq-len 256 \
    --save-every 10 \
    --val-split 0.1 \
    --device cuda \
    --save-dir ./trained_models/fineweb_large
```

### Resume from Checkpoint

```bash
python scripts/train_nflm.py \
    --resume ./trained_models/checkpoint_epoch_20.pt \
    --epochs 50
```

## Command-Line Arguments

### Dataset Options
- `--dataset`: Dataset name (wikitext, fineweb-edu, openwebtext, c4)
- `--sample-size`: Limit number of samples (useful for testing or limiting memory)
- `--max-text-length`: Maximum text length to process (default: 512)
- `--max-seq-len`: Maximum sequence length for training (default: 128)

### Training Options
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--coord-dim`: Coordinate space dimension (default: 8, try 16 for larger models)
- `--device`: Device to use (cuda/cpu, auto-detected if not specified)

### Checkpoint Options
- `--save-dir`: Directory to save checkpoints (default: ./trained_models)
- `--save-every`: Save checkpoint every N epochs (default: 5)
- `--resume`: Resume from checkpoint path

### Validation
- `--val-split`: Validation split ratio (default: 0.1)

## Output Files

Training produces several files in the save directory:

```
trained_models/
├── checkpoint_epoch_5.pt       # Checkpoint at epoch 5
├── checkpoint_epoch_10.pt      # Checkpoint at epoch 10
├── best_model.pt              # Best model (lowest validation loss)
├── final_model.pt             # Final model after all epochs
├── model_epoch_5.pt           # Model weights only (epoch 5)
├── model_epoch_10.pt          # Model weights only (epoch 10)
└── training_history.json      # Training metrics over time
```

### Checkpoint Contents

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (for resuming)
- `vocab_size`: Size of vocabulary
- `coord_dim`: Coordinate dimension
- `char_to_idx`: Character to index mapping
- `idx_to_char`: Index to character mapping
- `epoch`: Current epoch number
- `loss`: Training loss

## Testing Trained Models

### Basic Testing

```bash
python scripts/test_trained_model.py ./trained_models/final_model.pt
```

### Custom Prompts

```bash
python scripts/test_trained_model.py ./trained_models/final_model.pt \
    --prompts "the quick brown" "hello world" "in the beginning"
```

### Advanced Testing

```bash
python scripts/test_trained_model.py ./trained_models/best_model.pt \
    --prompts "the" "hello" \
    --max-length 150 \
    --temperature 0.9 \
    --device cuda
```

### Skip Certain Tests

```bash
# Skip interpolation and coordinate analysis for faster testing
python scripts/test_trained_model.py ./trained_models/final_model.pt \
    --skip-interpolation \
    --skip-analysis
```

## Training Tips

### 1. Start Small
Begin with WikiText-103 and a small sample size to verify everything works:

```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 1000 \
    --epochs 10 \
    --batch-size 16
```

### 2. Use GPU if Available
Training is much faster on GPU:

```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --device cuda
```

### 3. Monitor Training
Watch the training history:

```bash
cat trained_models/training_history.json
```

### 4. Curriculum Learning
Train progressively on increasingly diverse datasets:

```bash
# Stage 1: Clean baseline
python scripts/train_nflm.py --dataset wikitext --epochs 30 --save-dir ./models/stage1

# Stage 2: Quality web text
python scripts/train_nflm.py --dataset openwebtext --epochs 30 --save-dir ./models/stage2

# Stage 3: Diverse corpus
python scripts/train_nflm.py --dataset fineweb-edu --epochs 30 --save-dir ./models/stage3
```

### 5. Experiment with Hyperparameters

**Coordinate Dimension:**
- Smaller (8): Faster training, good for small datasets
- Larger (16-32): Better representation capacity for large/diverse datasets

**Learning Rate:**
- Higher (1e-3): Faster convergence but may be unstable
- Lower (1e-5): More stable but slower

**Sequence Length:**
- Shorter (64-128): Faster training, less memory
- Longer (256-512): Better long-range dependencies

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python scripts/train_nflm.py --batch-size 8

# Reduce sequence length
python scripts/train_nflm.py --max-seq-len 64

# Use smaller sample size
python scripts/train_nflm.py --sample-size 1000
```

### Slow Training
```bash
# Use GPU
python scripts/train_nflm.py --device cuda

# Reduce sample size for testing
python scripts/train_nflm.py --sample-size 5000

# Reduce epochs
python scripts/train_nflm.py --epochs 20
```

### Poor Generation Quality
- Train for more epochs
- Use a cleaner dataset (WikiText-103)
- Increase model capacity (--coord-dim 16)
- Reduce temperature during generation (--temperature 0.7)

### HuggingFace Download Issues
If you encounter issues downloading datasets:

1. Check your internet connection
2. Verify the HuggingFace token in `.hf_token`
3. Try a different dataset
4. Use `--sample-size` to limit downloads

## Expected Training Times

On CPU (approximate):
- WikiText-103 (1K samples, 10 epochs): ~5 minutes
- WikiText-103 (5K samples, 20 epochs): ~30 minutes
- FineWeb-Edu (10K samples, 30 epochs): ~2 hours

On GPU (approximate):
- WikiText-103 (5K samples, 20 epochs): ~5 minutes
- WikiText-103 (full, 50 epochs): ~30 minutes
- FineWeb-Edu (20K samples, 50 epochs): ~1 hour

## What Gets Saved

**Important:** The training script saves the **trained neural field model**, not the raw datasets.

Saved files are:
- Model weights (`.pt` files)
- Tokenizer vocabulary
- Training configuration
- Training history

Datasets are:
- Downloaded to HuggingFace cache (outside git)
- Not saved in the repository
- Can be re-downloaded as needed

The `.gitignore` is configured to exclude:
- Trained models (unless you explicitly add them)
- HuggingFace cache
- Training checkpoints
- Raw dataset files

## Example Workflow

```bash
# 1. Quick test (5 minutes)
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 1000 \
    --epochs 10 \
    --save-dir ./test_run

# 2. Test the model
python scripts/test_trained_model.py ./test_run/final_model.pt

# 3. Real training (30 minutes)
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 5000 \
    --epochs 30 \
    --save-dir ./trained_models/wikitext_30ep

# 4. Test again
python scripts/test_trained_model.py \
    ./trained_models/wikitext_30ep/best_model.pt \
    --prompts "the" "hello" "in the"

# 5. Advanced training (2+ hours)
python scripts/train_nflm.py \
    --dataset fineweb-edu \
    --sample-size 20000 \
    --epochs 50 \
    --coord-dim 16 \
    --batch-size 64 \
    --save-dir ./trained_models/fineweb_large
```

## Advanced: Using Custom Datasets

To add support for additional datasets, edit `scripts/train_nflm.py` and add a new case in the `load_dataset_texts()` function:

```python
elif dataset_name == "my_dataset":
    dataset = load_dataset("username/my_dataset", split=split)
    for item in tqdm(dataset, desc="Processing My Dataset"):
        text = item["text"].strip()
        if text and len(text) > 50:
            texts.append(text)
```

## Next Steps

After training:
1. Test generation quality with `test_trained_model.py`
2. Experiment with interpolation between sequences
3. Analyze the learned coordinate space
4. Try different datasets and hyperparameters
5. Train larger models with more data

For more information, see:
- `README.md` - General project overview
- `NFLM.md` - Neural field architecture details
- `IMPLEMENTATION_NFLM.md` - Implementation notes
