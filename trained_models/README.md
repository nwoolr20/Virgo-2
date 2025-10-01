# Trained Models Directory

This directory contains trained Neural Field Language Models.

## Directory Structure

```
trained_models/
├── wikitext_demo/              # Example: WikiText-103 trained model
│   ├── checkpoint_epoch_5.pt   # Checkpoint at epoch 5
│   ├── checkpoint_epoch_10.pt  # Checkpoint at epoch 10
│   ├── best_model.pt          # Best model (lowest validation loss)
│   ├── final_model.pt         # Final model after all epochs
│   ├── model_epoch_5.pt       # Model weights only (epoch 5)
│   ├── model_epoch_10.pt      # Model weights only (epoch 10)
│   └── training_history.json  # Training metrics
└── README.md                   # This file
```

## Model Files

### Checkpoint Files (*.pt)
Full checkpoints include:
- Model weights
- Optimizer state
- Training configuration
- Tokenizer vocabulary
- Epoch number and loss

### Model Files (model_*.pt)
Model-only files contain just the weights for inference.

### Training History (training_history.json)
JSON file with training metrics per epoch:
- Training loss
- Validation loss
- Timestamp

## Usage

### Load a trained model:

```python
from scripts.test_trained_model import load_model
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = load_model('trained_models/wikitext_demo/best_model.pt', device)
```

### Test a trained model:

```bash
python scripts/test_trained_model.py trained_models/wikitext_demo/best_model.pt \
    --prompts "the cat" "hello world"
```

### Resume training from checkpoint:

```bash
python scripts/train_nflm.py \
    --resume trained_models/wikitext_demo/checkpoint_epoch_5.pt \
    --epochs 20
```

## Creating New Models

Train a new model with:

```bash
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 1000 \
    --epochs 30 \
    --save-dir trained_models/my_model
```

See `TRAINING.md` for complete training documentation.

## Note on Git

Trained model files (*.pt) are excluded from git by default due to their large size (typically 10-50 MB per file). They are stored locally and not committed to the repository.

To share trained models:
1. Upload to HuggingFace Hub
2. Use external storage (Google Drive, Dropbox, etc.)
3. Include in releases as downloadable assets

## Provided Models

The repository includes training scripts but not pre-trained models. Train your own models using the provided scripts and datasets.

See `TRAINING.md` for step-by-step instructions.
