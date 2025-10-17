# Archive

This directory contains scripts and documentation that have been archived during the Virgo-2 cleanup and simplification.

## Archived Scripts

These scripts have been moved here as they are either:
- Redundant with the simplified training infrastructure
- Not fully functional
- Superseded by better implementations

### scripts/
- `demo.py` - Basic demo (superseded by demo_nflm.py)
- `continuous_training.py` - Progressive training framework (functionality integrated into train_nflm.py)
- `run_all_training.py` - Batch training script (superseded by launch_virgo.py train)
- `run_key_experiment.py` - Experimental script (experiments now run via train_nflm.py)
- `train_scaled_nflm.py` - Scaled training (consolidated into train_nflm.py)

## Archived Documentation

These documentation files have been consolidated or are no longer needed:

### docs/
- `SETUP.md` - Setup instructions (consolidated into README.md)
- `QUICKSTART.md` - Quick start guide (consolidated into README.md)
- `ARCHITECTURE_OPTIMIZATION.md` - Architecture details (covered in main docs)
- `ARCHITECTURE_VISUAL.md` - Visual architecture (covered in main docs)
- `ADDITIONAL_BASELINES.md` - Baseline comparisons (covered in comparison docs)
- `THREE_PHASE_SUMMARY.md` - Training phases (covered in TRAINING.md)
- `3B_TRAINING_GUIDE.md` - Large-scale training guide (covered in main docs)

## Active Files

The main repository now uses:
- `launch_virgo.py` - Main entry point for all operations
- `scripts/train_nflm.py` - Unified training script
- `scripts/chat_with_model.py` - Chat interface with trained models
- `scripts/demo_nflm.py` - Neural field LM demonstration
- `scripts/test_trained_model.py` - Model testing
- `scripts/evaluate.py` - Model evaluation

## Note

These files are kept in the archive for reference but are no longer maintained or supported. Use the active files in the main repository instead.
