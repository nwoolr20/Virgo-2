# Virgo-2 Simplified Structure

This document describes the simplified structure after the cleanup and consolidation.

## Core Files

### Entry Point
- **`launch_virgo.py`** - Main entry point for all operations
  - `python3 launch_virgo.py train` - Train the neural field model
  - `python3 launch_virgo.py chat` - Chat with trained model
  - `python3 launch_virgo.py demo` - Run demonstration
  - `python3 launch_virgo.py test` - Run test suite
  - `python3 launch_virgo.py evaluate` - Run evaluation

### Training Scripts (scripts/)
- **`train_nflm.py`** - Main training script for neural field language models
  - Supports multiple HuggingFace datasets (wikitext, fineweb-edu, openwebtext, c4)
  - Configurable epochs, batch size, coordinate dimensions
  - Automatic checkpointing and model saving
  - Validation metrics and perplexity tracking

- **`chat_with_model.py`** - Interactive chat interface with trained models
  - Loads trained neural field models
  - Provides conversational interface
  - Maintains conversation history for context

- **`demo_nflm.py`** - Quick demonstration of neural field LM capabilities
  - Shows training, generation, and interpolation
  - Useful for understanding how the system works

- **`test_trained_model.py`** - Test and evaluate trained models
  - Generation quality testing
  - Prompt-based evaluation
  - Model inspection utilities

- **`evaluate.py`** - Comprehensive model evaluation
  - Multiple test metrics
  - Comparison frameworks

### Core Library (virgo/)
- **`neural_field_lm.py`** - Neural Field Language Model (8D coordinates, learned)
- **`coordinates.py`** - Conversation Coordinate System (6D coordinates, hand-crafted)
- **`field.py`** - SIREN neural field implementation
- **`memory.py`** - Memory system for retrieval
- **`chat.py`** - Memory-based chat interface
- **`tokenizer.py`** - Character-level tokenizer
- **`bpe_tokenizer.py`** - BPE tokenizer
- **`baseline_transformer.py`** - Baseline transformer for comparisons
- **`additional_baselines.py`** - Additional baseline models

### Tests (tests/)
- **`test_neural_field_lm.py`** - Tests for generative neural field
- **`test_coordinates.py`** - Tests for coordinate system
- **`test_field.py`** - Tests for SIREN field
- **`test_memory.py`** - Tests for memory system
- **`test_integration.py`** - End-to-end integration tests

### Documentation
- **`README.md`** - Main documentation and quick start
- **`TRAINING.md`** - Training documentation
- **`NFLM.md`** - Neural Field Language Model documentation
- **`IMPLEMENTATION.md`** - Implementation details
- **`COMPARISON.md`** - Comparison with other approaches
- **`THE_KEY_EXPERIMENT.md`** - Key experiment tracking
- **`WIN_LOSE_ANALYSIS.md`** - Performance analysis framework
- **`TRAINED_MODEL_INFO.md`** - Trained model information

### Archive (archive/)
Contains archived scripts and documentation that have been superseded. See `archive/README.md` for details.

## Two Systems in One

Virgo-2 contains two distinct systems:

### 1. Memory System (Retrieval-Based)
- **Purpose**: Store and retrieve conversation history
- **Coordinates**: 6D hand-crafted (temporal, turn_id, semantic, importance, speaker, sentiment)
- **Use**: `virgo.MemorySystem`, `virgo.chat`
- **Files**: `coordinates.py`, `memory.py`, `chat.py`

### 2. Neural Field Language Model (Generative)
- **Purpose**: Generate new text autoregressively
- **Coordinates**: 8D learned end-to-end
- **Use**: `virgo.NeuralFieldLM`, `train_nflm.py`, `chat_with_model.py`
- **Files**: `neural_field_lm.py`, `train_nflm.py`, `chat_with_model.py`

**Note**: The 6D vs 8D distinction is intentional - they are different systems serving different purposes.

## Workflow

### Training a Model
```bash
# Quick training
python3 launch_virgo.py train

# Or with custom parameters
python scripts/train_nflm.py \
    --dataset wikitext \
    --sample-size 10000 \
    --epochs 30 \
    --coord-dim 8 \
    --save-dir ./trained_models/my_model
```

### Chatting with the Model
```bash
# Use default model location
python3 launch_virgo.py chat

# Or specify model path
python3 launch_virgo.py chat ./trained_models/my_model/best_model.pt
```

### Testing and Evaluation
```bash
# Run tests
python3 launch_virgo.py test

# Run evaluation
python3 launch_virgo.py evaluate

# Test specific model
python scripts/test_trained_model.py ./trained_models/my_model/best_model.pt
```

## Key Changes from Previous Version

1. **Simplified Entry Point**: All operations go through `launch_virgo.py`
2. **Unified Training**: Single `train_nflm.py` script handles all training scenarios
3. **Conversational Interface**: New `chat_with_model.py` provides chat with trained models
4. **Archived Redundancy**: Moved redundant scripts and docs to `archive/`
5. **Clearer Documentation**: Documentation clarifies 6D (memory) vs 8D (generative) distinction
6. **Updated Experiments**: THE_KEY_EXPERIMENT.md and WIN_LOSE_ANALYSIS.md marked as implemented

## Future Enhancements

- Continuous training on multiple datasets
- Improved chat interface with better context handling
- Fine-tuning on domain-specific data
- Multi-turn conversation support
- Model comparison dashboards
