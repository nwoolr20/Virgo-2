# Knowledge Distillation Guide for Virgo Neural Field

## Overview

This guide documents the knowledge distillation process to transfer knowledge from a 7B parameter teacher model into the Virgo neural field architecture.

## Why This Is Complex

Knowledge distillation with significant architectural changes requires:

1. **GPU Requirements**: 7B teacher model needs 14-28GB VRAM
2. **Training Time**: 2M samples × 3 epochs = days of training
3. **Architectural Changes**: Expanding model while preserving learned weights
4. **Dataset Management**: 200GB+ of training data
5. **Validation**: Ensuring quality doesn't degrade during upgrades

## Three-Phase Process

### Phase 1: Architecture Upgrade

**Goal**: Expand model capacity while preserving existing knowledge

**Current Model**:
- Vocabulary: 162 characters (character-level)
- Coordinate dim: 8
- Parameters: ~2.3M

**Target Model**:
- Vocabulary: 50,257 tokens (GPT-2 BPE)
- Coordinate dim: 16
- Coordinate encoder hidden: 2048
- SIREN field hidden: 4096
- Transformer layers: 12 (16 heads)
- Parameters: ~2B

**Challenges**:
- Vocabulary expansion requires embedding interpolation
- Coordinate dimension increase needs careful weight preservation
- Adding transformer layers without destroying existing field knowledge

**Implementation Status**: ⚠️ Framework created, core logic needed

### Phase 2: Knowledge Distillation

**Goal**: Transfer knowledge from 7B teacher to upgraded student

**Teacher Model**: `cognitivecomputations/dolphin-2.6-mistral-7b` (7B params)

**Dataset Mix** (Total: 2M samples):
```
Open-Orca/OpenOrca:        800,000 samples (40%)
teknium/OpenHermes-2.5:    600,000 samples (30%)
QuixiAI/samantha-data:     400,000 samples (20%)
stingning/ultrachat:       200,000 samples (10%)
```

**Training Configuration**:
- Epochs: 3
- Batch size: 4
- Gradient accumulation: 8 (effective batch size: 32)
- Learning rate: 1e-4, warmup 10K steps, cosine decay
- Temperature schedule: 2.0 → 1.5 → 1.0

**Loss Function**:
```python
total_loss = 0.5 * kl_divergence(student, teacher, temperature) + 
             0.5 * cross_entropy(student, targets)
```

**Checkpointing**: Every 25,000 steps

**Estimated Time**: 
- With 1x A100 (40GB): ~3-5 days
- With 4x A100: ~18-24 hours
- CPU: Not recommended (weeks/months)

**Implementation Status**: ⚠️ Framework created, training loop needed

### Phase 3: Extended Training

**Goal**: Continue training on high-quality web data

**Dataset**: HuggingFaceFW/fineweb-edu (streaming, 10B tokens)

**Training**:
- Pure language modeling loss (no distillation)
- Learning rate: 5e-5, cosine decay
- Stop when validation loss plateaus (min 1, max 3 epochs)

**Estimated Time**: 1-3 days on GPU

**Implementation Status**: ⚠️ Framework created, implementation needed

## Hardware Requirements

### Minimum (Not Recommended):
- CPU: Modern multi-core
- RAM: 32GB+
- Disk: 250GB
- Time: Weeks to months

### Recommended:
- GPU: 1x A100 (40GB) or 2x A6000 (48GB)
- RAM: 64GB+
- Disk: 500GB SSD
- Time: 3-7 days total

### Optimal:
- GPU: 4x A100 (40GB)
- RAM: 128GB+
- Disk: 1TB NVMe SSD
- Time: 1-2 days total

## Storage Requirements

**During Training**:
- Datasets (cached): ~200GB
- Teacher model: ~30GB
- Checkpoints (intermediate): ~50GB
- Working space: ~20GB
- **Total**: ~300GB

**After Training** (can delete):
- Datasets: ~200GB
- Teacher model: ~30GB
- Intermediate checkpoints: ~50GB

**Final Model** (keep):
- Trained field: ~27MB (if compression works as expected)
- Tokenizer: ~5MB
- Training metadata: ~1MB
- **Total**: ~33MB

## Current Implementation Status

### ✅ Complete:
- Framework and infrastructure code
- Dataset loading logic
- Command-line interface
- Phase orchestration
- Documentation

### ⚠️ Needs Implementation:
1. **Architecture Upgrade Methods**:
   - Vocabulary expansion with embedding interpolation
   - Coordinate dimension increase
   - Transformer layer insertion
   - Weight preservation strategy

2. **Distillation Training Loop**:
   - Teacher-student forward passes
   - Combined loss computation
   - Gradient accumulation
   - Checkpoint management
   - Validation metrics

3. **Extended Training**:
   - FineWeb-Edu streaming
   - Plateau detection
   - Final model saving

### 🔧 To Complete Implementation:

1. **Modify `virgo/neural_field_lm.py`**:
   ```python
   class NeuralFieldLM:
       def expand_vocabulary(self, new_vocab_size, tokenizer):
           # Interpolate embeddings for new tokens
           pass
       
       def increase_coord_dim(self, new_dim):
           # Expand coordinate space
           pass
       
       def add_transformer_layers(self, num_layers, num_heads):
           # Insert transformer between encoder and field
           pass
   ```

2. **Complete `scripts/train_distillation.py`**:
   - Implement `upgrade_architecture()` function
   - Implement distillation training loop
   - Implement extended training loop
   - Add validation and metrics

3. **Test on Small Scale**:
   ```bash
   # Test with minimal samples first
   python scripts/train_distillation.py --phase distill --quick-test
   ```

4. **Run Full Training** (on GPU cluster):
   ```bash
   # Phase 1: Upgrade
   python scripts/train_distillation.py --phase upgrade
   
   # Phase 2: Distill
   python scripts/train_distillation.py --phase distill --device cuda
   
   # Phase 3: Extend
   python scripts/train_distillation.py --phase extend --device cuda
   ```

## Validation Metrics

Track every 5,000 steps:
1. Validation loss
2. Validation perplexity (target: <15)
3. Teacher-student KL divergence
4. Interpolation coherence (100 sample pairs)
5. Generation quality (50 prompts, temperature=0.8)

## Expected Outcomes

**If Successful**:
- Final model: ~27MB (compression from 2B params)
- Perplexity: <15 on validation set
- Coherent multi-turn conversations
- Smooth interpolation in coordinate space
- Generation quality approaching teacher model

**If Unsuccessful**:
- May need to reduce compression ratio
- May need more training data
- May need architecture adjustments
- Fallback: Standard transformer with learned coordinates

## Cost Estimates

**Cloud GPU Training** (AWS p4d.24xlarge, 8x A100):
- $32.77/hour
- Estimated time: 24-48 hours
- **Total**: $800-$1,600

**Alternative** (Google Colab Pro+):
- $50/month
- Limited to A100 time caps
- More fragmented training

## Next Steps

1. **Immediate**: Test framework with `--quick-test` flag
2. **Short-term**: Implement architecture upgrade methods
3. **Medium-term**: Complete distillation training loop
4. **Long-term**: Run full training on GPU cluster

## Questions to Consider

1. Is 27MB final model size realistic? May need 100-500MB
2. Should we use smaller teacher model first? (1.3B instead of 7B)
3. Can we do progressive distillation? (Teacher → 1B → 500M → Field)
4. Should we train intermediate checkpoints at different scales?

## References

- Knowledge Distillation: Hinton et al., 2015
- SIREN Networks: Sitzmann et al., 2020
- Neural Field Language Models: (Your novel architecture)
- Teacher Models: Cognitive Computations Dolphin series
