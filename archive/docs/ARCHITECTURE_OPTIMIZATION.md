# Architecture Optimization Guide

## Purpose

This document provides systematic optimization strategies based on experimental results from neural field language model training and comparison experiments.

## Optimization Philosophy

**Principle**: Optimize based on evidence, not assumptions.

1. **Measure First**: Run experiments before optimizing
2. **One Change at a Time**: Isolate variables
3. **Document Results**: Track what works and what doesn't
4. **Scale Gradually**: Validate at small scale before large-scale training

---

## Performance Diagnosis

### Step 1: Identify the Problem

**Common Issues:**

| Symptom | Likely Cause | Section to Check |
|---------|--------------|------------------|
| High training loss plateau | Learning rate too low/high | [Learning Rate](#learning-rate-optimization) |
| Validation loss increasing | Overfitting | [Regularization](#regularization-strategies) |
| Slow convergence | Architecture capacity | [Model Scaling](#model-scaling) |
| Training instability | Initialization or gradients | [Training Stability](#training-stability) |
| Poor generation quality | Model too small or undertrained | [Model Scaling](#model-scaling) |
| Out of memory | Batch size or model size | [Memory Optimization](#memory-optimization) |

### Step 2: Diagnose Root Cause

**Analysis Checklist:**
- [ ] Plot training and validation loss curves
- [ ] Check gradient norms (should be stable, not exploding/vanishing)
- [ ] Monitor learning rate schedule
- [ ] Examine generation samples at different epochs
- [ ] Compare against baseline metrics

---

## Neural Field Specific Optimizations

### Coordinate Encoder Improvements

**Current Configuration:**
```python
ScaledCoordinateEncoder(
    vocab_size=50257,
    coord_dim=8,
    hidden_dim=512,
    num_layers=4
)
```

**Optimization Options:**

#### 1. Increase Transformer Layers

**When**: Model underperforms on long-range dependencies

**Change**:
```python
num_layers=4 → 6 or 8
```

**Expected Impact**:
- Better context modeling
- +20-30% parameters
- Slower training (~20-30%)
- Better perplexity (typically 5-15% improvement)

**Test**:
```bash
python scripts/train_scaled_nflm.py \
    --encoder-layers 6 \
    --num-samples 10000 \
    --epochs 10
```

#### 2. Adjust Hidden Dimensions

**When**: Model capacity seems insufficient

**Change**:
```python
hidden_dim=512 → 768 or 1024
```

**Expected Impact**:
- More expressive representations
- +50-100% parameters in encoder
- Proportional memory increase
- Potentially better performance (10-20%)

**Test**:
```bash
python scripts/train_scaled_nflm.py \
    --encoder-hidden-dim 768 \
    --num-samples 10000 \
    --epochs 10
```

#### 3. Coordinate Dimension

**When**: Interpolation quality is poor or model capacity is limited

**Change**:
```python
coord_dim=8 → 16 or 32
```

**Expected Impact**:
- Richer coordinate space
- Minimal parameter increase
- Better interpolation quality
- May help or hurt depending on task

**Test**:
```bash
python scripts/train_scaled_nflm.py \
    --coord-dim 16 \
    --num-samples 10000 \
    --epochs 10
```

**Warning**: Higher dimensions may require more data to avoid overfitting the coordinate space.

### SIREN Field Improvements

**Current Configuration:**
```python
ScaledGenerativeField(
    coord_dim=8,
    vocab_size=50257,
    hidden_dim=1024,
    num_layers=8
)
```

**Optimization Options:**

#### 1. Increase Depth

**When**: Generation quality is poor or model underfits

**Change**:
```python
num_layers=8 → 10 or 12
```

**Expected Impact**:
- More expressive field
- +25-50% parameters in field
- Better generation quality (potential 10-20% perplexity improvement)
- Slower inference

**Test**:
```bash
python scripts/train_scaled_nflm.py \
    --field-layers 10 \
    --num-samples 10000 \
    --epochs 10
```

#### 2. Adjust SIREN omega_0

**When**: Training is unstable or field doesn't learn well

**Current**: `omega_0=30.0` (default)

**Change**:
```python
omega_0=30.0 → 10.0 or 50.0
```

**Expected Impact**:
- Changes frequency of sine activations
- Lower = smoother field, may underfit
- Higher = more detailed field, may overfit
- Try 10, 30, 50 and measure

**Modification Required**: Add omega_0 parameter to ScaledGenerativeField.__init__

#### 3. Field Hidden Dimension

**When**: Field capacity is insufficient

**Change**:
```python
hidden_dim=1024 → 1536 or 2048
```

**Expected Impact**:
- Larger field capacity
- +50-100% parameters in field
- Significant memory increase
- Potentially much better performance (15-30%)

**Test**:
```bash
python scripts/train_scaled_nflm.py \
    --field-hidden-dim 1536 \
    --num-samples 10000 \
    --epochs 10
```

---

## Training Optimizations

### Learning Rate Optimization

**Current**: `lr=1e-4` (AdamW)

**Strategies:**

#### 1. Learning Rate Warmup

**When**: Training is unstable early on

**Implementation**:
```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=1000)
main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[1000])
```

**Expected Impact**:
- More stable early training
- Better final performance (typically 3-8%)
- Recommended for large models

#### 2. Cosine Annealing

**When**: Want better convergence

**Implementation**:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
```

**Expected Impact**:
- Smooth learning rate decay
- Better fine-tuning in later epochs
- Typically 2-5% better final performance

#### 3. Learning Rate Finder

**When**: Unsure what learning rate to use

**Implementation**:
```python
# Test range from 1e-6 to 1e-2
# Plot loss vs learning rate
# Choose lr just before loss starts increasing
```

**Typical Findings**:
- Neural fields: 1e-4 to 5e-4
- Transformers: 3e-4 to 1e-3
- Larger models: lower learning rates

### Batch Size Optimization

**Current**: Varies by hardware

**Guidelines:**

| Model Size | GPU Memory | Recommended Batch Size | Gradient Accumulation |
|-----------|------------|------------------------|----------------------|
| ~100M | 8GB | 4-8 | 4-8 steps |
| ~100M | 16GB | 16-32 | 2-4 steps |
| ~100M | 24GB+ | 32-64 | 1-2 steps |

**Effective Batch Size** = Batch Size × Gradient Accumulation Steps

**Recommendations:**
- Larger batches (32-128 effective) generally better
- Adjust learning rate: `new_lr = base_lr * sqrt(new_batch_size / base_batch_size)`
- Use gradient accumulation if memory limited

**Implementation**:
```python
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    loss = model(inputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Training Stability

**Common Issues and Fixes:**

#### 1. Gradient Explosion

**Symptoms**: Loss becomes NaN, weights explode

**Fixes**:
- Reduce learning rate (1e-4 → 5e-5)
- Increase gradient clipping (1.0 → 0.5)
- Add layer normalization
- Check initialization

**Implementation**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

#### 2. Gradient Vanishing

**Symptoms**: Loss doesn't decrease, very small gradients

**Fixes**:
- Increase learning rate
- Reduce model depth
- Add residual connections
- Use better initialization

#### 3. Loss Plateaus

**Symptoms**: Loss stops decreasing but hasn't converged

**Fixes**:
- Increase model capacity
- Reduce regularization
- Add learning rate warmup
- Check for data quality issues

---

## Regularization Strategies

### Prevent Overfitting

**When**: Validation loss increases while training loss decreases

**Options:**

#### 1. Dropout

**Current**: Some layers have dropout=0.1

**Adjust**:
```python
dropout=0.1 → 0.2 or 0.3  # More regularization
dropout=0.1 → 0.05  # Less regularization
```

**Guidelines**:
- Smaller datasets: higher dropout (0.2-0.3)
- Larger datasets: lower dropout (0.05-0.1)
- Monitor validation loss

#### 2. Weight Decay

**Current**: `weight_decay=0.01` in AdamW

**Adjust**:
```python
weight_decay=0.01 → 0.1  # Stronger regularization
weight_decay=0.01 → 0.001  # Weaker regularization
```

**Guidelines**:
- Start with 0.01
- Increase if overfitting
- Decrease if underfitting

#### 3. Early Stopping

**Current**: Implemented with patience=5

**Adjust**:
```python
patience=5 → 10  # More patient (larger models need more)
patience=5 → 3  # Less patient (faster experimentation)
```

**Guidelines**:
- Patience = 5-10 epochs typical
- Monitor validation loss
- Save best model automatically

---

## Model Scaling

### Scaling Laws

**General Principle**: Performance improves with scale, but with diminishing returns.

**Scaling Dimensions:**
1. **Data**: Most important, scales best
2. **Model Size**: Important, expensive
3. **Compute**: Enables above two

### Compute-Optimal Scaling

**Chinchilla Scaling Law** (approximate):
- For N parameters, optimal dataset size ≈ 20N tokens
- Example: 100M params → 2B tokens optimal
- More data better than bigger model if budget limited

### Progressive Scaling Strategy

**Phase 1: Validate** (1K-10K samples)
- Quick iteration
- Test architecture changes
- Identify obvious issues
- Time: minutes to hours

**Phase 2: Optimize** (50K-100K samples)
- Fine-tune hyperparameters
- Compare architectures
- Measure compression benefits
- Time: hours to day

**Phase 3: Scale** (500K-1M samples)
- Validated architecture
- Known hyperparameters
- Measure scaling properties
- Time: days

**Phase 4: Production** (5M-3B samples)
- Only if benefits proven
- Full training run
- Final model
- Time: weeks to months

---

## Memory Optimization

### Reduce Memory Usage

**Strategies:**

#### 1. Mixed Precision Training

**Current**: Optional with `--use-amp`

**Implementation**:
```bash
python scripts/train_scaled_nflm.py --use-amp
```

**Expected Impact**:
- 50% memory reduction
- 2x faster training on compatible GPUs
- Minimal accuracy loss (<1%)
- **Highly Recommended** for large models

#### 2. Gradient Checkpointing

**When**: Model too large for GPU memory

**Implementation**:
```python
from torch.utils.checkpoint import checkpoint

# In forward pass:
output = checkpoint(self.transformer, x)
```

**Expected Impact**:
- 30-50% memory reduction
- 20-30% slower training
- Allows larger models on same hardware

#### 3. Smaller Batch Size + Gradient Accumulation

**When**: Out of memory during training

**Strategy**:
- Reduce batch size: 16 → 8 → 4
- Increase accumulation: 1 → 2 → 4
- Keep effective batch size constant

**Example**:
```bash
# Instead of batch_size=16
python scripts/train_scaled_nflm.py --batch-size 4 --gradient-accumulation-steps 4
```

#### 4. Model Parallelism

**When**: Model too large for single GPU

**Options**:
- Pipeline parallelism (split layers across GPUs)
- Tensor parallelism (split tensors across GPUs)
- ZeRO optimizer (distributed optimizer states)

**Note**: Requires multi-GPU setup, complex implementation.

---

## Data Optimization

### Data Quality

**High Quality Data** > **Large Quantity Poor Data**

**Filtering Strategies:**

#### 1. Perplexity Filtering

**Method**: Train small model on high-quality subset, filter high-perplexity samples

**Implementation**:
```python
# Train small model on WikiText
# Score all data
# Keep samples with perplexity < threshold
```

**Impact**: 20-30% less data, 10-20% better final performance

#### 2. Deduplication

**Method**: Remove near-duplicate samples using MinHash

**Implementation**:
```python
from datasketch import MinHash, MinHashLSH
# Deduplicate corpus
```

**Impact**: 30-40% reduction in data, no performance loss, faster training

#### 3. Length Filtering

**Method**: Remove very short (<50 chars) and very long (>1M chars) samples

**Impact**: Faster training, more stable batch sizes

### Data Augmentation

**For Language Models:**
- Paraphrasing
- Back-translation
- Masking
- Span corruption

**Note**: Generally less effective than getting more real data.

### Curriculum Learning

**Strategy**: Train on easier examples first, then harder

**Implementation**:
```python
# Epoch 1-10: Short sequences
# Epoch 11-20: Medium sequences
# Epoch 21-30: Long sequences
```

**Expected Impact**:
- Faster initial convergence
- Better final performance (5-10%)
- More stable training

---

## Baseline Transformer Optimizations

### If Baseline Underperforms

**Since baseline is standard architecture, optimizations are well-known:**

#### 1. Increase Depth

**Change**:
```python
num_layers=12 → 16 or 24
```

**Impact**: More capacity, better performance, much slower

#### 2. Increase Width

**Change**:
```python
d_model=512 → 768 or 1024
```

**Impact**: More parameters, better performance

#### 3. More Attention Heads

**Change**:
```python
nhead=8 → 12 or 16
```

**Impact**: Better attention patterns, marginal improvement

#### 4. Better Hyperparameters

- Learning rate: 3e-4 to 1e-3
- Warmup: 2000-4000 steps
- Weight decay: 0.01-0.1
- Dropout: 0.1-0.2

---

## Hybrid Architecture Ideas

### If Both Approaches Have Merit

**Hybrid Option 1: Neural Field for Generation, Transformer for Encoding**
```python
class HybridLM(nn.Module):
    def __init__(self):
        self.encoder = TransformerEncoder()  # Strong sequence modeling
        self.coord_projection = CoordinateProjection()
        self.field = SIREN Field()  # Smooth generation
```

**Hybrid Option 2: Transformer with Coordinate Attention**
```python
class CoordinateAttention(nn.Module):
    # Attention mechanism in learned coordinate space
    # Combines discrete and continuous representations
```

**Hybrid Option 3: Ensemble**
```python
# Use both models
# Average predictions or use neural field for uncertainty cases
```

---

## Monitoring and Debugging

### Essential Metrics to Track

**During Training:**
1. Training loss (every batch)
2. Validation loss (every epoch)
3. Gradient norms (every N batches)
4. Learning rate (every step)
5. Perplexity (derived from loss)

**Periodic Evaluation:**
1. Generation samples (every 10 epochs)
2. Interpolation quality (neural field only)
3. Attention visualizations
4. Coordinate space structure

### Debugging Checklist

**If model isn't learning:**
- [ ] Check data loading (print samples)
- [ ] Verify loss calculation (should decrease)
- [ ] Check learning rate (not too low/high)
- [ ] Examine gradients (not zero, not exploding)
- [ ] Test on tiny dataset (should overfit)

**If model trains but doesn't generalize:**
- [ ] Check validation set (representative?)
- [ ] Add regularization (dropout, weight decay)
- [ ] More data or better data quality
- [ ] Reduce model size (if overfitting badly)

**If training is unstable:**
- [ ] Reduce learning rate
- [ ] Add gradient clipping
- [ ] Check initialization
- [ ] Add layer normalization
- [ ] Use mixed precision carefully

---

## Quick Reference: Common Optimizations

### To Improve Performance

| Goal | Action | Expected Gain |
|------|--------|---------------|
| Better perplexity | More data | 10-30% |
| Better perplexity | Larger model | 5-20% |
| Better perplexity | Better hyperparameters | 3-10% |
| Faster training | Mixed precision | 2x speed |
| Faster training | Larger batch size | 1.5-2x speed |
| Less memory | Smaller batch + gradient accumulation | 50% memory |
| Less memory | Mixed precision | 50% memory |
| Less memory | Gradient checkpointing | 30-40% memory |

### To Fix Issues

| Issue | Solution | Notes |
|-------|----------|-------|
| Loss = NaN | Reduce LR, clip gradients | Usually gradient explosion |
| No learning | Increase LR, check data | Might be too small LR |
| Overfit | More data, dropout, weight decay | Val loss > train loss |
| Underfit | Bigger model, less regularization | Val loss ≈ train loss |
| OOM | Smaller batch, mixed precision | Out of memory |
| Slow training | Mixed precision, larger batch | GPU underutilized |

---

## Optimization Workflow

### Step-by-Step Process

1. **Baseline**: Train with default settings, establish baseline performance
2. **Diagnose**: Identify specific issues or goals
3. **Hypothesize**: Choose one optimization to try
4. **Test**: Run experiment with changed parameter
5. **Measure**: Compare to baseline quantitatively
6. **Decide**: Keep change if improves performance >3%
7. **Document**: Record results in WIN_LOSE_ANALYSIS.md
8. **Repeat**: Iterate on next optimization

### Priority Order

**For Best ROI:**
1. Data quality and quantity (biggest impact)
2. Learning rate and schedule (easy wins)
3. Batch size and mixed precision (efficiency)
4. Model architecture (if above don't help)
5. Advanced techniques (last resort)

---

## Conclusion

**Key Principles:**
1. **Measure everything**: Track metrics religiously
2. **One change at a time**: Isolate variables
3. **Scale gradually**: Validate before committing compute
4. **Document results**: Build institutional knowledge
5. **Be pragmatic**: Sometimes baseline is best

**Remember**: The goal is not to make neural fields win at all costs, but to find the best solution for the task. If transformers work better, use transformers. If neural fields excel in specific scenarios, use them there.

**Optimization is empirical**: These guidelines are starting points. Your mileage may vary. Always validate on your specific task and data.

---

**Last Updated**: [Auto-updated]
**Version**: 1.0
**Status**: Living document - update based on experimental findings
