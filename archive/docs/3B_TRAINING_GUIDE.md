# 3B Sample Training Guide

## Overview

This guide provides the infrastructure and methodology for training neural field language models on 3 billion samples, representing production-scale training.

**Status**: This should only be attempted after validation at smaller scales (50K → 500K → 5M samples) shows clear benefits.

## Prerequisites

### Validation Requirements

Before attempting 3B sample training, you must have:

1. ✅ **Completed 50K sample experiment**
   - Neural field shows competitive or better performance
   - Architecture is stable and converging
   - No major training issues identified

2. ✅ **Completed 500K sample experiment**
   - Compression benefits are measurable (5%+ improvement)
   - Scaling curve looks promising
   - Generation quality is acceptable

3. ✅ **Completed 5M sample experiment**
   - Benefits persist at larger scale
   - No performance plateaus
   - Model continues to improve with more data

4. ✅ **Resource availability**
   - GPU cluster with 4-8 GPUs (A100 or V100 recommended)
   - 200-500 GB RAM
   - 2-5 TB storage for checkpoints
   - 2-4 weeks of continuous training time

### Decision Criteria

**Proceed to 3B if:**
- Neural field shows 10%+ perplexity improvement at 5M samples
- Benefits are consistent across scales
- Generation quality is high
- Have compute budget for 2-4 weeks

**Do NOT proceed if:**
- Baseline transformer performs equally or better
- Benefits are marginal (<3% improvement)
- Training is unstable at 5M samples
- Limited compute resources

## Infrastructure Setup

### Hardware Requirements

**Minimum Configuration:**
- 4x NVIDIA A100 (40GB) GPUs
- 256 GB RAM
- 2 TB NVMe storage
- 10 Gbps network connection

**Recommended Configuration:**
- 8x NVIDIA A100 (80GB) GPUs
- 512 GB RAM
- 5 TB NVMe storage
- 25 Gbps network connection

**Cost Estimate:**
- Cloud (AWS p4d.24xlarge): $32/hour × 336 hours = ~$10,750
- Cloud (8x A100): ~$20,000-$30,000 for full training
- On-premise: Significant upfront investment

### Software Requirements

```bash
# Install distributed training libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets huggingface-hub
pip install deepspeed accelerate
pip install wandb  # For experiment tracking

# Verify multi-GPU setup
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

### Data Preparation

**Dataset Selection:**

Use a high-quality mixture:
- 50% FineWeb-Edu (1.5B samples)
- 30% Dolma (900M samples)
- 20% C4 (600M samples)

**Preprocessing:**
```bash
# Preprocess and cache dataset (one-time setup)
python scripts/prepare_3b_dataset.py \
    --output-dir /data/3b_training_cache \
    --num-workers 16
```

## Training Configuration

### Model Configuration

**Optimized for 3B Samples:**

```python
model_config = {
    # Scale up from 98M to 200-300M parameters
    "vocab_size": 50257,
    "coord_dim": 16,  # Increased from 8
    "encoder_hidden_dim": 768,  # Increased from 512
    "encoder_layers": 6,  # Increased from 4
    "field_hidden_dim": 1536,  # Increased from 1024
    "field_layers": 10,  # Increased from 8
}

# Expected parameters: ~250M
```

### Training Hyperparameters

```python
training_config = {
    "num_samples": 3_000_000_000,
    "epochs": 1,  # One pass through 3B samples
    "batch_size": 32,  # Per GPU
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 32 * 4 * 8,  # 1024 samples
    "learning_rate": 5e-5,  # Lower for large-scale training
    "warmup_steps": 10_000,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "save_every": 50_000,  # Save every 50K steps
    "eval_every": 10_000,  # Evaluate every 10K steps
}
```

### Distributed Training Setup

**Using DeepSpeed ZeRO Stage 2:**

```json
{
  "train_batch_size": 1024,
  "gradient_accumulation_steps": 4,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 5e-5,
      "warmup_num_steps": 10000,
      "total_num_steps": 3000000
    }
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "steps_per_print": 100,
  "wall_clock_breakdown": false
}
```

## Training Scripts

### Main 3B Training Script

**Usage:**

```bash
# Single node, 8 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_3b_nflm.py \
    --config configs/3b_training.yaml \
    --output-dir /checkpoints/3b_nflm

# Multi-node (4 nodes, 8 GPUs each)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_3b_nflm.py \
    --config configs/3b_training.yaml \
    --output-dir /checkpoints/3b_nflm
```

### Using DeepSpeed

```bash
deepspeed --num_gpus=8 \
    scripts/train_3b_nflm.py \
    --deepspeed configs/deepspeed_config.json \
    --config configs/3b_training.yaml \
    --output-dir /checkpoints/3b_nflm
```

### Using Accelerate

```bash
accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --mixed_precision=fp16 \
    scripts/train_3b_nflm.py \
    --config configs/3b_training.yaml \
    --output-dir /checkpoints/3b_nflm
```

## Monitoring and Logging

### Weights & Biases Integration

```python
import wandb

wandb.init(
    project="virgo-3b-training",
    config={
        "model_params": 250_000_000,
        "samples": 3_000_000_000,
        "architecture": "neural_field_lm",
        "experiment": "production_scale"
    }
)

# Log metrics every step
wandb.log({
    "train/loss": loss,
    "train/perplexity": perplexity,
    "train/learning_rate": lr,
    "train/gradient_norm": grad_norm,
})
```

### Key Metrics to Track

**During Training:**
1. Training loss (every step)
2. Validation loss (every 10K steps)
3. Learning rate schedule
4. Gradient norms
5. Memory utilization
6. Throughput (samples/sec)
7. GPU utilization

**Periodic Evaluation:**
1. Generation quality (every 50K steps)
2. Perplexity on diverse domains
3. Interpolation quality
4. Checkpoint size and save time

### Alert Thresholds

**Set up alerts for:**
- Loss becomes NaN or Inf
- Gradient norm > 10.0 (explosion)
- Gradient norm < 0.001 (vanishing)
- GPU memory > 95%
- Validation loss increases 3 steps in a row
- Throughput drops > 20%

## Checkpointing Strategy

### Checkpoint Schedule

**Regular Checkpoints:**
- Every 50,000 steps (~1.5% of training)
- Keep last 5 checkpoints (rolling window)
- Full checkpoint ~100 GB each

**Milestone Checkpoints:**
- Every 500,000 steps (preserve indefinitely)
- 25%, 50%, 75%, 100% completion
- Include full training state

**Best Model:**
- Track validation perplexity
- Save whenever new best is achieved
- Keep separate from regular checkpoints

### Checkpoint Contents

```python
checkpoint = {
    "step": global_step,
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "training_config": config,
    "metrics": {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_perplexity": val_perplexity,
    },
    "samples_seen": samples_seen,
}
```

### Resume Training

```bash
# Resume from specific checkpoint
python scripts/train_3b_nflm.py \
    --resume /checkpoints/3b_nflm/checkpoint_step_500000.pt \
    --config configs/3b_training.yaml
```

## Expected Timeline

### Training Duration

**With 8x A100 GPUs:**

| Metric | Value |
|--------|-------|
| Samples per second | ~3,000 |
| Steps per hour | ~1,000 |
| Total steps needed | ~3,000,000 |
| **Estimated duration** | **~125 days** |

**Optimizations to reduce time:**
- Increase batch size → 75 days
- 16 GPUs → 60 days
- 32 GPUs → 30 days
- 64 GPUs → 15 days

**Realistic Timeline (8 GPUs):**
- With optimizations: **2-4 weeks**
- With interruptions: **4-6 weeks**

### Cost Estimate

**AWS p4d.24xlarge (8x A100 80GB):**
- $32.77/hour
- ~336-672 hours total
- **Total cost: $11,000 - $22,000**

**Alternative Options:**
- GCP: Similar pricing
- Azure: Slightly higher
- Lambda Labs: $4.40/hour (A100) = ~$3,000-$6,000
- Your own hardware: Upfront cost, no per-hour charges

## Validation Strategy

### During Training

**Every 10K steps:**
1. Evaluate on held-out validation set
2. Measure perplexity
3. Generate sample text
4. Check for mode collapse

**Every 50K steps:**
1. Full evaluation suite
2. Generation quality assessment
3. Interpolation tests
4. Save detailed results

### Final Evaluation

**After Training:**
1. Comprehensive benchmark suite
2. Compare to baseline transformer
3. Test on diverse domains
4. Measure semantic interpolation
5. Generate extensive samples
6. Analyze learned coordinate space

## Success Criteria

### Minimum Viable Success

**The model is successful if:**
1. Validation perplexity < 20
2. Generates coherent multi-sentence text
3. Maintains semantic interpolation capability
4. Outperforms baseline by 5%+

### Strong Success

**The model is highly successful if:**
1. Validation perplexity < 15
2. Generates coherent paragraphs
3. Smooth semantic interpolation
4. Outperforms baseline by 15%+
5. Shows novel capabilities (e.g., better few-shot learning)

### Breakthrough

**The model is groundbreaking if:**
1. Validation perplexity < 10
2. Human-like text generation
3. Strong conversational ability
4. Outperforms baseline by 30%+
5. Demonstrates unique neural field advantages

## Risk Mitigation

### Common Issues

**Issue: Training becomes unstable**
- Solution: Reduce learning rate by 2x
- Solution: Increase gradient clipping
- Solution: Add more warmup steps

**Issue: Loss plateaus early**
- Solution: Increase model capacity
- Solution: Adjust learning rate schedule
- Solution: Check data quality

**Issue: Out of memory**
- Solution: Reduce batch size, increase gradient accumulation
- Solution: Enable gradient checkpointing
- Solution: Use DeepSpeed ZeRO Stage 3

**Issue: Slow training**
- Solution: Profile and optimize data loading
- Solution: Increase batch size if memory allows
- Solution: Optimize model architecture

### Backup Plans

**Plan A (Preferred):** Full 3B sample training
**Plan B:** Stop at 1B if no improvement
**Plan C:** Focus on specific domains with less data
**Plan D:** Pivot to baseline transformer

## Baseline Comparisons (3B Scale)

Train baseline transformers at same scale:

**Baseline 1: GPT-2 Medium (355M params)**
```bash
python scripts/train_baseline_3b.py \
    --model gpt2-medium \
    --samples 3000000000 \
    --output-dir /checkpoints/gpt2_medium_3b
```

**Baseline 2: GPT-2 Large (774M params)**
```bash
python scripts/train_baseline_3b.py \
    --model gpt2-large \
    --samples 3000000000 \
    --output-dir /checkpoints/gpt2_large_3b
```

**Baseline 3: Custom Transformer (250M params)**
- Match neural field parameter count exactly
- Fair comparison for compression hypothesis

## Post-Training Analysis

### Comprehensive Evaluation

**1. Perplexity Comparison**
```python
# Evaluate on diverse test sets
test_sets = [
    "wikitext-103",
    "lambada",
    "hellaswag",
    "piqa",
    "winogrande"
]

for test_set in test_sets:
    nf_ppl = evaluate(neural_field_model, test_set)
    baseline_ppl = evaluate(baseline_model, test_set)
    print(f"{test_set}: NF={nf_ppl:.2f}, Baseline={baseline_ppl:.2f}")
```

**2. Generation Quality**
- Human evaluation (coherence, fluency, factuality)
- Automatic metrics (BLEU, ROUGE, BERTScore)
- Domain-specific tests

**3. Interpolation Analysis**
- Test semantic interpolation quality
- Measure smoothness of transitions
- Quantify novel generation capability

**4. Efficiency Metrics**
- Inference speed
- Memory footprint
- Parameter efficiency (performance per parameter)

### Update Documentation

**After completion, update:**
1. `WIN_LOSE_ANALYSIS.md` - Full results
2. `THE_KEY_EXPERIMENT.md` - Final findings
3. `THREE_PHASE_SUMMARY.md` - Production results
4. Create `3B_TRAINING_RESULTS.md` - Detailed analysis

## Conclusion

**3B sample training is the final validation of the neural field approach.**

**Only proceed if:**
- Smaller scales show clear benefits
- Resources are available
- Team is prepared for multi-week training

**Expected outcome:**
- Definitive answer on neural field viability
- Production-ready model if successful
- Valuable insights regardless of result

**Remember:** The goal is to answer definitively whether neural fields compress better than transformers. Even if they don't, the research is valuable.

---

**Status**: Ready to execute pending validation at smaller scales
**Next Step**: Complete 5M sample experiment first
**Decision Point**: Proceed only if 5M results are strongly positive
