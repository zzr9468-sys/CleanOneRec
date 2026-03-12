# Training Guide

This guide covers single-GPU, multi-GPU, and algorithm-specific training configurations.

## Quick Start

### Single-GPU GRPO Training (Constrained)

```bash
bash train_constrained.sh
```

This runs GRPO with constrained decoding (recommended for best performance).

### Multi-GPU GRPO Training

```bash
bash train_multi_gpu.sh
```

Uses 4 GPUs with Accelerate DDP.

### EEPO Training (Exploration)

```bash
bash train_eepo.sh
```

Runs EEPO with fast-weight exploration for better diversity.

## Configuration Files

All configs are in `configs/` directory:

- `grpo_video.yaml` - GRPO for video recommendation
- `eepo_video.yaml` - EEPO for video recommendation

### Basic Configuration

```yaml
# Model and Data
model_path:  /path/to/OneRec-1.7B
data_path:   /path/to/video_test.parquet
recif_path:  /path/to/OpenOneRec-RecIF
algo:        grpo  # or eepo
device:      cuda:0

# Training Hyperparameters
num_epochs:                1
per_device_batch_size:     2
gradient_accumulation_steps: 4
learning_rate:             5.0e-6
warmup_steps:              50
max_grad_norm:             1.0

# Generation
num_generations:           4
max_completion_length:     128
temperature:               0.7

# RL
beta: 0.04  # KL penalty coefficient

# Reward Weights (must sum to 1.0)
reward_weights:
  longview:  0.50
  semantic:  0.30
  novelty:   0.15
  diversity: 0.05

# Logging
logging_steps: 1
save_steps:    200
```

## Training Commands

### Basic Training

```bash
python train.py \
    --config configs/grpo_video.yaml \
    --output ./outputs/my_experiment
```

### With Constrained Decoding (Recommended)

```bash
python train.py \
    --config configs/grpo_video.yaml \
    --constrained \
    --output ./outputs/constrained_exp
```

**Why constrained?** Ensures model only generates valid item IDs, improving reward from -0.19 to 0.78 (488% improvement).

### With SwanLab Logging

```bash
python train.py \
    --config configs/grpo_video.yaml \
    --constrained \
    --report-to swanlab \
    --run-name my_experiment \
    --output ./outputs/my_experiment
```

### Multi-GPU Training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --config_file accelerate_config.yaml \
    train.py \
    --config configs/grpo_video.yaml \
    --constrained \
    --output ./outputs/multi_gpu_exp
```

## Hyperparameter Tuning

### Learning Rate

- **Default**: `5.0e-6`
- **Range**: `1.0e-6` to `1.0e-5`
- **Too high**: Loss spikes, unstable training
- **Too low**: Slow convergence, no improvement

### Batch Size and Gradient Accumulation

Effective batch size = `per_device_batch_size × gradient_accumulation_steps × num_gpus`

- **Small GPU (24GB)**: `per_device_batch_size=1`, `gradient_accumulation_steps=8`
- **Large GPU (80GB)**: `per_device_batch_size=4`, `gradient_accumulation_steps=2`

### Number of Generations

- **Default**: `4`
- **Range**: `2` to `16`
- **More generations**: Better advantage estimation, but slower
- **Fewer generations**: Faster training, but noisier gradients

### Temperature

- **Default**: `0.7`
- **Range**: `0.5` to `1.0`
- **Higher**: More diverse generations, better exploration
- **Lower**: More focused generations, faster convergence

### Beta (KL Penalty)

- **Default**: `0.04`
- **Range**: `0.01` to `0.1`
- **Higher**: Stays closer to reference model, more conservative
- **Lower**: Deviates more from reference, more aggressive

### Reward Weights

Adjust based on your task:

```yaml
# Emphasize relevance
reward_weights:
  longview:  0.60  # Increase
  semantic:  0.30
  novelty:   0.05  # Decrease
  diversity: 0.05

# Emphasize diversity
reward_weights:
  longview:  0.40  # Decrease
  semantic:  0.20
  novelty:   0.25  # Increase
  diversity: 0.15  # Increase
```

## EEPO-Specific Configuration

```yaml
algo: eepo

# EEPO Parameters
eepo_enabled:       true
eepo_stage1_ratio:  0.5      # 50% exploitation, 50% exploration
eepo_unlearn_lr:    1.0e-3   # Fast-weight learning rate
eepo_unlearn_weight: 0.1     # Unlearning strength
eepo_epsilon:       1.0e-8   # Numerical stability
add_gt:             false    # Inject ground truth
```

### EEPO Hyperparameters

- **eepo_stage1_ratio**: Exploitation vs exploration ratio
  - `0.5`: Balanced (recommended)
  - `0.7`: More exploitation, less exploration
  - `0.3`: More exploration, less exploitation

- **eepo_unlearn_lr**: Fast-weight update strength
  - `1.0e-3`: Default
  - Higher: Stronger exploration, may be unstable
  - Lower: Weaker exploration, more conservative

- **eepo_unlearn_weight**: Unlearning weight
  - `0.1`: Default
  - Controls how much to "forget" recent patterns

## Monitoring Training

### Check Training Progress

```bash
python check_training_progress.py logs/train_constrained.log
```

Output:
```
📊 Training Progress Analysis
============================================================
Total steps: 424
Latest step: 423

Recent 20 steps:
  Mean reward: 0.771
  Min reward:  0.443
  Max reward:  0.818

First 20 steps mean: 0.760
Last 20 steps mean:  0.771
Improvement:         +0.011
```

### SwanLab Dashboard

Visit https://swanlab.cn to view:
- Reward curves
- Loss curves
- Learning rate schedule
- Sample generations

### TensorBoard (Alternative)

```bash
tensorboard --logdir outputs/my_experiment/tb_logs
```

## Training Tips

### 1. Start with Constrained Decoding

Always use `--constrained` flag. It dramatically improves reward quality.

### 2. Monitor Reward, Not Loss

GRPO/EEPO loss ≈ 0 is normal (policy gradient method). Focus on reward trend.

### 3. Need 200+ Steps for Clear Improvement

Don't judge performance from first 50 steps. Wait for at least 200 steps.

### 4. GRPO vs EEPO

- **GRPO**: Faster, more stable, good for initial experiments
- **EEPO**: Better exploration, escapes local optima, good for final performance

### 5. Gradient Checkpointing

Already enabled in `train.py`. Saves ~40% GPU memory with minimal slowdown.

### 6. Reference Model CPU Offload

Already implemented. Keeps ref_model on CPU, temporarily moves to GPU for inference.

## Troubleshooting

### Loss = 0.0000 Throughout Training

**Cause**: Missing data fields or reward computation bug

**Solution**: Check logs for reward values. If rewards are also 0, check data loading.

### Reward Not Improving

**Possible causes**:
1. Learning rate too low → Increase to `5.0e-6`
2. Not using constrained decoding → Add `--constrained`
3. Need more steps → Wait for 200+ steps

### CUDA Out of Memory

**Solutions**:
1. Reduce `per_device_batch_size` to 1
2. Reduce `num_generations` to 2
3. Use multi-GPU training
4. Gradient checkpointing (already enabled)

### Multi-GPU Device Mismatch Error

**Solution**: Already fixed in latest code. Update to latest version.

## Advanced: Custom Training Script

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import CompositeReward, LongviewReward, SemanticReward
from recrl.constraints import SIDTrie, ConstrainedLogitsProcessor

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("path/to/model")
model.gradient_checkpointing_enable()
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Load data
train_dataset = DataEngine.from_recif_parquet("data/train.parquet")

# Setup engines
rollout_engine = RolloutEngine(tokenizer, device="cuda")

# Build SID trie for constrained decoding
trie = SIDTrie.from_recif("path/to/RecIF", tokenizer)
rollout_engine.set_trie(trie)

# Composite reward
reward_engine = CompositeReward([
    (LongviewReward("path/to/RecIF"), 0.5),
    (SemanticReward("path/to/RecIF"), 0.3),
])

# Training config
config = GRPOConfig(
    num_epochs=1,
    per_device_batch_size=2,
    learning_rate=5.0e-6,
    num_generations=4,
    temperature=0.7,
    beta=0.04,
)

# Train
trainer = GRPOTrainer(
    model=model,
    ref_model=None,  # Will use model as ref
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
)
trainer.train()
```

## Next Steps

- [Algorithm Documentation](ALGORITHMS.md) - Understand GRPO and EEPO
- [Benchmarks](BENCHMARKS.md) - See performance comparisons
- [Installation Guide](INSTALLATION.md) - Setup instructions
