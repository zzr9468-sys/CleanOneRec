# Migration Guide: MiniOneRec → RecRL

This guide helps you migrate from the old MiniOneRec framework to the new RecRL framework.

## Key Differences

### Architecture

**MiniOneRec:**
- Monolithic `minionerec_trainer.py` (500+ lines)
- All logic (GRPO, EEPO, rewards, data) in one file
- Hard to test, extend, or modify

**RecRL:**
- Modular design with clear separation
- Each component is independent and testable
- Easy to add new algorithms or rewards

### Code Comparison

#### Old Way (MiniOneRec)

```python
from minionerec_trainer import ReReTrainer
from reward import SemanticRewardEngine
from dataset import build_grpo_dataset

# Everything mixed together
trainer = ReReTrainer(
    model=model,
    base_model=base_model,
    reward_funcs=[reward_engine.compute_reward],
    args=training_args,
    train_dataset=train_dataset,
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,
    eepo_unlearn_lr=1e-5,
    eepo_unlearn_weight=1.0,
    add_gt=True,
)
trainer.train()
```

#### New Way (RecRL)

```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler

# Clean separation of concerns
data_engine = DataEngine.from_parquet("train.parquet")
rollout_engine = RolloutEngine(tokenizer)
reward_engine = TextSemanticReward(recif_path)

config = EEPOConfig(
    num_generations=16,
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,
    eepo_unlearn_lr=1e-5,
    add_gt=True
)

sampler = RepeatRandomSampler(data_engine, repeat_count=16)

trainer = EEPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=data_engine,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
    sampler=sampler
)
trainer.train()
```

## Component Mapping

### Data Loading

**Old:**
```python
from dataset import build_grpo_dataset
train_dataset = build_grpo_dataset(train_file, sample_num=1000)
```

**New:**
```python
from recrl.core import DataEngine
train_dataset = DataEngine.from_parquet(train_file, format="recif", sample_num=1000)
```

### Reward Functions

**Old:**
```python
from reward import SemanticRewardEngine
reward_engine = SemanticRewardEngine(recif_path=recif_path, device=device)
reward_funcs = [reward_engine.compute_reward]
```

**New:**
```python
from recrl.rewards import TextSemanticReward
reward_engine = TextSemanticReward(recif_path=recif_path, device=device)
# Can compose multiple rewards
from recrl.core import CompositeReward
reward_engine = CompositeReward([
    (TextSemanticReward(recif_path), 0.8),
    (ExactMatchReward(), 0.2)
])
```

### Training Configuration

**Old:**
```python
from trl import GRPOConfig
training_args = GRPOConfig(
    output_dir=output_dir,
    num_generations=num_generations,
    temperature=temperature,
    per_device_train_batch_size=train_batch_size,
    # ... many more params
)
```

**New:**
```python
from recrl.algorithms.eepo import EEPOConfig
config = EEPOConfig(
    num_generations=16,
    temperature=0.7,
    per_device_batch_size=4,
    learning_rate=1e-6,
    # EEPO-specific
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,
    output_dir=output_dir
)
```

### Trainer Initialization

**Old:**
```python
trainer = ReReTrainer(
    model=model,
    base_model=model_path,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[reward_engine.compute_reward],
    eepo_enabled=eepo_enabled,
    eepo_stage1_ratio=eepo_stage1_ratio,
    eepo_unlearn_lr=eepo_unlearn_lr,
    eepo_unlearn_weight=eepo_unlearn_weight,
    add_gt=add_gt,
)
```

**New:**
```python
trainer = EEPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,  # All params in config
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
    sampler=sampler
)
```

## Migration Steps

### Step 1: Install RecRL

```bash
cd CleanOneRec
pip install -e recrl/
```

### Step 2: Update Imports

Replace:
```python
from minionerec_trainer import ReReTrainer
from reward import SemanticRewardEngine
from dataset import build_grpo_dataset
```

With:
```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler
```

### Step 3: Refactor Data Loading

Replace:
```python
train_dataset = build_grpo_dataset(train_file, sample_num=1000)
```

With:
```python
train_dataset = DataEngine.from_parquet(
    path=train_file,
    format="recif",
    sample_num=1000
)
```

### Step 4: Setup Components

Add explicit component initialization:
```python
# Rollout engine
rollout_engine = RolloutEngine(
    tokenizer=tokenizer,
    device=device
)

# Reward engine
reward_engine = TextSemanticReward(
    recif_path=recif_path,
    device=device
)

# Sampler
sampler = RepeatRandomSampler(
    data_source=train_dataset,
    repeat_count=config.num_generations,
    seed=42
)
```

### Step 5: Update Trainer Initialization

Replace the old trainer with the new one, moving all hyperparameters into the config.

### Step 6: Test

Run your training script and verify it works correctly.

## Benefits of Migration

1. **Cleaner Code**: Explicit separation of concerns
2. **Easier Testing**: Mock individual components
3. **Better Extensibility**: Add new algorithms without modifying existing code
4. **Composable Rewards**: Easily combine multiple reward functions
5. **Type Safety**: Better IDE support and type hints

## Example: Full Migration

**Before (MiniOneRec):**
```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from dataset import build_grpo_dataset
from reward import SemanticRewardEngine
from minionerec_trainer import ReReTrainer

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
reward_engine = SemanticRewardEngine(recif_path=recif_path)
train_dataset = build_grpo_dataset(train_file, sample_num=1000)

training_args = GRPOConfig(
    output_dir=output_dir,
    num_generations=16,
    temperature=0.7,
    per_device_train_batch_size=4,
)

trainer = ReReTrainer(
    model=model,
    base_model=model_path,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=[reward_engine.compute_reward],
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,
    add_gt=True,
)
trainer.train()
```

**After (RecRL):**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler

# Load model
model = AutoModelForCausalLM.from_pretrained(model_path)
ref_model = AutoModelForCausalLM.from_pretrained(model_path)
ref_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Setup components
train_dataset = DataEngine.from_parquet(train_file, sample_num=1000)
rollout_engine = RolloutEngine(tokenizer, device="cuda")
reward_engine = TextSemanticReward(recif_path=recif_path)

# Configure
config = EEPOConfig(
    num_generations=16,
    temperature=0.7,
    per_device_batch_size=4,
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,
    add_gt=True,
    output_dir=output_dir
)

sampler = RepeatRandomSampler(train_dataset, repeat_count=16)

# Train
trainer = EEPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
    sampler=sampler
)
trainer.train()
```

## Troubleshooting

### Issue: Import errors

**Solution:** Make sure you've installed RecRL:
```bash
cd CleanOneRec
pip install -e recrl/
```

### Issue: Missing reference model

**Solution:** RecRL requires explicit reference model:
```python
ref_model = AutoModelForCausalLM.from_pretrained(model_path)
ref_model.eval()
```

### Issue: Sampler not working

**Solution:** Make sure to use RepeatRandomSampler:
```python
from recrl.data import RepeatRandomSampler
sampler = RepeatRandomSampler(train_dataset, repeat_count=config.num_generations)
```

## Need Help?

- Check `recrl/examples/` for complete working examples
- Read `recrl/README.md` for API documentation
- Review `ARCHITECTURE.md` for design principles
