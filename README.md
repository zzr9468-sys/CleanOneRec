# RecRL: Reinforcement Learning Framework for Recommendation LLMs

A clean, modular, and highly extensible framework for training Large Language Models (LLMs) for recommendation tasks using Reinforcement Learning. Inspired by Volcano Engine's VERL, **RecRL** provides an elegant architecture that completely decouples data, rollout, rewards, and training logic.

*Read this in [中文](README_zh.md).*

## ✨ Key Features

- **Modular "Engine" Design**: Clean separation of `DataEngine`, `RolloutEngine`, `RewardEngine`, and `TrainerEngine`.
- **Multiple Algorithms**: Native support for **GRPO** (Group Relative Policy Optimization) and **EEPO** (Explore-and-Evaluate Policy Optimization with fast-weight unlearning).
- **Composable Rewards**: Easily combine multiple reward functions (e.g., Semantic Similarity, Exact Match, NDCG) with custom weights like building blocks.
- **Constrained Decoding**: Built-in Trie-based `ConstrainedLogitsProcessor` ensures the LLM generates strictly valid Item IDs (SIDs), eliminating "hallucinated" recommendations and reward collapse.
- **Extensible API**: Implement new algorithms (like DPO or PPO) in under 100 lines of code without touching the core generation loop.

## 📦 Installation

RecRL requires Python 3.8+ and PyTorch 2.0+. 

```bash
# Clone the repository
https://github.com/zzr9468-sys/CleanOneRec.git
cd RecRL

# Install in editable mode
pip install -e .
```

## 🚀 Quick Start

### 1. Standard GRPO Training

```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler

# Load your recommendation dataset
train_dataset = DataEngine.from_parquet("data/train.parquet", format="recif")

# Setup core engines
rollout_engine = RolloutEngine(tokenizer, device="cuda")
reward_engine = TextSemanticReward(recif_path="data/recif_metadata")

# Configure training and sampler
config = GRPOConfig(num_generations=16, temperature=0.7)
sampler = RepeatRandomSampler(train_dataset, repeat_count=16)

# Initialize trainer and start
trainer = GRPOTrainer(
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

### 2. Fast-Weight EEPO Training

EEPO performs in-place `lm_head` fast-weight updates during generation to encourage exploratory recommendations and break student-teacher exposure bias.

```python
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig

config = EEPOConfig(
    num_generations=16,
    eepo_enabled=True,
    eepo_stage1_ratio=0.5,  # 50% exploitation, 50% exploration
    eepo_unlearn_lr=1e-5,
    add_gt=True
)

trainer = EEPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
    sampler=sampler
)
trainer.train()
```

### 3. Combining Multiple Rewards

Easily balance relevance, novelty, and exact-match constraints using the `CompositeReward` system:

```python
from recrl.core import CompositeReward
from recrl.rewards import TextSemanticReward, ExactMatchReward

reward_engine = CompositeReward([
    (TextSemanticReward(recif_path="data/"), 0.8),  # 80% weight on Semantic Similarity
    (ExactMatchReward(), 0.2)                       # 20% weight on Exact ID Match
])
```

## 🏗 Architecture

RecRL breaks away from monolithic trainer scripts by adopting a decoupled, Engine-based approach:

```text
recrl/
├── core/                       # Core abstractions
│   ├── base_trainer.py         # BaseRLTrainer loop
│   ├── rollout.py              # Generation & logprobs (RolloutEngine)
│   ├── reward.py               # Reward interfaces (CompositeReward)
│   └── data.py                 # Data parser (DataEngine)
├── algorithms/                 # RL implementations
│   ├── grpo/                   # Standard GRPO
│   └── eepo/                   # EEPO (Unlearn mechanism)
├── rewards/                    # Pluggable reward modules
│   ├── semantic.py             # Sentence-Transformer text similarity
│   └── rule.py                 # Exact Match / NDCG
├── constraints/                # Valid ID enforcement
│   ├── trie.py                 # Prefix Tree for Valid SIDs
│   └── processor.py            # NaN-safe ConstrainedLogitsProcessor
└── examples/                   # Ready-to-run training scripts
```

## 🤝 Contributing

We welcome contributions! To add a new algorithm (e.g., DPO):
1. Create a new folder under `recrl/algorithms/dpo/`.
2. Inherit from `BaseRLTrainer` and override the `compute_loss()` function.
3. Your algorithm will automatically inherit constrained decoding, reward composition, and dataset handling.

## 📄 License

This project is licensed under the Apache 2.0 License.
