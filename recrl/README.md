# RecRL - Reinforcement Learning Framework for Recommendation LLMs

A VERL-inspired modular framework for training recommendation large language models with reinforcement learning.

## Features

- **Modular Design**: Clean separation of data, rollout, reward, and training logic
- **Multiple Algorithms**: GRPO, EEPO (with fast-weight exploration), extensible to DPO
- **Composable Rewards**: Easily combine multiple reward functions with weights
- **Constrained Decoding**: Trie-based constraint enforcement for valid SIDs
- **Clean APIs**: Minimal code to implement new algorithms

## Installation

```bash
cd CleanOneRec
pip install -e .
```

## Quick Start

### GRPO Training

```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler

# Load data
train_dataset = DataEngine.from_parquet("train.parquet", format="recif")

# Setup components
rollout_engine = RolloutEngine(tokenizer)
reward_engine = TextSemanticReward(recif_path="path/to/recif")

# Configure and train
config = GRPOConfig(num_generations=16, temperature=0.7)
sampler = RepeatRandomSampler(train_dataset, repeat_count=16)

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

### EEPO Training

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

### Composite Rewards

```python
from recrl.core import CompositeReward
from recrl.rewards import TextSemanticReward, ExactMatchReward

reward_engine = CompositeReward([
    (TextSemanticReward(recif_path), 0.8),
    (ExactMatchReward(), 0.2)
])
```

## Architecture

```
recrl/
├── core/                       # Core abstractions
│   ├── base_trainer.py         # BaseRLTrainer
│   ├── rollout.py              # RolloutEngine
│   ├── reward.py               # BaseReward, CompositeReward
│   └── data.py                 # DataEngine
├── algorithms/                 # RL algorithms
│   ├── grpo/                   # GRPO implementation
│   └── eepo/                   # EEPO implementation
├── rewards/                    # Reward functions
│   ├── semantic.py             # Text semantic similarity
│   └── rule.py                 # Exact match, NDCG
├── constraints/                # Constrained decoding
│   ├── trie.py                 # SIDTrie
│   └── processor.py            # ConstrainedLogitsProcessor
├── data/                       # Data loading
│   └── sampler.py              # RepeatRandomSampler
└── examples/                   # Usage examples
    ├── train_grpo.py
    ├── train_eepo.py
    └── train_composite.py
```

## Key Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Composability**: Rewards, constraints, and algorithms are independently composable
3. **Extensibility**: Adding new algorithms requires minimal code (~100 lines)
4. **Testability**: Each module can be tested in isolation

## Comparison with MiniOneRec

| Aspect | MiniOneRec | RecRL |
|--------|------------|-------|
| Trainer size | 500+ lines | ~150 lines per algorithm |
| Modularity | Monolithic | Fully decoupled |
| Adding algorithm | Modify existing | Create new subclass |
| Reward composition | Hard-coded | Pluggable |
| Testing | Difficult | Each module testable |

## Examples

See `recrl/examples/` for complete working examples:
- `train_grpo.py` - Standard GRPO training
- `train_eepo.py` - EEPO with fast-weight exploration
- `train_composite.py` - Multiple reward composition

## Contributing

To add a new RL algorithm:

1. Create `recrl/algorithms/your_algo/`
2. Implement `YourAlgoConfig` (inherit from `RLConfig`)
3. Implement `YourAlgoTrainer` (inherit from `BaseRLTrainer`)
4. Override `compute_loss()` with your algorithm's loss

That's it! The framework handles data loading, generation, reward computation, and training loop.

## License

Apache 2.0
