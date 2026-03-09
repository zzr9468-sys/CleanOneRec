# RecRL - Reinforcement Learning Framework for Recommendation LLMs

A VERL-inspired modular framework for training recommendation large language models with reinforcement learning.

## Design Philosophy

**Separation of Concerns**: Following VERL's hybrid-controller model, RecRL decouples:
- **Data Loading** from **Rollout Generation**
- **Reward Computation** from **Policy Training**
- **Algorithm Logic** from **Infrastructure**

**Modularity**: Each component is independently testable and replaceable.

**Extensibility**: Adding new algorithms (GRPO, EEPO, DPO) requires minimal code.

## Architecture Overview

```
recrl/
├── core/                       # Core abstractions
│   ├── base_trainer.py         # BaseRLTrainer interface
│   ├── rollout.py              # RolloutEngine for generation
│   ├── reward.py               # BaseReward interface
│   └── data.py                 # DataEngine interface
├── algorithms/                 # RL algorithm implementations
│   ├── grpo/
│   │   ├── trainer.py          # GRPOTrainer
│   │   └── config.py           # GRPOConfig
│   ├── eepo/
│   │   ├── trainer.py          # EEPOTrainer
│   │   ├── config.py           # EEPOConfig
│   │   └── unlearn.py          # Fast-weight unlearning logic
│   └── dpo/                    # Future: DPO support
├── rewards/                    # Reward implementations
│   ├── semantic.py             # Text semantic similarity
│   ├── rule.py                 # Exact match, NDCG
│   └── composite.py            # Multi-reward composition
├── data/                       # Data loading
│   ├── builder.py              # DatasetBuilder
│   └── sampler.py              # RepeatRandomSampler
├── constraints/                # Constrained decoding
│   ├── trie.py                 # SIDTrie for valid tokens
│   └── processor.py            # ConstrainedLogitsProcessor
├── utils/                      # Utilities
│   ├── sid_helper.py           # SID parsing
│   └── metrics.py              # Evaluation metrics
└── examples/                   # Usage examples
    ├── train_grpo.py
    ├── train_eepo.py
    └── evaluate.py
```

## Key Components

### 1. Core Abstractions

#### BaseRLTrainer
```python
class BaseRLTrainer:
    def __init__(self, model, ref_model, config, data_engine, rollout_engine, reward_engine):
        pass

    def train(self):
        """Main training loop"""
        pass

    def compute_loss(self, batch):
        """Algorithm-specific loss computation"""
        raise NotImplementedError
```

#### RolloutEngine
```python
class RolloutEngine:
    """Handles all generation logic, decoupled from training"""

    def generate(self, prompts, model, generation_config, constraints=None):
        """Generate completions with optional constraints"""
        pass

    def compute_ref_logprobs(self, prompt_completion_ids, ref_model):
        """Compute reference model log probabilities"""
        pass
```

#### BaseReward
```python
class BaseReward:
    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        """Compute rewards for prompt-completion pairs"""
        raise NotImplementedError
```

### 2. Algorithm Implementations

Each algorithm (GRPO, EEPO) inherits from `BaseRLTrainer` and only implements:
- `compute_loss()` - Algorithm-specific loss
- `_prepare_batch()` - Batch preparation logic (if needed)

**GRPO**: Standard group relative policy optimization
**EEPO**: GRPO + Fast-weight exploration via temporary unlearning

### 3. Reward System

Rewards are composable:
```python
reward = CompositeReward([
    (TextSemanticReward(recif_path), 1.0),
    (NDCGReward(recif_path), 0.5),
    (NoveltyBonus(), 0.2)
])
```

### 4. Data Engine

Unified interface for different data sources:
```python
data_engine = DataEngine.from_parquet(
    path="train.parquet",
    format="recif",  # or "amazon_csv"
    sample_num=1000
)
```

## Comparison with MiniOneRec

| Aspect | MiniOneRec | RecRL |
|--------|------------|-------|
| Trainer size | 500+ lines | ~150 lines per algorithm |
| Modularity | Monolithic | Fully decoupled |
| Adding new algorithm | Modify existing trainer | Create new subclass |
| Reward composition | Hard-coded | Pluggable |
| Constrained decoding | Brittle array-based | Robust Trie-based |
| Testing | Difficult | Each module testable |

## Usage Example

```python
from recrl.core import RolloutEngine, DataEngine
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
from recrl.rewards import TextSemanticReward, CompositeReward
from recrl.constraints import SIDTrie

# Setup
model = AutoModelForCausalLM.from_pretrained("model_path")
tokenizer = AutoTokenizer.from_pretrained("model_path")

# Data
data_engine = DataEngine.from_parquet("train.parquet", sample_num=1000)

# Rollout
trie = SIDTrie.from_recif("recif_path")
rollout_engine = RolloutEngine(tokenizer, constraints=trie)

# Reward
reward = TextSemanticReward(recif_path="recif_path")

# Config
config = EEPOConfig(
    num_generations=16,
    temperature=0.7,
    eepo_stage1_ratio=0.5,
    eepo_unlearn_lr=1e-5,
    learning_rate=1e-6,
    beta=0.04
)

# Train
trainer = EEPOTrainer(
    model=model,
    config=config,
    data_engine=data_engine,
    rollout_engine=rollout_engine,
    reward_engine=reward
)
trainer.train()
```

## Benefits

1. **Clean Separation**: Each component has a single responsibility
2. **Easy Testing**: Mock any component for unit tests
3. **Algorithm Flexibility**: Switch between GRPO/EEPO/DPO with config change
4. **Reward Composition**: Combine multiple rewards easily
5. **Constraint Flexibility**: Enable/disable constrained decoding independently
6. **Future-Proof**: Easy to add new algorithms, rewards, or data sources
