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

RecRL requires Python 3.10+ and PyTorch 2.0+.

### Quick Install with uv (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### Alternative: pip

```bash
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec
pip install -e .
```

See [Installation Guide](docs/INSTALLATION.md) for detailed instructions.

## 🚀 Quick Start

### Single-GPU Training (Recommended)

```bash
# GRPO with constrained decoding
bash train_constrained.sh
```

This trains GRPO with trie-based constrained decoding, ensuring only valid item IDs are generated.

### Multi-GPU Training

```bash
# 4 GPUs with Accelerate
bash train_multi_gpu.sh
```

### EEPO Training (Enhanced Exploration)

```bash
# EEPO with fast-weight exploration
bash train_eepo.sh
```

### Custom Training

```python
from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import CompositeReward, LongviewReward, SemanticReward
from recrl.constraints import SIDTrie

# Load data
train_dataset = DataEngine.from_recif_parquet("data/train.parquet")

# Setup engines
rollout_engine = RolloutEngine(tokenizer, device="cuda")
trie = SIDTrie.from_recif("path/to/RecIF", tokenizer)
rollout_engine.set_trie(trie)

# Composite reward
reward_engine = CompositeReward([
    (LongviewReward("path/to/RecIF"), 0.5),
    (SemanticReward("path/to/RecIF"), 0.3),
])

# Train
config = GRPOConfig(num_generations=4, temperature=0.7)
trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    rollout_engine=rollout_engine,
    reward_engine=reward_engine,
    tokenizer=tokenizer,
)
trainer.train()
```

See [Training Guide](docs/TRAINING.md) for detailed instructions.

## 🏗 Architecture

RecRL breaks away from monolithic trainer scripts by adopting a decoupled, Engine-based approach:

```text
recrl/
├── core/                       # Core abstractions
│   ├── base_trainer.py         # BaseRLTrainer loop with multi-GPU support
│   ├── rollout.py              # Generation & logprobs (RolloutEngine)
│   ├── reward.py               # Reward interfaces (CompositeReward)
│   └── data.py                 # Data parser (DataEngine)
├── algorithms/                 # RL implementations
│   ├── grpo/                   # Standard GRPO
│   │   ├── trainer.py          # GRPO trainer
│   │   └── config.py           # GRPO configuration
│   └── eepo/                   # EEPO (Unlearn mechanism)
│       ├── trainer.py          # EEPO trainer
│       ├── unlearn.py          # Fast-weight unlearning
│       └── config.py           # EEPO configuration
├── rewards/                    # Pluggable reward modules
│   ├── longview_reward.py      # User history alignment
│   ├── semantic.py             # Sentence-Transformer similarity
│   ├── novelty_reward.py       # Exploration bonus
│   └── diversity_reward.py     # Intra-group diversity
├── constraints/                # Valid ID enforcement
│   ├── trie.py                 # Prefix Tree for Valid SIDs
│   └── processor.py            # ConstrainedLogitsProcessor
└── data/                       # Data utilities
    └── sampler.py              # RepeatRandomSampler
```

## 📊 Performance

### Constrained vs Unconstrained Decoding

| Metric | Unconstrained | Constrained | Improvement |
|--------|---------------|-------------|-------------|
| Mean Reward | -0.19 | **0.78** | **+488%** |
| Valid SID Rate | ~50% | **100%** | **+100%** |
| Positive Reward Rate | 48.9% | **99.7%** | **+104%** |

**Conclusion**: Constrained decoding is essential for RecRL.

### Training Efficiency

| Configuration | Steps/sec | GPU Memory | Time per Epoch |
|---------------|-----------|------------|----------------|
| Single GPU | 0.12 | 20GB | ~46 hours |
| 4 GPUs (DDP) | 0.45 | 20GB/GPU | ~12 hours |

See [Benchmarks](docs/BENCHMARKS.md) for detailed performance analysis.

## 📚 Documentation

- [Installation Guide](docs/INSTALLATION.md) - Setup instructions
- [Training Guide](docs/TRAINING.md) - Single/multi-GPU training, hyperparameter tuning
- [Algorithm Documentation](docs/ALGORITHMS.md) - GRPO, EEPO, constrained decoding, rewards
- [Benchmarks](docs/BENCHMARKS.md) - Performance comparisons and ablation studies

## 🎯 Key Features Explained

### Constrained Decoding

Trie-based logits processor ensures only valid Semantic IDs (SIDs) are generated:

```python
from recrl.constraints import SIDTrie

# Build trie from RecIF metadata
trie = SIDTrie.from_recif("path/to/RecIF", tokenizer)
rollout_engine.set_trie(trie)
```

**Impact**: 488% reward improvement over unconstrained generation.

### EEPO (Explore-and-Evaluate)

Two-stage generation with fast-weight exploration:

1. **Exploitation** (50%): Generate with current policy
2. **Exploration** (50%): Apply fast-weight mutation, then generate

**Benefit**: Escapes local optima, improves diversity.

### Composite Rewards

Flexible reward composition:

```python
reward_engine = CompositeReward([
    (LongviewReward(recif_path), 0.50),   # User history alignment
    (SemanticReward(recif_path), 0.30),   # Content similarity
    (NoveltyReward(), 0.15),              # Exploration bonus
    (DiversityReward(), 0.05),            # Intra-group diversity
])
```

### Memory Optimization

- **Gradient Checkpointing**: Saves ~40% GPU memory
- **Ref Model CPU Offload**: Saves ~3.4GB GPU memory
- **Mixed Precision (bf16)**: Saves ~50% memory

**Result**: Train 1.7B model on 24GB GPU.

## 🤝 Contributing

We welcome contributions! To add a new algorithm (e.g., DPO):

1. Create a new folder under `recrl/algorithms/dpo/`
2. Inherit from `BaseRLTrainer` and override the `compute_loss()` function
3. Your algorithm will automatically inherit constrained decoding, reward composition, and dataset handling

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 Citation

If you use RecRL in your research, please cite:

```bibtex
@software{recrl2025,
  title = {RecRL: Reinforcement Learning Framework for Recommendation LLMs},
  author = {Zhou, Ziren},
  year = {2025},
  url = {https://github.com/zzr9468-sys/CleanOneRec}
}
```

## 🙏 Acknowledgments

- [OpenOneRec](https://github.com/OpenOneRec) for the RecIF benchmark and OneRec model
- [TRL](https://github.com/huggingface/trl) for RL training utilities
- [Accelerate](https://github.com/huggingface/accelerate) for multi-GPU support
- [SwanLab](https://swanlab.cn) for experiment tracking

## 📄 License

This project is licensed under the Apache 2.0 License - see [LICENSE](LICENSE) for details.
