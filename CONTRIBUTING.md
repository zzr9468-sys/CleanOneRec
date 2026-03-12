# Contributing to RecRL

Thank you for your interest in contributing to RecRL! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/CleanOneRec.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and linting
6. Commit and push
7. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec

# Install with dev dependencies
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## Code Style

We use `ruff` for linting and formatting:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix linting issues
ruff check --fix .
```

Configuration is in `pyproject.toml`.

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rewards.py

# Run with coverage
pytest --cov=recrl tests/
```

## Adding a New Algorithm

To add a new RL algorithm (e.g., DPO):

1. **Create algorithm directory**:
   ```
   recrl/algorithms/dpo/
   ├── __init__.py
   ├── config.py
   └── trainer.py
   ```

2. **Define configuration** (`config.py`):
   ```python
   from dataclasses import dataclass
   from recrl.core.base_trainer import RLConfig

   @dataclass
   class DPOConfig(RLConfig):
       """DPO-specific configuration."""
       dpo_beta: float = 0.1  # DPO temperature
   ```

3. **Implement trainer** (`trainer.py`):
   ```python
   from recrl.core.base_trainer import BaseRLTrainer
   from .config import DPOConfig

   class DPOTrainer(BaseRLTrainer):
       def compute_loss(self, inputs: dict) -> torch.Tensor:
           """Compute DPO loss."""
           # Your DPO loss implementation
           ...
           return loss
   ```

4. **Export in `__init__.py`**:
   ```python
   from .trainer import DPOTrainer
   from .config import DPOConfig

   __all__ = ["DPOTrainer", "DPOConfig"]
   ```

5. **Add tests** (`tests/test_dpo.py`):
   ```python
   def test_dpo_trainer():
       # Test your trainer
       ...
   ```

## Adding a New Reward Function

To add a new reward function:

1. **Create reward file** (`recrl/rewards/your_reward.py`):
   ```python
   from recrl.core.reward import BaseReward

   class YourReward(BaseReward):
       def __init__(self, **kwargs):
           super().__init__()
           # Initialize your reward

       def __call__(self, prompts, completions, **kwargs):
           """Compute rewards for completions."""
           rewards = []
           for prompt, completion in zip(prompts, completions):
               reward = self.compute_reward(prompt, completion, **kwargs)
               rewards.append(reward)
           return rewards

       def compute_reward(self, prompt, completion, **kwargs):
           """Compute single reward."""
           # Your reward logic
           return reward_value
   ```

2. **Add tests** (`tests/test_rewards.py`):
   ```python
   def test_your_reward():
       reward_fn = YourReward()
       rewards = reward_fn(["prompt"], ["completion"])
       assert len(rewards) == 1
   ```

3. **Export in `recrl/rewards/__init__.py`**:
   ```python
   from .your_reward import YourReward
   ```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style (run `ruff format` and `ruff check`)
- [ ] Tests pass (`pytest`)
- [ ] New features have tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### PR Description

Please include:

1. **What**: Brief description of changes
2. **Why**: Motivation for the changes
3. **How**: Technical approach
4. **Testing**: How you tested the changes
5. **Screenshots**: If applicable (for UI changes)

### Example PR Description

```markdown
## What
Add DPO (Direct Preference Optimization) algorithm support.

## Why
DPO is a popular RL algorithm that doesn't require a value function,
making it simpler than PPO while achieving competitive performance.

## How
- Created `recrl/algorithms/dpo/` with trainer and config
- Implemented pairwise preference loss
- Added tests for DPO trainer

## Testing
- Unit tests for DPO loss computation
- Integration test with small dataset (10 samples, 5 steps)
- Verified training runs without errors

## Performance
Trained on video benchmark for 200 steps:
- Mean reward: 0.75
- Training time: ~30 min (single GPU)
```

## Code Review Process

1. Maintainer reviews your PR
2. Address feedback and update PR
3. Once approved, maintainer merges

## Reporting Issues

When reporting bugs, please include:

1. **Environment**: Python version, PyTorch version, GPU info
2. **Steps to reproduce**: Minimal code to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full error traceback
6. **Logs**: Relevant log files

### Example Issue

```markdown
## Bug: CUDA OOM during multi-GPU training

**Environment**:
- Python 3.10
- PyTorch 2.1.0+cu118
- 4x NVIDIA A100 40GB

**Steps to reproduce**:
```bash
bash train_multi_gpu.sh
```

**Expected**: Training runs successfully

**Actual**: CUDA OOM error after 10 steps

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Logs**: See attached `train_multi_gpu.log`
```

## Feature Requests

For feature requests, please include:

1. **Use case**: Why you need this feature
2. **Proposed solution**: How you envision it working
3. **Alternatives**: Other approaches you considered

## Documentation

When adding features, please update:

- **README.md**: If it's a major feature
- **docs/**: Relevant documentation files
- **Docstrings**: In-code documentation
- **Examples**: Usage examples if applicable

## Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Questions?

If you have questions about contributing, please open a GitHub Discussion or reach out to the maintainers.

Thank you for contributing to RecRL! 🎉
