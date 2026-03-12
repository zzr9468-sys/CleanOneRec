"""
Tests for reward functions.
"""

import pytest
import torch
from recrl.core.reward import BaseReward


class DummyReward(BaseReward):
    """Simple reward for testing."""

    def __call__(self, prompts, completions, **kwargs):
        return [1.0 for _ in completions]


def test_base_reward_interface():
    """Test BaseReward interface."""
    reward = DummyReward()

    prompts = ["prompt1", "prompt2"]
    completions = ["completion1", "completion2"]

    rewards = reward(prompts, completions)

    assert len(rewards) == 2
    assert all(r == 1.0 for r in rewards)


def test_reward_with_kwargs():
    """Test reward function with additional kwargs."""

    class KwargsReward(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            target = kwargs.get("target", "default")
            return [1.0 if target == "match" else 0.0 for _ in completions]

    reward = KwargsReward()

    prompts = ["p1", "p2"]
    completions = ["c1", "c2"]

    # With matching target
    rewards = reward(prompts, completions, target="match")
    assert all(r == 1.0 for r in rewards)

    # With non-matching target
    rewards = reward(prompts, completions, target="no_match")
    assert all(r == 0.0 for r in rewards)


def test_reward_empty_input():
    """Test reward with empty input."""
    reward = DummyReward()

    rewards = reward([], [])
    assert len(rewards) == 0


def test_reward_single_input():
    """Test reward with single input."""
    reward = DummyReward()

    rewards = reward(["prompt"], ["completion"])
    assert len(rewards) == 1
    assert rewards[0] == 1.0


def test_reward_batch_input():
    """Test reward with batch input."""
    reward = DummyReward()

    batch_size = 10
    prompts = [f"prompt_{i}" for i in range(batch_size)]
    completions = [f"completion_{i}" for i in range(batch_size)]

    rewards = reward(prompts, completions)

    assert len(rewards) == batch_size
    assert all(r == 1.0 for r in rewards)


def test_composite_reward():
    """Test CompositeReward combining multiple rewards."""
    from recrl.core.reward import CompositeReward

    class Reward1(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            return [1.0 for _ in completions]

    class Reward2(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            return [0.5 for _ in completions]

    composite = CompositeReward([
        (Reward1(), 0.6),
        (Reward2(), 0.4),
    ])

    prompts = ["p1", "p2"]
    completions = ["c1", "c2"]

    rewards = composite(prompts, completions)

    # Expected: 0.6 * 1.0 + 0.4 * 0.5 = 0.8
    assert len(rewards) == 2
    assert all(abs(r - 0.8) < 1e-6 for r in rewards)


def test_composite_reward_multiple_components():
    """Test CompositeReward with multiple components."""
    from recrl.core.reward import CompositeReward

    class Reward1(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            return [1.0 for _ in completions]

    class Reward2(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            return [0.5 for _ in completions]

    class Reward3(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            return [0.2 for _ in completions]

    composite = CompositeReward([
        (Reward1(), 0.5),
        (Reward2(), 0.3),
        (Reward3(), 0.2),
    ])

    prompts = ["p1", "p2"]
    completions = ["c1", "c2"]

    rewards = composite(prompts, completions)

    # Expected: 0.5 * 1.0 + 0.3 * 0.5 + 0.2 * 0.2 = 0.69
    assert len(rewards) == 2
    assert all(abs(r - 0.69) < 1e-6 for r in rewards)


def test_reward_with_invalid_completion():
    """Test reward handling invalid completions."""

    class ValidatingReward(BaseReward):
        def __call__(self, prompts, completions, **kwargs):
            rewards = []
            for completion in completions:
                if "invalid" in completion:
                    rewards.append(-1.0)
                else:
                    rewards.append(1.0)
            return rewards

    reward = ValidatingReward()

    prompts = ["p1", "p2", "p3"]
    completions = ["valid", "invalid", "valid"]

    rewards = reward(prompts, completions)

    assert rewards == [1.0, -1.0, 1.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
