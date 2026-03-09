"""
Base Reward Interface

All reward functions inherit from BaseReward and implement __call__.
Rewards can be composed using CompositeReward.
"""

from typing import Any


class BaseReward:
    """Base class for all reward functions."""

    def __init__(self, name: str = "base_reward"):
        self.name = name

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """
        Compute rewards for prompt-completion pairs.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            **kwargs: Additional context (e.g., target_sid, user_history)

        Returns:
            List of reward scores (one per completion)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement __call__")


class CompositeReward(BaseReward):
    """Combines multiple reward functions with weights."""

    def __init__(self, rewards: list[tuple[BaseReward, float]]):
        """
        Args:
            rewards: List of (reward_function, weight) tuples
        """
        super().__init__(name="composite_reward")
        self.rewards = rewards

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Compute weighted sum of all rewards."""
        num_samples = len(prompts)
        total_rewards = [0.0] * num_samples

        for reward_fn, weight in self.rewards:
            scores = reward_fn(prompts, completions, **kwargs)
            for i in range(num_samples):
                total_rewards[i] += scores[i] * weight

        return total_rewards
