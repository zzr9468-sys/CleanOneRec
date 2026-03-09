"""
GRPO Configuration
"""

from dataclasses import dataclass
from ...core.base_trainer import RLConfig


@dataclass
class GRPOConfig(RLConfig):
    """Configuration for GRPO (Group Relative Policy Optimization)."""

    # Inherited from RLConfig:
    # - num_epochs, per_device_batch_size, learning_rate, etc.
    # - num_generations, temperature, max_completion_length
    # - beta (KL penalty)

    pass  # GRPO uses base config without additional parameters
