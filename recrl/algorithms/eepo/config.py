"""
EEPO Configuration
"""

from dataclasses import dataclass
from ...core.base_trainer import RLConfig


@dataclass
class EEPOConfig(RLConfig):
    """
    Configuration for EEPO (Explore-and-Evaluate Policy Optimization).

    EEPO extends GRPO with fast-weight exploration to escape local optima.
    """

    # EEPO-specific parameters
    eepo_enabled: bool = True
    eepo_stage1_ratio: float = 0.5  # Ratio of generations for exploitation phase
    eepo_unlearn_lr: float = 1e-5  # Learning rate for fast-weight update
    eepo_unlearn_weight: float = 1.0  # Weight for unlearning loss
    eepo_epsilon: float = 1e-4  # Epsilon for probability clamping

    # Optional: Add ground truth to generation batch
    add_gt: bool = True
