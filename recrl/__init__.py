"""
RecRL - Reinforcement Learning Framework for Recommendation LLMs
A VERL-inspired modular framework for training recommendation models with RL.
"""

__version__ = "0.1.0"

from .core.base_trainer import BaseRLTrainer
from .core.rollout import RolloutEngine
from .core.reward import BaseReward, CompositeReward
from .core.data import DataEngine

__all__ = [
    "BaseRLTrainer",
    "RolloutEngine",
    "BaseReward",
    "CompositeReward",
    "DataEngine",
]
