"""
Custom Samplers

Samplers for RL training with repeated generations.
"""

import torch
from typing import Optional, Sized
from torch.utils.data import Sampler


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats each sample N times.

    Used in GRPO/EEPO where each prompt needs multiple generations.
    """

    def __init__(
        self,
        data_source: Sized,
        repeat_count: int,
        seed: Optional[int] = None
    ):
        """
        Args:
            data_source: Dataset to sample from
            repeat_count: Number of times to repeat each sample
            seed: Random seed
        """
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        """Generate indices with repetition."""
        indices = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indices)

    def __len__(self):
        return self.num_samples * self.repeat_count
