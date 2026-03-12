"""
Diversity Reward

Encourages diversity within each generation group.
Within a group of G completions for the same prompt,
rewards completions that are different from the others.
"""

import re
from typing import List, Optional
from ..core.reward import BaseReward
from ..utils.sid_helper import SIDHelper

SID_PATTERN = re.compile(r'<\|sid_begin\|>.*?<\|sid_end\|>')


class DiversityReward(BaseReward):
    """
    Intra-group diversity reward.

    For each completion in a group:
        diversity = 1 - (# of other completions with same SID) / (group_size - 1)

    Score ranges:
        1.0 = this SID is unique in the group (maximum diversity)
        0.0 = all other completions have the same SID (no diversity)

    This encourages the model to explore different items across
    the G generations for the same user, avoiding mode collapse.
    """

    def __init__(self, num_generations: int = 4):
        """
        Args:
            num_generations: Number of completions per prompt (group size G)
        """
        super().__init__(name="diversity_reward")
        self.num_generations = num_generations
        self.sid_helper = SIDHelper()

    def _extract_first_sid(self, completion: str) -> Optional[str]:
        """Extract the first SID token from completion string."""
        sids = SID_PATTERN.findall(completion)
        return sids[0] if sids else None

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute diversity rewards.

        Args:
            prompts: Input prompts (len = batch_size * num_generations)
            completions: Generated completions (same length)

        Returns:
            Diversity scores in [0.0, 1.0] for each completion
        """
        n = len(completions)
        G = self.num_generations

        # Extract first SID from each completion
        sids = [self._extract_first_sid(c) for c in completions]

        rewards = []

        for i in range(n):
            # Determine group boundaries
            group_start = (i // G) * G
            group_end = min(group_start + G, n)
            group_sids = sids[group_start:group_end]
            group_size = len(group_sids)

            if group_size <= 1:
                rewards.append(1.0)
                continue

            my_sid = sids[i]
            if my_sid is None:
                # Invalid SID, no diversity contribution
                rewards.append(0.0)
                continue

            # Count how many OTHER completions in the group share the same SID
            same_count = sum(
                1 for j, s in enumerate(group_sids)
                if s == my_sid and (group_start + j) != i
            )

            diversity = 1.0 - same_count / (group_size - 1)
            rewards.append(float(diversity))

        return rewards
