"""
Rule-based Rewards

Simple rule-based reward functions like exact match and NDCG.
"""

from ..core.reward import BaseReward
from ..utils.sid_helper import SIDHelper


class ExactMatchReward(BaseReward):
    """Exact match reward: 1.0 if generated SID matches target, else 0.0."""

    def __init__(self, validity_penalty: float = -10.0):
        super().__init__(name="exact_match_reward")
        self.validity_penalty = validity_penalty
        self.sid_helper = SIDHelper()

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        targets = kwargs.get("target_sid", [])
        if not targets:
            return [self.validity_penalty] * len(completions)

        rewards = []
        for completion, target in zip(completions, targets):
            gen_sid = completion.strip(" \n\"'")
            target_sid = target.strip(" \n\"'")

            # Check validity
            if not self.sid_helper.is_valid_sid(gen_sid):
                rewards.append(self.validity_penalty)
                continue

            # Exact match
            if gen_sid == target_sid:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


class NDCGReward(BaseReward):
    """
    NDCG-based reward.

    Placeholder implementation - requires ranking information.
    """

    def __init__(self, recif_path: str, k: int = 10):
        super().__init__(name="ndcg_reward")
        self.recif_path = recif_path
        self.k = k
        # TODO: Load ranking data if needed

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        # Placeholder: return 0.0 for now
        # Real implementation would compute NDCG@k based on ranking
        return [0.0] * len(completions)
