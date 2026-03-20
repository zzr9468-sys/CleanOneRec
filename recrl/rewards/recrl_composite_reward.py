"""
RecRL Composite Reward

Formula (multiplicative):
    composite = max(0, lv) * nov * 0.95  +  div * 0.05

Rationale for multiplicative combination:
- Additive (old): lv and nov fight each other — a popular item can score high on lv
  alone even with nov≈0, so EEPO's long-tail exploration never gets reinforced.
- Multiplicative (new): both signals must be satisfied simultaneously.
  EEPO is rewarded only when the generated item is BOTH semantically aligned with
  the user's interests (lv) AND rare in the training distribution (nov).
  This directly aligns EEPO's exploration objective with the reward.

Diversity (5%) stays additive so it's never zeroed out by low lv.
"""

import logging
from typing import List, Dict, Optional

from .longview_reward import LongviewBasedReward
from .novelty_reward import NoveltyReward
from .diversity_reward import DiversityReward

logger = logging.getLogger(__name__)


class RecRLCompositeReward:
    """
    Composite reward for generative recommendation RL training.

    composite = max(0, lv) * nov * 0.95  +  div * 0.05
    """

    # Diversity keeps a small additive weight so it is never multiplied to zero.
    DIV_WEIGHT = 0.05
    JOINT_WEIGHT = 0.95  # weight on the lv*nov product

    def __init__(
        self,
        recif_path: str,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
        num_generations: int = 4,
        longview_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.recif_path = recif_path
        self.device = device
        self.num_generations = num_generations

        # weights kwarg kept for backward compat but ignored in multiplicative mode
        if weights is not None:
            logger.warning(
                "RecRLCompositeReward: 'weights' kwarg is ignored — "
                "the reward now uses a fixed multiplicative formula."
            )

        logger.info("Initializing RecRL Composite Reward (multiplicative mode)...")

        self.longview_reward = LongviewBasedReward(
            recif_path,
            model_name=longview_model,
            device=device,
        )
        self.novelty_reward = NoveltyReward(recif_path)
        self.diversity_reward = DiversityReward(num_generations=num_generations)

        logger.info("RecRL Composite Reward ready")

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        longview_history: Optional[List[List[int]]] = None,
        **kwargs
    ) -> List[float]:
        n = len(completions)

        lv_rewards  = self.longview_reward(prompts, completions, longview_history=longview_history)
        nov_rewards = self.novelty_reward(prompts, completions)
        div_rewards = self.diversity_reward(prompts, completions)

        # Multiplicative combination: reward only items that are BOTH relevant AND novel.
        # lv is clipped to [0, 1] so negative validity penalties don't invert the product.
        # Multiplicative: only reward items that are BOTH relevant AND novel.
        # nov < 0 signals an invalid/unresolvable SID — always apply a flat penalty
        # so invalid generations can never score 0 through cancellation.
        INVALID_PENALTY = -0.5
        final_rewards = []
        for i in range(n):
            if nov_rewards[i] < 0:
                final_rewards.append(INVALID_PENALTY)
            else:
                final_rewards.append(
                    max(0.0, lv_rewards[i]) * nov_rewards[i] * self.JOINT_WEIGHT
                    + div_rewards[i] * self.DIV_WEIGHT
                )

        self._call_count = getattr(self, '_call_count', 0) + 1
        if self._call_count % 10 == 1:
            logger.info(
                f"[Reward breakdown step={self._call_count}]  "
                f"lv={sum(lv_rewards)/n:.3f}  "
                f"nov={sum(nov_rewards)/n:.3f}  "
                f"div={sum(div_rewards)/n:.3f}  "
                f"total={sum(final_rewards)/n:.3f}"
            )

        return final_rewards


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    reward = RecRLCompositeReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")
    prompts = ["user history..."] * 4
    completions = ["<|sid_begin|><s_a_0><s_b_0><s_c_1><|sid_end|>"] * 4
    longview_history = [[2360735, 9241153, 11239440]] * 4
    rewards = reward(prompts, completions, longview_history=longview_history)
    print(f"Composite Rewards: {rewards}")
