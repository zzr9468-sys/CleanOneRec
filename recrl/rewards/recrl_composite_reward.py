"""
RecRL Composite Reward

Two modes:

  multiplicative (default, new):
      composite = max(0, lv) * nov * 0.95  +  div * 0.05
      Invalid SID → -0.5 penalty

  additive (old baseline):
      composite = w_lv * lv  +  w_nov * nov  +  w_div * div
      Invalid SID → 0.0  (original behaviour, cannot distinguish from low-score)

Rationale for multiplicative combination:
- Additive: lv and nov fight each other — a popular item can score high on lv
  alone even with nov≈0, so EEPO's long-tail exploration never gets reinforced.
- Multiplicative: both signals must be satisfied simultaneously.
  EEPO is rewarded only when the generated item is BOTH semantically aligned with
  the user's interests (lv) AND rare in the training distribution (nov).
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

    mode='multiplicative':  composite = max(0, lv) * nov * 0.95  +  div * 0.05
    mode='additive':        composite = w_lv * lv + w_nov * nov + w_div * div
    """

    # Multiplicative mode fixed weights
    DIV_WEIGHT = 0.05
    JOINT_WEIGHT = 0.95

    # Additive mode default weights (can be overridden via `weights`)
    DEFAULT_ADDITIVE_WEIGHTS = {"longview": 0.70, "novelty": 0.25, "diversity": 0.05}

    def __init__(
        self,
        recif_path: str,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
        num_generations: int = 4,
        mode: str = "multiplicative",
        longview_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        assert mode in ("multiplicative", "additive"), \
            f"mode must be 'multiplicative' or 'additive', got '{mode}'"

        self.recif_path = recif_path
        self.device = device
        self.num_generations = num_generations
        self.mode = mode

        if mode == "multiplicative" and weights is not None:
            logger.warning(
                "RecRLCompositeReward: 'weights' kwarg is ignored in multiplicative mode."
            )
        if mode == "additive":
            self.additive_weights = {**self.DEFAULT_ADDITIVE_WEIGHTS, **(weights or {})}
            logger.info(
                f"Initializing RecRL Composite Reward (additive mode, "
                f"weights={self.additive_weights})..."
            )
        else:
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

        final_rewards = []
        if self.mode == "multiplicative":
            # nov < 0 → invalid SID, always penalise so model learns to avoid them
            for i in range(n):
                if nov_rewards[i] < 0:
                    final_rewards.append(-0.5)
                else:
                    final_rewards.append(
                        max(0.0, lv_rewards[i]) * nov_rewards[i] * self.JOINT_WEIGHT
                        + div_rewards[i] * self.DIV_WEIGHT
                    )
        else:  # additive
            w_lv  = self.additive_weights["longview"]
            w_nov = self.additive_weights["novelty"]
            w_div = self.additive_weights["diversity"]
            # Invalid SID: return 0.0 (original behaviour — no explicit penalty)
            for i in range(n):
                if nov_rewards[i] < 0:
                    final_rewards.append(0.0)
                else:
                    final_rewards.append(
                        w_lv * lv_rewards[i]
                        + w_nov * nov_rewards[i]
                        + w_div * div_rewards[i]
                    )

        self._call_count = getattr(self, '_call_count', 0) + 1
        if self._call_count % 10 == 1:
            logger.info(
                f"[Reward({self.mode}) step={self._call_count}]  "
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
