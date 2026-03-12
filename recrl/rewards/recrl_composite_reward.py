"""
RecRL Composite Reward

Combines four reward signals:
    1. Longview implicit feedback  (50%) — main signal, user true interest
    2. Semantic similarity         (30%) — generated item vs target item
    3. Novelty                     (15%) — encourage long-tail exploration
    4. Diversity                   ( 5%) — avoid mode collapse within group

All four are fully implemented. Weights are configurable.
"""

import logging
from typing import List, Dict, Optional

from .longview_reward import LongviewBasedReward
from .novelty_reward import NoveltyReward
from .semantic import TextSemanticReward
from .diversity_reward import DiversityReward

logger = logging.getLogger(__name__)


class RecRLCompositeReward:
    """
    Composite reward for generative recommendation RL training.

    Design rationale:
    - Longview (50%): strongest unbiased signal — user actively watched
    - Semantic (30%): content relevance to ground-truth target
    - Novelty (15%): pushes model toward long-tail, underexposed items
    - Diversity (5%): prevents all G generations collapsing to the same item
    """

    DEFAULT_WEIGHTS = {
        'longview':  0.50,
        'semantic':  0.30,
        'novelty':   0.15,
        'diversity': 0.05,
    }

    def __init__(
        self,
        recif_path: str,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None,
        num_generations: int = 4,
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        longview_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Args:
            recif_path:        Path to OpenOneRec-RecIF data directory
            device:            torch device string ("cuda", "cpu", "cuda:1", ...)
            weights:           Override default component weights (must sum to 1)
            num_generations:   G — number of completions per prompt (for diversity)
            semantic_model:    Sentence-transformer model for semantic reward
            longview_model:    Sentence-transformer model for longview reward
        """
        self.recif_path = recif_path
        self.device = device
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.num_generations = num_generations

        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-3:
            logger.warning(f"Reward weights sum to {total:.3f}, expected 1.0")

        logger.info("Initializing RecRL Composite Reward...")
        logger.info(f"   weights: {self.weights}")

        # 1. Longview
        self.longview_reward = LongviewBasedReward(
            recif_path,
            model_name=longview_model,
            device=device,
        )

        # 2. Semantic
        self.semantic_reward = TextSemanticReward(
            recif_path,
            device=device,
            model_name=semantic_model,
        )

        # 3. Novelty
        self.novelty_reward = NoveltyReward(recif_path)

        # 4. Diversity
        self.diversity_reward = DiversityReward(num_generations=num_generations)

        logger.info("RecRL Composite Reward ready")

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        longview_history: Optional[List[List[int]]] = None,
        target_pids: Optional[List[List[int]]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute composite reward for a batch of completions.

        Args:
            prompts:          Input prompts  (len = B * G)
            completions:      Generated SIDs (len = B * G)
            longview_history: Per-sample longview PID lists (len = B * G)
            target_pids:      Ground-truth PID lists (passed through)
            **kwargs:         Passed to sub-rewards (e.g. target_sid for semantic)

        Returns:
            Composite reward scores (len = B * G)
        """
        n = len(completions)
        w = self.weights

        lv_rewards = (
            self.longview_reward(prompts, completions, longview_history=longview_history)
            if w.get('longview', 0) > 0 else [0.0] * n
        )

        sem_rewards = (
            self.semantic_reward(prompts, completions, **kwargs)
            if w.get('semantic', 0) > 0 else [0.0] * n
        )

        nov_rewards = (
            self.novelty_reward(prompts, completions)
            if w.get('novelty', 0) > 0 else [0.0] * n
        )

        div_rewards = (
            self.diversity_reward(prompts, completions)
            if w.get('diversity', 0) > 0 else [0.0] * n
        )

        final_rewards = [
            w.get('longview',  0) * lv_rewards[i]  +
            w.get('semantic',  0) * sem_rewards[i]  +
            w.get('novelty',   0) * nov_rewards[i]  +
            w.get('diversity', 0) * div_rewards[i]
            for i in range(n)
        ]

        # Log component means on first call
        if not hasattr(self, '_logged_once'):
            self._logged_once = True
            logger.info(
                f"[Reward breakdown]  "
                f"lv={sum(lv_rewards)/n:.3f}  "
                f"sem={sum(sem_rewards)/n:.3f}  "
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
