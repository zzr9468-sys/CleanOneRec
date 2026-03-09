import logging
from .semantic import TextSemanticReward

logger = logging.getLogger(__name__)

from .rule import BaseReward

class CompositeReward(BaseReward):
    """
    Combines multiple reward components. 
    e.g. TextSemanticReward + NoveltyBonus (Tail Exploring Bonus)
    """
    def __init__(self, 
                 recif_path: str, 
                 device: str = "cuda", 
                 validity_penalty: float = -10.0,
                 tail_bonus: float = 0.1,
                 tail_freq_threshold: int = 50):
        super().__init__("composite_semantic_reward")
        self.semantic_reward = TextSemanticReward(recif_path, device, validity_penalty)
        self.tail_bonus = tail_bonus
        self.tail_freq_threshold = tail_freq_threshold
        self.validity_penalty = validity_penalty
        
        # To strictly implement the tail bonus, we need sid_freq.
        # For simplicity, we can pass it via kwargs or load it during init.
        # Assuming we receive `sid_freq_dict` in __call__ kwargs for now, 
        # or we could compute it statically if we pass the dataset path.
        self.sid_freq_dict = None 

    def load_freq_dict_from_csv(self, csv_path):
        import pandas as pd
        import os
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "item_sid" in df.columns:
                self.sid_freq_dict = df["item_sid"].value_counts().to_dict()
                logger.info(f"[CompositeReward] Loaded frequency dict with {len(self.sid_freq_dict)} items.")

    def __call__(self, prompts, completions, **kwargs):
        # 1. Get base semantic reward
        rewards = self.semantic_reward(prompts, completions, **kwargs)
        
        # 2. Add Novelty / Tail Bonus
        if self.sid_freq_dict is not None and self.tail_bonus > 0:
            for i, completion in enumerate(completions):
                # Only reward valid outputs (not heavily penalized)
                if rewards[i] > self.validity_penalty + 1.0:
                    clean_sid = completion.strip(" \n\"'")
                    freq = self.sid_freq_dict.get(clean_sid, 0)
                    if freq < self.tail_freq_threshold:
                        rewards[i] += self.tail_bonus
                        
        return rewards