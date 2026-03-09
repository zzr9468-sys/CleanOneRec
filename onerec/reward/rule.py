import math

class BaseReward:
    """Base class for all reward functions"""
    def __init__(self, name: str):
        self.__name__ = name # TRL GRPO requires __name__

    def __call__(self, prompts, completions, **kwargs):
        raise NotImplementedError("Reward function must implement __call__")

class RuleReward(BaseReward):
    """Strict matching reward: 1.0 if generated SID exactly matches target SID, else 0.0"""
    def __init__(self):
        super().__init__("rule_reward")

    def __call__(self, prompts, completions, **kwargs):
        targets = kwargs.get("target_sid", [])
        if not targets:
            return [0.0] * len(completions)
            
        rewards = []
        for i, completion in enumerate(completions):
            gen_sid = completion.strip(" \n\"'")
            target_sid = targets[i].strip(" \n\"'")
            if gen_sid == target_sid:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

class NDCGReward(BaseReward):
    """
    Decaying penalty based on generation position. 
    Requires num_generations to be passed or derived from the batch.
    """
    def __init__(self, num_generations: int = 16):
        super().__init__("ndcg_reward")
        self.num_generations = num_generations
        
        # Calculate NDCG decay weights
        raw_rewards = [-1.0 / math.log2(i + 2) for i in range(num_generations)]
        self.decay_weights = [-elm / sum(raw_rewards) for elm in raw_rewards]

    def __call__(self, prompts, completions, **kwargs):
        targets = kwargs.get("target_sid", [])
        if not targets:
            return [0.0] * len(completions)
            
        rewards = []
        batch_size = len(completions) // self.num_generations
        
        for b in range(batch_size):
            start_idx = b * self.num_generations
            end_idx = start_idx + self.num_generations
            
            group_completions = completions[start_idx:end_idx]
            target_sid = targets[start_idx].strip(" \n\"'")
            
            group_rewards = []
            hit_found = False
            
            for j, comp in enumerate(group_completions):
                gen_sid = comp.strip(" \n\"'")
                if gen_sid == target_sid:
                    hit_found = True
                    group_rewards.append(0.0) # Best score relative to penalty
                else:
                    group_rewards.append(self.decay_weights[j])
                    
            if not hit_found:
                group_rewards = [0.0] * self.num_generations
                
            rewards.extend(group_rewards)
            
        return rewards