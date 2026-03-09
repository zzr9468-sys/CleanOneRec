import math
from typing import List, Dict

class Evaluator:
    @staticmethod
    def compute_metrics(predictions: List[List[str]], targets: List[str], k_list: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Compute Hit Rate (HR) and Normalized Discounted Cumulative Gain (NDCG) at different K.
        
        Args:
            predictions: List of list of predicted SIDs for each user.
            targets: List of ground truth SIDs.
            k_list: List of K values to evaluate at.
            
        Returns:
            Dict containing HR@K and NDCG@K
        """
        metrics = {}
        for k in k_list:
            metrics[f'HR@{k}'] = 0.0
            metrics[f'NDCG@{k}'] = 0.0
            
        valid_samples = 0
        
        for preds, target in zip(predictions, targets):
            if not target:
                continue
                
            valid_samples += 1
            
            for k in k_list:
                preds_at_k = preds[:k]
                if target in preds_at_k:
                    metrics[f'HR@{k}'] += 1.0
                    rank = preds_at_k.index(target)
                    metrics[f'NDCG@{k}'] += 1.0 / math.log2(rank + 2)
                    
        if valid_samples > 0:
            for key in metrics:
                metrics[key] /= valid_samples
                
        return metrics
