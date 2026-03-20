"""
GRPO Trainer

Group Relative Policy Optimization (GRPO) implementation.
Clean, minimal implementation inheriting from BaseRLTrainer.
"""

import torch
from trl.trainer.utils import selective_log_softmax

from ...core.base_trainer import BaseRLTrainer
from .config import GRPOConfig


class GRPOTrainer(BaseRLTrainer):
    """
    GRPO Trainer.

    GRPO normalizes rewards within each generation group and uses
    advantages for policy gradient updates with KL regularization.
    """

    def __init__(self, config: GRPOConfig, **kwargs):
        """
        Args:
            config: GRPOConfig instance
            **kwargs: Passed to BaseRLTrainer (model, ref_model, etc.)
        """
        super().__init__(config=config, **kwargs)

    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """
        Compute GRPO loss.

        Loss = -E[exp(log_pi - log_pi.detach()) * A] + beta * KL

        Where:
        - A is the advantage (normalized reward)
        - KL is the KL divergence from reference policy
        """
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        old_per_token_logps = inputs["old_per_token_logps"]
        advantages = inputs["advantages"]

        # Concatenate prompt and completion
        # Ensure prompt_ids matches completion_ids batch size
        num_gens = completion_ids.size(0) // prompt_ids.size(0)
        if num_gens > 1:
            prompt_ids = prompt_ids.repeat_interleave(num_gens, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(num_gens, dim=0)
            
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Forward pass through policy model
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        # Compute per-token log probabilities
        logits = logits[:, :-1, :]  # Shift for next-token prediction
        target_ids = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, target_ids)

        # Extract only completion part
        per_token_logps = per_token_logps[:, -completion_ids.size(1):]

        # Compute KL divergence (per token)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                       (ref_per_token_logps - per_token_logps) - 1

        # Policy gradient loss with importance sampling
        # Use old_per_token_logps (rollout snapshot) for correct off-policy ratio
        per_token_loss = torch.exp(per_token_logps - old_per_token_logps.detach()) * \
                         advantages.unsqueeze(1)

        # Combine with KL penalty
        per_token_loss = -(per_token_loss - self.config.beta * per_token_kl)

        # Average over valid tokens and batch
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) /
            completion_mask.sum(dim=1).clamp(min=1)
        ).mean()

        return loss
