"""
Fast-Weight Unlearner

Implements the temporary weight mutation logic for EEPO exploration.
"""

import torch
from typing import Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.utils import selective_log_softmax


class FastWeightUnlearner:
    """
    Applies temporary fast-weight updates to push model out of comfort zone.

    The unlearning process:
    1. Freeze all parameters except LM head
    2. Compute unlearning loss: -log(1 - P(completion))
    3. Apply single SGD step with high learning rate
    4. Generate with mutated weights
    5. Restore original weights
    """

    def __init__(
        self,
        unlearn_lr: float = 1e-5,
        unlearn_weight: float = 1.0,
        epsilon: float = 1e-4
    ):
        """
        Args:
            unlearn_lr: Learning rate for fast-weight update
            unlearn_weight: Weight for unlearning loss
            epsilon: Epsilon for probability clamping (avoid log(0))
        """
        self.unlearn_lr = unlearn_lr
        self.unlearn_weight = unlearn_weight
        self.epsilon = epsilon

    def apply_unlearn_update(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase
    ) -> Tuple[Optional[torch.nn.Module], Optional[torch.Tensor]]:
        """
        Apply fast-weight unlearning update.

        Args:
            model: Model to mutate
            prompt_ids: Prompt token IDs [batch, prompt_len]
            prompt_mask: Prompt attention mask
            completion_ids: Completion token IDs [batch, completion_len]
            tokenizer: Tokenizer for EOS token

        Returns:
            Tuple of (lm_head_module, original_weights) for restoration
        """
        if self.unlearn_lr <= 0 or self.unlearn_weight <= 0:
            return None, None

        # Locate LM head
        lm_head = self._get_lm_head(model)
        original_head_weight = lm_head.weight.data.clone()

        # Freeze all except LM head
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            if any(id(param) == id(p) for p in lm_head.parameters()):
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.train()

        # Expand prompts to match completion batch size
        num_prompts = prompt_ids.size(0)
        repeat_count = completion_ids.size(0) // num_prompts
        prompt_ids_expanded = prompt_ids.repeat_interleave(repeat_count, dim=0)
        prompt_mask_expanded = prompt_mask.repeat_interleave(repeat_count, dim=0)

        # Compute completion mask
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=completion_ids.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(
            is_eos.size(1),
            device=completion_ids.device
        ).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Forward pass
        input_ids = torch.cat([prompt_ids_expanded, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask_expanded, completion_mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, target_ids)

        # Extract completion part
        per_token_logps = per_token_logps[:, -completion_mask.shape[1]:]

        # Compute sequence log probabilities
        token_logps = per_token_logps * completion_mask
        denom = completion_mask.sum(dim=1).clamp(min=1)
        seq_logp = token_logps.sum(dim=1) / denom

        # Unlearning loss: -log(1 - P(completion))
        probs = torch.exp(seq_logp).clamp(max=1 - self.epsilon)
        unlearn_loss = (-torch.log(1 - probs)).mean() * self.unlearn_weight

        # Single SGD step
        temp_optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.unlearn_lr
        )
        temp_optimizer.zero_grad()
        unlearn_loss.backward()
        temp_optimizer.step()

        model.eval()

        # Restore requires_grad flags
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]

        return lm_head, original_head_weight

    def restore_weights(
        self,
        lm_head: Optional[torch.nn.Module],
        original_weights: Optional[torch.Tensor]
    ):
        """Restore original LM head weights."""
        if lm_head is not None and original_weights is not None:
            lm_head.weight.data.copy_(original_weights)

    def _get_lm_head(self, model: PreTrainedModel) -> torch.nn.Module:
        """Locate the LM head module."""
        if hasattr(model, "lm_head"):
            return model.lm_head
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        elif hasattr(model, "embed_out"):
            return model.embed_out
        else:
            # Fallback: last child module
            return list(model.children())[-1]
