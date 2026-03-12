"""
Constrained Logits Processor

Modifies logits during generation to enforce constraints.
"""

import torch
from typing import Callable, Set
from transformers import LogitsProcessor


class ConstrainedLogitsProcessor(LogitsProcessor):
    """
    Logits processor that enforces trie-based constraints.

    Sets logits of disallowed tokens to -inf.
    """

    def __init__(
        self,
        get_allowed_tokens_fn: Callable[[torch.Tensor], Set[int]],
        prompt_length: int
    ):
        """
        Args:
            get_allowed_tokens_fn: Function that returns allowed tokens
            prompt_length: Length of prompt (to extract completion part)
        """
        self.get_allowed_tokens_fn = get_allowed_tokens_fn
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Modify logits to enforce constraints.

        Args:
            input_ids: Current token IDs [batch, seq_len]
            scores: Logits [batch, vocab_size]

        Returns:
            Modified logits
        """
        batch_size, vocab_size = scores.shape

        for i in range(batch_size):
            # Extract completion part (after prompt)
            completion_ids = input_ids[i, self.prompt_length:]

            # Get allowed next tokens
            allowed_tokens = self.get_allowed_tokens_fn(completion_ids)

            if allowed_tokens:
                # Create mask on the same device as scores
                mask = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
                allowed_indices = torch.tensor(list(allowed_tokens), dtype=torch.long, device=scores.device)
                mask[allowed_indices] = False

                # Set disallowed tokens to -inf
                scores[i, mask] = float('-inf')

        return scores
