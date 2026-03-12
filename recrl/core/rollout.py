"""
Rollout Engine

Handles all generation logic, decoupled from training.
Responsible for:
- Generating completions from prompts
- Computing reference model log probabilities
- Managing constrained decoding
"""

import torch
from typing import Optional, Callable
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
    LogitsProcessorList,
    TemperatureLogitsWarper
)
from trl.trainer.utils import selective_log_softmax


class RolloutEngine:
    """Manages generation and reference log probability computation."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        constraints: Optional[Callable] = None,
        device: str = "cuda"
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding/decoding
            constraints: Optional constraint function (e.g., SIDTrie)
            device: Device for computation
        """
        self.tokenizer = tokenizer
        self.constraints = constraints
        self.device = device

    def generate(
        self,
        prompts: list[str],
        model: PreTrainedModel,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        num_return_sequences: int = 1,
        do_sample: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate completions for prompts.

        Args:
            prompts: List of prompt strings
            model: Model to generate with
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of completions per prompt
            do_sample: Whether to sample or use greedy decoding

        Returns:
            Tuple of (prompt_ids, completion_ids, completion_mask)
        """
        # Tokenize prompts
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        ).to(self.device)

        prompt_ids = prompt_inputs["input_ids"]
        prompt_mask = prompt_inputs["attention_mask"]

        # Setup logits processors
        logits_processor = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=temperature))

        if self.constraints is not None:
            from ..constraints.processor import ConstrainedLogitsProcessor
            logits_processor.append(
                ConstrainedLogitsProcessor(
                    self.constraints,
                    prompt_length=prompt_ids.size(1)
                )
            )

        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            prompt_completion_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
                logits_processor=logits_processor if len(logits_processor) > 0 else None,
            )

        # Extract completions
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Compute completion mask (mask out tokens after EOS)
        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),),
            is_eos.size(1),
            dtype=torch.long,
            device=self.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        return prompt_ids, completion_ids, completion_mask

    def compute_ref_logprobs(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        ref_model: PreTrainedModel
    ) -> torch.Tensor:
        """
        Compute reference model log probabilities.

        Args:
            prompt_ids: Prompt token IDs [batch, prompt_len]
            completion_ids: Completion token IDs [batch, completion_len]
            prompt_mask: Prompt attention mask
            completion_mask: Completion attention mask
            ref_model: Reference model

        Returns:
            Per-token log probabilities [batch, completion_len]
        """
        # Ensure prompt_ids matches completion_ids batch size
        num_gens = completion_ids.size(0) // prompt_ids.size(0)
        if num_gens > 1:
            prompt_ids = prompt_ids.repeat_interleave(num_gens, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(num_gens, dim=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        with torch.inference_mode():
            ref_device = next(ref_model.parameters()).device
            prompt_completion_ids = prompt_completion_ids.to(ref_device)
            attention_mask = attention_mask.to(ref_device)
            logits = ref_model(
                input_ids=prompt_completion_ids,
                attention_mask=attention_mask
            ).logits

            # Shift for next-token prediction
            ref_logits = logits[:, :-1, :]
            target_ids = prompt_completion_ids[:, 1:]

            # Compute log probabilities
            ref_per_token_logps = selective_log_softmax(ref_logits, target_ids)

            # Extract only completion part
            ref_per_token_logps = ref_per_token_logps[:, -completion_ids.size(1):]

        return ref_per_token_logps

    def decode_completions(self, completion_ids: torch.Tensor) -> list[str]:
        """Decode completion token IDs to strings."""
        return self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
