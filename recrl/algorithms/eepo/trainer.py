"""
EEPO Trainer

Explore-and-Evaluate Policy Optimization (EEPO) implementation.
Extends GRPO with fast-weight exploration to escape local optima.
"""

import torch
from trl.trainer.utils import selective_log_softmax, pad

from ...core.base_trainer import BaseRLTrainer
from .config import EEPOConfig
from .unlearn import FastWeightUnlearner


class EEPOTrainer(BaseRLTrainer):
    """
    EEPO Trainer.

    EEPO performs two-stage generation:
    1. Exploitation: Generate with current policy
    2. Exploration: Apply fast-weight mutation, then generate
    """

    def __init__(self, config: EEPOConfig, **kwargs):
        """
        Args:
            config: EEPOConfig instance
            **kwargs: Passed to BaseRLTrainer (model, ref_model, etc.)
        """
        super().__init__(config=config, **kwargs)

        # Initialize fast-weight unlearner
        self.unlearner = FastWeightUnlearner(
            unlearn_lr=config.eepo_unlearn_lr,
            unlearn_weight=config.eepo_unlearn_weight,
            epsilon=config.eepo_epsilon
        )

    def _prepare_inputs(self, batch: list[dict]) -> dict:
        """
        Override to implement EEPO's two-stage generation.

        Stage 1: Exploitation (generate with current policy)
        Stage 2: Exploration (mutate weights, generate, restore)
        """
        if not self.config.eepo_enabled:
            # Fall back to standard generation
            return super()._prepare_inputs(batch)

        prompts = [x["prompt"] for x in batch]
        targets = [x.get("target_sid", "") for x in batch]

        # Deduplicate prompts (each prompt repeated num_generations times in batch)
        unique_prompts = prompts[::self.config.num_generations]

        # Tokenize unique prompts
        prompt_inputs = self.tokenizer(
            unique_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        ).to(self.device)

        dedup_prompt_ids = prompt_inputs["input_ids"]
        dedup_prompt_mask = prompt_inputs["attention_mask"]

        # Calculate generation splits
        g1 = max(1, int(self.config.num_generations * self.config.eepo_stage1_ratio))
        g2 = self.config.num_generations - g1

        # Disable gradient checkpointing during generation
        grad_ckpt_was_enabled = self.model.is_gradient_checkpointing
        if grad_ckpt_was_enabled:
            self.model.gradient_checkpointing_disable()

        # Stage 1: Exploitation
        _, completion_ids_1, _ = self.rollout_engine.generate(
            prompts=unique_prompts,
            model=self.model,
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            num_return_sequences=g1
        )

        # Stage 2: Fast-weight mutation
        lm_head, original_weights = self.unlearner.apply_unlearn_update(
            model=self.model,
            prompt_ids=dedup_prompt_ids,
            prompt_mask=dedup_prompt_mask,
            completion_ids=completion_ids_1,
            tokenizer=self.tokenizer
        )

        # Stage 3: Exploration with mutated weights
        if g2 > 0:
            _, completion_ids_2, _ = self.rollout_engine.generate(
                prompts=unique_prompts,
                model=self.model,
                max_new_tokens=self.config.max_completion_length,
                temperature=self.config.temperature,
                num_return_sequences=g2
            )

            # Pad to same length
            max_len = max(completion_ids_1.size(1), completion_ids_2.size(1))
            if completion_ids_1.size(1) < max_len:
                completion_ids_1 = torch.nn.functional.pad(
                    completion_ids_1,
                    (0, max_len - completion_ids_1.size(1)),
                    value=self.tokenizer.pad_token_id
                )
            if completion_ids_2.size(1) < max_len:
                completion_ids_2 = torch.nn.functional.pad(
                    completion_ids_2,
                    (0, max_len - completion_ids_2.size(1)),
                    value=self.tokenizer.pad_token_id
                )

            # Combine: [batch, g1+g2, seq_len]
            comp1 = completion_ids_1.view(len(unique_prompts), g1, -1)
            comp2 = completion_ids_2.view(len(unique_prompts), g2, -1)
            completion_ids = torch.cat([comp1, comp2], dim=1).view(
                len(unique_prompts) * self.config.num_generations, -1
            )
        else:
            completion_ids = completion_ids_1

        # Re-enable gradient checkpointing for training
        if grad_ckpt_was_enabled:
            self.model.gradient_checkpointing_enable()

        # Stage 4: Restore original weights
        self.unlearner.restore_weights(lm_head, original_weights)

        # Expand prompts to match completion batch size
        prompt_ids = dedup_prompt_ids.repeat_interleave(self.config.num_generations, dim=0)
        prompt_mask = dedup_prompt_mask.repeat_interleave(self.config.num_generations, dim=0)

        # Optional: Add ground truth to avoid reward collapse
        if self.config.add_gt and targets:
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            prompt_completion_ids = self._inject_ground_truth(
                prompt_completion_ids,
                prompt_ids,
                targets,
                len(unique_prompts)
            )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Compute completion mask
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

        # Decode completions
        completions = self.rollout_engine.decode_completions(completion_ids)

        # Compute rewards
        longview_histories = [x.get("longview_history", []) for x in batch]
        target_pids_list = [x.get("target_pids", []) for x in batch]
        reward_kwargs = {
            "target_sid": targets * self.config.num_generations,
            "longview_history": longview_histories * self.config.num_generations,
            "target_pids": target_pids_list * self.config.num_generations,
        }
        rewards = self.reward_engine(
            prompts=prompts * self.config.num_generations,
            completions=completions,
            **reward_kwargs
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Compute advantages
        mean_grouped_rewards = rewards.view(-1, self.config.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.config.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.config.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.config.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute reference log probabilities (ref_model offloaded to CPU)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            ref_model_for_logp = self.ref_model
        else:
            ref_model_for_logp = self.model
        ref_per_token_logps = self.rollout_engine.compute_ref_logprobs(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            ref_model=ref_model_for_logp
        )
        if self.ref_model is not None:
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "rewards": rewards
        }

    def _inject_ground_truth(
        self,
        prompt_completion_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        targets: list[str],
        num_unique_prompts: int
    ) -> torch.Tensor:
        """Inject ground truth as last generation for each prompt group."""
        repeat = len(targets) // num_unique_prompts
        new_prompt_completions = []

        for i in range(len(prompt_completion_ids)):
            if (i + 1) % max(repeat, 1) == 0:
                # Replace last generation with ground truth
                target_idx = i // max(repeat, 1)
                target_ids = self.tokenizer(
                    targets[target_idx],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=True
                )["input_ids"].squeeze().to(self.device)

                added_ids = torch.cat([prompt_ids[i], target_ids], dim=0)
                new_prompt_completions.append(added_ids)
            else:
                new_prompt_completions.append(prompt_completion_ids[i])

        return pad(new_prompt_completions, padding_value=self.tokenizer.pad_token_id)

    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """
        Compute EEPO loss (same as GRPO).

        EEPO's innovation is in generation, not loss computation.
        """
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        # Concatenate prompt and completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Forward pass
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits

        # Compute per-token log probabilities
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, target_ids)
        per_token_logps = per_token_logps[:, -completion_ids.size(1):]

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                       (ref_per_token_logps - per_token_logps) - 1

        # Policy gradient loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * \
                         advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.config.beta * per_token_kl)

        # Average over valid tokens and batch
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) /
            completion_mask.sum(dim=1).clamp(min=1)
        ).mean()

        return loss
