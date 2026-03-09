"""
Base RL Trainer

Abstract base class for all RL algorithms (GRPO, EEPO, DPO, etc.).
Provides common training loop infrastructure while delegating
algorithm-specific logic to subclasses.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler
)
from datasets import Dataset
from tqdm import tqdm
import logging

from .rollout import RolloutEngine
from .reward import BaseReward
from .data import DataEngine

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Base configuration for RL training."""

    # Training
    num_epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    warmup_steps: int = 0
    max_grad_norm: float = 1.0

    # Generation
    num_generations: int = 16
    max_completion_length: int = 128
    temperature: float = 0.7

    # RL specific
    beta: float = 0.04  # KL penalty coefficient

    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    output_dir: str = "./outputs"

    # Device
    device: str = "cuda"
    seed: int = 42


class BaseRLTrainer(ABC):
    """
    Base trainer for RL algorithms.

    Subclasses must implement:
    - compute_loss(): Algorithm-specific loss computation
    """

    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel],
        config: RLConfig,
        train_dataset: Dataset,
        rollout_engine: RolloutEngine,
        reward_engine: BaseReward,
        tokenizer: PreTrainedTokenizerBase,
        sampler: Optional[Sampler] = None
    ):
        """
        Args:
            model: Policy model to train
            ref_model: Reference model (frozen)
            config: Training configuration
            train_dataset: HuggingFace Dataset
            rollout_engine: RolloutEngine for generation
            reward_engine: Reward function
            tokenizer: Tokenizer
            sampler: Optional custom sampler
        """
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.train_dataset = train_dataset
        self.rollout_engine = rollout_engine
        self.reward_engine = reward_engine
        self.tokenizer = tokenizer
        self.sampler = sampler

        self.device = config.device
        self.global_step = 0

        # Move models to device
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

        # Setup scheduler
        total_steps = (len(train_dataset) // config.per_device_batch_size) * config.num_epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_batch_size,
            sampler=self.sampler,
            collate_fn=lambda x: x  # No collation needed
        )

        self.model.train()

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for step, batch in enumerate(tqdm(dataloader, desc="Training")):
                loss = self._training_step(batch)

                if (step + 1) % self.config.logging_steps == 0:
                    logger.info(f"Step {self.global_step}, Loss: {loss:.4f}")

                if (step + 1) % self.config.save_steps == 0:
                    self._save_checkpoint()

                self.global_step += 1

        logger.info("Training completed!")
        self._save_checkpoint(final=True)

    def _training_step(self, batch: list[dict]) -> float:
        """Single training step."""
        # Prepare inputs (rollout + reward computation)
        inputs = self._prepare_inputs(batch)

        # Compute loss (algorithm-specific)
        loss = self.compute_loss(inputs)

        # Backward pass
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return loss.item() * self.config.gradient_accumulation_steps

    def _prepare_inputs(self, batch: list[dict]) -> dict:
        """
        Prepare inputs for loss computation.

        This includes:
        1. Generating completions via RolloutEngine
        2. Computing rewards via RewardEngine
        3. Computing reference log probabilities
        4. Computing advantages (for policy gradient methods)
        """
        prompts = [x["prompt"] for x in batch]
        targets = [x.get("target_sid", "") for x in batch]

        # Generate completions
        prompt_ids, completion_ids, completion_mask = self.rollout_engine.generate(
            prompts=prompts,
            model=self.model,
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            num_return_sequences=self.config.num_generations
        )

        # Decode completions
        completions = self.rollout_engine.decode_completions(completion_ids)

        # Compute rewards
        reward_kwargs = {"target_sid": targets * self.config.num_generations}
        rewards = self.reward_engine(
            prompts=prompts * self.config.num_generations,
            completions=completions,
            **reward_kwargs
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Compute advantages (group-wise normalization)
        mean_grouped_rewards = rewards.view(-1, self.config.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.config.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.config.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.config.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Compute reference log probabilities
        prompt_mask = (prompt_ids != self.tokenizer.pad_token_id).int()
        ref_per_token_logps = self.rollout_engine.compute_ref_logprobs(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            ref_model=self.ref_model if self.ref_model is not None else self.model
        )

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "rewards": rewards
        }

    @abstractmethod
    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """
        Compute algorithm-specific loss.

        Args:
            inputs: Dictionary containing:
                - prompt_ids: [batch, prompt_len]
                - completion_ids: [batch, completion_len]
                - prompt_mask: [batch, prompt_len]
                - completion_mask: [batch, completion_len]
                - ref_per_token_logps: [batch, completion_len]
                - advantages: [batch]
                - rewards: [batch]

        Returns:
            Scalar loss tensor
        """
        raise NotImplementedError

    def _save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        import os
        suffix = "final" if final else f"step_{self.global_step}"
        save_path = os.path.join(self.config.output_dir, suffix)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Checkpoint saved to {save_path}")
