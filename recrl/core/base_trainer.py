"""
Base RL Trainer

Abstract base class for all RL algorithms (GRPO, EEPO, DPO, etc.).
Provides common training loop infrastructure while delegating
algorithm-specific logic to subclasses.

Features:
- Multi-GPU via Accelerate (automatic DDP / FSDP)
- swanlab + tensorboard logging
- Rich tqdm progress bar with live reward/loss stats
- Gradient accumulation, clipping, LR scheduling
"""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
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
    report_to: str = "none"          # "swanlab" | "tensorboard" | "none"
    run_name: Optional[str] = None   # swanlab run name

    # Device / distributed
    device: str = "cuda"
    seed: int = 42


class BaseRLTrainer(ABC):
    """
    Base trainer for RL algorithms.

    Supports single-GPU, multi-GPU (via Accelerate), and CPU training.
    Pass ACCELERATE=1 env var or use `accelerate launch` to enable multi-GPU.

    Subclasses must implement:
        compute_loss(inputs: dict) -> torch.Tensor
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
        sampler: Optional[Sampler] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.rollout_engine = rollout_engine
        self.reward_engine = reward_engine
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.global_step = 0
        self._metrics_history: list[dict] = []

        # ── Accelerate setup ──────────────────────────────────────────────
        try:
            from accelerate import Accelerator
            self.accelerator = Accelerator(
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            self._use_accelerate = True
        except ImportError:
            self.accelerator = None
            self._use_accelerate = False
            logger.warning(
                "accelerate not installed — running single-GPU. "
                "Install with: pip install accelerate"
            )

        self.device = (
            self.accelerator.device
            if self._use_accelerate
            else torch.device(config.device)
        )

        # ── Models ────────────────────────────────────────────────────────
        self.model = model
        self.ref_model = ref_model
        if not self._use_accelerate:
            self.model.to(self.device)
        if self.ref_model is not None:
            # ref_model stays on same device as model; never wrapped by accelerate
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        # ── Optimizer ─────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # ── Scheduler ─────────────────────────────────────────────────────
        steps_per_epoch = len(train_dataset) // config.per_device_batch_size
        total_steps = steps_per_epoch * config.num_epochs
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        # ── Accelerate: wrap model + optimizer ────────────────────────────
        if self._use_accelerate:
            self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler
            )

        # ── SwanLab logging ───────────────────────────────────────────────
        self._swanlab = None
        if config.report_to == "swanlab":
            try:
                import swanlab
                is_main = (not self._use_accelerate) or self.accelerator.is_main_process
                if is_main:
                    self._swanlab = swanlab.init(
                        project="CleanOneRec",
                        experiment_name=config.run_name or "run",
                        config=vars(config),
                    )
                    logger.info(f"SwanLab logging enabled (experiment: {config.run_name or 'run'})")
            except ImportError:
                logger.warning("swanlab not installed — pip install swanlab")

        # ── TensorBoard logging ───────────────────────────────────────────
        self._tb_writer = None
        if config.report_to == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(config.output_dir, "tb_logs")
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging → {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed, skipping TB logging")

        is_main = (not self._use_accelerate) or self.accelerator.is_main_process
        if is_main:
            n_gpu = (
                self.accelerator.num_processes
                if self._use_accelerate
                else (1 if str(self.device) != "cpu" else 0)
            )
            logger.info(f"Training on {n_gpu} GPU(s), device={self.device}")
            logger.info(f"Dataset: {len(train_dataset)} samples, "
                        f"{total_steps} total steps")

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def train(self):
        """Main training loop."""
        is_main = (not self._use_accelerate) or self.accelerator.is_main_process

        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_batch_size,
            sampler=self.sampler,
            collate_fn=lambda x: x,
        )
        if self._use_accelerate:
            dataloader = self.accelerator.prepare(dataloader)

        self.model.train()
        t_start = time.time()

        for epoch in range(self.config.num_epochs):
            if is_main:
                logger.info(f"{'='*60}")
                logger.info(f"Epoch {epoch + 1} / {self.config.num_epochs}")

            pbar = tqdm(
                dataloader,
                desc=f"Epoch {epoch+1}",
                disable=not is_main,
                dynamic_ncols=True,
            )

            for step, batch in enumerate(pbar):
                loss, metrics = self._training_step(batch)

                if is_main and (self.global_step + 1) % self.config.logging_steps == 0:
                    self._log_metrics(metrics, step=self.global_step)
                    pbar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        reward=f"{metrics['reward_mean']:.3f}",
                        lr=f"{metrics['lr']:.2e}",
                    )

                if is_main and (self.global_step + 1) % self.config.save_steps == 0:
                    self._save_checkpoint()

                self.global_step += 1

        if is_main:
            elapsed = time.time() - t_start
            logger.info(f"Training done in {elapsed/60:.1f} min")
            self._save_checkpoint(final=True)

        if self._swanlab is not None:
            self._swanlab.finish()

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _training_step(self, batch: list[dict]) -> tuple[float, dict]:
        """Single training step. Returns (loss_value, metrics_dict)."""
        inputs = self._prepare_inputs(batch)
        loss = self.compute_loss(inputs)

        if self._use_accelerate:
            self.accelerator.backward(loss / self.config.gradient_accumulation_steps)
        else:
            (loss / self.config.gradient_accumulation_steps).backward()

        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self._use_accelerate:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        rewards = inputs["rewards"]
        metrics = {
            "loss":         loss.item(),
            "reward_mean":  rewards.mean().item(),
            "reward_max":   rewards.max().item(),
            "reward_min":   rewards.min().item(),
            "reward_std":   rewards.std().item(),
            "advantage_mean": inputs["advantages"].mean().item(),
            "lr":           self.scheduler.get_last_lr()[0],
        }
        return loss.item(), metrics

    def _prepare_inputs(self, batch: list[dict]) -> dict:
        """
        Rollout + reward + reference logprobs + advantages.
        Called once per training step.
        """
        prompts              = [x["prompt"]                       for x in batch]
        targets              = [x.get("target_sid", "")           for x in batch]
        longview_histories   = [x.get("longview_history", [])     for x in batch]
        target_pids_list     = [x.get("target_pids", [])          for x in batch]

        # ── Generation ────────────────────────────────────────────────────
        # unwrap DDP model for generation
        gen_model = (
            self.accelerator.unwrap_model(self.model)
            if self._use_accelerate else self.model
        )
        prompt_ids, completion_ids, completion_mask = self.rollout_engine.generate(
            prompts=prompts,
            model=gen_model,
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            num_return_sequences=self.config.num_generations,
        )

        completions = self.rollout_engine.decode_completions(completion_ids)

        # ── Debug log on first step ───────────────────────────────────────
        if self.global_step == 0:
            logger.info("[step-0] Sample completions:")
            for i, c in enumerate(completions[:3]):
                logger.info(f"  [{i}] {c[:120]}")

        # ── Rewards ───────────────────────────────────────────────────────
        G = self.config.num_generations
        reward_kwargs = {
            "target_sid":       targets             * G,
            "longview_history": longview_histories  * G,
            "target_pids":      target_pids_list    * G,
        }
        rewards_list = self.reward_engine(
            prompts=prompts * G,
            completions=completions,
            **reward_kwargs,
        )
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

        if self.global_step == 0:
            logger.info(
                f"[step-0] Rewards  min={rewards.min():.4f}  "
                f"max={rewards.max():.4f}  mean={rewards.mean():.4f}"
            )

        # ── Advantages (group-wise normalisation) ─────────────────────────
        r_grouped  = rewards.view(-1, G)
        mean_r     = r_grouped.mean(dim=1, keepdim=True).repeat_interleave(G, dim=0).squeeze()
        std_r      = r_grouped.std(dim=1, keepdim=True).repeat_interleave(G, dim=0).squeeze()
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # ── Reference log-probs ───────────────────────────────────────────
        prompt_mask = (prompt_ids != self.tokenizer.pad_token_id).int()
        ref_model   = self.ref_model if self.ref_model is not None else gen_model
        ref_per_token_logps = self.rollout_engine.compute_ref_logprobs(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            ref_model=ref_model,
        )
        ref_per_token_logps = ref_per_token_logps.to(self.device)

        return {
            "prompt_ids":           prompt_ids,
            "prompt_mask":          prompt_mask,
            "completion_ids":       completion_ids,
            "completion_mask":      completion_mask,
            "ref_per_token_logps":  ref_per_token_logps,
            "advantages":           advantages,
            "rewards":              rewards,
        }

    def _log_metrics(self, metrics: dict, step: int):
        """Write metrics to all enabled backends."""
        self._metrics_history.append({"step": step, **metrics})

        # Console
        logger.info(
            f"step={step:4d}  loss={metrics['loss']:.4f}  "
            f"reward={metrics['reward_mean']:.3f}±{metrics['reward_std']:.3f}  "
            f"lr={metrics['lr']:.2e}"
        )

        # TensorBoard
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(f"train/{k}", v, step)

        # SwanLab
        if self._swanlab is not None:
            import swanlab
            swanlab.log({f"train/{k}": v for k, v in metrics.items()}, step=step)

    def _save_checkpoint(self, final: bool = False):
        """Save model + tokenizer checkpoint (only on main process)."""
        suffix    = "final" if final else f"step_{self.global_step}"
        save_path = os.path.join(self.config.output_dir, suffix)
        os.makedirs(save_path, exist_ok=True)

        save_model = (
            self.accelerator.unwrap_model(self.model)
            if self._use_accelerate else self.model
        )
        save_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Checkpoint saved → {save_path}")

    # ─────────────────────────────────────────────────────────────────────
    # Abstract
    # ─────────────────────────────────────────────────────────────────────

    @abstractmethod
    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """
        Compute algorithm-specific loss.

        Args:
            inputs: dict with keys:
                prompt_ids, prompt_mask, completion_ids, completion_mask,
                ref_per_token_logps, advantages, rewards

        Returns:
            Scalar loss tensor (do NOT call .backward() — trainer handles it)
        """
        raise NotImplementedError
