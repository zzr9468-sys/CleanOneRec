"""
EEPO Trainer

Explore-and-Evaluate Policy Optimization (EEPO) implementation.
Extends GRPO with fast-weight exploration to escape local optima.

Bug fixes vs original:
  - unique_prompts was wrongly sliced with [::num_generations]; fixed to use all prompts
  - _inject_ground_truth repeat factor was always 1; fixed to use actual num_generations
  - G2 exploration bonus incorrectly included the GT slot; now excludes it
  - Importance sampling used logps - logps.detach() = 0; now uses old_per_token_logps

New features (4 directions):
  Direction 1 — eepo_expand_scope: also mutate last-block v_proj (via FastWeightUnlearner)
  Direction 2 — _get_current_stage1_ratio: linear annealing of exploitation/exploration ratio
  Direction 3 — directed unlearn (head items) + remember loss (tail items)
  Direction 4 — eepo_separate_adv: separate G1/G2 advantage normalisation
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

import torch
from trl.trainer.utils import selective_log_softmax, pad

from ...core.base_trainer import BaseRLTrainer
from .config import EEPOConfig
from .unlearn import FastWeightUnlearner

logger = logging.getLogger(__name__)


class EEPOTrainer(BaseRLTrainer):
    """
    EEPO Trainer.

    Two-stage generation:
      Stage 1 (G1): Exploit with current policy
      Stage 2 (G2): Mutate lm_head (+ optionally v_proj), then explore
    """

    def __init__(self, config: EEPOConfig, **kwargs):
        super().__init__(config=config, **kwargs)

        # FastWeightUnlearner — created with Direction 1/3 settings
        self.unlearner = FastWeightUnlearner(
            unlearn_lr=config.eepo_unlearn_lr,
            unlearn_weight=config.eepo_unlearn_weight,
            epsilon=config.eepo_epsilon,
            expand_scope=config.eepo_expand_scope,
            remember_weight=config.eepo_remember_weight if config.eepo_remember_tail else 0.0,
        )

        # Direction 3: pre-build head/tail item SID corpora
        self._head_unlearn_completions: Optional[List[str]] = None
        self._tail_remember_completions: Optional[List[str]] = None

        if config.eepo_recif_path:
            self._head_unlearn_completions = self._build_sid_corpus(
                config.eepo_recif_path, config.eepo_head_k, mode="head"
            )
            logger.info(
                f"[EEPO] Directed unlearn: {len(self._head_unlearn_completions)} "
                f"head SID strings (top-{config.eepo_head_k} by freq)"
            )
            if config.eepo_remember_tail:
                self._tail_remember_completions = self._build_sid_corpus(
                    config.eepo_recif_path, config.eepo_tail_k, mode="tail"
                )
                logger.info(
                    f"[EEPO] Bidirectional remember: {len(self._tail_remember_completions)} "
                    f"tail SID strings (bottom-{config.eepo_tail_k} by freq)"
                )
        else:
            logger.warning(
                "[EEPO] eepo_recif_path not set — falling back to unlearning G1 rollout. "
                "Set eepo_recif_path (= reward recif_path) for directed unlearning."
            )

    # ── SID corpus helpers (Direction 3) ─────────────────────────────────

    def _build_sid_corpus(
        self, recif_path: str, k: int, mode: str
    ) -> List[str]:
        """
        Build list of SID strings for top-K (mode='head') or bottom-K (mode='tail')
        items by training-set frequency.
        """
        sid2pid_path = Path(recif_path) / "benchmark_data" / "sid2pid.json"
        with open(sid2pid_path, "r") as f:
            sid2pid = json.load(f)

        counts: Counter = Counter()
        for hk_str, pid_list in sid2pid.items():
            total = sum(
                item.get("count_after_downsample", item.get("count", 1))
                for item in pid_list
            )
            counts[int(hk_str)] = total

        if mode == "head":
            selected = [hk for hk, _ in counts.most_common(k)]
        else:  # tail
            sorted_asc = sorted(counts.items(), key=lambda x: x[1])
            selected = [hk for hk, cnt in sorted_asc if cnt >= 1][:k]

        # Reverse hash: hash_key = a * 8192^2 + b * 8192 + c
        sids: List[str] = []
        for hk in selected:
            c = hk % 8192
            b = (hk // 8192) % 8192
            a = hk // (8192 * 8192)
            sids.append(f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>")
        return sids

    def _sample_sids(self, corpus: List[str], n: int) -> torch.Tensor:
        """Sample n SIDs from corpus, tokenize, return [n, seq_len] tensor."""
        sampled = random.choices(corpus, k=n)
        enc = self.tokenizer(
            sampled,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,
        ).to(self.device)
        return enc["input_ids"]

    # ── Direction 2: dynamic stage1 ratio ────────────────────────────────

    def _get_current_stage1_ratio(self) -> float:
        """
        Linearly anneal stage1_ratio from eepo_stage1_ratio → eepo_stage1_ratio_end
        over eepo_stage1_anneal_steps steps.  After that, stays at ratio_end.
        """
        cfg = self.config
        if cfg.eepo_stage1_anneal_steps <= 0:
            return cfg.eepo_stage1_ratio
        progress = min(self.global_step / cfg.eepo_stage1_anneal_steps, 1.0)
        return cfg.eepo_stage1_ratio + progress * (
            cfg.eepo_stage1_ratio_end - cfg.eepo_stage1_ratio
        )

    # ── Main _prepare_inputs ─────────────────────────────────────────────

    def _prepare_inputs(self, batch: list[dict]) -> dict:
        if not self.config.eepo_enabled:
            return super()._prepare_inputs(batch)

        prompts  = [x["prompt"]               for x in batch]
        targets  = [x.get("target_sid", "")   for x in batch]

        # BUG-FIX: DataLoader already returns per_device_batch_size unique prompts;
        # the original [::num_generations] slicing was wrong.
        unique_prompts = prompts

        # Tokenize prompts
        prompt_inputs = self.tokenizer(
            unique_prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(self.device)
        dedup_prompt_ids  = prompt_inputs["input_ids"]
        dedup_prompt_mask = prompt_inputs["attention_mask"]

        # Direction 2: dynamic ratio
        stage1_ratio = self._get_current_stage1_ratio()
        g1 = max(1, int(self.config.num_generations * stage1_ratio))
        g2 = self.config.num_generations - g1

        # Disable gradient checkpointing during generation
        grad_ckpt = self.model.is_gradient_checkpointing
        if grad_ckpt:
            self.model.gradient_checkpointing_disable()

        # ── Stage 1: Exploitation ─────────────────────────────────────────
        _, completion_ids_1, _ = self.rollout_engine.generate(
            prompts=unique_prompts,
            model=self.model,
            max_new_tokens=self.config.max_completion_length,
            temperature=self.config.temperature,
            num_return_sequences=g1,
        )

        # ── Stage 2: Fast-weight mutation (Directions 1, 3) ──────────────
        # Unlearn target: pre-built head corpus (directed) or G1 fallback
        unlearn_ids = (
            self._sample_sids(self._head_unlearn_completions, len(unique_prompts) * g1)
            if self._head_unlearn_completions is not None
            else completion_ids_1
        )
        # Remember target: pre-built tail corpus (Direction 3)
        tail_ids = (
            self._sample_sids(self._tail_remember_completions, len(unique_prompts) * g1)
            if self._tail_remember_completions is not None
            else None
        )

        target_modules, original_weights = self.unlearner.apply_unlearn_update(
            model=self.model,
            prompt_ids=dedup_prompt_ids,
            prompt_mask=dedup_prompt_mask,
            completion_ids=unlearn_ids,
            tokenizer=self.tokenizer,
            tail_completion_ids=tail_ids,
        )

        # ── Stage 3: Exploration with mutated weights ─────────────────────
        if g2 > 0:
            _, completion_ids_2, _ = self.rollout_engine.generate(
                prompts=unique_prompts,
                model=self.model,
                max_new_tokens=self.config.max_completion_length,
                temperature=self.config.temperature,
                num_return_sequences=g2,
            )
            max_len = max(completion_ids_1.size(1), completion_ids_2.size(1))
            if completion_ids_1.size(1) < max_len:
                completion_ids_1 = torch.nn.functional.pad(
                    completion_ids_1, (0, max_len - completion_ids_1.size(1)),
                    value=self.tokenizer.pad_token_id,
                )
            if completion_ids_2.size(1) < max_len:
                completion_ids_2 = torch.nn.functional.pad(
                    completion_ids_2, (0, max_len - completion_ids_2.size(1)),
                    value=self.tokenizer.pad_token_id,
                )
            n_uniq = len(unique_prompts)
            comp1 = completion_ids_1.view(n_uniq, g1, -1)
            comp2 = completion_ids_2.view(n_uniq, g2, -1)
            completion_ids = torch.cat([comp1, comp2], dim=1).view(
                n_uniq * self.config.num_generations, -1
            )
        else:
            completion_ids = completion_ids_1

        if grad_ckpt:
            self.model.gradient_checkpointing_enable()

        # ── Stage 4: Restore mutated weights ─────────────────────────────
        self.unlearner.restore_weights(target_modules, original_weights)

        # Expand prompts to [batch * num_gen, prompt_len]
        prompt_ids = dedup_prompt_ids.repeat_interleave(self.config.num_generations, dim=0)
        prompt_mask = dedup_prompt_mask.repeat_interleave(self.config.num_generations, dim=0)

        # Optional: inject ground truth as last slot per group
        if self.config.add_gt and targets:
            pc_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            pc_ids = self._inject_ground_truth(
                pc_ids, prompt_ids, targets, len(unique_prompts)
            )
            pl = prompt_ids.size(1)
            prompt_ids     = pc_ids[:, :pl]
            completion_ids = pc_ids[:, pl:]

        # Completion mask
        is_eos  = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

        completions = self.rollout_engine.decode_completions(completion_ids)

        # ── Rewards ───────────────────────────────────────────────────────
        longview_histories = [x.get("longview_history", []) for x in batch]
        target_pids_list   = [x.get("target_pids", [])      for x in batch]
        G = self.config.num_generations
        reward_kwargs = {
            "target_sid":       targets             * G,
            "longview_history": longview_histories  * G,
            "target_pids":      target_pids_list    * G,
        }
        rewards = torch.tensor(
            self.reward_engine(
                prompts=unique_prompts * G,
                completions=completions,
                **reward_kwargs,
            ),
            dtype=torch.float32, device=self.device,
        )

        # ── Direction 4: Separate G1/G2 advantage normalisation ──────────
        n_uniq  = len(unique_prompts)
        r_mat   = rewards.view(n_uniq, G)

        if self.config.eepo_separate_adv:
            # G1 sub-group normalisation
            r_g1 = r_mat[:, :g1]
            if g1 > 1:
                adv_g1 = (
                    (r_g1 - r_g1.mean(dim=1, keepdim=True))
                    / (r_g1.std(dim=1, keepdim=True) + 1e-4)
                )
            else:
                # Single sample per group — fall back to global stats
                adv_g1 = (r_g1 - rewards.mean()) / (rewards.std() + 1e-4)

            adv_parts = [adv_g1]

            if g2 > 0:
                r_g2 = r_mat[:, g1:]
                if g2 > 1:
                    adv_g2 = (
                        (r_g2 - r_g2.mean(dim=1, keepdim=True))
                        / (r_g2.std(dim=1, keepdim=True) + 1e-4)
                    )
                else:
                    adv_g2 = torch.zeros_like(r_g2)

                # Exploration bonus: only real G2 slots, exclude GT slot if add_gt
                if self.config.eepo_exploration_bonus > 0:
                    bonus_slots = g2 - (1 if self.config.add_gt else 0)
                    if bonus_slots > 0:
                        adv_g2 = adv_g2.clone()
                        adv_g2[:, :bonus_slots] += self.config.eepo_exploration_bonus

                adv_parts.append(adv_g2)

            advantages = torch.cat(adv_parts, dim=1).view(-1)
        else:
            # Shared baseline (original GRPO style)
            mean_r = r_mat.mean(dim=1, keepdim=True).repeat_interleave(G, dim=0).view(-1)
            std_r  = r_mat.std(dim=1, keepdim=True).repeat_interleave(G, dim=0).view(-1)
            advantages = (rewards - mean_r) / (std_r + 1e-4)
            if g2 > 0 and self.config.eepo_exploration_bonus > 0:
                adv_mat   = advantages.view(n_uniq, G).clone()
                bonus_end = G - (1 if self.config.add_gt else 0)
                if bonus_end > g1:
                    adv_mat[:, g1:bonus_end] += self.config.eepo_exploration_bonus
                advantages = adv_mat.view(-1)

        # ── Reference log-probs ───────────────────────────────────────────
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            ref_m = self.ref_model
        else:
            ref_m = self.model
        ref_per_token_logps = self.rollout_engine.compute_ref_logprobs(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            ref_model=ref_m,
        )
        if self.ref_model is not None:
            self.ref_model.to("cpu")
            torch.cuda.empty_cache()

        # ── Old log-probs (policy snapshot, needed for importance sampling) ──
        full_ids  = torch.cat([prompt_ids, completion_ids], dim=1)
        full_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        with torch.no_grad():
            logits_old = self.model(
                input_ids=full_ids, attention_mask=full_mask
            ).logits
        logits_old = logits_old[:, :-1, :]
        old_per_token_logps = selective_log_softmax(logits_old, full_ids[:, 1:])
        old_per_token_logps = old_per_token_logps[:, -completion_ids.size(1):]

        return {
            "prompt_ids":           prompt_ids,
            "prompt_mask":          prompt_mask,
            "completion_ids":       completion_ids,
            "completion_mask":      completion_mask,
            "ref_per_token_logps":  ref_per_token_logps,
            "old_per_token_logps":  old_per_token_logps,
            "advantages":           advantages,
            "rewards":              rewards,
        }

    # ── Ground-truth injection ─────────────────────────────────────────────

    def _inject_ground_truth(
        self,
        prompt_completion_ids: torch.Tensor,
        prompt_ids: torch.Tensor,
        targets: list[str],
        num_unique_prompts: int,
    ) -> torch.Tensor:
        """Replace the last generation in each group with the ground-truth SID."""
        # BUG-FIX: was len(targets)//num_unique_prompts = 1 always; now uses actual G
        repeat = prompt_completion_ids.size(0) // num_unique_prompts
        new_seqs = []

        for i in range(len(prompt_completion_ids)):
            if (i + 1) % max(repeat, 1) == 0:
                target_idx = i // max(repeat, 1)
                gt_ids = self.tokenizer(
                    targets[target_idx],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=True,
                )["input_ids"].squeeze().to(self.device)
                new_seqs.append(torch.cat([prompt_ids[i], gt_ids], dim=0))
            else:
                new_seqs.append(prompt_completion_ids[i])

        return pad(new_seqs, padding_value=self.tokenizer.pad_token_id)

    # ── Loss computation ───────────────────────────────────────────────────

    def compute_loss(self, inputs: dict) -> torch.Tensor:
        """GRPO-style loss with correct importance sampling."""
        prompt_ids          = inputs["prompt_ids"]
        prompt_mask         = inputs["prompt_mask"]
        completion_ids      = inputs["completion_ids"]
        completion_mask     = inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        old_per_token_logps = inputs["old_per_token_logps"]
        advantages          = inputs["advantages"]

        input_ids      = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits

        logits          = logits[:, :-1, :]
        target_ids      = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, target_ids)
        per_token_logps = per_token_logps[:, -completion_ids.size(1):]

        # KL penalty (reverse KL from ref policy)
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps) - 1
        )

        # Policy-gradient loss with genuine off-policy importance-sampling ratio
        per_token_loss = (
            torch.exp(per_token_logps - old_per_token_logps.detach())
            * advantages.unsqueeze(1)
        )
        per_token_loss = -(per_token_loss - self.config.beta * per_token_kl)

        loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        ).mean()
        return loss
