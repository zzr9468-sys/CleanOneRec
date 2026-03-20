"""
Fast-Weight Unlearner

Implements the temporary weight mutation logic for EEPO exploration.

Changes vs original:
  Direction 1 — expand_scope: optionally also mutate the last transformer block's
                v_proj so the representation space shifts, not just token probabilities.
  Direction 3 — bidirectional loss: unlearn head items (push away) and simultaneously
                remember tail items (pull toward), giving fast-weight mutation direction.
"""

import torch
from typing import Optional, Tuple, List
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.utils import selective_log_softmax


class FastWeightUnlearner:
    """
    Applies temporary fast-weight updates to push model out of comfort zone.

    Process:
      1. Select target modules (lm_head + optionally last-block v_proj)
      2. Compute total loss:
           unlearn_loss  = -log(1 - P(head_completion))   [push away from heads]
           remember_loss = cross-entropy on tail_completion [pull toward tails]
      3. Apply a single SGD step with high learning rate
      4. Caller generates G2 with mutated weights
      5. Caller calls restore_weights() to undo the mutation
    """

    def __init__(
        self,
        unlearn_lr: float = 1e-5,
        unlearn_weight: float = 1.0,
        epsilon: float = 1e-4,
        expand_scope: bool = False,
        remember_weight: float = 0.0,
    ):
        self.unlearn_lr      = unlearn_lr
        self.unlearn_weight  = unlearn_weight
        self.epsilon         = epsilon
        self.expand_scope    = expand_scope
        self.remember_weight = remember_weight

    def apply_unlearn_update(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        tail_completion_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.nn.Module], List[torch.Tensor]]:
        """
        Apply fast-weight unlearning update.

        Args:
            model:               Model to mutate
            prompt_ids:          Prompt token IDs [batch, prompt_len]
            prompt_mask:         Prompt attention mask [batch, prompt_len]
            completion_ids:      Head-item SID IDs [batch*g1, completion_len]
            tokenizer:           Tokenizer (used for EOS detection)
            tail_completion_ids: Optional tail-item SID IDs for remember loss

        Returns:
            (target_modules, original_weights) — pass to restore_weights() after G2 gen
        """
        if self.unlearn_lr <= 0 or self.unlearn_weight <= 0:
            return [], []

        # ── Collect target modules and save original weights ───────────────
        target_modules  = self._get_target_modules(model)
        original_weights = [m.weight.data.clone() for m in target_modules]

        # ── Freeze everything except target modules ────────────────────────
        target_param_ids = {id(p) for m in target_modules for p in m.parameters()}
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = (id(param) in target_param_ids)

        model.train()

        # Expand prompts to match completion batch size
        num_prompts  = prompt_ids.size(0)
        repeat_count = completion_ids.size(0) // num_prompts
        if repeat_count > 1:
            p_ids  = prompt_ids.repeat_interleave(repeat_count, dim=0)
            p_mask = prompt_mask.repeat_interleave(repeat_count, dim=0)
        else:
            p_ids, p_mask = prompt_ids, prompt_mask

        # ── Unlearn loss ───────────────────────────────────────────────────
        total_loss = self._unlearn_loss(
            model, p_ids, p_mask, completion_ids, tokenizer
        ) * self.unlearn_weight

        # ── Remember loss (Direction 3) ────────────────────────────────────
        if self.remember_weight > 0 and tail_completion_ids is not None:
            t_repeat = tail_completion_ids.size(0) // num_prompts
            if t_repeat > 1:
                t_ids  = prompt_ids.repeat_interleave(t_repeat, dim=0)
                t_mask = prompt_mask.repeat_interleave(t_repeat, dim=0)
            else:
                t_ids, t_mask = prompt_ids, prompt_mask
            total_loss = total_loss + self._remember_loss(
                model, t_ids, t_mask, tail_completion_ids, tokenizer
            ) * self.remember_weight

        # ── Single SGD step ────────────────────────────────────────────────
        opt = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.unlearn_lr,
        )
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        model.eval()

        # Restore requires_grad
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]

        return target_modules, original_weights

    def restore_weights(
        self,
        modules: List[torch.nn.Module],
        original_weights: List[torch.Tensor],
    ):
        """Restore all mutated modules to their pre-mutation weights."""
        for module, weights in zip(modules, original_weights):
            module.weight.data.copy_(weights)

    # ── Private helpers ───────────────────────────────────────────────────

    def _unlearn_loss(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        """Make model LESS likely to generate head-item completions."""
        comp_mask = self._build_completion_mask(completion_ids, tokenizer)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attn_mask = torch.cat([prompt_mask, comp_mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        logits = logits[:, :-1, :]
        per_token_logps = selective_log_softmax(logits, input_ids[:, 1:])
        per_token_logps = per_token_logps[:, -comp_mask.shape[1]:]

        denom    = comp_mask.sum(dim=1).clamp(min=1)
        seq_logp = (per_token_logps * comp_mask).sum(dim=1) / denom
        probs    = torch.exp(seq_logp).clamp(max=1 - self.epsilon)
        return (-torch.log(1 - probs)).mean()

    def _remember_loss(
        self,
        model: PreTrainedModel,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        """Make model MORE likely to generate tail-item completions (cross-entropy)."""
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        comp_mask = (completion_ids != pad_id).int()
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attn_mask = torch.cat([prompt_mask, comp_mask], dim=1)

        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        logits = logits[:, :-1, :]
        per_token_logps = selective_log_softmax(logits, input_ids[:, 1:])
        per_token_logps = per_token_logps[:, -comp_mask.shape[1]:]

        denom = comp_mask.sum(dim=1).clamp(min=1)
        return ((-per_token_logps * comp_mask).sum(dim=1) / denom).mean()

    def _get_target_modules(self, model: PreTrainedModel) -> List[torch.nn.Module]:
        """Collect modules to be mutated during fast-weight update."""
        modules = []
        lm_head = self._get_lm_head(model)
        if lm_head is not None:
            modules.append(lm_head)
        if self.expand_scope:  # Direction 1
            v_proj = self._get_last_v_proj(model)
            if v_proj is not None:
                modules.append(v_proj)
        return modules

    def _get_lm_head(self, model: PreTrainedModel) -> Optional[torch.nn.Module]:
        if hasattr(model, "lm_head"):
            return model.lm_head
        if hasattr(model, "embed_out"):
            return model.embed_out
        children = list(model.children())
        return children[-1] if children else None

    def _get_last_v_proj(self, model: PreTrainedModel) -> Optional[torch.nn.Module]:
        """Locate the v_proj Linear layer of the last transformer block."""
        last_block = None
        # LLaMA / Qwen / Mistral
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            last_block = model.model.layers[-1]
        # GPT-2 / Falcon
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            last_block = model.transformer.h[-1]
        # GPT-NeoX
        elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
            last_block = model.gpt_neox.layers[-1]

        if last_block is None:
            return None

        for _, module in last_block.named_modules():
            if isinstance(module, torch.nn.Linear):
                n = _.lower()
                if "v_proj" in n or ("value" in n and "proj" in n):
                    return module
        return None

    def _build_completion_mask(
        self,
        completion_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.Tensor:
        """Mask covering all tokens up to and including the first EOS."""
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1),
            dtype=torch.long, device=completion_ids.device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        seq_idx = torch.arange(
            is_eos.size(1), device=completion_ids.device
        ).expand(is_eos.size(0), -1)
        return (seq_idx <= eos_idx.unsqueeze(1)).int()
