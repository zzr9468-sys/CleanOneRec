import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Optional, Union, Sized
from packaging import version
import transformers
from datasets import Dataset, IterableDataset
from torch.utils.data import Sampler
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
)
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import pad, selective_log_softmax
from accelerate.utils import gather, gather_object
from transformers import LogitsProcessorList, TemperatureLogitsWarper
from ..utils.logit_processor import ConstrainedLogitsProcessor, SIDTrie

# --- Sampler ---
class RepeatRandomSampler(Sampler):
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

# --- Trainer ---
class EEPOTrainer(GRPOTrainer):
    """
    Explore-and-Evaluate Policy Optimization (EEPO) Trainer.
    Built on top of TRL's GRPOTrainer but injects the Fast-Weight Mutate logic 
    to force exploration outside the "Teacher Bias" comfort zone.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[Any, list[Any]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        # EEPO Specific Hyperparams
        eepo_enabled: bool = True,
        eepo_stage1_ratio: float = 0.5,
        eepo_unlearn_lr: float = 1e-5,
        eepo_unlearn_weight: float = 1.0,
        eepo_epsilon: float = 1e-4,
        add_gt: bool = True,
        allowed_trie: Optional[SIDTrie] = None,
        **kwargs
    ):
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs
        )
        
        self.eepo_enabled = eepo_enabled
        self.eepo_stage1_ratio = eepo_stage1_ratio
        self.eepo_unlearn_lr = eepo_unlearn_lr
        self.eepo_unlearn_weight = eepo_unlearn_weight
        self.eepo_epsilon = eepo_epsilon
        self.add_gt = add_gt
        self.allowed_trie = allowed_trie
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self) -> Sampler:
        return RepeatRandomSampler(self.train_dataset, self.args.num_generations, seed=self.args.seed)

    def _sample_with_model(self, model, prompt_ids, prompt_mask, num_generations):
        logits_processor = LogitsProcessorList()
        if self.args.temperature > 0 and self.args.temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=self.args.temperature))
        if self.allowed_trie is not None:
            logits_processor.append(ConstrainedLogitsProcessor(self.allowed_trie.get_allowed_next_tokens, prompt_length=prompt_ids.size(1)))

        generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            num_return_sequences=num_generations,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
            # Prevent multinomial failure by setting sensible min_p
            # This is optional but good safety measure
        )
        with torch.no_grad():
            prompt_completion_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
                logits_processor=logits_processor,
            )
        prompt_length = prompt_ids.size(1)
        return prompt_completion_ids[:, prompt_length:]

    def _apply_unlearn_update_fast(self, model, prompt_ids, prompt_mask, completion_ids):
        """
        In-place Fast-Weight mutation on LM_Head to push model out of comfort zone.
        """
        if self.eepo_unlearn_lr <= 0 or self.eepo_unlearn_weight <= 0:
            return None, None

        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
        elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            lm_head = model.model.embed_tokens
        elif hasattr(model, "embed_out"):
            lm_head = model.embed_out
        else:
            lm_head = list(model.children())[-1]

        original_head_weight = lm_head.weight.data.clone()

        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            if any(id(param) == id(p) for p in lm_head.parameters()):
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.train()
        num_prompts = prompt_ids.size(0)
        repeat_count = completion_ids.size(0) // num_prompts
        prompt_ids_expanded = prompt_ids.repeat_interleave(repeat_count, dim=0)
        prompt_mask_expanded = prompt_mask.repeat_interleave(repeat_count, dim=0)

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        input_ids = torch.cat([prompt_ids_expanded, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask_expanded, completion_mask], dim=1)
        
        # Micro-Backward
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, target_ids)
        
        # Align shape with completion_mask
        per_token_logps = per_token_logps[:, -completion_mask.shape[1]:]

        token_logps = per_token_logps * completion_mask
        denom = completion_mask.sum(dim=1).clamp(min=1)
        seq_logp = token_logps.sum(dim=1) / denom
        probs = torch.exp(seq_logp).clamp(max=1 - self.eepo_epsilon)
        unlearn_loss = (-torch.log(1 - probs)).mean() * self.eepo_unlearn_weight

        temp_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.eepo_unlearn_lr)
        temp_optimizer.zero_grad()
        unlearn_loss.backward()
        temp_optimizer.step()
        model.eval()

        # Restore
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]

        return lm_head, original_head_weight

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """Overrides GRPOTrainer to inject EEPO Rollout logic."""
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        targets = [x.get("target_sid", "") for x in inputs] 
        num_categories = len(set(targets)) if targets else 1
        
        prompt_inputs = self.processing_class(
            prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        ).to(device)
        
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            if self.eepo_enabled:
                dedup_prompt, dedup_mask = [], []
                for i in range(len(prompt_ids)):
                    if i % self.args.num_generations == 0:
                        dedup_prompt.append(prompt_ids[i])
                        dedup_mask.append(prompt_mask[i])
                dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                dedup_prompt_mask = torch.stack(dedup_mask).to(device)

                g1 = max(1, int(self.args.num_generations * self.eepo_stage1_ratio))
                g2 = self.args.num_generations - g1

                # Step 1: Base Strategy Exploitation
                completion_ids_1 = self._sample_with_model(
                    unwrapped_model, dedup_prompt_ids, dedup_prompt_mask, g1
                )
                
                # Step 2: Unlearn / Mutate
                lm_head, original_head_weight = self._apply_unlearn_update_fast(
                    unwrapped_model, dedup_prompt_ids, dedup_prompt_mask, completion_ids_1
                )
                
                # Step 3: Fast-Weight Exploration
                if g2 > 0:
                    completion_ids_2 = self._sample_with_model(
                        unwrapped_model, dedup_prompt_ids, dedup_prompt_mask, g2
                    )
                    max_len = max(completion_ids_1.size(1), completion_ids_2.size(1))
                    
                    if completion_ids_1.size(1) < max_len:
                        completion_ids_1 = torch.nn.functional.pad(completion_ids_1, (0, max_len - completion_ids_1.size(1)), value=self.processing_class.pad_token_id)
                    if completion_ids_2.size(1) < max_len:
                        completion_ids_2 = torch.nn.functional.pad(completion_ids_2, (0, max_len - completion_ids_2.size(1)), value=self.processing_class.pad_token_id)
                        
                    comp1 = completion_ids_1.view(len(dedup_prompt), g1, -1)
                    comp2 = completion_ids_2.view(len(dedup_prompt), g2, -1)
                    completion_ids = torch.cat([comp1, comp2], dim=1).view(len(dedup_prompt) * self.args.num_generations, -1)
                else:
                    completion_ids = completion_ids_1

                # Step 4: Restore Original LM_Head
                if lm_head is not None and original_head_weight is not None:
                    lm_head.weight.data.copy_(original_head_weight)

                prompt_ids = dedup_prompt_ids.repeat_interleave(self.args.num_generations, dim=0)
                prompt_mask = dedup_prompt_mask.repeat_interleave(self.args.num_generations, dim=0)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                
            else:
                logits_processor = LogitsProcessorList()
                if self.args.temperature > 0 and self.args.temperature != 1.0:
                    logits_processor.append(TemperatureLogitsWarper(temperature=self.args.temperature))
                if self.allowed_trie is not None:
                    logits_processor.append(ConstrainedLogitsProcessor(self.allowed_trie.get_allowed_next_tokens, prompt_length=prompt_ids.size(1)))

                generation_config = GenerationConfig(
                    max_new_tokens=self.args.max_completion_length,
                    do_sample=True,
                    temperature=self.args.temperature,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                )
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=generation_config, logits_processor=logits_processor
                )

        # Bootstrapping GT to avoid early reward collapse
        if self.add_gt and targets:
            repeat = len(prompts) // num_categories
            new_prompt_completions = []
            for i in range(len(prompts)):
                if (i+1) % max(repeat, 1) == 0:
                    target_ids = self.processing_class(
                        targets[i], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True
                    )["input_ids"].squeeze().to(device)
                    added_ids = torch.cat([prompt_ids[i], target_ids], dim=0)
                    new_prompt_completions.append(added_ids)
                else:
                    new_prompt_completions.append(prompt_completion_ids[i])
            prompt_completion_ids = pad(new_prompt_completions, padding_value=self.processing_class.pad_token_id)
            
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1) 
        
        with torch.inference_mode():
            if self.ref_model is not None:
                logits = self.ref_model(input_ids=prompt_completion_ids, attention_mask=attention_mask).logits
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    logits = self.model(input_ids=prompt_completion_ids, attention_mask=attention_mask).logits
            
            ref_logits = logits[:, :-1, :]
            target_ids = prompt_completion_ids[:, 1:]
            ref_per_token_logps = selective_log_softmax(ref_logits, target_ids)
            ref_per_token_logps = ref_per_token_logps[:, -logits_to_keep:]

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions = completions_text
        
        # Rewards Computation
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_weight) in enumerate(zip(self.reward_funcs, getattr(self, "reward_weights", [1.0]*len(self.reward_funcs)))):
            reward_kwargs = {key: [example[key] for example in inputs] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device) * reward_weight

        rewards = rewards_per_func.sum(dim=1)

        # GPRO Advantage Normalization
        mean_grouped_rewards = rewards.view(-1, self.args.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.args.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.args.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.args.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        if getattr(self, "log_completions", False) and self.state.global_step % self.args.logging_steps == 0:
            if self.accelerator.is_main_process and is_wandb_available():
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["Prompt", "Completion", "Reward"])
                    for row in zip(prompts, completions_text, rewards.tolist()):
                        table.add_data(*row)
                    wandb.log({"completions": table}, step=self.state.global_step)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        per_token_logps = selective_log_softmax(logits, target_ids)
        per_token_logps = per_token_logps[:, -completion_ids.size(1):]

        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.args.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        if hasattr(self, "_metrics"):
            # (since advantage is just a standardized reward, but let's safely append)
            if "reward" not in self._metrics:
                self._metrics["reward"] = []
            if "reward_std" not in self._metrics:
                self._metrics["reward_std"] = []
            if "kl" not in self._metrics:
                self._metrics["kl"] = []
                
            self._metrics["reward"].append(inputs["advantages"].mean().item())
            self._metrics["reward_std"].append(inputs["advantages"].std().item())
            self._metrics["kl"].append((per_token_kl * completion_mask).sum(dim=1).mean().item())

        return loss
