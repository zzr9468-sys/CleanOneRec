# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
import copy
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl import SyncRefModelCallback
from trl import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import random

from transformers import (
    is_wandb_available, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    TemperatureLogitsWarper, 
    LogitsProcessorList,
    Trainer
)
import math

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.
    """

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


class ReReTrainer(Trainer):
    """
    Cleaner ReReTrainer tailored for MiniOneRec GRPO training.
    Removes unused legacy code (like VLLM patches and constraint generation that wasn't used).
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        base_model: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        # sample
        add_gt: bool = False,
        # EEPO
        eepo_enabled: bool = False,
        eepo_stage1_ratio: float = 0.5,
        eepo_unlearn_lr: float = 1e-5,
        eepo_unlearn_weight: float = 1.0,
        eepo_epsilon: float = 1e-4,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        **kwargs # Accept legacy kwargs but ignore them
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.base_model = base_model
        model_init_kwargs = args.model_init_kwargs or {}
        
        if isinstance(model, str):
            model_init_kwargs["use_cache"] = (False if args.gradient_checkpointing else model_init_kwargs.get("use_cache"))
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
            
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model.config._name_or_path, **model_init_kwargs)
        elif not is_peft_model(model):
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
            if processing_class.pad_token is None:
                processing_class.pad_token = processing_class.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        def data_collator(features):  # No data collation is needed in GRPO
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  
        self.num_generations = args.num_generations 
        self.beta = args.beta
        
        model.warnings_issued["estimate_tokens"] = True
        self._metrics = defaultdict(list)   
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.add_gt = add_gt
        self.temperature = args.temperature
        self.eepo_enabled = eepo_enabled
        self.eepo_stage1_ratio = eepo_stage1_ratio
        self.eepo_unlearn_lr = eepo_unlearn_lr
        self.eepo_unlearn_weight = eepo_unlearn_weight
        self.eepo_epsilon = eepo_epsilon

        set_seed(args.seed, device_specific=True)
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id,
        )

        self.model_accepts_loss_kwargs = False
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _sample_with_model(self, model, prompt_ids, prompt_mask, num_generations):
        generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=num_generations,
            pad_token_id=self.processing_class.pad_token_id,
            eos_token_id=self.processing_class.eos_token_id,
        )
        with torch.no_grad():
            prompt_completion_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config,
            )
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        return completion_ids

    def _apply_unlearn_update_fast(self, model, prompt_ids, prompt_mask, completion_ids):
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
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
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

        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]

        return lm_head, original_head_weight

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "target_sid"] # Keep target_sid

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        targets = [x["target_sid"] for x in inputs] 
        num_categories = len(set(targets))
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            
            if self.eepo_enabled:
                dedup_prompt, dedup_mask = [], []
                for i in range(len(prompt_ids)):
                    if i % self.num_generations == 0:
                        dedup_prompt.append(prompt_ids[i])
                        dedup_mask.append(prompt_mask[i])
                dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                dedup_prompt_mask = torch.stack(dedup_mask).to(device)

                g1 = max(1, int(self.num_generations * self.eepo_stage1_ratio))
                g2 = self.num_generations - g1

                completion_ids_1 = self._sample_with_model(
                    unwrapped_model, dedup_prompt_ids, dedup_prompt_mask, g1
                )
                
                lm_head, original_head_weight = self._apply_unlearn_update_fast(
                    unwrapped_model, dedup_prompt_ids, dedup_prompt_mask, completion_ids_1
                )
                
                if g2 > 0:
                    completion_ids_2 = self._sample_with_model(
                        unwrapped_model, dedup_prompt_ids, dedup_prompt_mask, g2
                    )
                    comp1 = completion_ids_1.view(len(dedup_prompt), g1, -1)
                    comp2 = completion_ids_2.view(len(dedup_prompt), g2, -1)
                    completion_ids = torch.cat([comp1, comp2], dim=1).view(len(dedup_prompt) * self.num_generations, -1)
                else:
                    completion_ids = completion_ids_1

                if lm_head is not None and original_head_weight is not None:
                    lm_head.weight.data.copy_(original_head_weight)

                prompt_ids = dedup_prompt_ids.repeat_interleave(self.num_generations, dim=0)
                prompt_mask = dedup_prompt_mask.repeat_interleave(self.num_generations, dim=0)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                
            else:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

        if self.add_gt:
            repeat = len(prompts) // num_categories
            new_prompt_completions = []
            for i in range(len(prompts)):
                if (i+1)%repeat == 0:
                    target_ids = self.processing_class(targets[i], return_tensors="pt", padding=True, padding_side="left", \
                        add_special_tokens=True)["input_ids"].squeeze()
                    target_ids = target_ids.to(device)
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
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Calculate Diversity for Wandb
        div_lis = [len(set(completions_text[i:i+self.num_generations]))/self.num_generations for i in range(0, len(completions_text), self.num_generations)]
        cate_diversity = sum(div_lis)/len(div_lis) if len(div_lis) > 0 else 0.0
        self._metrics["categorical_diversity"].append(cate_diversity)
        
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_weight) in enumerate(zip(self.reward_funcs, self.reward_weights)):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion", "target_sid"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].append(example[key])
            
            # Pass target_sid explicitly to reward function
            reward_kwargs["target_sid"] = targets 
            
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device) * reward_weight

        rewards = rewards_per_func.sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = gather(rewards).tolist()

            if self.accelerator.is_main_process:
                if is_wandb_available():
                    import wandb
                    if wandb.run is not None:
                        table = wandb.Table(columns=["Prompt", "Completion", "Reward"])
                        for row in zip(prompts_to_log, completions_to_log, rewards_to_log):
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
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
        ref_per_token_logps = inputs["ref_per_token_logps"]

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        completion_mask = inputs["completion_mask"]

        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        self._metrics["reward"].append(inputs["advantages"].mean().item()) # log original rewards if needed by passing, but standard GRPO logs advantage mean which is 0
        self._metrics["reward_std"].append(inputs["advantages"].std().item())
        self._metrics["kl"].append((per_token_kl * completion_mask).sum(dim=1).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        pass # Optional to implement based on your needs