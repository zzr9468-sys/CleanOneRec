#!/usr/bin/env python3
"""
Minimal example of GRPO training with RecRL.

This example demonstrates the core API without requiring full dataset setup.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from recrl.core import RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import BaseReward


class DummyReward(BaseReward):
    """Simple reward that returns random values for demonstration."""

    def __call__(self, prompts, completions, **kwargs):
        return [torch.rand(1).item() for _ in completions]


def main():
    # 1. Load model and tokenizer
    print("Loading model...")
    model_path = "gpt2"  # Use small model for demo
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Create dummy dataset
    print("Creating dataset...")
    train_dataset = Dataset.from_dict({
        "prompt": [
            "Recommend a movie:",
            "Suggest a book:",
            "What video should I watch?",
        ] * 10  # 30 samples
    })

    # 3. Setup engines
    print("Setting up engines...")
    rollout_engine = RolloutEngine(tokenizer, device="cpu")
    reward_engine = DummyReward()

    # 4. Configure training
    config = GRPOConfig(
        num_epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        num_generations=2,
        max_completion_length=32,
        temperature=0.7,
        beta=0.04,
        logging_steps=1,
        save_steps=10,
        output_dir="./outputs/minimal_example",
        device="cpu",
    )

    # 5. Create trainer
    print("Creating trainer...")
    trainer = GRPOTrainer(
        model=model,
        ref_model=None,  # Will use model as reference
        config=config,
        train_dataset=train_dataset,
        rollout_engine=rollout_engine,
        reward_engine=reward_engine,
        tokenizer=tokenizer,
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    print("\nTraining complete!")
    print(f"Model saved to {config.output_dir}/final")


if __name__ == "__main__":
    main()
