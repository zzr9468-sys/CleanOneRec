"""
GRPO Training Example

Clean example showing how to train with GRPO algorithm.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import TextSemanticReward
from recrl.data import RepeatRandomSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    model_path = "gpt2" # Using dummy model for test
    train_file = "dummy_train" 
    recif_path = "dummy_recif"
    output_dir = "./outputs/grpo_test_mac"

    device = "cpu" # Forced to CPU for quick test

    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Create reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(model_path)
    ref_model.eval()

    # Create a tiny dummy dataset
    logger.info(f"Creating dummy dataset")
    from datasets import Dataset
    train_dataset = Dataset.from_dict({
        "prompt": ["### User Input:\nRecommend item.\n### Response:\n", "### User Input:\nNext item?\n### Response:\n"],
        "target_sid": ["<s_a_1><s_b_2><s_c_3>", "<s_a_4><s_b_5><s_c_6>"]
    })

    # Setup rollout engine
    rollout_engine = RolloutEngine(
        tokenizer=tokenizer,
        constraints=None,  # Optional: add SIDTrie for constrained decoding
        device=device
    )

    # Setup dummy reward function (ExactMatch to avoid loading sentence-transformers/files)
    from recrl.rewards import ExactMatchReward
    reward_engine = ExactMatchReward()

    # Create sampler
    config = GRPOConfig(
        num_epochs=1,
        per_device_batch_size=2, # Must divide train_dataset len * num_generations
        gradient_accumulation_steps=1,
        learning_rate=1e-6,
        num_generations=2, # Small generation for quick test
        temperature=0.7,
        max_completion_length=10,
        beta=0.04,
        output_dir=output_dir,
        device=device
    )

    sampler = RepeatRandomSampler(
        data_source=train_dataset,
        repeat_count=config.num_generations,
        seed=42
    )

    # Create trainer
    logger.info("Initializing GRPO Trainer")
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        train_dataset=train_dataset,
        rollout_engine=rollout_engine,
        reward_engine=reward_engine,
        tokenizer=tokenizer,
        sampler=sampler
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Training completed! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
