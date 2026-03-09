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
    model_path = "path/to/your/model"
    train_file = "path/to/train.parquet"
    recif_path = "path/to/OpenOneRec-RecIF"
    output_dir = "./outputs/grpo"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Create reference model (frozen copy)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()

    # Load dataset
    logger.info(f"Loading dataset from {train_file}")
    train_dataset = DataEngine.from_parquet(
        path=train_file,
        format="recif",
        sample_num=1000  # Use -1 for full dataset
    )

    # Setup rollout engine
    rollout_engine = RolloutEngine(
        tokenizer=tokenizer,
        constraints=None,  # Optional: add SIDTrie for constrained decoding
        device=device
    )

    # Setup reward function
    reward_engine = TextSemanticReward(
        recif_path=recif_path,
        device=device
    )

    # Create sampler
    config = GRPOConfig(
        num_epochs=1,
        per_device_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        num_generations=16,
        temperature=0.7,
        max_completion_length=128,
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
