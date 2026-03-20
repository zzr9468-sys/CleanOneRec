"""
EEPO Training Example

Example showing how to train with EEPO (Explore-and-Evaluate Policy Optimization).

NOTE: RepeatRandomSampler is intentionally NOT used here.
The EEPO trainer calls rollout_engine.generate(num_return_sequences=G) internally,
so each unique prompt in the batch already produces G completions.
Using RepeatRandomSampler(repeat_count=G) would give G copies of the same prompt
in a single batch, wasting compute and causing a reward-shape mismatch.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from recrl.core import DataEngine, RolloutEngine
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
from recrl.rewards import RecRLCompositeReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    model_path = "path/to/your/model"
    train_file = "path/to/train.parquet"
    recif_path = "path/to/OpenOneRec-RecIF"
    output_dir = "./outputs/eepo"

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

    # Setup reward function (multiplicative composite: lv * nov * 0.95 + div * 0.05)
    reward_engine = RecRLCompositeReward(
        recif_path=recif_path,
        device=device,
        num_generations=16,
    )

    # Create EEPO config
    config = EEPOConfig(
        # Standard RL params
        num_epochs=1,
        per_device_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        num_generations=16,
        temperature=0.7,
        max_completion_length=128,
        beta=0.04,

        # EEPO core
        eepo_enabled=True,
        eepo_unlearn_lr=1e-5,
        eepo_unlearn_weight=1.0,
        eepo_epsilon=1e-4,
        add_gt=True,

        # Direction 1: also mutate last-block v_proj (more aggressive exploration)
        eepo_expand_scope=False,

        # Direction 2: start exploitation-heavy, gradually shift to exploration
        eepo_stage1_ratio=0.7,
        eepo_stage1_ratio_end=0.3,
        eepo_stage1_anneal_steps=500,

        # Direction 3: directed unlearn top-500 head items, remember bottom-1000 tail items
        eepo_recif_path=recif_path,
        eepo_head_k=500,
        eepo_remember_tail=True,
        eepo_remember_weight=0.3,
        eepo_tail_k=1000,

        # Direction 4: separate G1/G2 advantage normalisation + exploration bonus
        eepo_separate_adv=True,
        eepo_exploration_bonus=0.1,

        output_dir=output_dir,
        device=device
    )

    # No RepeatRandomSampler: EEPO generates num_generations completions per prompt
    # internally, so each batch element must be a distinct prompt.

    # Create trainer
    logger.info("Initializing EEPO Trainer")
    trainer = EEPOTrainer(
        model=model,
        ref_model=ref_model,
        config=config,
        train_dataset=train_dataset,
        rollout_engine=rollout_engine,
        reward_engine=reward_engine,
        tokenizer=tokenizer,
        sampler=None,   # standard random sampling
    )

    # Train
    logger.info("Starting EEPO training...")
    trainer.train()

    logger.info(f"Training completed! Model saved to {output_dir}")


if __name__ == "__main__":
    main()
