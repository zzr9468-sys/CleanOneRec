"""
Composite Reward Example

Shows how to combine multiple reward functions.
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from recrl.core import DataEngine, RolloutEngine, CompositeReward
from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.rewards import TextSemanticReward, ExactMatchReward
from recrl.data import RepeatRandomSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    model_path = "path/to/your/model"
    train_file = "path/to/train.parquet"
    recif_path = "path/to/OpenOneRec-RecIF"
    output_dir = "./outputs/composite_reward"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model.eval()

    # Load dataset
    train_dataset = DataEngine.from_parquet(
        path=train_file,
        format="recif",
        sample_num=1000
    )

    # Setup rollout engine
    rollout_engine = RolloutEngine(
        tokenizer=tokenizer,
        device=device
    )

    # Compose multiple rewards
    semantic_reward = TextSemanticReward(recif_path=recif_path, device=device)
    exact_match_reward = ExactMatchReward()

    # Weighted combination: 80% semantic + 20% exact match
    reward_engine = CompositeReward([
        (semantic_reward, 0.8),
        (exact_match_reward, 0.2)
    ])

    # Config and sampler
    config = GRPOConfig(
        num_epochs=1,
        per_device_batch_size=4,
        num_generations=16,
        temperature=0.7,
        learning_rate=1e-6,
        output_dir=output_dir,
        device=device
    )

    sampler = RepeatRandomSampler(
        data_source=train_dataset,
        repeat_count=config.num_generations,
        seed=42
    )

    # Train
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

    logger.info("Training with composite reward (semantic + exact match)")
    trainer.train()


if __name__ == "__main__":
    main()
