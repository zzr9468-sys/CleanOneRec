import os
import sys
import logging
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

# Ensure onerec_rl is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec import (
    DatasetBuilder,
    EEPOTrainer,
    CompositeReward
)
from onerec.utils.logit_processor import SIDTrie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    # Model & Data
    model_path: str,
    train_file: str,
    recif_path: str = "/Users/zhouziren/onerec/OpenOneRec-RecIF",
    output_dir: str = "./outputs/onerec_rl_run",
    
    # Wandb Logging
    wandb_project: str = "OneRecRL_Experiments",
    wandb_run_name: str = "eepo_semantic_run_01",
    
    # Dataset Limits
    sample_train: int = 2000,
    
    # EEPO Conf
    eepo_enabled: bool = True,
    eepo_stage1_ratio: float = 0.5,
    add_gt: bool = True,
    
    # Constrained Decoding
    constrained_decoding: bool = True,
    
    # Reward Conf
    validity_penalty: float = -10.0,
    tail_bonus: float = 0.1,
    tail_freq_threshold: int = 50,
    
    # Hyperparams
    train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-6,
    num_generations: int = 16,
    temperature: float = 0.7,
    beta: float = 0.04
):
    os.environ['WANDB_PROJECT'] = wandb_project
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Initializing Dataset...")
    train_dataset = DatasetBuilder.build_from_parquet(train_file, sample_num=sample_train)
    
    logger.info("Initializing Composite Semantic Reward Engine...")
    reward_engine = CompositeReward(
        recif_path=recif_path,
        device=device,
        validity_penalty=validity_penalty,
        tail_bonus=tail_bonus,
        tail_freq_threshold=tail_freq_threshold
    )
    
    # You could optionally load freq dict here if needed:
    # reward_engine.load_freq_dict_from_csv(some_csv_path)

    logger.info(f"Loading Model: {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    allowed_trie = None
    if constrained_decoding:
        logger.info("Building Constrained Trie from Dataset Target SIDs...")
        allowed_trie = SIDTrie(eos_token_id=tokenizer.eos_token_id)
        # Assuming training dataset contains valid target_sids, we build Trie from them
        unique_sids = set(train_dataset["target_sid"])
        for sid in unique_sids:
            if sid:
                sid_tokens = tokenizer(sid, add_special_tokens=False)["input_ids"]
                allowed_trie.add(sid_tokens)
        logger.info(f"Trie initialized with {len(unique_sids)} valid SIDs.")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        temperature=temperature,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_completion_length=128,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",
        run_name=wandb_run_name,
        bf16=(device == "cuda"),
        optim="paged_adamw_32bit"
    )

    trainer = EEPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_engine],
        processing_class=tokenizer,
        
        # EEPO Params
        eepo_enabled=eepo_enabled,
        eepo_stage1_ratio=eepo_stage1_ratio,
        eepo_unlearn_lr=1e-5,
        eepo_unlearn_weight=1.0,
        add_gt=add_gt,
        allowed_trie=allowed_trie
    )

    logger.info("Starting EEPO/GRPO Training Loop...")
    trainer.train()

    final_output = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info(f"Training Done. Checkpoint saved to {final_output}")

if __name__ == "__main__":
    Fire(main)