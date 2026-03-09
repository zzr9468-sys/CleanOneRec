import os
import torch
import logging
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

# Local module imports
from dataset import build_grpo_dataset
from reward import SemanticRewardEngine
from minionerec_trainer import ReReTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(
    # --- Paths ---
    model_path: str,
    train_file: str,
    recif_path: str = "/Users/zhouziren/onerec/OpenOneRec-RecIF",
    output_dir: str = "./outputs/clean_grpo",
    
    # --- Data & Metrics ---
    sample_train: int = 1000,
    wandb_project: str = "clean_onerec_grpo",
    wandb_run_name: str = "run_1",
    
    # --- EEPO (Fast-Weight) Configurations ---
    eepo_enabled: bool = True,
    eepo_stage1_ratio: float = 0.5,
    eepo_unlearn_lr: float = 1e-5,
    eepo_unlearn_weight: float = 1.0,
    
    # --- GRPO Bootstrapping & Generation ---
    add_gt: bool = True,           # Recommended: Force 1 Ground Truth in generation batch to avoid early reward collapse
    temperature: float = 0.7,      # Recommended: <1.0 for RL early phase to reduce hallucinated SIDs
    num_generations: int = 16,     # G in GRPO paper
    
    # --- Standard Training Hyperparams ---
    train_batch_size: int = 4,     # Keep this small to avoid OOM
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 1e-6,
    beta: float = 0.04,            # KL Divergence penalty
):
    # Setup WandB
    os.environ['WANDB_PROJECT'] = wandb_project
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Initializing Semantic Reward Engine...")
    reward_engine = SemanticRewardEngine(recif_path=recif_path, device=device)
    
    logger.info(f"Loading base model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    logger.info(f"Preparing dataset from {train_file}...")
    train_dataset = build_grpo_dataset(train_file, sample_num=sample_train)
    
    # Configure Trainer
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
    
    logger.info("Initializing Custom ReReTrainer (EEPO)...")
    trainer = ReReTrainer(
        model=model,
        base_model=model_path,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_engine.compute_reward],
        
        # EEPO Injected Params
        eepo_enabled=eepo_enabled,
        eepo_stage1_ratio=eepo_stage1_ratio,
        eepo_unlearn_lr=eepo_unlearn_lr,
        eepo_unlearn_weight=eepo_unlearn_weight,
        
        # Bootstrapping flag
        add_gt=add_gt,
    )
    
    logger.info("Starting GRPO Training...")
    trainer.train()
    
    # Save Model
    final_output = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info(f"Training successfully completed. Model saved to {final_output}")

if __name__ == "__main__":
    Fire(train)
