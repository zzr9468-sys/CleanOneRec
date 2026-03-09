import os
import sys
import logging
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]}{example['completion'][i]}"
        output_texts.append(text)
    return output_texts

def main(
    # --- Paths ---
    model_path: str,
    train_file: str,
    output_dir: str = "./outputs/onerec_sft_run",
    
    # --- Data & Metrics ---
    sample_train: int = 50000,
    wandb_project: str = "OneRec_SFT_Experiments",
    wandb_run_name: str = "sft_run_01",
    
    # --- LoRA Parameters ---
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    
    # --- Standard Training Hyperparams ---
    train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    max_seq_length: int = 512,
):
    os.environ['WANDB_PROJECT'] = wandb_project
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Initializing SFT Dataset...")
    train_dataset = DatasetBuilder.build_from_parquet(train_file, sample_num=sample_train)
    
    logger.info(f"Loading Model and Tokenizer: {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if use_lora:
        logger.info("Injecting LoRA adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"] # Customize based on your base model (e.g. LLaMA/Qwen)
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        peft_config = None

    response_template = "### Response:\n"
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name=wandb_run_name,
        bf16=(device == "cuda"),
        optim="paged_adamw_32bit"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=max_seq_length,
        peft_config=peft_config if use_lora else None,
    )

    logger.info("Starting SFT Training...")
    trainer.train()

    final_output = os.path.join(output_dir, "final_checkpoint")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info(f"SFT Training Done. Checkpoint saved to {final_output}")

if __name__ == "__main__":
    Fire(main)