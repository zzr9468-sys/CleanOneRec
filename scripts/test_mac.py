import os
import sys
import torch
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig
from datasets import Dataset

# Make sure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec import (
    EEPOTrainer,
    RuleReward
)
from onerec.utils.logit_processor import SIDTrie

logging.basicConfig(level=logging.INFO)

def test_trainer_mac():
    os.environ["WANDB_MODE"] = "disabled"
    
    # Use dummy model
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create dummy dataset
    data = {
        "prompt": [
            "### User Input:\nRecommend me something.\n### Response:\n",
            "### User Input:\nWhat's next?\n### Response:\n"
        ],
        "target_sid": [
            "<s_a_1><s_b_2><s_c_3>",
            "<s_a_4><s_b_5><s_c_6>"
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Build a tiny Trie
    trie = SIDTrie(eos_token_id=tokenizer.eos_token_id)
    sid_tokens = tokenizer("<s_a_1><s_b_2><s_c_3>", add_special_tokens=False)["input_ids"]
    trie.add(sid_tokens)
    sid_tokens = tokenizer("<s_a_4><s_b_5><s_c_6>", add_special_tokens=False)["input_ids"]
    trie.add(sid_tokens)
    
    # Dummy Reward
    reward_fn = RuleReward()

    # Training args
    training_args = GRPOConfig(
        output_dir="./test_output",
        num_generations=2,
        temperature=0.7,
        per_device_train_batch_size=2, # Must be divisible by num_generations
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        beta=0.04,
        max_completion_length=10,
        logging_steps=1,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = EEPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        
        # EEPO Params
        eepo_enabled=True,
        eepo_stage1_ratio=0.5,
        eepo_unlearn_lr=1e-5,
        eepo_unlearn_weight=1.0,
        add_gt=True,
        allowed_trie=trie
    )
    
    logging.info("Starting dummy training loop...")
    trainer.train()
    logging.info("Test passed successfully!")

if __name__ == "__main__":
    test_trainer_mac()
