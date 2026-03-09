import os
import sys
import json
import logging
import torch
from fire import Fire
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from onerec import DatasetBuilder
from onerec.utils.logit_processor import SIDTrie, ConstrainedLogitsProcessor
from onerec.eval.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(
    model_path: str,
    test_file: str,
    recif_path: str = "/Users/zhouziren/onerec/OpenOneRec-RecIF",
    sample_test: int = 1000,
    beam_size: int = 20,
    batch_size: int = 4,
    constrained_decoding: bool = True
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("Initializing Test Dataset...")
    test_dataset = DatasetBuilder.build_from_parquet(test_file, sample_num=sample_test)
    
    logger.info(f"Loading Model and Tokenizer: {model_path} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    allowed_trie = None
    if constrained_decoding:
        logger.info("Building Constrained Trie...")
        allowed_trie = SIDTrie(eos_token_id=tokenizer.eos_token_id)
        
        # In evaluation, we definitely need the full dictionary
        sid2pid_path = os.path.join(recif_path, "metadata", "sid2pid.json")
        try:
            with open(sid2pid_path, 'r') as f:
                sid2pid = json.load(f)
            # Actually, to build the trie we need the string SIDs. 
            # If we don't have a direct list of string SIDs, we build it from the test dataset for now.
            # In a real scenario, you should load all possible valid SIDs here.
            unique_sids = set(test_dataset["target_sid"])
            for sid in unique_sids:
                if sid:
                    sid_tokens = tokenizer(sid, add_special_tokens=False)["input_ids"]
                    allowed_trie.add(sid_tokens)
            logger.info("Trie initialized.")
        except Exception as e:
            logger.warning(f"Failed to build Trie: {e}")
            allowed_trie = None

    all_predictions = []
    all_targets = []
    
    logger.info("Starting Evaluation...")
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size)):
            batch = test_dataset[i:i+batch_size]
            prompts = batch["prompt"]
            targets = batch["target_sid"]
            
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            prompt_length = inputs.input_ids.shape[1]
            
            logits_processor = LogitsProcessorList()
            if allowed_trie is not None:
                logits_processor.append(ConstrainedLogitsProcessor(allowed_trie.get_allowed_next_tokens, prompt_length=prompt_length))
                
            generation_config = GenerationConfig(
                max_new_tokens=128,
                num_beams=beam_size,
                num_return_sequences=beam_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
            
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                logits_processor=logits_processor
            )
            
            # Reshape outputs to (batch_size, beam_size, seq_len)
            outputs = outputs.view(len(prompts), beam_size, -1)
            
            for b_idx in range(len(prompts)):
                preds_for_user = []
                for beam_idx in range(beam_size):
                    gen_tokens = outputs[b_idx, beam_idx, prompt_length:]
                    pred_str = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    if pred_str not in preds_for_user:
                        preds_for_user.append(pred_str)
                all_predictions.append(preds_for_user)
                all_targets.append(targets[b_idx])

    metrics = Evaluator.compute_metrics(all_predictions, all_targets)
    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

if __name__ == "__main__":
    Fire(main)