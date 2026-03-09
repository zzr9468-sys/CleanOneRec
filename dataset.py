import json
import random
import pandas as pd
from datasets import Dataset

def build_grpo_dataset(parquet_path, sample_num=-1, seed=42):
    """
    Builds a clean, single-purpose dataset for GRPO training.
    Directly parses the `messages` column from the parquet file to match SFT formatting.
    """
    random.seed(seed)
    df = pd.read_parquet(parquet_path)
    
    if sample_num > 0 and sample_num < len(df):
        df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)
        
    dataset_dict = {
        "prompt": [],
        "completion": [],
        "target_sid": []
    }
    
    for _, row in df.iterrows():
        # Parse messages
        msgs = row['messages']
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
            
        # Reconstruct Prompt
        prompt_str = ""
        for m in msgs:
            if m['role'] == 'system':
                prompt_str += f"{m['content'][0]['text']}\n"
            elif m['role'] == 'user':
                prompt_str += f"### User Input:\n{m['content'][0]['text']}\n### Response:\n"
                
        # Parse Target (Ground Truth)
        meta = row['metadata']
        if isinstance(meta, str):
            meta = json.loads(meta)
            
        target_sid = meta.get('answer', '')
        if not target_sid:
            continue
            
        dataset_dict["prompt"].append(prompt_str)
        dataset_dict["completion"].append(target_sid)
        # Passing target_sid separately so it can be extracted by ReReTrainer kwargs
        dataset_dict["target_sid"].append(target_sid) 
        
    hf_dataset = Dataset.from_dict(dataset_dict)
    return hf_dataset
