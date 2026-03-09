import json
import random
import logging
import pandas as pd
from datasets import Dataset

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Factory for building unified HF Datasets for SFT, GRPO, and EEPO from various sources.
    Supports OpenOneRec-RecIF Parquet format and Legacy Amazon CSV format.
    """
    
    @staticmethod
    def build_from_parquet(parquet_path: str, sample_num: int = -1, seed: int = 42) -> Dataset:
        """
        Parses OpenOneRec-RecIF `messages` style DataFrames.
        """
        random.seed(seed)
        logger.info(f"Loading parquet dataset from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        if sample_num > 0 and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)
            
        dataset_dict = {
            "prompt": [],
            "completion": [],
            "target_sid": []
        }
        
        for _, row in df.iterrows():
            msgs = row['messages']
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
                
            prompt_str = ""
            for m in msgs:
                if m['role'] == 'system':
                    prompt_str += f"{m['content'][0]['text']}\n"
                elif m['role'] == 'user':
                    prompt_str += f"### User Input:\n{m['content'][0]['text']}\n### Response:\n"
                    
            meta = row['metadata']
            if isinstance(meta, str):
                meta = json.loads(meta)
                
            target_sid = meta.get('answer', '')
            if not target_sid:
                continue
                
            dataset_dict["prompt"].append(prompt_str)
            dataset_dict["completion"].append(target_sid)
            dataset_dict["target_sid"].append(target_sid)
            
        hf_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Dataset built successfully. Total samples: {len(hf_dataset)}")
        return hf_dataset

    @staticmethod
    def build_from_csv(csv_path: str, sample_num: int = -1, seed: int = 42) -> Dataset:
        """
        Legacy parser for older Amazon CSV format.
        """
        random.seed(seed)
        logger.info(f"Loading CSV dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if sample_num > 0 and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)
            
        dataset_dict = {
            "prompt": [],
            "completion": [],
            "target_sid": []
        }
        
        for _, row in df.iterrows():
            try:
                history_sid = eval(row['history_item_sid'])
            except:
                continue
                
            history_str = ", ".join(history_sid)
            target_sid = str(row['item_sid'])
            
            prompt_str = (
                f"### User Input:\nThe user has interacted with items {history_str} "
                f"in chronological order. Can you predict the next possible item that the user may expect?\n\n"
                f"### Response:\n"
            )
            
            dataset_dict["prompt"].append(prompt_str)
            dataset_dict["completion"].append(target_sid)
            dataset_dict["target_sid"].append(target_sid)
            
        hf_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Dataset built successfully. Total samples: {len(hf_dataset)}")
        return hf_dataset