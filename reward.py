import os
import json
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class SemanticRewardEngine:
    """
    A standalone module handling the Text Semantic Reward logic.
    Loads necessary dicts and models only once.
    """
    def __init__(self, recif_path="/Users/zhouziren/onerec/OpenOneRec-RecIF", device="cuda"):
        self.device = device
        self.sid2pid_dict = {}
        self.pid2caption_dict = {}
        self.text_emb_model = None
        self.text_emb_tokenizer = None
        
        self._load_resources(recif_path)

    def _load_resources(self, recif_path):
        logger.info(f"Loading dictionaries from {recif_path}...")
        
        sid2pid_path = os.path.join(recif_path, "benchmark_data/sid2pid.json")
        pid2cap_path = os.path.join(recif_path, "pid2caption.parquet")
        
        with open(sid2pid_path, 'r') as f:
            self.sid2pid_dict = json.load(f)
            
        df_cap = pd.read_parquet(pid2cap_path)
        self.pid2caption_dict = dict(zip(df_cap['pid'].astype(str), df_cap['caption']))
        
        logger.info("Loading sentence-transformer (all-MiniLM-L6-v2) for Semantic Reward...")
        self.text_emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.text_emb_model.eval()

    def _parse_sid_to_int(self, sid_str):
        """
        Parses '<s_a_123><s_b_456><s_c_789>' into the hashed string key used in sid2pid.json.
        Formula: s_a * 8192^2 + s_b * 8192 + s_c
        """
        matches = re.findall(r'<s_[abc]_(\d+)>', sid_str)
        if len(matches) == 3:
            a, b, c = int(matches[0]), int(matches[1]), int(matches[2])
            return str(a * 8192 * 8192 + b * 8192 + c)
        return None

    def compute_reward(self, prompts, completions, **kwargs):
        """
        The main reward function to be passed to GRPOTrainer.
        kwargs must contain 'target_sid' list of the same length as prompts/completions.
        """
        targets = kwargs.get("target_sid", [])
        if not targets:
            logger.error("No target_sid found in kwargs! Did you configure the dataset correctly?")
            return [-10.0] * len(completions)

        validity_penalty = -10.0
        gen_texts = []
        target_texts = []
        valid_mask = []
        
        for i, completion in enumerate(completions):
            sid = completion.strip(" \n\"'")
            target_sid = targets[i].strip(" \n\"'")
            
            is_format_valid = True
            if not sid or sid.count("<") < 3:
                is_format_valid = False
                
            parsed_sid_key = self._parse_sid_to_int(sid) if is_format_valid else None
            parsed_target_key = self._parse_sid_to_int(target_sid)
            
            if not parsed_sid_key:
                gen_texts.append("")
                target_texts.append("")
                valid_mask.append(0.0)
                continue
                
            sid_mapping = self.sid2pid_dict.get(parsed_sid_key)
            target_mapping = self.sid2pid_dict.get(parsed_target_key)
            
            pid = sid_mapping[0]['pid'] if sid_mapping else None
            target_pid = target_mapping[0]['pid'] if target_mapping else None
            
            if not pid or str(pid) not in self.pid2caption_dict or not target_pid or str(target_pid) not in self.pid2caption_dict:
                gen_texts.append("")
                target_texts.append("")
                valid_mask.append(0.0)
            else:
                gen_texts.append(self.pid2caption_dict[str(pid)])
                target_texts.append(self.pid2caption_dict[str(target_pid)])
                valid_mask.append(1.0)
                
        if not any(valid_mask):
            return [validity_penalty] * len(completions)

        # Batch compute similarity
        inputs_gen = self.text_emb_tokenizer(gen_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        inputs_target = self.text_emb_tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            emb_gen = self.text_emb_model(**inputs_gen).last_hidden_state.mean(dim=1)
            emb_target = self.text_emb_model(**inputs_target).last_hidden_state.mean(dim=1)
            
        sims = torch.cosine_similarity(emb_gen, emb_target, dim=-1).tolist()
        rewards = [sim if valid == 1.0 else validity_penalty for sim, valid in zip(sims, valid_mask)]
        
        return rewards
