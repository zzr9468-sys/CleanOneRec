import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import logging
from ..utils.sid_helper import SIDHelper
from .rule import BaseReward

logger = logging.getLogger(__name__)

class TextSemanticReward(BaseReward):
    """
    Evaluates the content similarity between the generated SID and Target SID
    using their text captions (from pid2caption.parquet) via Sentence-Transformers.
    """
    def __init__(self, recif_path: str, device: str = "cuda", validity_penalty: float = -10.0):
        super().__init__("text_semantic_reward")
        self.device = device
        self.validity_penalty = validity_penalty
        self.sid_helper = SIDHelper()
        
        self.sid2pid_dict = {}
        self.pid2caption_dict = {}
        self.text_emb_model = None
        self.text_emb_tokenizer = None
        
        self._init_resources(recif_path)

    def _init_resources(self, recif_path: str):
        logger.info(f"[{self.name}] Loading mapping dictionaries from {recif_path}...")
        
        sid2pid_path = os.path.join(recif_path, "benchmark_data/sid2pid.json")
        pid2cap_path = os.path.join(recif_path, "pid2caption.parquet")
        
        import json
        with open(sid2pid_path, 'r') as f:
            self.sid2pid_dict = json.load(f)
            
        df_cap = pd.read_parquet(pid2cap_path)
        self.pid2caption_dict = dict(zip(df_cap['pid'].astype(str), df_cap['caption']))
        
        logger.info(f"[{self.name}] Loading MiniLM for semantic similarity...")
        self.text_emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.text_emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
        self.text_emb_model.eval()

    def __call__(self, prompts, completions, **kwargs):
        targets = kwargs.get("target_sid", [])
        if not targets:
            logger.error("No target_sid found in kwargs!")
            return [self.validity_penalty] * len(completions)

        gen_texts = []
        target_texts = []
        valid_mask = []
        
        for i, completion in enumerate(completions):
            gen_sid = completion.strip(" \n\"'")
            target_sid = targets[i].strip(" \n\"'")
            
            parsed_gen_key = self.sid_helper.sid_to_hash_key(gen_sid)
            parsed_target_key = self.sid_helper.sid_to_hash_key(target_sid)
            
            if not parsed_gen_key:
                gen_texts.append("")
                target_texts.append("")
                valid_mask.append(0.0)
                continue
                
            gen_mapping = self.sid2pid_dict.get(parsed_gen_key)
            target_mapping = self.sid2pid_dict.get(parsed_target_key)
            
            gen_pid = gen_mapping[0]['pid'] if gen_mapping else None
            target_pid = target_mapping[0]['pid'] if target_mapping else None
            
            if not gen_pid or str(gen_pid) not in self.pid2caption_dict or not target_pid or str(target_pid) not in self.pid2caption_dict:
                gen_texts.append("")
                target_texts.append("")
                valid_mask.append(0.0)
            else:
                gen_texts.append(self.pid2caption_dict[str(gen_pid)])
                target_texts.append(self.pid2caption_dict[str(target_pid)])
                valid_mask.append(1.0)
                
        # If absolutely nothing is valid in this batch, return early
        if not any(valid_mask):
            return [self.validity_penalty] * len(completions)

        # Compute batched embeddings
        inputs_gen = self.text_emb_tokenizer(gen_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        inputs_target = self.text_emb_tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            emb_gen = self.text_emb_model(**inputs_gen).last_hidden_state.mean(dim=1)
            emb_target = self.text_emb_model(**inputs_target).last_hidden_state.mean(dim=1)
            
        sims = torch.cosine_similarity(emb_gen, emb_target, dim=-1).tolist()
        
        # Apply mask
        rewards = [sim if valid == 1.0 else self.validity_penalty for sim, valid in zip(sims, valid_mask)]
        return rewards