"""
Text Semantic Reward

Computes semantic similarity between generated and target SIDs
using their text captions via sentence transformers.
"""

import os
import json
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from ..core.reward import BaseReward
from ..utils.sid_helper import SIDHelper

logger = logging.getLogger(__name__)


class TextSemanticReward(BaseReward):
    """
    Semantic similarity reward using sentence transformers.

    Compares text captions of generated vs target SIDs.
    """

    def __init__(
        self,
        recif_path: str,
        device: str = "cuda",
        validity_penalty: float = -10.0,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Args:
            recif_path: Path to OpenOneRec-RecIF directory
            device: Device for computation
            validity_penalty: Penalty for invalid SIDs
            model_name: Sentence transformer model name
        """
        super().__init__(name="text_semantic_reward")
        self.device = device
        self.validity_penalty = validity_penalty
        self.sid_helper = SIDHelper()

        # Load mappings
        self.sid2pid_dict = {}
        self.pid2caption_dict = {}

        logger.info(f"[{self.name}] Loading mappings from {recif_path}")
        sid2pid_path = os.path.join(recif_path, "benchmark_data/sid2pid.json")
        pid2cap_path = os.path.join(recif_path, "pid2caption.parquet")

        with open(sid2pid_path, 'r') as f:
            self.sid2pid_dict = json.load(f)

        df_cap = pd.read_parquet(pid2cap_path)
        self.pid2caption_dict = dict(zip(df_cap['pid'].astype(str), df_cap['caption']))

        # Load sentence transformer
        logger.info(f"[{self.name}] Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Compute semantic similarity rewards."""
        targets = kwargs.get("target_sid", [])
        if not targets:
            logger.error(f"[{self.name}] No target_sid in kwargs!")
            return [self.validity_penalty] * len(completions)

        gen_texts = []
        target_texts = []
        valid_mask = []

        for i, completion in enumerate(completions):
            gen_sid = completion.strip(" \n\"'")
            target_sid = targets[i].strip(" \n\"'")

            # Parse SIDs
            gen_key = self.sid_helper.sid_to_hash_key(gen_sid)
            target_key = self.sid_helper.sid_to_hash_key(target_sid)

            if not gen_key:
                gen_texts.append("")
                target_texts.append("")
                valid_mask.append(0.0)
                continue

            # Get PIDs
            gen_mapping = self.sid2pid_dict.get(gen_key)
            target_mapping = self.sid2pid_dict.get(target_key)

            gen_pid = gen_mapping[0]['pid'] if gen_mapping else None
            target_pid = target_mapping[0]['pid'] if target_mapping else None

            # Get captions
            if (gen_pid and str(gen_pid) in self.pid2caption_dict and
                target_pid and str(target_pid) in self.pid2caption_dict):
                gen_texts.append(self.pid2caption_dict[str(gen_pid)])
                target_texts.append(self.pid2caption_dict[str(target_pid)])
                valid_mask.append(1.0)
            else:
                gen_texts.append("")
                target_texts.append("")
                valid_mask.append(0.0)

        # Early exit if nothing valid
        if not any(valid_mask):
            return [self.validity_penalty] * len(completions)

        # Compute embeddings
        inputs_gen = self.tokenizer(
            gen_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        inputs_target = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            emb_gen = self.model(**inputs_gen).last_hidden_state.mean(dim=1)
            emb_target = self.model(**inputs_target).last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        sims = torch.cosine_similarity(emb_gen, emb_target, dim=-1).tolist()

        # Apply validity mask
        rewards = [
            sim if valid == 1.0 else self.validity_penalty
            for sim, valid in zip(sims, valid_mask)
        ]

        return rewards
