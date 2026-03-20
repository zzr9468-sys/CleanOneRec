"""
Text Semantic Reward

Computes semantic similarity between generated and target SIDs
using their text captions via sentence transformers.
"""

import re
import os
import json
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from ..core.reward import BaseReward
from ..utils.sid_helper import SIDHelper

logger = logging.getLogger(__name__)

SID_PATTERN = re.compile(r'<\|sid_begin\|>.*?<\|sid_end\|>')


class TextSemanticReward(BaseReward):
    """
    Semantic similarity reward using sentence transformers.

    Compares text captions of generated vs target SIDs.
    Batches all encoding in a single forward pass for efficiency.
    """

    def __init__(
        self,
        recif_path: str,
        device: str = "cuda",
        validity_penalty: float = -1.0,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Args:
            recif_path: Path to OpenOneRec-RecIF directory
            device: Device for computation
            validity_penalty: Penalty for invalid SIDs (changed from -10 to -1 to avoid dominating composite reward)
            model_name: Sentence transformer model name
        """
        super().__init__(name="text_semantic_reward")
        self.device = device
        self.validity_penalty = validity_penalty
        self.sid_helper = SIDHelper()

        # Load mappings
        logger.info(f"[{self.name}] Loading mappings from {recif_path}")
        sid2pid_path = os.path.join(recif_path, "benchmark_data/sid2pid.json")
        pid2cap_path = os.path.join(recif_path, "pid2caption.parquet")

        with open(sid2pid_path, 'r') as f:
            self.sid2pid_dict = json.load(f)

        df_cap = pd.read_parquet(pid2cap_path)
        # Column is 'dense_caption', not 'caption'
        self.pid2caption_dict = dict(zip(df_cap['pid'].astype(str), df_cap['dense_caption']))

        # Load sentence transformer
        logger.info(f"[{self.name}] Loading {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def _mean_pool(self, model_output, attention_mask):
        """Mean pooling over token embeddings."""
        token_embs = model_output.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
        return torch.sum(token_embs * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

    def _get_caption(self, completion: str) -> str | None:
        """Extract first SID from completion and return its caption."""
        sids = SID_PATTERN.findall(completion)
        if not sids:
            return None
        sid = sids[0]
        key = self.sid_helper.sid_to_hash_key(sid)
        if not key:
            return None
        mapping = self.sid2pid_dict.get(key)
        if not mapping:
            return None
        pid = str(mapping[0]['pid'])
        return self.pid2caption_dict.get(pid)

    def __call__(self, prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """Compute semantic similarity rewards (batched encoding)."""
        targets = kwargs.get("target_sid", [])
        if not targets:
            logger.warning(f"[{self.name}] No target_sid provided, returning penalty")
            return [self.validity_penalty] * len(completions)

        gen_captions = []
        target_captions = []
        valid_mask = []

        for i, completion in enumerate(completions):
            gen_cap = self._get_caption(completion)
            target_cap = self._get_caption(targets[i]) if i < len(targets) else None

            if gen_cap and target_cap:
                gen_captions.append(gen_cap)
                target_captions.append(target_cap)
                valid_mask.append(True)
            else:
                gen_captions.append("")
                target_captions.append("")
                valid_mask.append(False)

        if not any(valid_mask):
            return [self.validity_penalty] * len(completions)

        # Batch encode gen and target together in one pass
        all_texts = gen_captions + target_captions
        encoded = self.tokenizer(
            all_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)

        all_embs = self._mean_pool(output, encoded['attention_mask'])
        n = len(completions)
        gen_embs = all_embs[:n]
        target_embs = all_embs[n:]

        sims = torch.cosine_similarity(gen_embs, target_embs, dim=-1).tolist()

        rewards = [
            sim if valid else self.validity_penalty
            for sim, valid in zip(sims, valid_mask)
        ]
        return rewards
