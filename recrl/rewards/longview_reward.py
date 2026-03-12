"""
Longview-based Implicit Feedback Reward

Core idea:
- Longview (long-watch) is the strongest signal of true user interest
- Not subject to exposure bias (user actively chose to watch)
- Reward generated items that are semantically similar to user's longview history
"""

import re
import torch
import json
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer

SID_PATTERN = re.compile(r'<\|sid_begin\|>.*?<\|sid_end\|>')


class LongviewBasedReward:
    """
    Longview-based implicit feedback reward.

    Steps:
    1. Parse generated SID -> PID -> Caption
    2. Get user longview history captions
    3. Compute max cosine similarity between generated and history
    4. Return max similarity as reward

    Performance: all encoding is batched across the full call in one
    sentence-transformer forward pass.
    """

    def __init__(
        self,
        recif_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        validity_penalty: float = -1.0,
        encode_batch_size: int = 256,
    ):
        self.recif_path = Path(recif_path)
        self.device = device
        self.validity_penalty = validity_penalty
        self.encode_batch_size = encode_batch_size

        self.sid2pid_dict = self._load_sid2pid()
        self.pid2caption_dict = self._load_pid2caption()
        self.text_model = SentenceTransformer(model_name, device=device)

        print(f"✅ LongviewBasedReward initialized")
        print(f"   SID->PID mappings : {len(self.sid2pid_dict):,}")
        print(f"   PID->Caption entries: {len(self.pid2caption_dict):,}")
        print(f"   Encoder           : {model_name}")

    def _load_sid2pid(self) -> Dict[str, List[Dict]]:
        sid2pid_path = self.recif_path / "benchmark_data" / "sid2pid.json"
        with open(sid2pid_path, 'r') as f:
            return json.load(f)

    def _load_pid2caption(self) -> Dict[str, str]:
        pid2caption_path = self.recif_path / "pid2caption.parquet"
        df = pd.read_parquet(pid2caption_path)
        return dict(zip(df['pid'].astype(str), df['dense_caption']))

    def _parse_sid_to_hash_key(self, sid_str: str) -> Optional[str]:
        """<s_a_X><s_b_Y><s_c_Z> -> hash key string."""
        matches = re.findall(r'<s_[abc]_(\d+)>', sid_str)
        if len(matches) == 3:
            a, b, c = int(matches[0]), int(matches[1]), int(matches[2])
            return str(a * 8192 * 8192 + b * 8192 + c)
        return None

    def _get_caption_from_sid(self, sid_str: str) -> Optional[str]:
        """SID string -> caption text (via hash key -> PID -> caption)."""
        hash_key = self._parse_sid_to_hash_key(sid_str)
        if hash_key is None or hash_key not in self.sid2pid_dict:
            return None
        pid_list = self.sid2pid_dict[hash_key]
        if not pid_list:
            return None
        pid = str(pid_list[0]['pid'])
        return self.pid2caption_dict.get(pid)

    def _get_longview_captions(self, hist_pids: List[int]) -> List[str]:
        """Get captions for a list of longview PIDs."""
        captions = []
        for pid in hist_pids:
            cap = self.pid2caption_dict.get(str(int(pid)))
            if cap:
                captions.append(cap)
        return captions

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        longview_history: Optional[List[List[int]]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.

        All sentence-transformer encoding is done in one batched call.
        """
        if longview_history is None:
            return [0.0] * len(completions)

        # ── Step 1: resolve captions for each completion ──────────────────
        gen_captions: List[Optional[str]] = []
        hist_captions_per_sample: List[List[str]] = []
        valid: List[bool] = []

        for i, completion in enumerate(completions):
            hist_pids = longview_history[i] if i < len(longview_history) else []
            hist_caps = self._get_longview_captions(hist_pids)

            sids = SID_PATTERN.findall(completion)
            gen_cap = self._get_caption_from_sid(sids[0]) if sids else None

            # Debug first sample on first call
            if i == 0 and not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                first_sid = sids[0] if sids else "none"
                print(f"[DEBUG Longview] first_sid : {first_sid}")
                print(f"[DEBUG Longview] gen_caption: {(gen_cap or 'None')[:80]}")
                print(f"[DEBUG Longview] hist_size  : {len(hist_caps)}")

            gen_captions.append(gen_cap)
            hist_captions_per_sample.append(hist_caps)
            valid.append(gen_cap is not None and len(hist_caps) > 0)

        if not any(valid):
            return [self.validity_penalty] * len(completions)

        # ── Step 2: build flat text list for ONE batched encode ───────────
        # Layout: [gen_cap_i, hist_cap_i_0, hist_cap_i_1, ...] for valid i
        flat_texts: List[str] = []
        # Track slices: for each sample i, (gen_idx, hist_start, hist_end) in flat_texts
        sample_slices: List[Optional[tuple]] = []

        for i in range(len(completions)):
            if not valid[i]:
                sample_slices.append(None)
                continue
            gen_idx = len(flat_texts)
            flat_texts.append(gen_captions[i])
            hist_start = len(flat_texts)
            flat_texts.extend(hist_captions_per_sample[i])
            hist_end = len(flat_texts)
            sample_slices.append((gen_idx, hist_start, hist_end))

        # ── Step 3: encode all texts in one call ──────────────────────────
        all_embs = self.text_model.encode(
            flat_texts,
            convert_to_tensor=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )  # [N, D]

        # ── Step 4: compute max cosine similarity per sample ──────────────
        rewards: List[float] = []
        for i in range(len(completions)):
            if not valid[i]:
                rewards.append(self.validity_penalty)
                continue

            gen_idx, hist_start, hist_end = sample_slices[i]
            gen_emb = all_embs[gen_idx]          # [D]
            hist_embs = all_embs[hist_start:hist_end]  # [H, D]

            sims = torch.nn.functional.cosine_similarity(
                gen_emb.unsqueeze(0), hist_embs
            )  # [H]
            rewards.append(sims.max().item())

        return rewards


if __name__ == "__main__":
    reward = LongviewBasedReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")
    prompts = ["user history..."]
    completions = ["<|sid_begin|><s_a_0><s_b_0><s_c_1><|sid_end|>"]
    longview_history = [[2360735, 9241153, 11239440]]
    rewards = reward(prompts, completions, longview_history=longview_history)
    print(f"Rewards: {rewards}")
