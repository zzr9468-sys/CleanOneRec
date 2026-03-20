"""
Data Engine

Unified interface for loading datasets from various sources.
Supports OpenOneRec-RecIF parquet format and legacy CSV formats.
"""

import json
import random
import logging
import pandas as pd
from datasets import Dataset
from typing import Optional

logger = logging.getLogger(__name__)


class DataEngine:
    """Factory for building HuggingFace Datasets for RL training."""

    @staticmethod
    def from_parquet(
        path: str,
        format: str = "recif",
        sample_num: int = -1,
        seed: int = 42
    ) -> Dataset:
        """
        Load dataset from parquet file.

        Args:
            path: Path to parquet file
            format: Data format ("recif" or "amazon")
            sample_num: Number of samples to use (-1 for all)
            seed: Random seed for sampling

        Returns:
            HuggingFace Dataset with columns: prompt, completion, target_sid
        """
        if format == "recif":
            return DataEngine._load_recif_parquet(path, sample_num, seed)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def from_csv(
        path: str,
        format: str = "amazon",
        sample_num: int = -1,
        seed: int = 42
    ) -> Dataset:
        """
        Load dataset from CSV file.

        Args:
            path: Path to CSV file
            format: Data format (currently only "amazon")
            sample_num: Number of samples to use (-1 for all)
            seed: Random seed for sampling

        Returns:
            HuggingFace Dataset with columns: prompt, completion, target_sid
        """
        if format == "amazon":
            return DataEngine._load_amazon_csv(path, sample_num, seed)
        else:
            raise ValueError(f"Unsupported CSV format: {format}")

    @staticmethod
    def _load_recif_parquet(path: str, sample_num: int, seed: int) -> Dataset:
        """Parse OpenOneRec-RecIF parquet format."""
        random.seed(seed)
        logger.info(f"Loading RecIF parquet from {path}")
        df = pd.read_parquet(path)

        if sample_num > 0 and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)

        dataset_dict = {
            "prompt": [],
            "completion": [],
            "target_sid": [],
            "longview_history": [],
            "target_pids": []
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

            # Extract longview history: use hist_longview_video_list (items the user
            # genuinely watched long), NOT hist_pid (all exposed items including skipped).
            # hist_pid includes exposure-biased head items; hist_longview_video_list is ~75%
            # long-tail and is the correct anchor for LongviewBasedReward.
            longview_pids = []
            for lv_field in ('hist_longview_video_list', 'hist_pid'):
                if lv_field in row and row[lv_field] is not None:
                    longview_pids = (
                        row[lv_field].tolist()
                        if hasattr(row[lv_field], 'tolist')
                        else list(row[lv_field])
                    )
                    break

            # Extract target PIDs
            target_pids = meta.get('answer_pid', [])

            dataset_dict["prompt"].append(prompt_str)
            dataset_dict["completion"].append(target_sid)
            dataset_dict["target_sid"].append(target_sid)
            dataset_dict["longview_history"].append(longview_pids)
            dataset_dict["target_pids"].append(target_pids)

        hf_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Loaded {len(hf_dataset)} samples")
        return hf_dataset

    @staticmethod
    def _load_amazon_csv(path: str, sample_num: int, seed: int) -> Dataset:
        """Parse legacy Amazon CSV format."""
        random.seed(seed)
        logger.info(f"Loading Amazon CSV from {path}")
        df = pd.read_csv(path)

        if sample_num > 0 and sample_num < len(df):
            df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)

        dataset_dict = {"prompt": [], "completion": [], "target_sid": []}

        for _, row in df.iterrows():
            try:
                import ast
                history_sid = ast.literal_eval(row['history_item_sid'])
            except (ValueError, SyntaxError):
                continue

            history_str = ", ".join(history_sid)
            target_sid = str(row['item_sid'])

            prompt_str = (
                f"### User Input:\nThe user has interacted with items {history_str} "
                f"in chronological order. Can you predict the next possible item?\n\n"
                f"### Response:\n"
            )

            dataset_dict["prompt"].append(prompt_str)
            dataset_dict["completion"].append(target_sid)
            dataset_dict["target_sid"].append(target_sid)

        hf_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Loaded {len(hf_dataset)} samples")
        return hf_dataset
