"""
基于 Longview 的隐式反馈 Reward

核心思想：
- Longview（长时观看）是用户真实兴趣的最强信号
- 不受曝光偏差影响（用户主动选择观看）
- 如果推荐的物品与用户 longview 历史语义相似，给高分
"""

import torch
import json
import pandas as pd
import re
from typing import List, Dict, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer


class LongviewBasedReward:
    """
    基于 Longview 的隐式反馈 Reward

    计算逻辑：
    1. 解析生成的 SID -> PID -> Caption
    2. 获取用户 longview 历史的 Captions
    3. 计算语义相似度
    4. 返回最大相似度作为 reward
    """

    def __init__(
        self,
        recif_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        self.recif_path = Path(recif_path)
        self.device = device

        # 加载映射
        self.sid2pid_dict = self._load_sid2pid()
        self.pid2caption_dict = self._load_pid2caption()

        # 加载文本模型
        self.text_model = SentenceTransformer(model_name, device=device)

        print(f"✅ LongviewBasedReward 初始化完成")
        print(f"   - SID->PID 映射: {len(self.sid2pid_dict)} 条")
        print(f"   - PID->Caption 映射: {len(self.pid2caption_dict)} 条")
        print(f"   - 文本模型: {model_name}")

    def _load_sid2pid(self) -> Dict[str, List[Dict]]:
        """加载 SID -> PID 映射"""
        sid2pid_path = self.recif_path / "benchmark_data" / "sid2pid.json"
        with open(sid2pid_path, 'r') as f:
            return json.load(f)

    def _load_pid2caption(self) -> Dict[str, str]:
        """加载 PID -> Caption 映射"""
        pid2caption_path = self.recif_path / "pid2caption.parquet"
        df = pd.read_parquet(pid2caption_path)

        pid2caption = {}
        for _, row in df.iterrows():
            pid = str(row['pid'])
            caption = row['dense_caption']  # 注意：列名是 dense_caption
            pid2caption[pid] = caption

        return pid2caption

    def _parse_sid_to_hash_key(self, sid_str: str) -> Optional[str]:
        """
        将 <s_a_X><s_b_Y><s_c_Z> 转换为 hash key

        公式: hash_key = a * 8192 * 8192 + b * 8192 + c
        """
        matches = re.findall(r'<s_[abc]_(\d+)>', sid_str)
        if len(matches) == 3:
            a, b, c = int(matches[0]), int(matches[1]), int(matches[2])
            hash_key = str(a * 8192 * 8192 + b * 8192 + c)
            return hash_key
        return None

    def _get_caption_from_sid(self, sid_str: str) -> Optional[str]:
        """
        从 SID 获取 Caption

        流程: SID -> Hash Key -> PID -> Caption
        """
        # 1. SID -> Hash Key
        hash_key = self._parse_sid_to_hash_key(sid_str)
        if hash_key is None:
            return None

        # 2. Hash Key -> PID
        if hash_key not in self.sid2pid_dict:
            return None

        pid_list = self.sid2pid_dict[hash_key]
        if len(pid_list) == 0:
            return None

        # 取第一个 PID（通常是最常见的）
        pid = str(pid_list[0]['pid'])

        # 3. PID -> Caption
        if pid not in self.pid2caption_dict:
            return None

        return self.pid2caption_dict[pid]

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        emb1 = self.text_model.encode(text1, convert_to_tensor=True)
        emb2 = self.text_model.encode(text2, convert_to_tensor=True)

        # 余弦相似度
        sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item()

        return sim

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        longview_history: Optional[List[List[int]]] = None,
        **kwargs
    ) -> List[float]:
        """
        计算 Reward

        参数:
            prompts: 输入 prompts
            completions: 生成的 SID 列表
            longview_history: 每个样本的 longview 历史 PID 列表

        返回:
            rewards: 每个样本的 reward
        """
        if longview_history is None:
            # 如果没有提供 longview 历史，返回 0
            return [0.0] * len(completions)

        rewards = []

        for i, completion in enumerate(completions):
            # 获取该样本的 longview 历史
            hist_pids = longview_history[i] if i < len(longview_history) else []

            if len(hist_pids) == 0:
                rewards.append(0.0)
                continue

            # 获取 longview 物品的 captions
            longview_captions = []
            for pid in hist_pids:
                pid_str = str(int(pid))
                if pid_str in self.pid2caption_dict:
                    longview_captions.append(self.pid2caption_dict[pid_str])

            if len(longview_captions) == 0:
                rewards.append(0.0)
                continue

            # 解析生成的 SID -> Caption
            gen_caption = self._get_caption_from_sid(completion)

            if gen_caption is None:
                # 无效的 SID，给予惩罚
                rewards.append(-10.0)
                continue

            # 计算与所有 longview 物品的最大相似度
            max_sim = 0.0
            for lv_caption in longview_captions:
                sim = self._compute_similarity(gen_caption, lv_caption)
                max_sim = max(max_sim, sim)

            rewards.append(max_sim)

        return rewards


if __name__ == "__main__":
    # 测试
    reward = LongviewBasedReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 测试数据
    prompts = ["用户历史..."]
    completions = ["<s_a_0><s_b_0><s_c_1>"]
    longview_history = [[2360735, 9241153, 11239440]]

    # 计算 reward
    rewards = reward(prompts, completions, longview_history=longview_history)
    print(f"Rewards: {rewards}")
