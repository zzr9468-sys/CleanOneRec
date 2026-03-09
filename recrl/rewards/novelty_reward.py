"""
新颖性 Reward

核心思想：
- 鼓励推荐长尾/冷门物品
- 打破"师生偏见"，探索未曝光的物品
- 基于物品的曝光频率计算新颖性
"""

import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter


class NoveltyReward:
    """
    新颖性 Reward

    计算逻辑：
    1. 统计训练数据中每个 PID 的曝光频率
    2. 曝光频率越低，新颖性越高
    3. 新颖性 = 1 - log(freq + 1) / log(max_freq + 1)
    """

    def __init__(self, recif_path: str, freq_threshold: int = 100):
        self.recif_path = Path(recif_path)
        self.freq_threshold = freq_threshold

        # 加载映射
        self.sid2pid_dict = self._load_sid2pid()

        # 统计 PID 频率
        self.pid_freq = self._compute_pid_frequency()
        self.max_freq = max(self.pid_freq.values()) if self.pid_freq else 1

        print(f"✅ NoveltyReward 初始化完成")
        print(f"   - 总 PID 数: {len(self.pid_freq)}")
        print(f"   - 最高频率: {self.max_freq}")
        print(f"   - 长尾物品 (freq < {freq_threshold}): {sum(1 for f in self.pid_freq.values() if f < freq_threshold)}")

    def _load_sid2pid(self) -> Dict[str, List[Dict]]:
        """加载 SID -> PID 映射"""
        sid2pid_path = self.recif_path / "benchmark_data" / "sid2pid.json"
        with open(sid2pid_path, 'r') as f:
            return json.load(f)

    def _compute_pid_frequency(self) -> Dict[str, int]:
        """
        统计 PID 的曝光频率

        从 sid2pid.json 中提取 count 字段
        """
        pid_freq = Counter()

        for hash_key, pid_list in self.sid2pid_dict.items():
            for item in pid_list:
                pid = str(item['pid'])
                count = item.get('count_after_downsample', item.get('count', 1))
                pid_freq[pid] += count

        return dict(pid_freq)

    def _parse_sid_to_hash_key(self, sid_str: str) -> Optional[str]:
        """将 <s_a_X><s_b_Y><s_c_Z> 转换为 hash key"""
        matches = re.findall(r'<s_[abc]_(\d+)>', sid_str)
        if len(matches) == 3:
            a, b, c = int(matches[0]), int(matches[1]), int(matches[2])
            hash_key = str(a * 8192 * 8192 + b * 8192 + c)
            return hash_key
        return None

    def _get_pid_from_sid(self, sid_str: str) -> Optional[str]:
        """从 SID 获取 PID"""
        hash_key = self._parse_sid_to_hash_key(sid_str)
        if hash_key is None or hash_key not in self.sid2pid_dict:
            return None

        pid_list = self.sid2pid_dict[hash_key]
        if len(pid_list) == 0:
            return None

        return str(pid_list[0]['pid'])

    def _compute_novelty_score(self, pid: str) -> float:
        """
        计算新颖性分数

        公式: novelty = 1 - log(freq + 1) / log(max_freq + 1)

        - 高频物品 (freq = max_freq): novelty ≈ 0
        - 低频物品 (freq = 1): novelty ≈ 1
        """
        import math

        freq = self.pid_freq.get(pid, 0)

        if freq == 0:
            # 完全未见过的物品，给予最高新颖性
            return 1.0

        novelty = 1.0 - math.log(freq + 1) / math.log(self.max_freq + 1)
        return max(0.0, novelty)  # 确保非负

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        计算新颖性 Reward

        参数:
            prompts: 输入 prompts
            completions: 生成的 SID 列表

        返回:
            rewards: 每个样本的新颖性分数
        """
        rewards = []

        for completion in completions:
            # 解析 SID -> PID
            pid = self._get_pid_from_sid(completion)

            if pid is None:
                # 无效的 SID，给予惩罚
                rewards.append(-10.0)
                continue

            # 计算新颖性
            novelty = self._compute_novelty_score(pid)
            rewards.append(novelty)

        return rewards


if __name__ == "__main__":
    # 测试
    reward = NoveltyReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 测试数据
    prompts = ["用户历史..."]
    completions = ["<s_a_0><s_b_0><s_c_1>"]

    # 计算 reward
    rewards = reward(prompts, completions)
    print(f"Novelty Rewards: {rewards}")
