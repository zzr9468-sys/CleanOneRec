"""
RecIF 数据加载器
专门处理 OpenOneRec-RecIF 的 parquet 格式
"""

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from pathlib import Path


class RecIFDataLoader:
    """
    加载 OpenOneRec-RecIF 数据集

    数据格式：
    - hist_video_pid: 用户历史观看视频 PID
    - target_video_pid: 目标推荐视频 PID
    - hist_longview_video_list: 用户长时观看的视频列表
    - inter_user_profile_with_sid: 用户画像（带 SID）
    - reco_cot: 推荐理由（Chain-of-Thought）
    """

    def __init__(self, recif_path: str, load_captions: bool = False):
        """
        初始化 RecIF 数据加载器

        参数:
            recif_path: RecIF 数据集路径
            load_captions: 是否立即加载 caption（5.5GB，很慢）。默认 False，延迟加载
        """
        self.recif_path = Path(recif_path)

        # 加载主数据
        df = pd.read_parquet(self.recif_path / "onerec_bench_release.parquet")

        # 过滤掉无效行
        print(f"原始数据: {len(df)} 条")

        # 过滤条件：必须有 target_video_pid 和 hist_longview_video_list
        valid_mask = (
            df['target_video_pid'].notna() &
            df['hist_longview_video_list'].notna() &
            df['inter_user_profile_with_sid'].notna()
        )

        self.df = df[valid_mask].reset_index(drop=True)
        print(f"过滤后数据: {len(self.df)} 条 (移除了 {len(df) - len(self.df)} 条无效数据)")

        # 加载映射
        self.sid2pid_dict = self._load_sid2pid()

        # 延迟加载 PID -> Caption 映射（5.5GB，很慢）
        self._pid2caption_dict = None
        if load_captions:
            self._pid2caption_dict = self._load_pid2caption()

        print(f"✅ 加载 RecIF 数据: {len(self.df)} 条")
        print(f"✅ 加载 SID->PID 映射: {len(self.sid2pid_dict)} 条")
        if load_captions:
            print(f"✅ 加载 PID->Caption 映射: {len(self._pid2caption_dict)} 条")
        else:
            print(f"⏳ PID->Caption 映射将在首次使用时加载")

    @property
    def pid2caption_dict(self):
        """延迟加载 caption 字典"""
        if self._pid2caption_dict is None:
            print("⏳ 正在加载 pid2caption.parquet (5.5GB)，这可能需要 1-2 分钟...")
            self._pid2caption_dict = self._load_pid2caption()
            print(f"✅ Caption 加载完成: {len(self._pid2caption_dict)} 条")
        return self._pid2caption_dict

    def _load_sid2pid(self) -> Dict[str, List[Dict]]:
        """加载 SID -> PID 映射"""
        sid2pid_path = self.recif_path / "benchmark_data" / "sid2pid.json"
        with open(sid2pid_path, 'r') as f:
            return json.load(f)

    def _load_pid2caption(self) -> Dict[str, str]:
        """加载 PID -> Caption 映射"""
        pid2caption_path = self.recif_path / "pid2caption.parquet"
        df = pd.read_parquet(pid2caption_path)

        # 转换为字典
        pid2caption = {}
        for _, row in df.iterrows():
            pid = str(row['pid'])
            caption = row['dense_caption']  # 注意：列名是 dense_caption
            pid2caption[pid] = caption

        return pid2caption

    def get_train_data(self) -> pd.DataFrame:
        """获取训练集"""
        return self.df[self.df['split'] == 0]

    def get_val_data(self) -> pd.DataFrame:
        """获取验证集"""
        return self.df[self.df['split'] == 1]

    def get_test_data(self) -> pd.DataFrame:
        """获取测试集"""
        return self.df[self.df['split'] == 2]

    def format_prompt(self, row: pd.Series) -> str:
        """
        将数据行格式化为 LLM prompt

        格式：
        用户画像: {user_profile}
        历史观看: {hist_video_pids}
        请推荐下一个视频。
        """
        # 获取用户画像（处理可能的 JSON 解析错误）
        try:
            profile_str = row['inter_user_profile_with_sid']
            if isinstance(profile_str, str) and profile_str.strip():
                user_profile = json.loads(profile_str)
            else:
                user_profile = {}
        except (json.JSONDecodeError, TypeError):
            user_profile = {}

        # 获取历史观看
        hist_pids = row['hist_video_pid']

        # 构造 prompt
        prompt = f"""用户画像:
性别: {user_profile.get('性别', '未知')}
年龄: {user_profile.get('年龄', '未知')}

历史观看视频 (最近 {len(hist_pids)} 个):
"""

        # 添加历史视频的 caption（最多显示最近 10 个）
        for pid in hist_pids[-10:]:
            pid_str = str(int(pid))
            if pid_str in self.pid2caption_dict:
                caption = self.pid2caption_dict[pid_str]
                prompt += f"- {caption[:100]}...\n"

        prompt += "\n请推荐下一个视频 (输出格式: <s_a_X><s_b_Y><s_c_Z>):"

        return prompt

    def format_target(self, row: pd.Series) -> List[int]:
        """获取目标 PID 列表"""
        target_pids = row['target_video_pid']
        if target_pids is None:
            return []
        return target_pids.tolist() if hasattr(target_pids, 'tolist') else list(target_pids)

    def get_longview_history(self, row: pd.Series) -> List[int]:
        """获取用户长时观看历史"""
        longview_list = row['hist_longview_video_list']
        if longview_list is None or (hasattr(longview_list, '__len__') and len(longview_list) == 0):
            return []
        return longview_list.tolist() if hasattr(longview_list, 'tolist') else list(longview_list)

    def get_behavior_signals(self, row: pd.Series) -> Dict[str, float]:
        """
        获取用户行为信号

        返回：
        {
            'longview_ratio': 长时观看比例,
            'like_ratio': 点赞比例,
            'forward_ratio': 转发比例,
            'follow_ratio': 关注比例
        }
        """
        hist_longview = row['hist_video_longview']
        hist_like = row['hist_video_like']
        hist_forward = row['hist_video_forward']
        hist_follow = row['hist_video_follow']

        # 处理可能为 None 的情况
        if hist_longview is None or len(hist_longview) == 0:
            return {
                'longview_ratio': 0.0,
                'like_ratio': 0.0,
                'forward_ratio': 0.0,
                'follow_ratio': 0.0,
            }

        total = len(hist_longview)

        return {
            'longview_ratio': float(hist_longview.sum() / total if total > 0 else 0.0),
            'like_ratio': float(hist_like.sum() / total if total > 0 else 0.0),
            'forward_ratio': float(hist_forward.sum() / total if total > 0 else 0.0),
            'follow_ratio': float(hist_follow.sum() / total if total > 0 else 0.0),
        }

    def to_recrl_format(self, split: str = "train") -> List[Dict[str, Any]]:
        """
        转换为 RecRL 训练格式

        返回：
        [
            {
                'prompt': str,
                'target_pids': List[int],
                'longview_history': List[int],
                'behavior_signals': Dict[str, float]
            },
            ...
        ]
        """
        if split == "train":
            df = self.get_train_data()
        elif split == "val":
            df = self.get_val_data()
        else:
            df = self.get_test_data()

        data = []
        for _, row in df.iterrows():
            data.append({
                'prompt': self.format_prompt(row),
                'target_pids': self.format_target(row),
                'longview_history': self.get_longview_history(row),
                'behavior_signals': self.get_behavior_signals(row),
            })

        return data


if __name__ == "__main__":
    # 测试
    loader = RecIFDataLoader("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 获取训练数据
    train_data = loader.to_recrl_format("train")
    print(f"\n训练数据: {len(train_data)} 条")

    # 查看第一条
    print("\n第一条数据示例:")
    print(f"Prompt: {train_data[0]['prompt'][:200]}...")
    print(f"Target PIDs: {train_data[0]['target_pids'][:5]}")
    print(f"Longview History: {len(train_data[0]['longview_history'])} 个")
    print(f"Behavior Signals: {train_data[0]['behavior_signals']}")
