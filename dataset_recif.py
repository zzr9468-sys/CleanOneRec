"""
RecIF 数据集构建器
兼容 OpenOneRec-RecIF 的实际数据格式
"""

import json
import random
import pandas as pd
from datasets import Dataset
from pathlib import Path
from typing import Optional, Dict, List


def build_recif_dataset(
    recif_path: str,
    split: str = "train",
    sample_num: int = -1,
    seed: int = 42
) -> Dataset:
    """
    构建 RecIF 格式的数据集

    参数:
        recif_path: RecIF 数据路径
        split: 数据集划分 (train/val/test)
        sample_num: 采样数量，-1 表示全部
        seed: 随机种子

    返回:
        HuggingFace Dataset
    """
    random.seed(seed)
    recif_path = Path(recif_path)

    # 加载主数据
    df = pd.read_parquet(recif_path / "onerec_bench_release.parquet")

    # 加载映射
    with open(recif_path / "benchmark_data" / "sid2pid.json", 'r') as f:
        sid2pid_dict = json.load(f)

    pid2caption_df = pd.read_parquet(recif_path / "pid2caption.parquet")
    pid2caption_dict = {
        str(row['pid']): row['dense_caption']
        for _, row in pid2caption_df.iterrows()
    }

    # 选择数据集划分
    split_map = {"train": 0, "val": 1, "test": 2}
    df = df[df['split'] == split_map[split]]

    # 采样
    if sample_num > 0 and sample_num < len(df):
        df = df.sample(n=sample_num, random_state=seed).reset_index(drop=True)

    # 构建数据集
    dataset_dict = {
        "prompt": [],
        "target_pids": [],
        "target_sids": [],
        "longview_history": [],
        "behavior_signals": []
    }

    for _, row in df.iterrows():
        # 构造 prompt
        prompt = _format_prompt(row, pid2caption_dict)

        # 获取目标 PID
        target_pids = row['target_video_pid'].tolist()

        # 获取目标 SID（如果有的话）
        # 注意：实际数据可能没有 SID，需要从 PID 反推
        target_sids = _pids_to_sids(target_pids, sid2pid_dict)

        # 获取 longview 历史
        longview_history = row['hist_longview_video_list'].tolist()

        # 获取行为信号
        behavior_signals = _get_behavior_signals(row)

        dataset_dict["prompt"].append(prompt)
        dataset_dict["target_pids"].append(target_pids)
        dataset_dict["target_sids"].append(target_sids)
        dataset_dict["longview_history"].append(longview_history)
        dataset_dict["behavior_signals"].append(behavior_signals)

    hf_dataset = Dataset.from_dict(dataset_dict)
    return hf_dataset


def _format_prompt(row: pd.Series, pid2caption_dict: Dict[str, str]) -> str:
    """格式化 prompt"""
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

历史观看视频 (最近 10 个):
"""

    # 添加历史视频的 caption（最多显示最近 10 个）
    for pid in hist_pids[-10:]:
        pid_str = str(int(pid))
        if pid_str in pid2caption_dict:
            caption = pid2caption_dict[pid_str]
            prompt += f"- {caption[:100]}...\n"

    prompt += "\n请推荐下一个视频 (输出格式: <s_a_X><s_b_Y><s_c_Z>):"

    return prompt


def _pids_to_sids(pids: List[int], sid2pid_dict: Dict[str, List[Dict]]) -> List[str]:
    """
    从 PID 反推 SID

    注意：这是一个反向查找，可能比较慢
    """
    sids = []

    # 构建 PID -> SID 的反向索引
    pid_to_sid = {}
    for hash_key, pid_list in sid2pid_dict.items():
        for item in pid_list:
            pid = str(item['pid'])
            if pid not in pid_to_sid:
                pid_to_sid[pid] = []
            pid_to_sid[pid].append(hash_key)

    # 查找每个 PID 对应的 SID
    for pid in pids:
        pid_str = str(int(pid))
        if pid_str in pid_to_sid:
            # 取第一个 SID（可能有多个）
            hash_key = pid_to_sid[pid_str][0]
            # 将 hash_key 转换为 SID 格式
            sid = _hash_key_to_sid(hash_key)
            sids.append(sid)
        else:
            sids.append("")  # 找不到对应的 SID

    return sids


def _hash_key_to_sid(hash_key: str) -> str:
    """
    将 hash_key 转换为 SID 格式

    公式: hash_key = a * 8192 * 8192 + b * 8192 + c
    反推: a = hash_key // (8192 * 8192)
          b = (hash_key % (8192 * 8192)) // 8192
          c = hash_key % 8192
    """
    key = int(hash_key)
    a = key // (8192 * 8192)
    b = (key % (8192 * 8192)) // 8192
    c = key % 8192
    return f"<s_a_{a}><s_b_{b}><s_c_{c}>"


def _get_behavior_signals(row: pd.Series) -> Dict[str, float]:
    """获取用户行为信号"""
    hist_longview = row['hist_video_longview']
    hist_like = row['hist_video_like']
    hist_forward = row['hist_video_forward']
    hist_follow = row['hist_video_follow']

    total = len(hist_longview)

    return {
        'longview_ratio': float(hist_longview.sum() / total if total > 0 else 0.0),
        'like_ratio': float(hist_like.sum() / total if total > 0 else 0.0),
        'forward_ratio': float(hist_forward.sum() / total if total > 0 else 0.0),
        'follow_ratio': float(hist_follow.sum() / total if total > 0 else 0.0),
    }


if __name__ == "__main__":
    # 测试
    dataset = build_recif_dataset(
        recif_path="/Users/zhouziren/onerec/OpenOneRec-RecIF",
        split="train",
        sample_num=10
    )

    print(f"数据集大小: {len(dataset)}")
    print(f"\n第一条数据:")
    print(f"Prompt: {dataset[0]['prompt'][:200]}...")
    print(f"Target PIDs: {dataset[0]['target_pids'][:3]}")
    print(f"Target SIDs: {dataset[0]['target_sids'][:3]}")
    print(f"Longview History: {len(dataset[0]['longview_history'])} 个")
    print(f"Behavior Signals: {dataset[0]['behavior_signals']}")
