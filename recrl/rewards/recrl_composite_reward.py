"""
RecRL 组合 Reward

组合多个 reward 函数：
1. Longview 隐式反馈 (50%) - 主要信号
2. 语义相似度 (30%) - 辅助信号
3. 新颖性 (15%) - 探索长尾
4. 多样性 (5%) - 避免重复
"""

from typing import List, Dict, Optional
from .longview_reward import LongviewBasedReward
from .novelty_reward import NoveltyReward


class RecRLCompositeReward:
    """
    RecRL 推荐的组合 Reward

    核心思想：
    - 用 Longview 作为主要信号（用户真实兴趣）
    - 语义相似度作为辅助（内容匹配）
    - 新颖性鼓励探索长尾
    - 多样性避免重复推荐
    """

    def __init__(
        self,
        recif_path: str,
        device: str = "cuda",
        weights: Optional[Dict[str, float]] = None
    ):
        """
        参数:
            recif_path: RecIF 数据路径
            device: 设备 (cuda/cpu)
            weights: 权重字典，默认为:
                {
                    'longview': 0.5,
                    'semantic': 0.3,
                    'novelty': 0.15,
                    'diversity': 0.05
                }
        """
        self.recif_path = recif_path
        self.device = device

        # 默认权重
        if weights is None:
            weights = {
                'longview': 0.5,
                'semantic': 0.3,
                'novelty': 0.15,
                'diversity': 0.05
            }
        self.weights = weights

        # 初始化各个 reward 函数
        print("🚀 初始化 RecRL Composite Reward...")

        # 1. Longview 隐式反馈
        self.longview_reward = LongviewBasedReward(recif_path, device=device)

        # 2. 语义相似度（复用 Longview 的逻辑，但针对 target）
        # TODO: 实现专门的 semantic reward

        # 3. 新颖性
        self.novelty_reward = NoveltyReward(recif_path)

        # 4. 多样性
        # TODO: 实现 diversity reward

        print(f"✅ RecRL Composite Reward 初始化完成")
        print(f"   权重: {self.weights}")

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        longview_history: Optional[List[List[int]]] = None,
        target_pids: Optional[List[List[int]]] = None,
        **kwargs
    ) -> List[float]:
        """
        计算组合 Reward

        参数:
            prompts: 输入 prompts
            completions: 生成的 SID 列表
            longview_history: 每个样本的 longview 历史
            target_pids: 每个样本的目标 PID 列表

        返回:
            rewards: 每个样本的组合 reward
        """
        # 1. Longview 隐式反馈
        longview_rewards = self.longview_reward(
            prompts, completions, longview_history=longview_history
        )

        # 2. 新颖性
        novelty_rewards = self.novelty_reward(prompts, completions)

        # 3. 组合
        final_rewards = []
        for i in range(len(completions)):
            lv_r = longview_rewards[i]
            nov_r = novelty_rewards[i]

            # 加权组合
            final_r = (
                self.weights['longview'] * lv_r +
                self.weights['novelty'] * nov_r
            )

            final_rewards.append(final_r)

        return final_rewards


if __name__ == "__main__":
    # 测试
    reward = RecRLCompositeReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 测试数据
    prompts = ["用户历史..."]
    completions = ["<s_a_0><s_b_0><s_c_1>"]
    longview_history = [[2360735, 9241153, 11239440]]

    # 计算 reward
    rewards = reward(prompts, completions, longview_history=longview_history)
    print(f"Composite Rewards: {rewards}")
