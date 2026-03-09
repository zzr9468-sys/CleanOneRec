"""
测试 RecRL 完整流程

测试内容：
1. 数据加载
2. SID 解析
3. Reward 计算
"""

import sys
sys.path.insert(0, '/Users/zhouziren/onerec/CleanOneRec')

from recrl.data.recif_loader import RecIFDataLoader
from recrl.rewards.longview_reward import LongviewBasedReward
from recrl.rewards.novelty_reward import NoveltyReward
from recrl.rewards.recrl_composite_reward import RecRLCompositeReward


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)

    loader = RecIFDataLoader("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 获取训练数据
    train_data = loader.to_recrl_format("train")
    print(f"✅ 训练数据: {len(train_data)} 条")

    # 查看第一条
    sample = train_data[0]
    print(f"\n第一条数据:")
    print(f"  Prompt 长度: {len(sample['prompt'])} 字符")
    print(f"  Target PIDs: {sample['target_pids'][:3]}...")
    print(f"  Longview History: {len(sample['longview_history'])} 个")
    print(f"  Behavior Signals: {sample['behavior_signals']}")

    return loader, train_data


def test_sid_parsing():
    """测试 SID 解析"""
    print("\n" + "=" * 60)
    print("测试 2: SID 解析")
    print("=" * 60)

    reward = LongviewBasedReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 测试 SID
    test_sids = [
        "<s_a_0><s_b_0><s_c_1>",
        "<s_a_1><s_b_2><s_c_3>",
        "<s_a_10><s_b_20><s_c_30>",
    ]

    for sid in test_sids:
        hash_key = reward._parse_sid_to_hash_key(sid)
        caption = reward._get_caption_from_sid(sid)
        print(f"\nSID: {sid}")
        print(f"  Hash Key: {hash_key}")
        print(f"  Caption: {caption[:100] if caption else 'None'}...")


def test_longview_reward(loader, train_data):
    """测试 Longview Reward"""
    print("\n" + "=" * 60)
    print("测试 3: Longview Reward")
    print("=" * 60)

    reward = LongviewBasedReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 取第一条数据
    sample = train_data[0]

    # 模拟生成的 SID（这里用真实的 target PID 反推 SID）
    # 实际使用时，这些是 LLM 生成的
    test_completions = ["<s_a_0><s_b_0><s_c_1>"]

    # 计算 reward
    rewards = reward(
        prompts=[sample['prompt']],
        completions=test_completions,
        longview_history=[sample['longview_history']]
    )

    print(f"✅ Longview Reward: {rewards}")


def test_novelty_reward():
    """测试 Novelty Reward"""
    print("\n" + "=" * 60)
    print("测试 4: Novelty Reward")
    print("=" * 60)

    reward = NoveltyReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 测试不同频率的物品
    test_completions = [
        "<s_a_0><s_b_0><s_c_1>",
        "<s_a_1><s_b_2><s_c_3>",
    ]

    rewards = reward(
        prompts=["test"],
        completions=test_completions
    )

    print(f"✅ Novelty Rewards: {rewards}")


def test_composite_reward(loader, train_data):
    """测试组合 Reward"""
    print("\n" + "=" * 60)
    print("测试 5: 组合 Reward")
    print("=" * 60)

    reward = RecRLCompositeReward("/Users/zhouziren/onerec/OpenOneRec-RecIF")

    # 取第一条数据
    sample = train_data[0]

    # 测试
    test_completions = ["<s_a_0><s_b_0><s_c_1>"]

    rewards = reward(
        prompts=[sample['prompt']],
        completions=test_completions,
        longview_history=[sample['longview_history']]
    )

    print(f"✅ Composite Reward: {rewards}")


def main():
    """运行所有测试"""
    print("\n🚀 开始测试 RecRL 流程...\n")

    try:
        # 1. 数据加载
        loader, train_data = test_data_loading()

        # 2. SID 解析
        test_sid_parsing()

        # 3. Longview Reward
        test_longview_reward(loader, train_data)

        # 4. Novelty Reward
        test_novelty_reward()

        # 5. 组合 Reward
        test_composite_reward(loader, train_data)

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
