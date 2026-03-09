"""
简化测试：只测试数据加载
"""

import sys
sys.path.insert(0, '/Users/zhouziren/onerec/CleanOneRec')

from recrl.data.recif_loader import RecIFDataLoader


def test_basic_loading():
    """测试基础数据加载"""
    print("=" * 60)
    print("测试: 基础数据加载")
    print("=" * 60)

    try:
        # 不立即加载 caption（5.5GB）
        loader = RecIFDataLoader("/Users/zhouziren/onerec/OpenOneRec-RecIF", load_captions=False)
        print("✅ RecIFDataLoader 初始化成功")

        # 获取训练数据（只取 5 条）
        train_df = loader.get_train_data()
        print(f"✅ 训练数据: {len(train_df)} 条")

        # 测试目标 PID（不需要 caption）
        first_row = train_df.iloc[0]
        target_pids = loader.format_target(first_row)
        print(f"\n✅ Target PIDs: {target_pids[:3]}...")

        # 测试 longview 历史（不需要 caption）
        longview_history = loader.get_longview_history(first_row)
        print(f"✅ Longview History: {len(longview_history)} 个")

        # 测试行为信号（不需要 caption）
        behavior_signals = loader.get_behavior_signals(first_row)
        print(f"✅ Behavior Signals: {behavior_signals}")

        print("\n" + "=" * 60)
        print("✅ 基础测试通过（未加载 caption）")
        print("=" * 60)

        # 现在测试 prompt 生成（会触发 caption 加载）
        print("\n" + "=" * 60)
        print("测试: Prompt 生成（会加载 5.5GB caption）")
        print("=" * 60)
        prompt = loader.format_prompt(first_row)
        print(f"✅ Prompt 生成成功")
        print(f"   长度: {len(prompt)} 字符")
        print(f"   前 200 字符: {prompt[:200]}...")

        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_loading()
