"""
完整功能测试（不加载 caption）
"""

import sys
sys.path.insert(0, '/Users/zhouziren/onerec/CleanOneRec')

from recrl.data.recif_loader import RecIFDataLoader


def test_complete():
    """完整功能测试"""
    print("=" * 60)
    print("完整功能测试")
    print("=" * 60)

    # 1. 初始化（不加载 caption）
    print("\n1. 初始化 RecIFDataLoader...")
    loader = RecIFDataLoader(
        "/Users/zhouziren/onerec/OpenOneRec-RecIF",
        load_captions=False
    )

    # 2. 获取训练数据
    print("\n2. 获取训练数据...")
    train_df = loader.get_train_data()
    print(f"   训练数据: {len(train_df)} 条")

    # 3. 测试第一条数据
    print("\n3. 测试第一条数据...")
    row = train_df.iloc[0]

    # 3.1 目标 PID
    target_pids = loader.format_target(row)
    print(f"   ✅ Target PIDs: {len(target_pids)} 个 - {target_pids[:3]}...")

    # 3.2 Longview 历史
    longview_history = loader.get_longview_history(row)
    print(f"   ✅ Longview History: {len(longview_history)} 个 - {longview_history[:3]}...")

    # 3.3 行为信号
    behavior_signals = loader.get_behavior_signals(row)
    print(f"   ✅ Behavior Signals: {behavior_signals}")

    # 4. 转换为 RecRL 格式（只测试 5 条，不加载 caption）
    print("\n4. 转换为 RecRL 格式（前 5 条，不生成 prompt）...")
    data = []
    for i in range(min(5, len(train_df))):
        row = train_df.iloc[i]
        data.append({
            'target_pids': loader.format_target(row),
            'longview_history': loader.get_longview_history(row),
            'behavior_signals': loader.get_behavior_signals(row),
        })

    print(f"   ✅ 转换成功: {len(data)} 条")

    # 5. 显示第一条
    print("\n5. 第一条数据示例:")
    print(f"   Target PIDs: {data[0]['target_pids'][:3]}...")
    print(f"   Longview History: {len(data[0]['longview_history'])} 个")
    print(f"   Behavior Signals: {data[0]['behavior_signals']}")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)

    print("\n📊 数据统计:")
    print(f"   总数据: {len(loader.df)} 条")
    print(f"   训练集: {len(train_df)} 条")
    print(f"   验证集: {len(loader.get_val_data())} 条")
    print(f"   测试集: {len(loader.get_test_data())} 条")

    print("\n💡 提示:")
    print("   - 如需生成 prompt，调用 loader.format_prompt(row)")
    print("   - 这会触发加载 5.5GB 的 caption 数据（需要 1-2 分钟）")


if __name__ == "__main__":
    try:
        test_complete()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
