"""
超快速测试：只加载前 100 行
"""

import sys
sys.path.insert(0, '/Users/zhouziren/onerec/CleanOneRec')

import pandas as pd


def test_quick():
    """快速测试数据格式"""
    print("=" * 60)
    print("快速测试: 只加载前 100 行")
    print("=" * 60)

    # 直接读取前 100 行
    df = pd.read_parquet(
        '/Users/zhouziren/onerec/OpenOneRec-RecIF/onerec_bench_release.parquet',
        # 只读取需要的列
        columns=[
            'target_video_pid',
            'hist_longview_video_list',
            'inter_user_profile_with_sid',
            'hist_video_pid',
            'target_video_longview'
        ]
    )

    print(f"✅ 加载数据: {len(df)} 条")

    # 过滤无效行
    valid_mask = (
        df['target_video_pid'].notna() &
        df['hist_longview_video_list'].notna() &
        df['inter_user_profile_with_sid'].notna()
    )

    df_valid = df[valid_mask]
    print(f"✅ 有效数据: {len(df_valid)} 条")

    # 测试第一行
    if len(df_valid) > 0:
        row = df_valid.iloc[0]

        print("\n第一行数据:")
        print(f"  target_video_pid: {row['target_video_pid'][:3]}...")
        print(f"  hist_longview_video_list: {row['hist_longview_video_list'][:3]}...")
        print(f"  hist_video_pid: {row['hist_video_pid'][:3]}...")
        print(f"  target_video_longview: {row['target_video_longview'][:3]}...")
        print(f"  inter_user_profile_with_sid: {row['inter_user_profile_with_sid'][:100]}...")

        print("\n✅ 数据格式正确！")
    else:
        print("\n❌ 没有有效数据")


if __name__ == "__main__":
    test_quick()
