"""
Demo 1: 头部/尾部物品分布分析

展示为什么需要 Directed Unlearn：
- 少数头部物品垄断了训练集的大多数交互
- EEPO 应该主动压制这些物品，鼓励探索长尾

运行: python eval/demo_data_analysis.py --recif_path /path/to/OpenOneRec-RecIF
"""

import json
import argparse
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_item_counts(recif_path: str):
    sid2pid_path = Path(recif_path) / "benchmark_data" / "sid2pid.json"
    print(f"Loading {sid2pid_path} ...")
    with open(sid2pid_path) as f:
        sid2pid = json.load(f)

    counts = {}
    for hk_str, pid_list in sid2pid.items():
        total = sum(
            item.get("count_after_downsample", item.get("count", 1))
            for item in pid_list
        )
        counts[hk_str] = total

    return counts


def print_stats(counts: dict):
    vals = sorted(counts.values(), reverse=True)
    n = len(vals)
    total = sum(vals)

    print(f"\n{'='*50}")
    print(f"  总物品数: {n:,}")
    print(f"  总交互数: {total:,}")
    print(f"  中位数频次: {np.median(vals):.1f}")
    print(f"  平均频次:   {np.mean(vals):.1f}")
    print(f"{'='*50}")

    for pct in [1, 5, 10, 20]:
        k = max(1, int(n * pct / 100))
        head_sum = sum(vals[:k])
        print(f"  Top {pct:2d}% ({k:,} 个物品) 占总交互的 {head_sum/total*100:.1f}%")

    long_tail = sum(1 for v in vals if v <= 10)
    print(f"\n  频次 ≤ 10 的物品: {long_tail:,} ({long_tail/n*100:.1f}%)")
    print(f"  (这些是目标：EEPO 应该学会推荐它们)")
    print(f"{'='*50}\n")


def plot_distribution(counts: dict, save_path: str = "eval/fig_item_distribution.png"):
    vals = sorted(counts.values(), reverse=True)
    n = len(vals)
    total = sum(vals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── 左图：频次分布（log scale）────────────────────────────────
    ax = axes[0]
    freq_counter = collections.Counter(vals)
    x = sorted(freq_counter.keys())
    y = [freq_counter[k] for k in x]
    ax.bar(x[:50], y[:50], color="#4C72B0", alpha=0.8, width=0.8)
    ax.set_xlabel("Item frequency (count)", fontsize=11)
    ax.set_ylabel("Number of items", fontsize=11)
    ax.set_title("Item Frequency Distribution (low-freq range)", fontsize=12)
    ax.axvline(x=10, color="red", linestyle="--", alpha=0.7, label="freq=10 (long-tail boundary)")
    ax.legend()

    # ── Right: Lorenz Curve ───────────────────────────────────────
    ax = axes[1]
    cumsum = np.cumsum(vals) / total
    x_axis = np.arange(1, n + 1) / n * 100
    ax.plot(x_axis, cumsum * 100, color="#4C72B0", linewidth=2, label="Actual distribution")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Perfect equality")
    ax.fill_between(x_axis, cumsum * 100, x_axis, alpha=0.15, color="#4C72B0")

    idx_10 = int(n * 0.1)
    ax.annotate(
        f"Top 10% items\n= {cumsum[idx_10]*100:.0f}% of interactions",
        xy=(10, cumsum[idx_10] * 100),
        xytext=(25, 55),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )
    ax.set_xlabel("Item rank percentile", fontsize=11)
    ax.set_ylabel("Cumulative interaction share (%)", fontsize=11)
    ax.set_title("Lorenz Curve: Interaction Inequality", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recif_path", default="/data/zhouziren/ms/OpenOneRec-RecIF")
    parser.add_argument("--save", default="eval/fig_item_distribution.png")
    args = parser.parse_args()

    counts = load_item_counts(args.recif_path)
    print_stats(counts)
    plot_distribution(counts, args.save)

    # ── 展示 EEPO head/tail corpus 大小 ──────────────────────────
    vals_sorted = sorted(counts.values(), reverse=True)
    head_k, tail_k = 500, 1000
    head_total = sum(vals_sorted[:head_k])
    tail_total = sum(vals_sorted[-tail_k:])
    grand_total = sum(vals_sorted)

    print(f"EEPO Directed Unlearn 目标（Top-{head_k}）：")
    print(f"  占训练集交互总数的 {head_total/grand_total*100:.1f}%")
    print(f"EEPO Remember 目标（Bottom-{tail_k}）：")
    print(f"  占训练集交互总数的 {tail_total/grand_total*100:.2f}%")
    print(f"\n→ Unlearn 500 个头部物品可以抑制 ~{head_total/grand_total*100:.0f}% 的交互，")
    print(f"  为长尾 {tail_k} 个物品腾出探索空间。")


if __name__ == "__main__":
    main()
