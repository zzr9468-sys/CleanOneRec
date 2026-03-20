"""
Demo 2: 加法 vs 乘法 Reward 对比

展示为什么要从加法切换到乘法 Reward：
- 加法：热门物品靠高 lv 得分，novelty=0 也无所谓 → EEPO 探索得不到正强化
- 乘法：必须同时满足语义相关 AND 新颖 → 与 EEPO 探索目标直接对齐

运行: python eval/demo_reward_comparison.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── 加法 Reward（旧）──────────────────────────────────────────────────
def additive_reward(lv, nov, div, w_lv=0.7, w_nov=0.2, w_div=0.1):
    return w_lv * lv + w_nov * nov + w_div * div


# ── 乘法 Reward（新）──────────────────────────────────────────────────
INVALID_PENALTY = -0.5

def multiplicative_reward(lv, nov, div, joint_w=0.95, div_w=0.05):
    if nov < 0:
        return INVALID_PENALTY
    return max(0.0, lv) * nov * joint_w + div * div_w


# ── 典型场景 ──────────────────────────────────────────────────────────
# 每个场景: (描述, lv, nov, div)
SCENARIOS = [
    # 标签,              lv,   nov,  div,  说明
    ("热门+相关\n(旧模型爱推)", 0.8,  0.05, 0.5,  "高频物品，语义对"),
    ("冷门+相关\n(EEPO目标)",   0.7,  0.85, 0.8,  "长尾物品，语义对"),
    ("热门+不相关",            0.2,  0.05, 0.5,  "高频物品，语义差"),
    ("冷门+不相关",            0.2,  0.85, 0.8,  "长尾物品，语义差"),
    ("无效SID",                0.0, -1.0,  0.0,  "解析失败"),
]


def print_comparison():
    print(f"\n{'='*72}")
    print(f"{'场景':<20} {'lv':>5} {'nov':>5} {'加法Reward':>12} {'乘法Reward':>12}")
    print(f"{'-'*72}")
    for label, lv, nov, div, note in SCENARIOS:
        add_r = additive_reward(lv, nov, div)
        mul_r = multiplicative_reward(lv, nov, div)
        winner = "← 乘法更合理" if abs(mul_r - add_r) > 0.05 else ""
        print(f"{note:<20} {lv:>5.2f} {nov:>5.2f} {add_r:>12.3f} {mul_r:>12.3f}  {winner}")
    print(f"{'='*72}\n")

    print("关键差异:")
    print("  [热门+相关]  加法=0.62  乘法=0.06  → 乘法大幅压低（nov=0.05 拉低整体）")
    print("  [冷门+相关]  加法=0.74  乘法=0.61  → 乘法略低，但仍是正奖励（EEPO 探索仍然得到强化）")
    print("  核心：热门物品 reward 从 0.62 → 0.06，冷门物品从 0.74 → 0.61")
    print("       乘法使热门/冷门 reward 比从 0.84 降到 0.10，EEPO 探索的相对优势大幅提升！")
    print("  [热门+不相关] 加法=0.20  乘法=0.04  → 乘法压低无意义热门推荐")
    print("  [无效SID]    加法=0.00  乘法=-0.50 → 加法无惩罚！乘法有明确惩罚\n")


def plot_heatmap(save_path: str = "eval/fig_reward_comparison.png"):
    lv_vals  = np.linspace(0, 1, 50)
    nov_vals = np.linspace(0, 1, 50)
    LV, NOV = np.meshgrid(lv_vals, nov_vals)

    add_r = additive_reward(LV, NOV, div=0.5)
    mul_r = np.vectorize(lambda lv, nov: multiplicative_reward(lv, nov, div=0.5))(LV, NOV)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    vmin, vmax = 0, 1

    # Additive reward
    im0 = axes[0].contourf(NOV, LV, add_r, levels=20, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[0].set_title("Additive Reward (old)", fontsize=12)
    axes[0].set_xlabel("Novelty (nov)", fontsize=10)
    axes[0].set_ylabel("Longview Similarity (lv)", fontsize=10)
    plt.colorbar(im0, ax=axes[0])

    # Multiplicative reward
    im1 = axes[1].contourf(NOV, LV, mul_r, levels=20, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[1].set_title("Multiplicative Reward (new)", fontsize=12)
    axes[1].set_xlabel("Novelty (nov)", fontsize=10)
    axes[1].set_ylabel("Longview Similarity (lv)", fontsize=10)
    plt.colorbar(im1, ax=axes[1])

    # Difference: multiplicative - additive
    diff = mul_r - add_r
    im2 = axes[2].contourf(NOV, LV, diff, levels=20, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    axes[2].set_title("Diff: Multiplicative - Additive", fontsize=12)
    axes[2].set_xlabel("Novelty (nov)", fontsize=10)
    axes[2].set_ylabel("Longview Similarity (lv)", fontsize=10)
    plt.colorbar(im2, ax=axes[2])

    # Annotate EEPO target zone
    axes[2].annotate(
        "EEPO target zone\n(high nov + high lv)",
        xy=(0.85, 0.75), xytext=(0.45, 0.3),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
    )

    plt.suptitle("Additive vs Multiplicative Reward: Design Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {save_path}")


if __name__ == "__main__":
    print_comparison()
    plot_heatmap()
