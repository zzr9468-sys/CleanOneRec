#!/usr/bin/env python3
"""
Check if training is making progress by analyzing SwanLab logs or training metrics.
"""

import re
import sys
from pathlib import Path


def parse_log_file(log_path: str):
    """Parse training log and extract reward metrics."""
    rewards = []
    steps = []

    with open(log_path) as f:
        for line in f:
            # Match lines like: "step=  50  loss=0.0000  reward=0.012±1.397  lr=1.00e-06"
            match = re.search(r'step=\s*(\d+)\s+loss=.*?reward=([-\d.]+)±', line)
            if match:
                step = int(match.group(1))
                reward_mean = float(match.group(2))
                steps.append(step)
                rewards.append(reward_mean)

    return steps, rewards


def compute_moving_average(values, window=10):
    """Compute moving average."""
    if len(values) < window:
        return values

    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(sum(values[start:i+1]) / (i - start + 1))
    return ma


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs/train_exp006.log"

    if not Path(log_path).exists():
        print(f"❌ Log file not found: {log_path}")
        return

    steps, rewards = parse_log_file(log_path)

    if len(rewards) < 10:
        print(f"⚠️  Only {len(rewards)} steps logged, need more data")
        return

    print(f"📊 Training Progress Analysis")
    print(f"=" * 60)
    print(f"Total steps: {len(steps)}")
    print(f"Latest step: {steps[-1]}")
    print()

    # Recent stats
    recent_20 = rewards[-20:] if len(rewards) >= 20 else rewards
    print(f"Recent 20 steps:")
    print(f"  Mean reward: {sum(recent_20)/len(recent_20):.3f}")
    print(f"  Min reward:  {min(recent_20):.3f}")
    print(f"  Max reward:  {max(recent_20):.3f}")
    print()

    # Compare first 20 vs last 20
    if len(rewards) >= 40:
        first_20 = rewards[:20]
        last_20 = rewards[-20:]
        first_mean = sum(first_20) / len(first_20)
        last_mean = sum(last_20) / len(last_20)
        improvement = last_mean - first_mean

        print(f"First 20 steps mean: {first_mean:.3f}")
        print(f"Last 20 steps mean:  {last_mean:.3f}")
        print(f"Improvement:         {improvement:+.3f}")
        print()

        if improvement > 0.1:
            print("✅ Reward is improving! Training is working.")
        elif improvement > -0.1:
            print("⚠️  Reward is stable. May need more steps to see improvement.")
        else:
            print("❌ Reward is decreasing. Check hyperparameters.")
    else:
        print("⚠️  Need at least 40 steps to compare progress")

    print()
    print(f"💡 Tips:")
    print(f"  - GRPO loss ≈ 0 is normal (policy gradient method)")
    print(f"  - Focus on reward trend, not loss value")
    print(f"  - Need 200+ steps to see clear improvement")
    print(f"  - Check SwanLab for detailed curves: https://swanlab.cn/@arenmy/CleanOneRec")


if __name__ == "__main__":
    main()
