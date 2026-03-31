"""
Parse exp_A_baseline.log and plot training curves.
"""
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_PATH = "logs/exp_A_baseline.log"
OUT_PATH = "logs/exp_A_curves.png"

# ── Parse ──────────────────────────────────────────────────────────────────
step_re    = re.compile(r'\[INFO\] step=(\d+)\s+loss=([\d.]+)\s+reward=([\d.]+)±([\d.]+)\s+lr=([\de.\-+]+)')
reward_re  = re.compile(r'\[Reward\(additive\) step=(\d+)\]\s+lv=([\d.]+)\s+nov=([\d.]+)\s+div=([\d.]+)\s+total=([\d.]+)')

steps, losses, rewards, reward_stds, lrs = [], [], [], [], []
r_steps, lvs, novs, divs, totals = [], [], [], [], []

with open(LOG_PATH) as f:
    for line in f:
        m = step_re.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            rewards.append(float(m.group(3)))
            reward_stds.append(float(m.group(4)))
            lrs.append(float(m.group(5)))
            continue
        m = reward_re.search(line)
        if m:
            r_steps.append(int(m.group(1)))
            lvs.append(float(m.group(2)))
            novs.append(float(m.group(3)))
            divs.append(float(m.group(4)))
            totals.append(float(m.group(5)))

steps       = np.array(steps)
losses      = np.array(losses)
rewards     = np.array(rewards)
reward_stds = np.array(reward_stds)
lrs         = np.array(lrs)
r_steps     = np.array(r_steps)
lvs         = np.array(lvs)
novs        = np.array(novs)
divs        = np.array(divs)
totals      = np.array(totals)

print(f"Parsed {len(steps)} step entries, {len(r_steps)} reward-detail entries")
print(f"Step range: {steps[0]} ~ {steps[-1]}")
print(f"Latest: loss={losses[-1]:.4f}  reward={rewards[-1]:.3f}  lr={lrs[-1]:.2e}")

def smooth(x, w=200):
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

def smooth_steps(s, w=200):
    return s[w-1:]

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle(f"Exp A — Baseline (GRPO + Additive Reward)\nstep {steps[0]}~{steps[-1]} / 19391  ({steps[-1]/19391*100:.1f}%)",
             fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

# 1. Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(steps, losses, color='#aec6e8', alpha=0.3, linewidth=0.5, label='raw')
ax1.plot(smooth_steps(steps), smooth(losses), color='#1f77b4', linewidth=1.8, label=f'smooth(w=200)')
ax1.set_title('Training Loss', fontsize=12)
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# 2. Reward mean ± std
ax2 = fig.add_subplot(gs[0, 1])
ax2.fill_between(steps,
                 np.clip(rewards - reward_stds, 0, 1),
                 np.clip(rewards + reward_stds, 0, 1),
                 color='#ffbb78', alpha=0.25, label='±std')
ax2.plot(steps, rewards, color='#ffbb78', alpha=0.3, linewidth=0.5)
ax2.plot(smooth_steps(steps), smooth(rewards), color='#ff7f0e', linewidth=1.8, label='reward (smooth)')
ax2.set_title('Reward Mean ± Std', fontsize=12)
ax2.set_xlabel('Step')
ax2.set_ylabel('Reward')
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# 3. Reward components (lv, nov, div, total) — from detailed reward lines
ax3 = fig.add_subplot(gs[1, 0])
w2 = 50
colors = {'lv': '#2ca02c', 'nov': '#d62728', 'div': '#9467bd', 'total': '#8c564b'}
for name, arr, color in [('longview', lvs, colors['lv']),
                          ('novelty',  novs, colors['nov']),
                          ('diversity',divs, colors['div']),
                          ('total',    totals, colors['total'])]:
    ax3.plot(r_steps, arr, alpha=0.2, color=color, linewidth=0.5)
    if len(arr) >= w2:
        ax3.plot(r_steps[w2-1:], smooth(arr, w2), color=color, linewidth=1.8, label=name)
ax3.set_title('Reward Components (Additive)', fontsize=12)
ax3.set_xlabel('Step')
ax3.set_ylabel('Score')
ax3.set_ylim(0, 1.1)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# 4. LR schedule
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(steps, lrs, color='#17becf', linewidth=1.5)
ax4.set_title('Learning Rate Schedule', fontsize=12)
ax4.set_xlabel('Step')
ax4.set_ylabel('LR')
ax4.grid(alpha=0.3)
# annotate current value
ax4.annotate(f"current: {lrs[-1]:.2e}", xy=(steps[-1], lrs[-1]),
             xytext=(-80, 20), textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=9, color='#17becf')

plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to {OUT_PATH}")
