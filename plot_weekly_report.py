"""
Generate figures for weekly report:
  fig1: exp_B_status_v2.png  — B training curves (updated, 89.2%)
  fig2: exp_C_status.png     — C training curves (82.1%)
  fig3: exp_ABC_compare.png  — A vs B vs C aligned on prompt count (key figure)
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os

OUT_DIR = "Project_Docs"   # relative to CleanOneRec/
os.makedirs(OUT_DIR, exist_ok=True)

# ── regex ───────────────────────────────────────────────────────────────────
step_re_add  = re.compile(r'\[INFO\] step=(\d+)\s+loss=([-\d.]+)\s+reward=([\d.]+)±([\d.]+)\s+lr=([\de.\-+]+)')
step_re_mul  = re.compile(r'\[INFO\] step=(\d+)\s+loss=([-\d.]+)\s+reward=([-\d.]+)±([\d.]+)\s+lr=([\de.\-+]+)')
rwd_add_re   = re.compile(r'\[Reward\(additive\) step=(\d+)\]\s+lv=([-\d.]+)\s+nov=([\d.]+)\s+div=([\d.]+)\s+total=([-\d.]+)')
rwd_mul_re   = re.compile(r'\[Reward\(multiplicative\) step=(\d+)\]\s+lv=([-\d.]+)\s+nov=([\d.]+)\s+div=([\d.]+)\s+total=([-\d.]+)')

def parse_log(path, reward_type='additive'):
    steps, losses, rewards, reward_stds, lrs = [], [], [], [], []
    r_steps, lvs, novs, divs, totals = [], [], [], [], []
    rwd_re = rwd_add_re if reward_type == 'additive' else rwd_mul_re
    with open(path) as f:
        for line in f:
            m = step_re_mul.search(line)
            if m and '[INFO] step=' in line:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                rewards.append(float(m.group(3)))
                reward_stds.append(float(m.group(4)))
                lrs.append(float(m.group(5)))
                continue
            m = rwd_re.search(line)
            if m:
                r_steps.append(int(m.group(1)))
                lvs.append(float(m.group(2)))
                novs.append(float(m.group(3)))
                divs.append(float(m.group(4)))
                totals.append(float(m.group(5)))
    return (np.array(steps), np.array(losses), np.array(rewards),
            np.array(reward_stds), np.array(lrs),
            np.array(r_steps), np.array(lvs), np.array(novs),
            np.array(divs), np.array(totals))

def smooth(x, w=200):
    if len(x) < w:
        w = max(1, len(x) // 5)
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='valid')

def smooth_steps(s, w=200):
    if len(s) < w:
        w = max(1, len(s) // 5)
    return s[w-1:]

COLORS = {
    'nov':   '#e07b39',
    'lv':    '#7b5ea7',
    'div':   '#3a9a6e',
    'total': '#c0392b',
    'loss':  '#c0392b',
    'reward':'#2471a3',
}

# ══════════════════════════════════════════════════════════════════════════════
# Parse logs
# ══════════════════════════════════════════════════════════════════════════════
print("Parsing logs...")
A = parse_log("logs/exp_A_baseline.log",   reward_type='additive')
B = parse_log("logs/exp_B_full_eepo.log",  reward_type='multiplicative')
C = parse_log("logs/exp_C_grpo_multi.log", reward_type='multiplicative')

A_steps, A_loss, A_rwd, A_rwd_std, A_lr, A_rs, A_lv, A_nov, A_div, A_tot = A
B_steps, B_loss, B_rwd, B_rwd_std, B_lr, B_rs, B_lv, B_nov, B_div, B_tot = B
C_steps, C_loss, C_rwd, C_rwd_std, C_lr, C_rs, C_lv, C_nov, C_div, C_tot = C

print(f"A: {len(A_steps)} steps, last={A_steps[-1] if len(A_steps) else 'N/A'}")
print(f"B: {len(B_steps)} steps, last={B_steps[-1] if len(B_steps) else 'N/A'}")
print(f"C: {len(C_steps)} steps, last={C_steps[-1] if len(C_steps) else 'N/A'}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Exp B status (updated)
# ══════════════════════════════════════════════════════════════════════════════
print("Drawing Fig 1: Exp B status...")

B_total_steps = 38781
B_pct = B_steps[-1] / B_total_steps * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"Experiment B — Full EEPO  (multiplicative reward) | "
    f"INTERRUPTED by server restart  [{B_steps[-1]}/{B_total_steps}  {B_pct:.1f}%]",
    fontsize=13, fontweight='bold', color='#c0392b'
)

# reward
ax = axes[0, 0]
w = 300
last_k = min(1000, len(B_rwd))
rwd_mean_last = np.mean(B_rwd[-last_k:])
ax.plot(B_steps, B_rwd, color='#f0a030', alpha=0.2, linewidth=0.4)
if len(B_rwd) >= w:
    ax.plot(smooth_steps(B_steps, w), smooth(B_rwd, w),
            color='#e67e22', linewidth=1.8, label='reward (smooth)')
ax.axhline(rwd_mean_last, color='#e67e22', linestyle='--', linewidth=1.2,
           label=f'last-{last_k} mean = {rwd_mean_last:.3f}')
ax.set_title('Reward Mean  (multiplicative, not comparable to A)', fontsize=11)
ax.set_xlabel('Step'); ax.set_ylabel('Reward')
ax.set_ylim(0.4, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# reward components vs A reference
ax = axes[0, 1]
w2 = 40
B_nov_last = float(np.mean(B_nov[-30:])) if len(B_nov) >= 30 else float(B_nov[-1])
B_lv_last  = float(np.mean(B_lv[-30:]))  if len(B_lv)  >= 30 else float(B_lv[-1])
B_div_last = float(np.mean(B_div[-30:])) if len(B_div) >= 30 else float(B_div[-1])
A_nov_ref = float(np.mean(A_nov[-50:])) if len(A_nov) >= 50 else float(A_nov[-1])
A_lv_ref  = float(np.mean(A_lv[-50:]))  if len(A_lv)  >= 50 else float(A_lv[-1])

for arr, rs, col, lbl, ref in [
    (B_nov, B_rs, COLORS['nov'], f'nov  cur={B_nov_last:.3f}', A_nov_ref),
    (B_lv,  B_rs, COLORS['lv'],  f'lv   cur={B_lv_last:.3f}',  A_lv_ref),
    (B_div, B_rs, COLORS['div'], f'div  cur={B_div_last:.3f}', None),
]:
    ax.plot(rs, arr, color=col, alpha=0.2, linewidth=0.5)
    if len(arr) >= w2:
        ax.plot(smooth_steps(rs, w2), smooth(arr, w2),
                color=col, linewidth=1.8, label=lbl)
    if ref is not None:
        ax.axhline(ref, color=col, linestyle=':', linewidth=1.1, alpha=0.6)

ax.axhline(A_nov_ref, color=COLORS['nov'], linestyle=':', linewidth=1, alpha=0.5,
           label=f'A nov ref={A_nov_ref:.3f}')
ax.axhline(A_lv_ref,  color=COLORS['lv'],  linestyle=':', linewidth=1, alpha=0.5,
           label=f'A lv  ref={A_lv_ref:.3f}')
ax.set_title('Reward Components  (solid=B, dotted=A final)', fontsize=11)
ax.set_xlabel('Step'); ax.set_ylabel('Value')
ax.set_ylim(0, 1.15); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# loss
ax = axes[1, 0]
w3 = 500
ax.plot(B_steps, B_loss, color='#e74c3c', alpha=0.15, linewidth=0.4)
if len(B_loss) >= w3:
    ax.plot(smooth_steps(B_steps, w3), smooth(B_loss, w3),
            color='#c0392b', linewidth=1.8, label='loss (smooth)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Training Loss  (negative values normal in GRPO)', fontsize=11)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.set_ylim(-0.6, 0.8); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# summary text box
ax = axes[1, 1]
ax.axis('off')
summary = (
    f"Summary  (vs Experiment A)\n"
    f"{'─'*38}\n"
    f"Progress : {B_steps[-1]:,} / {B_total_steps:,}   ({B_pct:.1f}%)\n"
    f"Interrupted by server restart\n"
    f"Last ckpt: outputs/exp_B_full_eepo/step_34599\n\n"
    f"reward (last 1000) : {np.mean(B_rwd[-1000:]):.4f}  [diff scale]\n"
    f"nov    (last 50)   : {B_nov_last:.4f}  ({B_nov_last - A_nov_ref:+.3f} vs A)\n"
    f"lv     (last 50)   : {B_lv_last:.4f}  ({B_lv_last - A_lv_ref:+.3f} vs A)\n"
    f"div    (last 50)   : {B_div_last:.4f}  ({B_div_last - 0.839:+.3f} vs A)\n\n"
    f"Key signals:\n"
    f"  nov > A  (+{B_nov_last - A_nov_ref:.3f}) => long-tail novelty improved\n"
    f"  lv  ≈ A  ({B_lv_last - A_lv_ref:+.3f})  => relevance maintained\n"
    f"  div << A => diversity problem persists\n\n"
    f"=> Resume from step_34599 next week"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.9, edgecolor='#e67e22'))

plt.tight_layout()
out = os.path.join(OUT_DIR, "exp_B_status_v2.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Exp C status
# ══════════════════════════════════════════════════════════════════════════════
print("Drawing Fig 2: Exp C status...")

C_total_steps = 19391
C_pct = C_steps[-1] / C_total_steps * 100

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    f"Experiment C — GRPO + Multiplicative Reward  (no EEPO) | "
    f"INTERRUPTED by server restart  [{C_steps[-1]}/{C_total_steps}  {C_pct:.1f}%]",
    fontsize=13, fontweight='bold', color='#c0392b'
)

# reward
ax = axes[0, 0]
w = 200
last_k = min(500, len(C_rwd))
rwd_mean_last_C = np.mean(C_rwd[-last_k:])
ax.plot(C_steps, C_rwd, color='#76b7d4', alpha=0.25, linewidth=0.4)
if len(C_rwd) >= w:
    ax.plot(smooth_steps(C_steps, w), smooth(C_rwd, w),
            color='#2980b9', linewidth=1.8, label='reward (smooth)')
ax.axhline(rwd_mean_last_C, color='#2980b9', linestyle='--', linewidth=1.2,
           label=f'last-{last_k} mean = {rwd_mean_last_C:.3f}')
ax.set_title('Reward Mean  (multiplicative)', fontsize=11)
ax.set_xlabel('Step'); ax.set_ylabel('Reward')
ax.set_ylim(0.4, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

# reward components vs A reference
ax = axes[0, 1]
w2 = 30
C_nov_last = float(np.mean(C_nov[-30:])) if len(C_nov) >= 30 else float(C_nov[-1])
C_lv_last  = float(np.mean(C_lv[-30:]))  if len(C_lv)  >= 30 else float(C_lv[-1])
C_div_last = float(np.mean(C_div[-30:])) if len(C_div) >= 30 else float(C_div[-1])

for arr, rs, col, lbl, ref in [
    (C_nov, C_rs, COLORS['nov'], f'nov  cur={C_nov_last:.3f}', A_nov_ref),
    (C_lv,  C_rs, COLORS['lv'],  f'lv   cur={C_lv_last:.3f}',  A_lv_ref),
    (C_div, C_rs, COLORS['div'], f'div  cur={C_div_last:.3f}', None),
]:
    ax.plot(rs, arr, color=col, alpha=0.2, linewidth=0.5)
    if len(arr) >= w2:
        ax.plot(smooth_steps(rs, w2), smooth(arr, w2),
                color=col, linewidth=1.8, label=lbl)

ax.axhline(A_nov_ref, color=COLORS['nov'], linestyle=':', linewidth=1, alpha=0.6,
           label=f'A nov ref={A_nov_ref:.3f}')
ax.axhline(A_lv_ref,  color=COLORS['lv'],  linestyle=':', linewidth=1, alpha=0.6,
           label=f'A lv  ref={A_lv_ref:.3f}')
ax.set_title('Reward Components  (solid=C, dotted=A final)', fontsize=11)
ax.set_xlabel('Step'); ax.set_ylabel('Value')
ax.set_ylim(0, 1.15); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)

# loss
ax = axes[1, 0]
w3 = 300
ax.plot(C_steps, C_loss, color='#e74c3c', alpha=0.2, linewidth=0.4)
if len(C_loss) >= w3:
    ax.plot(smooth_steps(C_steps, w3), smooth(C_loss, w3),
            color='#c0392b', linewidth=1.8, label='loss (smooth)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Training Loss', fontsize=11)
ax.set_xlabel('Step'); ax.set_ylabel('Loss')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# summary
ax = axes[1, 1]
ax.axis('off')
summary = (
    f"Summary  (vs Experiment A)\n"
    f"{'─'*38}\n"
    f"Progress : {C_steps[-1]:,} / {C_total_steps:,}   ({C_pct:.1f}%)\n"
    f"Interrupted by server restart\n"
    f"Last ckpt: outputs/exp_C_grpo_multi/step_15799\n\n"
    f"reward (last 500)  : {rwd_mean_last_C:.4f}  [diff scale]\n"
    f"nov    (last 30)   : {C_nov_last:.4f}  ({C_nov_last - A_nov_ref:+.3f} vs A)\n"
    f"lv     (last 30)   : {C_lv_last:.4f}  ({C_lv_last - A_lv_ref:+.3f} vs A)\n"
    f"div    (last 30)   : {C_div_last:.4f}  ({C_div_last - 0.839:+.3f} vs A)\n\n"
    f"Key signals:\n"
    f"  nov >> A  ({C_nov_last - A_nov_ref:+.3f}) => strong novelty improvement\n"
    f"  lv  >  A  ({C_lv_last - A_lv_ref:+.3f}) => relevance also improved\n"
    f"  div << A  => same div issue as B (reward design)\n\n"
    f"=> GRPO + mult-reward alone looks strong\n"
    f"=> Resume from step_15799 next week"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#eaf4fb', alpha=0.9, edgecolor='#2980b9'))

plt.tight_layout()
out = os.path.join(OUT_DIR, "exp_C_status.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — A vs B vs C comparison (prompt-aligned, key figure)
# ══════════════════════════════════════════════════════════════════════════════
print("Drawing Fig 3: A vs B vs C comparison...")

# Align on prompt count
# A: batch=2 → prompts = step × 2
# B: batch=1 → prompts = step × 1
# C: batch=1 → prompts = step × 1
A_prompts   = A_steps * 2
A_r_prompts = A_rs   * 2
B_prompts   = B_steps
B_r_prompts = B_rs
C_prompts   = C_steps
C_r_prompts = C_rs

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(
    "A vs B vs C — Training Metrics Aligned on Prompts Seen\n"
    "A: GRPO+additive  |  B: Full EEPO+multiplicative  |  C: GRPO+multiplicative (no EEPO)",
    fontsize=13, fontweight='bold'
)

pal = {
    'A': '#2471a3',   # blue
    'B': '#e67e22',   # orange
    'C': '#27ae60',   # green
}

smooth_w = {'reward': 300, 'nov': 40, 'lv': 40, 'div': 40}

# ── Panel 1: nov (novelty) ──
ax = axes[0]
for name, prompts, r_prompts, arr, col in [
    ('A (Baseline, additive)',       A_r_prompts, A_r_prompts, A_nov, pal['A']),
    ('B (Full EEPO, mult)',           B_r_prompts, B_r_prompts, B_nov, pal['B']),
    ('C (GRPO+mult, no EEPO)',        C_r_prompts, C_r_prompts, C_nov, pal['C']),
]:
    w = smooth_w['nov']
    ax.plot(r_prompts, arr, color=col, alpha=0.15, linewidth=0.5)
    if len(arr) >= w:
        ax.plot(smooth_steps(r_prompts, w), smooth(arr, w),
                color=col, linewidth=2.2, label=name)

ax.set_title('Novelty (nov) — Long-tail Coverage', fontsize=11, fontweight='bold')
ax.set_xlabel('Prompts Seen', fontsize=10)
ax.set_ylabel('nov Score', fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(alpha=0.3)
# annotate final values
for name, arr, col in [('A', A_nov, pal['A']), ('B', B_nov, pal['B']), ('C', C_nov, pal['C'])]:
    val = float(np.mean(arr[-30:]))
    ax.annotate(f'{name}: {val:.3f}',
                xy=(1.0, val), xycoords=('axes fraction', 'data'),
                xytext=(3, 0), textcoords='offset points',
                fontsize=8, color=col, va='center')

# ── Panel 2: lv (relevance) ──
ax = axes[1]
for name, r_prompts, arr, col in [
    ('A (Baseline, additive)',  A_r_prompts, A_lv, pal['A']),
    ('B (Full EEPO, mult)',      B_r_prompts, B_lv, pal['B']),
    ('C (GRPO+mult, no EEPO)',   C_r_prompts, C_lv, pal['C']),
]:
    w = smooth_w['lv']
    ax.plot(r_prompts, arr, color=col, alpha=0.15, linewidth=0.5)
    if len(arr) >= w:
        ax.plot(smooth_steps(r_prompts, w), smooth(arr, w),
                color=col, linewidth=2.2, label=name)

ax.set_title('Relevance (lv) — Semantic Alignment', fontsize=11, fontweight='bold')
ax.set_xlabel('Prompts Seen', fontsize=10)
ax.set_ylabel('lv Score', fontsize=10)
ax.set_ylim(0.5, 1.05)
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(alpha=0.3)
for name, arr, col in [('A', A_lv, pal['A']), ('B', B_lv, pal['B']), ('C', C_lv, pal['C'])]:
    val = float(np.mean(arr[-30:]))
    ax.annotate(f'{name}: {val:.3f}',
                xy=(1.0, val), xycoords=('axes fraction', 'data'),
                xytext=(3, 0), textcoords='offset points',
                fontsize=8, color=col, va='center')

# ── Panel 3: div (diversity) ──
ax = axes[2]
for name, r_prompts, arr, col in [
    ('A (Baseline, additive)',  A_r_prompts, A_div, pal['A']),
    ('B (Full EEPO, mult)',      B_r_prompts, B_div, pal['B']),
    ('C (GRPO+mult, no EEPO)',   C_r_prompts, C_div, pal['C']),
]:
    w = smooth_w['div']
    ax.plot(r_prompts, arr, color=col, alpha=0.15, linewidth=0.5)
    if len(arr) >= w:
        ax.plot(smooth_steps(r_prompts, w), smooth(arr, w),
                color=col, linewidth=2.2, label=name)

ax.set_title('Diversity (div) — Unique SID Rate', fontsize=11, fontweight='bold')
ax.set_xlabel('Prompts Seen', fontsize=10)
ax.set_ylabel('div Score', fontsize=10)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(alpha=0.3)
# highlight the div problem
ax.axhspan(0, 0.55, alpha=0.06, color='red')
ax.text(2000, 0.25, '⚠ div low in B & C\n(mult-reward design issue)',
        fontsize=8, color='#c0392b', style='italic')
for name, arr, col in [('A', A_div, pal['A']), ('B', B_div, pal['B']), ('C', C_div, pal['C'])]:
    val = float(np.mean(arr[-30:]))
    ax.annotate(f'{name}: {val:.3f}',
                xy=(1.0, val), xycoords=('axes fraction', 'data'),
                xytext=(3, 0), textcoords='offset points',
                fontsize=8, color=col, va='center')

# vertical lines marking where B and C are interrupted
for ax_i in axes:
    ax_i.axvline(B_prompts[-1], color=pal['B'], linestyle=':', linewidth=1.2, alpha=0.7)
    ax_i.axvline(C_prompts[-1], color=pal['C'], linestyle=':', linewidth=1.2, alpha=0.7)

axes[0].text(B_prompts[-1] + 200, 0.52, 'B interrupted', color=pal['B'], fontsize=7.5, rotation=90)
axes[0].text(C_prompts[-1] + 200, 0.52, 'C interrupted', color=pal['C'], fontsize=7.5, rotation=90)

plt.tight_layout()
out = os.path.join(OUT_DIR, "exp_ABC_compare.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {out}")

print("\nAll figures generated.")
