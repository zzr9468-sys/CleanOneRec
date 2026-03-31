#!/bin/bash
# =============================================================
# Experiment D — EEPO 消融：关闭 Direction 3
# EEPO（Dir1+2+4 开，Dir3 关）+ 乘法 Reward
# 量化定向 Unlearn/Remember 的净贡献
# GPU: 7  预计时间: 周四～周五
# =============================================================

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_D_ablate_no_dir3

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Experiment D: Ablation - No Direction 3 (no directed unlearn)"

uv run python train.py \
    --config configs/exp_D_ablate_no_dir3.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp_D_ablate_no_dir3 \
    --output ./outputs/exp_D_ablate_no_dir3 \
    --seed 42 \
    2>&1 | tee logs/exp_D_ablate_no_dir3.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment D finished. Log: logs/exp_D_ablate_no_dir3.log"
