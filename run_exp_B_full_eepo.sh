#!/bin/bash
# =============================================================
# Experiment B — Full EEPO
# EEPO（4 个 Direction 全开）+ 乘法 Reward
# GPU: 7  预计时间: 周三～周四
# =============================================================

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_B_full_eepo

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Experiment B: Full EEPO (4 Directions + Multiplicative Reward)"

uv run python train.py \
    --config configs/exp_B_full_eepo.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp_B_full_eepo \
    --output ./outputs/exp_B_full_eepo \
    --seed 42 \
    2>&1 | tee logs/exp_B_full_eepo.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment B finished. Log: logs/exp_B_full_eepo.log"
