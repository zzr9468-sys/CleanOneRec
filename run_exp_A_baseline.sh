#!/bin/bash
# =============================================================
# Experiment A — Baseline
# GRPO + Additive Reward（旧加法公式）
# GPU: 2  预计时间: 周一～周二
# =============================================================

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_A_baseline

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Experiment A: Baseline (GRPO + Additive Reward)"

uv run python train.py \
    --config configs/exp_A_baseline.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp_A_baseline \
    --output ./outputs/exp_A_baseline \
    --seed 42 \
    2>&1 | tee logs/exp_A_baseline.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment A finished. Log: logs/exp_A_baseline.log"
