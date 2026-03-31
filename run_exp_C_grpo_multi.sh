#!/bin/bash
# =============================================================
# Experiment C — GRPO + Multiplicative Reward（无 EEPO）
# 核心问题：乘法 reward 本身够不够？EEPO 是否冗余？
# GPU: 5  预计时间: 周三～周四
# =============================================================

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=5

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_C_grpo_multi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Experiment C: GRPO + Multiplicative Reward (no EEPO)"

uv run python train.py \
    --config configs/exp_C_grpo_multi.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp_C_grpo_multi \
    --output ./outputs/exp_C_grpo_multi \
    --seed 42 \
    2>&1 | tee logs/exp_C_grpo_multi.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment C finished. Log: logs/exp_C_grpo_multi.log"
