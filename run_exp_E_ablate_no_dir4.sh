#!/bin/bash
# =============================================================
# Experiment E — EEPO 消融：关闭 Direction 4
# EEPO（Dir1+2+3 开，Dir4 关）+ 乘法 Reward
# 量化 G1/G2 独立 Advantage 归一化的净贡献
# GPU: 7  预计时间: 周四～周五（D 完成后接着跑）
# =============================================================

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_E_ablate_no_dir4

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Experiment E: Ablation - No Direction 4 (merged adv norm)"

uv run python train.py \
    --config configs/exp_E_ablate_no_dir4.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp_E_ablate_no_dir4 \
    --output ./outputs/exp_E_ablate_no_dir4 \
    --seed 42 \
    2>&1 | tee logs/exp_E_ablate_no_dir4.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment E finished. Log: logs/exp_E_ablate_no_dir4.log"
