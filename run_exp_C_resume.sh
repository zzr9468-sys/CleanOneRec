#!/bin/bash
# Resume Experiment C from step_15799 on GPU 3
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=6

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_C_grpo_multi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming Experiment C from step_15799 on GPU 3"

uv run python train.py \
    --config configs/exp_C_grpo_multi.yaml \
    --constrained \
    --report-to none \
    --run-name exp_C_grpo_multi_resume \
    --output ./outputs/exp_C_grpo_multi \
    --seed 42 \
    --resume-from ./outputs/exp_C_grpo_multi/step_15799 \
    2>&1 | tee logs/exp_C_resume.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment C resume finished."
