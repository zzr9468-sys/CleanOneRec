#!/bin/bash
# Resume Experiment B from step_34599 on GPU 0
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

cd /data/zhouziren/ms/CleanOneRec
mkdir -p logs outputs/exp_B_full_eepo

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Resuming Experiment B from step_34599 on GPU 0"

uv run python train.py \
    --config configs/exp_B_full_eepo.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp_B_full_eepo_resume \
    --output ./outputs/exp_B_full_eepo \
    --seed 42 \
    --resume-from ./outputs/exp_B_full_eepo/step_34399 \
    2>&1 | tee logs/exp_B_resume.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment B resume finished."
