#!/bin/bash
# EEPO training script (single GPU)

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2  # Use GPU 2

cd /data/zhouziren/ms/CleanOneRec

uv run python train.py \
    --config configs/eepo_video.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp-eepo \
    --output ./outputs/exp-eepo \
    2>&1 | tee logs/train_eepo.log
