#!/bin/bash
# Single-GPU training with constrained decoding

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2  # Use free GPU

nohup uv run python train.py \
    --config configs/grpo_video.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp-constrained-grpo \
    --output ./outputs/exp-constrained \
    > logs/train_constrained.log 2>&1 &

echo "Training started with PID=$!"
echo "Monitor with: tail -f logs/train_constrained.log"
