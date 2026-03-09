#!/bin/bash

# Configuration
export WANDB_MODE="offline"

MODEL_PATH="/path/to/your/sft/model"
RECIF_PATH="/Users/zhouziren/onerec/OpenOneRec-RecIF"
TRAIN_FILE="$RECIF_PATH/benchmark_data/video/video_test.parquet"

# Run Script
/opt/anaconda3/bin/python scripts/train.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --recif_path "$RECIF_PATH" \
    --output_dir "./outputs/run_01" \
    --wandb_project "OneRecRL_Experiments" \
    --wandb_run_name "run_01" \
    --sample_train 2000 \
    --eepo_enabled True \
    --add_gt True \
    --temperature 0.7
