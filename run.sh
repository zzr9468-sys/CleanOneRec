#!/bin/bash
# 启动 CleanOneRec 强化学习训练

# 设置环境变量，强制使用离线wandb或者指定你的key
export WANDB_MODE="offline"

# 你的 SFT 模型路径 (请替换为真实路径)
MODEL_PATH="/path/to/your/sft/model"

# 数据路径
RECIF_PATH="/Users/zhouziren/onerec/OpenOneRec-RecIF"
TRAIN_FILE="$RECIF_PATH/benchmark_data/video/video_test.parquet"

# 运行脚本
/opt/anaconda3/bin/python train.py \
    --model_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --recif_path "$RECIF_PATH" \
    --output_dir "./outputs/eepo_semantic_grpo" \
    --wandb_project "CleanOneRec" \
    --wandb_run_name "eepo_semantic_run1" \
    --sample_train 2000 \
    --eepo_enabled True \
    --eepo_stage1_ratio 0.5 \
    --add_gt True \
    --temperature 0.7 \
    --num_generations 16 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --beta 0.04
