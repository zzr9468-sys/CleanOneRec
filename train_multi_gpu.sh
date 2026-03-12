#!/bin/bash
# Multi-GPU training script with accelerate

# Check if accelerate config exists
if [ ! -f "accelerate_config.yaml" ]; then
    echo "Creating accelerate config for 4 GPUs..."
    cat > accelerate_config.yaml << 'ACCEL_EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
ACCEL_EOF
fi

# Set environment variables
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2,3,5,6  # Use 4 free GPUs

# Launch training with accelerate
accelerate launch \
    --config_file accelerate_config.yaml \
    train.py \
    --config configs/grpo_video.yaml \
    --constrained \
    --report-to swanlab \
    --run-name exp-multi-gpu \
    --output ./outputs/exp-multi-gpu \
    "$@"
