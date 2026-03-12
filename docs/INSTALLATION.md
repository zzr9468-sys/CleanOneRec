# Installation Guide

## System Requirements

- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher (for GPU training)
- **GPU Memory**:
  - Single GPU: 24GB+ (e.g., RTX 3090, RTX 4090, A5000)
  - Multi-GPU: 4x 24GB+ recommended
- **Disk Space**: 20GB+ for model and data

## Quick Installation with uv

We recommend using [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## Alternative: pip Installation

```bash
# Clone the repository
git clone https://github.com/zzr9468-sys/CleanOneRec.git
cd CleanOneRec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check RecRL
python -c "from recrl.core import BaseRLTrainer; print('RecRL installed successfully!')"
```

Expected output:
```
PyTorch: 2.x.x+cu118
CUDA: True
RecRL installed successfully!
```

## Data Preparation

### 1. Download RecIF Benchmark Data

```bash
# Download OpenOneRec-RecIF dataset
git clone https://huggingface.co/datasets/OpenOneRec/RecIF /path/to/RecIF

# Or download specific domain
wget https://huggingface.co/datasets/OpenOneRec/RecIF/resolve/main/benchmark_data/video/video_test.parquet
```

### 2. Prepare SID Trie (for Constrained Decoding)

The SID trie is built automatically from `sid2pid.json` in the RecIF metadata:

```python
from recrl.constraints import SIDTrie

# Build trie from RecIF metadata
trie = SIDTrie.from_recif(
    recif_path="/path/to/RecIF",
    tokenizer=tokenizer
)
```

## Multi-GPU Setup

For multi-GPU training, install Accelerate:

```bash
pip install accelerate

# Generate accelerate config
accelerate config
```

Or use our pre-configured `accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 4
gpu_ids: all
```

## Optional Dependencies

### SwanLab (Experiment Tracking)

```bash
pip install swanlab

# Login to SwanLab
swanlab login
```

### TensorBoard (Alternative Logging)

```bash
pip install tensorboard

# View logs
tensorboard --logdir outputs/
```

## Common Issues

### CUDA Out of Memory

**Solution 1**: Enable gradient checkpointing (already enabled in `train.py`)

**Solution 2**: Reduce batch size or num_generations:
```yaml
per_device_batch_size: 1  # Reduce from 2
num_generations: 2        # Reduce from 4
```

**Solution 3**: Use CPU offload for ref_model (already implemented)

### Import Error: No module named 'recrl'

Make sure you installed in editable mode:
```bash
pip install -e .
```

### Accelerate Not Found

Install accelerate for multi-GPU training:
```bash
pip install accelerate
```

### SwanLab Login Failed

Check your API key:
```bash
swanlab login --relogin
```

Or disable SwanLab:
```bash
python train.py --config configs/grpo_video.yaml --report-to none
```

## Next Steps

- [Training Guide](TRAINING.md) - Learn how to train models
- [Algorithm Documentation](ALGORITHMS.md) - Understand GRPO and EEPO
- [Benchmarks](BENCHMARKS.md) - See performance comparisons
