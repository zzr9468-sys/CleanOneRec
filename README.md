# CleanOneRec

A clean, modular, and highly extensible framework for LLM-based Recommendation, designed as a modern replacement for `MiniOneRec`. 

Instead of treating SFT, RL, and Evaluation as tangled scripts, `CleanOneRec` separates concerns into robust components (Data, Reward, Trainer, Evaluator), making it easy to experiment with new algorithms like **GRPO**, **EEPO**, and **DPO** without fighting the codebase.

## 🏗️ Architecture

```
CleanOneRec/
├── onerec/                         # Core Library
│   ├── data/                       # Unified Dataset Builders
│   │   └── builder.py              # Supports OpenOneRec-RecIF & Legacy CSV
│   ├── trainer/                    # Training Engines
│   │   └── eepo_trainer.py         # Advanced GRPO/EEPO Trainer
│   ├── reward/                     # Reward Modeling
│   │   ├── rule.py                 # Exact Match & NDCG rules
│   │   ├── semantic.py             # Sentence-Transformer text similarity
│   │   └── composite.py            # Combines semantic + novelty + penalties
│   ├── eval/                       # Evaluation & Metrics
│   │   └── evaluator.py            # HitRate & NDCG calculators
│   └── utils/                      # Utilities
│       ├── sid_helper.py           # SID parsing & hashing
│       └── logit_processor.py      # Trie-based Constrained Decoding
├── scripts/                        # Executable Entry Points
│   ├── train_sft.py                # Supervised Fine-Tuning (SFT) with LoRA
│   ├── train_rl.py                 # Reinforcement Learning (GRPO/EEPO)
│   └── evaluate.py                 # Inference & Generation with Constraints
└── run_experiment.sh               # Easy Bash Wrapper
```

## 🚀 Key Improvements over MiniOneRec

1. **Decoupled Trainers**: No more massive `minionerec_trainer.py` with infinite `if/else` branches. We use clean inheritance from TRL (e.g., `SFTTrainer`, `GRPOTrainer`).
2. **Robust Constrained Decoding**: Replaced the brittle array-based `ConstrainedLogitsProcessor` with a robust Prefix Tree (`SIDTrie`). This strictly enforces that the LLM *cannot* hallucinate invalid SIDs during inference and RL sampling.
3. **Unified Data Loading**: `DatasetBuilder` provides a single entry point for parsing both new `OpenOneRec-RecIF` (parquet/json) and old `Amazon` (csv) datasets.
4. **Pluggable Rewards**: Reward functions are now proper Python classes, making it trivial to chain them (e.g., `CompositeReward = Semantic + TailBonus + InvalidPenalty`).

## 🛠️ Usage Guide

### 1. Supervised Fine-Tuning (SFT)
Train the base recommendation model using LoRA:
```bash
python scripts/train_sft.py \
    --model_path "/path/to/base/model" \
    --train_file "/path/to/train.parquet" \
    --output_dir "./outputs/sft" \
    --use_lora True
```

### 2. Reinforcement Learning (GRPO / EEPO)
Align the model using Group Relative Policy Optimization or Fast-Weight EEPO. The Trie-based constraint ensures you don't get stuck at `-10` invalid penalty!
```bash
python scripts/train_rl.py \
    --model_path "./outputs/sft/final_checkpoint" \
    --train_file "/path/to/train.parquet" \
    --recif_path "/path/to/OpenOneRec-RecIF" \
    --output_dir "./outputs/rl" \
    --constrained_decoding True \
    --eepo_enabled True
```

### 3. Evaluation & Inference
Generate predictions using Beam Search with Constrained Decoding, and compute HitRate (HR) & NDCG:
```bash
python scripts/evaluate.py \
    --model_path "./outputs/rl/final_checkpoint" \
    --test_file "/path/to/test.parquet" \
    --recif_path "/path/to/OpenOneRec-RecIF" \
    --beam_size 20 \
    --constrained_decoding True
```

## 🔧 Installation
Ensure you have PyTorch, Transformers, TRL, PEFT, and Pandas installed:
```bash
pip install torch transformers trl peft datasets pandas pyarrow sentence-transformers fire
```
