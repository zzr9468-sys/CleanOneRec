#!/usr/bin/env python3
"""
CleanOneRec Training Launcher

Single-GPU:
    python train.py --config configs/grpo_video.yaml

Multi-GPU (e.g. 4 cards):
    accelerate launch --num_processes 4 train.py --config configs/grpo_video.yaml

Config fields can be overridden via CLI:
    python train.py --config configs/grpo_video.yaml --lr 5e-7 --num_generations 8
"""

import argparse
import logging
import os
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
from recrl.core.data import DataEngine
from recrl.core.rollout import RolloutEngine
from recrl.rewards.recrl_composite_reward import RecRLCompositeReward
from recrl.constraints.trie import SIDTrie

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  type=str, default=None, help="YAML config file")
    p.add_argument("--model",   type=str, default=None, help="Model path")
    p.add_argument("--data",    type=str, default=None, help="RecIF parquet path")
    p.add_argument("--recif",   type=str, default=None, help="OpenOneRec-RecIF root")
    p.add_argument("--output",  type=str, default="./outputs")
    p.add_argument("--algo",    type=str, default="grpo", choices=["grpo", "eepo"])
    p.add_argument("--device",  type=str, default="cuda")
    p.add_argument("--epochs",  type=int, default=None)
    p.add_argument("--bs",      type=int, default=None, help="Batch size per device")
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--num-generations", type=int, default=None, dest="num_generations")
    p.add_argument("--samples", type=int, default=-1, help="Subset size (-1 = all)")
    p.add_argument("--report-to", type=str, default="none",
                   choices=["none", "swanlab", "tensorboard"], dest="report_to")
    p.add_argument("--run-name", type=str, default=None, dest="run_name")
    p.add_argument("--constrained", action="store_true",
                   help="Enable SID trie constrained decoding (slower but always valid)")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    cfg: dict = {}
    if args.config:
        cfg = load_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")

    # CLI overrides
    if args.model:           cfg["model_path"]       = args.model
    if args.data:            cfg["data_path"]        = args.data
    if args.recif:           cfg["recif_path"]       = args.recif
    if args.epochs:          cfg["num_epochs"]       = args.epochs
    if args.bs:              cfg["per_device_batch_size"] = args.bs
    if args.lr:              cfg["learning_rate"]    = args.lr
    if args.num_generations: cfg["num_generations"]  = args.num_generations

    MODEL_PATH = cfg.get("model_path",  "/data/zhouziren/ms/OneRec-1.7B")
    DATA_PATH  = cfg.get("data_path",   "/data/zhouziren/ms/OpenOneRec-RecIF/benchmark_data/video/video_test.parquet")
    RECIF_PATH = cfg.get("recif_path",  "/data/zhouziren/ms/OpenOneRec-RecIF")
    ALGO       = cfg.get("algo",        args.algo)
    DEVICE     = cfg.get("device",      args.device)

    # ── Training config ───────────────────────────────────────────────────
    common = dict(
        num_epochs              = cfg.get("num_epochs",             1),
        per_device_batch_size   = cfg.get("per_device_batch_size",  4),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 2),
        learning_rate           = cfg.get("learning_rate",          1e-6),
        warmup_steps            = cfg.get("warmup_steps",           0),
        max_grad_norm           = cfg.get("max_grad_norm",          1.0),
        num_generations         = cfg.get("num_generations",        4),
        max_completion_length   = cfg.get("max_completion_length",  128),
        temperature             = cfg.get("temperature",            0.7),
        beta                    = cfg.get("beta",                   0.04),
        logging_steps           = cfg.get("logging_steps",         1),
        save_steps              = cfg.get("save_steps",             100),
        output_dir              = args.output,
        report_to               = args.report_to,
        run_name                = args.run_name,
        device                  = DEVICE,
        seed                    = args.seed,
    )

    if ALGO == "grpo":
        train_config = GRPOConfig(**common)
    else:
        train_config = EEPOConfig(
            eepo_enabled         = cfg.get("eepo_enabled",      True),
            eepo_stage1_ratio    = cfg.get("eepo_stage1_ratio", 0.5),
            eepo_unlearn_lr      = cfg.get("eepo_unlearn_lr",   1e-5),
            eepo_unlearn_weight  = cfg.get("eepo_unlearn_weight", 1.0),
            eepo_epsilon         = cfg.get("eepo_epsilon",       1e-4),
            add_gt               = cfg.get("add_gt",            False),
            **common,
        )

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info(f"Loading dataset from {DATA_PATH}")
    train_dataset = DataEngine.from_parquet(
        DATA_PATH,
        format="recif",
        sample_num=args.samples,
        seed=args.seed,
    )
    logger.info(f"Dataset: {len(train_dataset)} samples")

    # ── Model & Tokenizer ─────────────────────────────────────────────────
    logger.info(f"Loading model from {MODEL_PATH}")
    dtype = torch.bfloat16 if "cuda" in DEVICE else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype, trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()  # Save GPU memory at cost of ~20% slower training

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype, trust_remote_code=True,
    )

    # ── Constrained decoding (optional) ───────────────────────────────────
    constraints = None
    if args.constrained:
        logger.info("Building SID trie for constrained decoding...")
        trie = SIDTrie.from_recif(RECIF_PATH, tokenizer)
        constraints = trie.get_allowed_next_tokens
        logger.info("SID trie ready")

    # ── Rollout engine ────────────────────────────────────────────────────
    rollout_engine = RolloutEngine(
        tokenizer=tokenizer,
        constraints=constraints,
        device=DEVICE,
    )

    # ── Reward engine ─────────────────────────────────────────────────────
    logger.info("Initializing reward engine...")
    reward_engine = RecRLCompositeReward(
        recif_path=RECIF_PATH,
        device=DEVICE,
        num_generations=train_config.num_generations,
        weights=cfg.get("reward_weights", None),
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    TrainerCls = GRPOTrainer if ALGO == "grpo" else EEPOTrainer
    logger.info(f"Starting {ALGO.upper()} training...")

    trainer = TrainerCls(
        model=model,
        ref_model=ref_model,
        config=train_config,
        train_dataset=train_dataset,
        rollout_engine=rollout_engine,
        reward_engine=reward_engine,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
