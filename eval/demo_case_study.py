"""
Demo 3: G1 vs G2 Case Study

对同一批用户 prompt 同时跑 G1（当前策略）和 G2（Fast-Weight 变异后），
对比生成物品的：
  1. 物品频次（是否更长尾）
  2. 物品多样性（SID 去重率）
  3. 实际推荐内容（如果有 pid2caption 则打印标题）

运行:
  python eval/demo_case_study.py \
    --model_path /path/to/model \
    --recif_path /path/to/OpenOneRec-RecIF \
    --data_path  /path/to/train.parquet \
    --n_prompts  8
"""

import re
import json
import argparse
import collections
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SID_PATTERN = re.compile(r"<\|sid_begin\|>.*?<\|sid_end\|>")


# ── 数据 helpers ──────────────────────────────────────────────────────

def load_prompts(data_path: str, n: int, seed: int = 42):
    import pandas as pd
    df = pd.read_parquet(data_path).sample(n=n, random_state=seed)
    prompts = []
    for _, row in df.iterrows():
        msgs = row.get("messages", row.get("prompt", ""))
        if isinstance(msgs, list):
            # messages format: [{"role": ..., "content": ...}]
            text = "\n".join(m["content"] for m in msgs if m.get("role") != "assistant")
        else:
            text = str(msgs)
        prompts.append(text[:1500])   # truncate to avoid OOM
    return prompts


def load_sid_freq(recif_path: str):
    p = Path(recif_path) / "benchmark_data" / "sid2pid.json"
    with open(p) as f:
        sid2pid = json.load(f)
    freq = {}
    for hk_str, pid_list in sid2pid.items():
        freq[hk_str] = sum(
            item.get("count_after_downsample", item.get("count", 1))
            for item in pid_list
        )
    return freq, sid2pid


def sid_to_hashkey(sid_str: str):
    m = re.findall(r"<s_[abc]_(\d+)>", sid_str)
    if len(m) == 3:
        a, b, c = int(m[0]), int(m[1]), int(m[2])
        return str(a * 8192 * 8192 + b * 8192 + c)
    return None


def get_caption(hk, sid2pid, pid2caption):
    if hk is None or hk not in sid2pid:
        return "[未知]"
    pid_list = sid2pid[hk]
    if not pid_list:
        return "[未知]"
    pid = str(pid_list[0]["pid"])
    return pid2caption.get(pid, "[无标题]")[:60]


def load_caption_dict(recif_path: str):
    import pandas as pd
    p = Path(recif_path) / "pid2caption.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return dict(zip(df["pid"].astype(str), df["dense_caption"]))


# ── 生成 helpers ──────────────────────────────────────────────────────

def generate_batch(model, tokenizer, prompts, n_seq, temperature, max_new_tokens, device):
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        padding_side="left", add_special_tokens=False,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=n_seq,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].size(1)
    completions = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return completions


def apply_fast_weight(model, tokenizer, prompts, unlearn_lr=1e-5, device="cuda"):
    """One-step SGD on lm_head to push model away from current outputs."""
    # Backup lm_head
    lm_head = model.lm_head
    backup = lm_head.weight.data.clone()

    # Get G1 completions as unlearn target
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        padding_side="left", add_special_tokens=False,
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=32,
            do_sample=False,   # greedy for unlearn target
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = out[:, enc["input_ids"].size(1):]
    comp_mask = (completion_ids != tokenizer.pad_token_id).int()
    full_ids = torch.cat([enc["input_ids"], completion_ids], dim=1)
    full_mask = torch.cat([enc["attention_mask"], comp_mask], dim=1)

    # Freeze all except lm_head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lm_head.parameters():
        p.requires_grad_(True)

    model.train()
    logits = model(input_ids=full_ids, attention_mask=full_mask).logits[:, :-1, :]
    from trl.trainer.utils import selective_log_softmax
    logps = selective_log_softmax(logits, full_ids[:, 1:])[:, -comp_mask.shape[1]:]
    denom = comp_mask.sum(dim=1).clamp(min=1)
    seq_logp = (logps * comp_mask).sum(dim=1) / denom
    prob = torch.exp(seq_logp).clamp(max=1 - 1e-4)
    loss = (-torch.log(1 - prob)).mean()

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=unlearn_lr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.eval()

    for p in model.parameters():
        p.requires_grad_(True)

    return lm_head, backup


# ── 分析 helpers ──────────────────────────────────────────────────────

def analyze_completions(completions, freq_dict, label):
    sids = [SID_PATTERN.findall(c) for c in completions]
    first_sids = [s[0] if s else None for s in sids]
    valid = [s for s in first_sids if s is not None]
    unique = set(valid)

    freqs = []
    for s in valid:
        hk = sid_to_hashkey(s)
        freqs.append(freq_dict.get(hk, 0) if hk else 0)

    print(f"\n  [{label}]")
    print(f"    生成数量:   {len(completions)}")
    print(f"    有效 SID:   {len(valid)} / {len(completions)} ({len(valid)/len(completions)*100:.0f}%)")
    print(f"    唯一 SID:   {len(unique)}")
    print(f"    多样性:     {len(unique)/len(valid)*100:.0f}%" if valid else "    多样性:     N/A")
    if freqs:
        long_tail = sum(1 for f in freqs if f <= 10)
        print(f"    平均频次:   {sum(freqs)/len(freqs):.1f}")
        print(f"    长尾比例:   {long_tail}/{len(freqs)} ({long_tail/len(freqs)*100:.0f}%，频次≤10)")

    return first_sids, freqs


def print_case_study(prompts, g1_sids, g2_sids, freq_dict, sid2pid, pid2caption):
    print(f"\n{'='*70}")
    print("  Case Study: G1（利用）vs G2（探索）逐条对比")
    print(f"{'='*70}")
    for i, prompt in enumerate(prompts):
        if i >= len(g1_sids) or i >= len(g2_sids):
            break
        s1 = g1_sids[i]
        s2 = g2_sids[i]
        hk1 = sid_to_hashkey(s1) if s1 else None
        hk2 = sid_to_hashkey(s2) if s2 else None
        f1 = freq_dict.get(hk1, 0) if hk1 else 0
        f2 = freq_dict.get(hk2, 0) if hk2 else 0
        cap1 = get_caption(hk1, sid2pid, pid2caption)
        cap2 = get_caption(hk2, sid2pid, pid2caption)
        print(f"\n  Prompt #{i+1}: {prompt[:80]}...")
        print(f"    G1 → {s1 or '[无效]':<55} 频次={f1:>6}  {cap1}")
        print(f"    G2 → {s2 or '[无效]':<55} 频次={f2:>6}  {cap2}")
        if f1 > 0 and f2 > 0:
            if f2 < f1:
                print(f"         ✓ G2 更长尾（频次降低 {(1-f2/f1)*100:.0f}%）")
            elif f2 > f1:
                print(f"         ! G2 更热门（频次升高）")
            else:
                print(f"         = 频次相同")


def plot_freq_comparison(g1_freqs, g2_freqs, save_path="eval/fig_g1_vs_g2.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 频次箱线图
    ax = axes[0]
    data = [g1_freqs, g2_freqs]
    bp = ax.boxplot(data, labels=["G1 (exploit)", "G2 (explore)"],
                    patch_artist=True, widths=0.5)
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Item frequency in training set", fontsize=11)
    ax.set_title("G1 vs G2: Item Frequency Distribution", fontsize=12)
    ax.set_yscale("log")

    # Long-tail ratio comparison
    ax = axes[1]
    thresholds = [1, 5, 10, 50, 100]
    g1_rates = [sum(1 for f in g1_freqs if f <= t) / max(len(g1_freqs), 1) * 100
                for t in thresholds]
    g2_rates = [sum(1 for f in g2_freqs if f <= t) / max(len(g2_freqs), 1) * 100
                for t in thresholds]
    x = np.arange(len(thresholds))
    w = 0.35
    ax.bar(x - w/2, g1_rates, w, label="G1 (exploit)", color="#4C72B0", alpha=0.8)
    ax.bar(x + w/2, g2_rates, w, label="G2 (explore)", color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"freq<={t}" for t in thresholds])
    ax.set_ylabel("Long-tail item ratio (%)", fontsize=11)
    ax.set_title("G2 should generate more long-tail items", fontsize=12)
    ax.legend()

    plt.suptitle("EEPO G1 (exploit) vs G2 (explore): Item Novelty Comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {save_path}")


# ── 主程序 ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--recif_path",   default="/data/zhouziren/ms/OpenOneRec-RecIF")
    parser.add_argument("--data_path",    required=True)
    parser.add_argument("--n_prompts",    type=int, default=8)
    parser.add_argument("--g1",           type=int, default=4)
    parser.add_argument("--g2",           type=int, default=4)
    parser.add_argument("--temperature",  type=float, default=0.9)
    parser.add_argument("--max_tokens",   type=int, default=64)
    parser.add_argument("--unlearn_lr",   type=float, default=1e-5)
    parser.add_argument("--save",         default="eval/fig_g1_vs_g2.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 加载模型
    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    prompts = load_prompts(args.data_path, args.n_prompts)
    print(f"Loaded {len(prompts)} prompts")

    # 加载物品数据
    freq_dict, sid2pid = load_sid_freq(args.recif_path)
    pid2caption = load_caption_dict(args.recif_path)
    print(f"Loaded {len(freq_dict):,} item frequencies")

    # ── G1：当前策略生成 ──────────────────────────────────────────
    print(f"\nGenerating G1 ({args.g1} per prompt) ...")
    g1_completions = generate_batch(
        model, tokenizer, prompts, args.g1, args.temperature, args.max_tokens, device
    )

    # ── Fast-Weight 变异 ──────────────────────────────────────────
    print(f"Applying Fast-Weight unlearn (lr={args.unlearn_lr}) ...")
    lm_head, backup = apply_fast_weight(model, tokenizer, prompts, args.unlearn_lr, device)

    # ── G2：变异后生成 ────────────────────────────────────────────
    print(f"Generating G2 ({args.g2} per prompt) ...")
    g2_completions = generate_batch(
        model, tokenizer, prompts, args.g2, args.temperature, args.max_tokens, device
    )

    # 恢复权重
    lm_head.weight.data.copy_(backup)
    print("Weights restored.")

    # ── 分析 ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  统计对比")
    print(f"{'='*70}")
    g1_sids, g1_freqs = analyze_completions(g1_completions, freq_dict, "G1 利用")
    g2_sids, g2_freqs = analyze_completions(g2_completions, freq_dict, "G2 探索")

    # Case study（每个 prompt 取第一个生成）
    first_g1 = g1_sids[::args.g1]
    first_g2 = g2_sids[::args.g2]
    print_case_study(prompts, first_g1, first_g2, freq_dict, sid2pid, pid2caption)

    # 图表
    if g1_freqs and g2_freqs:
        plot_freq_comparison(g1_freqs, g2_freqs, args.save)


if __name__ == "__main__":
    main()
