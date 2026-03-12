#!/usr/bin/env python3
"""
Inference script for trained RecRL models.

Usage:
    python inference.py \
        --model-path ./outputs/exp-constrained/final \
        --recif-path /path/to/RecIF \
        --prompt "User history: [item1, item2, ...]. Recommend next item:" \
        --num-return 5 \
        --constrained
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from recrl.constraints import SIDTrie


def parse_args():
    p = argparse.ArgumentParser(description="Inference with trained RecRL model")
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to trained model checkpoint")
    p.add_argument("--recif-path", type=str, required=True,
                   help="Path to RecIF metadata")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single prompt for inference")
    p.add_argument("--prompt-file", type=str, default=None,
                   help="File containing prompts (one per line)")
    p.add_argument("--num-return", type=int, default=5,
                   help="Number of recommendations to generate")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature")
    p.add_argument("--max-length", type=int, default=128,
                   help="Maximum generation length")
    p.add_argument("--constrained", action="store_true",
                   help="Use constrained decoding (recommended)")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to use (cuda/cpu)")
    p.add_argument("--output", type=str, default=None,
                   help="Output file for results (JSON)")
    return p.parse_args()


def load_model_and_tokenizer(model_path, device):
    """Load trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device}")
    return model, tokenizer


def build_trie(recif_path, tokenizer):
    """Build SID trie for constrained decoding."""
    print("Building SID trie...")
    trie = SIDTrie.from_recif(recif_path, tokenizer)
    print(f"Trie built with {len(trie.root.children)} root nodes")
    return trie


def get_allowed_tokens_fn(trie, tokenizer):
    """Create function to get allowed tokens from trie."""
    def fn(completion_ids):
        return trie.get_next_tokens(completion_ids.tolist())
    return fn


def generate_recommendations(
    model,
    tokenizer,
    prompt,
    num_return=5,
    temperature=0.7,
    max_length=128,
    constrained=False,
    trie=None,
    device="cuda"
):
    """Generate recommendations for a single prompt."""
    # Tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)

    prompt_length = inputs["input_ids"].shape[1]

    # Setup logits processor for constrained decoding
    logits_processor = None
    if constrained and trie is not None:
        from recrl.constraints import ConstrainedLogitsProcessor
        logits_processor = [ConstrainedLogitsProcessor(
            get_allowed_tokens_fn=lambda ids: trie.get_next_tokens(ids.tolist()),
            prompt_length=prompt_length
        )]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            num_return_sequences=num_return,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )

    # Decode completions
    completions = []
    for output in outputs:
        completion_ids = output[prompt_length:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
        completions.append(completion)

    return completions


def extract_sid(completion):
    """Extract SID from completion."""
    import re
    pattern = r'<\|sid_begin\|>(.*?)<\|sid_end\|>'
    match = re.search(pattern, completion)
    if match:
        return match.group(0)
    return None


def sid_to_item_info(sid, recif_path):
    """Get item information from SID."""
    # Load sid2pid mapping
    sid2pid_path = f"{recif_path}/sid2pid.json"
    with open(sid2pid_path) as f:
        sid2pid = json.load(f)

    # Parse SID to get hash key
    import re
    pattern = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'
    match = re.search(pattern, sid)
    if not match:
        return None

    a, b, c = map(int, match.groups())
    hash_key = a * 8192 * 8192 + b * 8192 + c

    # Get product ID
    pid = sid2pid.get(str(hash_key))
    if not pid:
        return None

    # Load metadata (if available)
    try:
        meta_path = f"{recif_path}/video_meta.json"
        with open(meta_path) as f:
            metadata = json.load(f)
        item_info = metadata.get(pid, {})
        item_info["pid"] = pid
        return item_info
    except FileNotFoundError:
        return {"pid": pid}


def main():
    args = parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Build trie if constrained
    trie = None
    if args.constrained:
        trie = build_trie(args.recif_path, tokenizer)

    # Get prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Must provide --prompt or --prompt-file")

    # Generate recommendations
    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {i+1}/{len(prompts)}:")
        print(f"{prompt[:100]}...")

        completions = generate_recommendations(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_return=args.num_return,
            temperature=args.temperature,
            max_length=args.max_length,
            constrained=args.constrained,
            trie=trie,
            device=args.device,
        )

        print(f"\nGenerated {len(completions)} recommendations:")
        result = {
            "prompt": prompt,
            "recommendations": []
        }

        for j, completion in enumerate(completions):
            sid = extract_sid(completion)
            item_info = sid_to_item_info(sid, args.recif_path) if sid else None

            print(f"\n  [{j+1}] {completion[:80]}...")
            if sid:
                print(f"      SID: {sid}")
            if item_info:
                print(f"      PID: {item_info.get('pid')}")
                if 'title' in item_info:
                    print(f"      Title: {item_info['title'][:50]}...")

            result["recommendations"].append({
                "rank": j + 1,
                "completion": completion,
                "sid": sid,
                "item_info": item_info,
            })

        all_results.append(result)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
