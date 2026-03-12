# Algorithm Documentation

This document explains the core algorithms and techniques used in RecRL.

## Table of Contents

1. [GRPO (Group Relative Policy Optimization)](#grpo)
2. [EEPO (Explore-and-Evaluate Policy Optimization)](#eepo)
3. [Constrained Decoding](#constrained-decoding)
4. [Composite Reward System](#composite-reward-system)
5. [Memory Optimization](#memory-optimization)

---

## GRPO (Group Relative Policy Optimization)

### Overview

GRPO is a policy gradient method that normalizes advantages within generation groups, making training more stable and sample-efficient.

### Algorithm

For each prompt, generate `G` completions:

```
1. Generate G completions: {y₁, y₂, ..., yG} ~ π_θ(·|x)
2. Compute rewards: {r₁, r₂, ..., rG}
3. Normalize advantages within group:
   A_i = (r_i - mean(r)) / (std(r) + ε)
4. Compute policy gradient loss:
   L = -E[A_i · log π_θ(y_i|x)] + β · KL(π_θ || π_ref)
```

### Key Features

- **Group-wise normalization**: Reduces variance, improves stability
- **KL penalty**: Prevents policy from deviating too far from reference
- **No value function**: Simpler than PPO, no critic network needed

### Implementation

```python
# recrl/algorithms/grpo/trainer.py

def compute_loss(self, inputs: dict) -> torch.Tensor:
    # Forward pass through policy
    logits = self.model(input_ids, attention_mask).logits
    per_token_logps = selective_log_softmax(logits, target_ids)

    # KL divergence with reference model
    per_token_kl = torch.exp(ref_logps - per_token_logps) - \
                   (ref_logps - per_token_logps) - 1

    # Policy gradient loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * \
                     advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - self.config.beta * per_token_kl)

    # Average over valid tokens
    loss = (per_token_loss * completion_mask).sum(dim=1) / \
           completion_mask.sum(dim=1).clamp(min=1)

    return loss.mean()
```

### Hyperparameters

- **beta** (KL penalty): `0.04` (default)
  - Higher: More conservative, stays closer to reference
  - Lower: More aggressive, deviates more from reference

- **num_generations**: `4` to `16`
  - More: Better advantage estimation, slower training
  - Fewer: Faster training, noisier gradients

### When to Use GRPO

- Initial experiments and baselines
- When computational budget is limited
- When reward signal is clear and stable
- When you want proven, stable performance

---

## EEPO (Explore-and-Evaluate Policy Optimization)

### Overview

EEPO extends GRPO with **fast-weight exploration** to escape local optima and improve sample diversity.

### Motivation

**Problem**: GRPO samples all completions from current policy → limited exploration → may get stuck in local optima

**Solution**: EEPO uses two-stage generation:
1. **Exploitation**: Sample from current policy (like GRPO)
2. **Exploration**: Apply fast-weight mutation, then sample

### Fast-Weight Unlearning

The key innovation is **temporary weight modification** to encourage exploration:

```python
# Compute "unlearning" gradient
with torch.no_grad():
    logits = model(prompt_ids, completion_ids)
    loss = cross_entropy(logits, target_ids)

# Compute gradient w.r.t. lm_head
grad = torch.autograd.grad(loss, lm_head.weight)[0]

# Apply fast-weight update (temporary)
lm_head.weight.data -= unlearn_lr * grad

# Generate with mutated weights
exploration_completions = model.generate(...)

# Restore original weights
lm_head.weight.data += unlearn_lr * grad
```

This temporarily "unlearns" recent patterns, forcing the model to explore different outputs.

### Algorithm

```
For each prompt x:
  1. Exploitation (G₁ samples):
     y₁, ..., y_{G₁} ~ π_θ(·|x)

  2. Fast-weight mutation:
     θ' = θ - α · ∇_θ L_unlearn

  3. Exploration (G₂ samples):
     y_{G₁+1}, ..., y_G ~ π_θ'(·|x)

  4. Restore weights:
     θ ← θ

  5. Compute rewards and train (same as GRPO):
     L = -E[A_i · log π_θ(y_i|x)] + β · KL(π_θ || π_ref)
```

### Implementation

```python
# recrl/algorithms/eepo/trainer.py

def _prepare_inputs(self, batch):
    # Stage 1: Exploitation
    _, completion_ids_1, _ = self.rollout_engine.generate(
        prompts=prompts,
        model=self.model,
        num_return_sequences=g1  # 50% of generations
    )

    # Stage 2: Fast-weight mutation
    lm_head, original_weights = self.unlearner.apply_unlearn_update(
        model=self.model,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids_1,
    )

    # Stage 3: Exploration with mutated weights
    _, completion_ids_2, _ = self.rollout_engine.generate(
        prompts=prompts,
        model=self.model,
        num_return_sequences=g2  # 50% of generations
    )

    # Stage 4: Restore original weights
    self.unlearner.restore_weights(lm_head, original_weights)

    # Combine and compute loss (same as GRPO)
    ...
```

### Hyperparameters

- **eepo_stage1_ratio**: `0.5` (default)
  - `0.5`: Balanced exploitation/exploration
  - `0.7`: More exploitation, less exploration
  - `0.3`: More exploration, less exploitation

- **eepo_unlearn_lr**: `1.0e-3` (default)
  - Higher: Stronger exploration, may be unstable
  - Lower: Weaker exploration, more conservative

- **eepo_unlearn_weight**: `0.1` (default)
  - Controls unlearning strength

### When to Use EEPO

- When GRPO gets stuck in local optima
- When you need better exploration
- When reward collapse is a concern
- When you want more diverse recommendations
- For research on exploration methods

### GRPO vs EEPO Comparison

| Aspect | GRPO | EEPO |
|--------|------|------|
| **Generation** | Single-stage | Two-stage |
| **Exploration** | Limited | Enhanced |
| **Computational Cost** | 1x | ~2x |
| **Stability** | High | Medium |
| **Diversity** | Medium | High |
| **Local Optima** | May get stuck | Can escape |
| **Use Case** | Baselines, stable tasks | Research, exploration-critical tasks |

---

## Constrained Decoding

### Problem

LLMs may generate invalid item IDs (hallucinations), leading to:
- Invalid recommendations
- Reward computation errors
- Training instability

### Solution: Trie-Based Constrained Decoding

We use a **prefix tree (trie)** to enforce that only valid Semantic IDs (SIDs) can be generated.

### Semantic ID (SID) Format

```
<|sid_begin|><s_a_123><s_b_456><s_c_789><|sid_end|>
```

Where `(a, b, c)` are hash components: `hash_key = a × 8192² + b × 8192 + c`

### Trie Construction

```python
# recrl/constraints/trie.py

class SIDTrie:
    def insert(self, token_ids: list[int]):
        """Insert a valid SID token sequence into trie."""
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.is_end = True

    @classmethod
    def from_recif(cls, recif_path: str, tokenizer):
        """Build trie from sid2pid.json."""
        trie = cls()
        sid2pid = json.load(open(f"{recif_path}/sid2pid.json"))

        for hash_key_str in sid2pid.keys():
            key = int(hash_key_str)
            a = key // (8192 * 8192)
            b = (key % (8192 * 8192)) // 8192
            c = key % 8192

            sid_str = f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>"
            token_ids = tokenizer.encode(sid_str, add_special_tokens=False)
            trie.insert(token_ids)

        return trie
```

### Logits Processor

```python
# recrl/constraints/processor.py

class ConstrainedLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        """Modify logits to enforce trie constraints."""
        for i in range(batch_size):
            # Extract completion part
            completion_ids = input_ids[i, self.prompt_length:]

            # Get allowed next tokens from trie
            allowed_tokens = self.get_allowed_tokens_fn(completion_ids)

            if allowed_tokens:
                # Set disallowed tokens to -inf
                mask = torch.ones(vocab_size, dtype=torch.bool, device=scores.device)
                mask[list(allowed_tokens)] = False
                scores[i, mask] = float('-inf')

        return scores
```

### Performance Impact

**Without constrained decoding**:
- Reward: -0.19
- Positive reward rate: 48.9%
- Many invalid SIDs generated

**With constrained decoding**:
- Reward: 0.78
- Positive reward rate: 99.7%
- All SIDs are valid

**Improvement**: 488% reward increase!

### Usage

```python
# In train.py
if args.constrained:
    trie = SIDTrie.from_recif(RECIF_PATH, tokenizer)
    rollout_engine.set_trie(trie)
```

Or use `--constrained` flag:
```bash
python train.py --config configs/grpo_video.yaml --constrained
```

---

## Composite Reward System

### Overview

RecRL supports combining multiple reward functions with custom weights.

### Reward Components

#### 1. Longview Reward (Relevance)

Measures alignment with user's long-term viewing history.

```python
# recrl/rewards/longview_reward.py

def compute_reward(self, prompt, completion, longview_history):
    # Extract first SID from completion
    sid = extract_first_sid(completion)
    if not sid:
        return -1.0  # Invalid SID penalty

    # Get item embedding
    item_emb = self.get_item_embedding(sid)

    # Get history embeddings
    history_embs = [self.get_item_embedding(h) for h in longview_history]

    # Compute similarity
    similarities = [cosine_similarity(item_emb, h_emb) for h_emb in history_embs]

    return max(similarities)  # Best match with history
```

#### 2. Semantic Reward (Content Similarity)

Measures semantic similarity between generated item and target item.

```python
# recrl/rewards/semantic.py

def compute_reward(self, prompt, completion, target_sid):
    # Extract SIDs
    pred_sid = extract_first_sid(completion)

    # Get captions
    pred_caption = self.sid_to_caption[pred_sid]
    target_caption = self.sid_to_caption[target_sid]

    # Compute sentence embedding similarity
    pred_emb = self.sentence_model.encode(pred_caption)
    target_emb = self.sentence_model.encode(target_caption)

    return cosine_similarity(pred_emb, target_emb)
```

#### 3. Novelty Reward (Exploration)

Penalizes items that appear in user's history (encourages exploration).

```python
# recrl/rewards/novelty_reward.py

def compute_reward(self, prompt, completion, longview_history):
    pred_sid = extract_first_sid(completion)

    # Check if item is in history
    if pred_sid in longview_history:
        return 0.0  # No novelty
    else:
        return 1.0  # Novel item
```

#### 4. Diversity Reward (Intra-Group)

Rewards diversity within the generation group.

```python
# recrl/rewards/diversity_reward.py

def compute_reward(self, completions_group):
    # Extract all SIDs in group
    sids = [extract_first_sid(c) for c in completions_group]

    # Count unique SIDs
    unique_ratio = len(set(sids)) / len(sids)

    return unique_ratio
```

### Composite Reward

```python
# recrl/core/reward.py

class CompositeReward:
    def __init__(self, reward_components: list[tuple[BaseReward, float]]):
        self.components = reward_components
        assert sum(w for _, w in reward_components) == 1.0

    def __call__(self, prompts, completions, **kwargs):
        total_rewards = []

        for prompt, completion in zip(prompts, completions):
            reward = 0.0
            for reward_fn, weight in self.components:
                r = reward_fn(prompt, completion, **kwargs)
                reward += weight * r
            total_rewards.append(reward)

        return total_rewards
```

### Configuration

```yaml
reward_weights:
  longview:  0.50  # Relevance to history
  semantic:  0.30  # Content similarity
  novelty:   0.15  # Exploration bonus
  diversity: 0.05  # Intra-group diversity
```

### Tuning Reward Weights

**Emphasize relevance** (conservative recommendations):
```yaml
reward_weights:
  longview:  0.60
  semantic:  0.30
  novelty:   0.05
  diversity: 0.05
```

**Emphasize exploration** (diverse recommendations):
```yaml
reward_weights:
  longview:  0.40
  semantic:  0.20
  novelty:   0.25
  diversity: 0.15
```

---

## Memory Optimization

### Techniques

#### 1. Gradient Checkpointing

Trade compute for memory by recomputing activations during backward pass.

```python
# In train.py
model.gradient_checkpointing_enable()
```

**Savings**: ~40% GPU memory
**Cost**: ~20% slower training

**Important**: Disable during generation (breaks KV cache):
```python
# In base_trainer.py
grad_ckpt_was_enabled = model.is_gradient_checkpointing
if grad_ckpt_was_enabled:
    model.gradient_checkpointing_disable()

# Generate...

if grad_ckpt_was_enabled:
    model.gradient_checkpointing_enable()
```

#### 2. Reference Model CPU Offload

Keep reference model on CPU, temporarily move to GPU for inference.

```python
# In base_trainer.py
self.ref_model.to("cpu")
self.ref_model.eval()

# During training step:
self.ref_model.to(self.device)  # Temporarily move to GPU
ref_logps = compute_ref_logprobs(...)
self.ref_model.to("cpu")  # Move back to CPU
torch.cuda.empty_cache()
```

**Savings**: ~3.4GB GPU memory (for 1.7B model)
**Cost**: Minimal (ref model only used once per step)

#### 3. Mixed Precision (bf16)

Use bfloat16 for forward/backward pass.

```python
# In accelerate_config.yaml
mixed_precision: bf16
```

**Savings**: ~50% memory
**Cost**: Negligible (modern GPUs have bf16 hardware)

### Memory Budget

For **OneRec-1.7B** on **24GB GPU**:

| Component | Memory |
|-----------|--------|
| Model weights | 3.4GB |
| Optimizer states | 6.8GB |
| Activations (batch=2, seq=512) | 8GB |
| Reference model (CPU) | 0GB |
| **Total** | **~18GB** |

With gradient checkpointing: **~12GB** (fits comfortably on 24GB GPU)

---

## Summary

- **GRPO**: Stable, proven policy gradient method with group-wise advantage normalization
- **EEPO**: Enhanced exploration via fast-weight unlearning, better for escaping local optima
- **Constrained Decoding**: Trie-based enforcement of valid SIDs, 488% reward improvement
- **Composite Rewards**: Flexible combination of relevance, semantic, novelty, and diversity
- **Memory Optimization**: Gradient checkpointing + ref model CPU offload enables training on 24GB GPUs

For more details, see:
- [Training Guide](TRAINING.md)
- [Benchmarks](BENCHMARKS.md)
- [Installation](INSTALLATION.md)
