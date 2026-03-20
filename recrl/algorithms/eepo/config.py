"""
EEPO Configuration
"""

from dataclasses import dataclass
from ...core.base_trainer import RLConfig


@dataclass
class EEPOConfig(RLConfig):
    """
    Configuration for EEPO (Explore-and-Evaluate Policy Optimization).

    EEPO extends GRPO with fast-weight exploration to escape local optima.

    Four improvement directions implemented:
      Direction 1 — Extended fast-weight scope  (eepo_expand_scope)
      Direction 2 — Dynamic stage1 ratio anneal  (eepo_stage1_anneal_steps)
      Direction 3 — Directed unlearn + remember  (eepo_recif_path, eepo_remember_tail)
      Direction 4 — Separate G1/G2 advantage     (eepo_separate_adv)
    """

    # ── Core EEPO ──────────────────────────────────────────────────────────
    eepo_enabled: bool = True

    # Starting exploitation ratio (Direction 2 anneals this toward eepo_stage1_ratio_end)
    eepo_stage1_ratio: float = 0.7

    eepo_unlearn_lr: float = 1e-5      # Learning rate for fast-weight SGD step
    eepo_unlearn_weight: float = 1.0   # Weight on unlearning loss
    eepo_epsilon: float = 1e-4         # Clamp prob away from 1.0 to avoid log(0)

    # ── Direction 1: Extended fast-weight mutation scope ───────────────────
    # When True, also mutate the last transformer block's v_proj in addition to
    # lm_head. This shifts the representation space itself, producing more
    # semantically meaningful exploration than logit-level noise alone.
    eepo_expand_scope: bool = False

    # ── Direction 2: Dynamic stage1 ratio annealing ────────────────────────
    # Linearly anneal stage1_ratio: eepo_stage1_ratio → eepo_stage1_ratio_end
    # over eepo_stage1_anneal_steps steps.
    # Rationale: start with more exploitation (model is weak), progressively shift
    # to more exploration (model is stable enough to benefit from divergence).
    eepo_stage1_ratio_end: float = 0.3    # Final exploitation ratio after annealing
    eepo_stage1_anneal_steps: int = 0     # 0 = fixed ratio (no annealing)

    # ── Direction 3: Directed unlearn target + bidirectional remember ──────
    # Use top-K high-frequency items as the unlearn target (instead of G1 rollout).
    # Optionally also add a "remember" loss that pulls the model toward tail items,
    # forming a bidirectional signal: push away from heads AND toward tails.
    eepo_recif_path: str = ""          # Path to RecIF data dir (same as reward's recif_path)
    eepo_head_k: int = 500             # Top-K most frequent items to unlearn
    eepo_remember_tail: bool = True    # Add remember loss toward low-frequency tail items
    eepo_remember_weight: float = 0.3  # Scale for remember loss relative to unlearn loss
    eepo_tail_k: int = 1000            # Bottom-K least frequent items to remember

    # ── Direction 4: Separate G1/G2 advantage normalisation ───────────────
    # Normalise G1 (exploitation) and G2 (exploration) advantages within their own
    # sub-groups rather than sharing a single group baseline. This prevents G2 from
    # always being disadvantaged relative to G1's higher-probability outputs.
    eepo_separate_adv: bool = True
    eepo_exploration_bonus: float = 0.1  # Flat bonus added to real G2 advantages

    # Optional: Add ground truth to generation batch to avoid reward collapse
    add_gt: bool = True
