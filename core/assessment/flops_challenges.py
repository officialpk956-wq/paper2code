"""
core/assessment/flops_challenges.py

Deterministic FLOPs challenges backed by FLOPsEngine calculations.
No LLM. All answers computed by AssessmentEngine.

Challenge scenarios:
  - Double channels       → new FLOPs
  - Increase image size   → new FLOPs
  - Add residual blocks   → delta FLOPs
  - Increase transformer heads → complexity class change

Every challenge returns:
  {
    "challenge_id": str,
    "question": str,
    "choices": list[str],
    "answer": str,
    "answer_index": int,
    "explanation": str,
    "difficulty": str,
    "computation": dict   ← exact numbers from FLOPsEngine formulas
  }
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


def _make_id(label: str) -> str:
    return "flops_" + hashlib.md5(label.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Core FLOPs formulas — mirroring FLOPsEngine exactly
# ---------------------------------------------------------------------------


def _conv2d_flops(C_in: int, C_out: int, K: int, H_out: int, W_out: int) -> float:
    """2 × C_in × C_out × K² × H_out × W_out  (multiply-adds × 2)"""
    return 2 * C_in * C_out * K * K * H_out * W_out / 1e6  # MFLOPs


def _mha_flops(N: int, D: int) -> float:
    """QKV: 3·2·N·D² + Scores: 2·N²·D + AV: 2·N²·D + Out: 2·N·D²"""
    qkv = 3 * 2 * N * D * D
    scores = 2 * N * N * D
    av = 2 * N * N * D
    out_p = 2 * N * D * D
    return (qkv + scores + av + out_p) / 1e6


def _residual_block_flops(C_in: int, C_out: int, H: int, W: int, K: int = 3) -> float:
    f1 = 2 * C_in * C_out * K * K * H * W
    f2 = 2 * C_out * C_out * K * K * H * W
    return (f1 + f2) / 1e6


def _fmt(mflops: float) -> str:
    """Format MFLOPs into human-readable string."""
    if mflops >= 1000:
        return f"{mflops / 1000:.1f} GFLOPs"
    return f"{mflops:.1f} MFLOPs"


# ---------------------------------------------------------------------------
# Challenge 1: Double channels
# ---------------------------------------------------------------------------


def _double_channels_challenge() -> dict[str, Any]:
    C_in, C_out, K, H, W = 64, 64, 3, 56, 56
    base_flops = _conv2d_flops(C_in, C_out, K, H, W)
    # Doubling both C_in and C_out → 4× more FLOPs (C_in×C_out term)
    new_C_in, new_C_out = C_in * 2, C_out * 2
    new_flops = _conv2d_flops(new_C_in, new_C_out, K, H, W)
    ratio = new_flops / base_flops

    answer = f"~{ratio:.0f}× more ({_fmt(new_flops)} vs {_fmt(base_flops)})"
    choices = [
        answer,
        f"~2× more ({_fmt(base_flops * 2)} vs {_fmt(base_flops)})",
        f"~8× more ({_fmt(base_flops * 8)} vs {_fmt(base_flops)})",
        "Same FLOPs — only parameter count changes.",
    ]
    random.shuffle(choices)

    return {
        "challenge_id": _make_id("double_channels_conv"),
        "question": (
            f"A Conv2D layer has C_in={C_in}, C_out={C_out}, kernel=3×3, output spatial size={H}×{W}. "
            f"Using FLOPs = 2×C_in×C_out×K²×H×W, it costs {_fmt(base_flops)}. "
            f"If both C_in and C_out are DOUBLED (to {new_C_in} and {new_C_out}), how much do FLOPs increase?"
        ),
        "choices": choices,
        "answer": answer,
        "answer_index": choices.index(answer),
        "explanation": (
            f"FLOPs ∝ C_in × C_out. Doubling both multiplies the product by 2×2 = 4×.\n"
            f"Base: {_fmt(base_flops)} → New: {_fmt(new_flops)} (exactly {ratio:.0f}× more).\n"
            f"Key insight: channel scaling has a quadratic effect on FLOPs."
        ),
        "difficulty": "beginner",
        "computation": {
            "formula": "2 × C_in × C_out × K² × H × W",
            "base": {"C_in": C_in, "C_out": C_out, "flops_mflops": round(base_flops, 2)},
            "modified": {"C_in": new_C_in, "C_out": new_C_out, "flops_mflops": round(new_flops, 2)},
            "ratio": round(ratio, 2),
        },
    }


# ---------------------------------------------------------------------------
# Challenge 2: Increase image size
# ---------------------------------------------------------------------------


def _increase_image_size_challenge() -> dict[str, Any]:
    C_in, C_out, K = 64, 64, 3
    H1, W1 = 56, 56
    H2, W2 = 112, 112
    base_flops = _conv2d_flops(C_in, C_out, K, H1, W1)
    new_flops = _conv2d_flops(C_in, C_out, K, H2, W2)
    ratio = new_flops / base_flops

    answer = f"~{ratio:.0f}× more ({_fmt(new_flops)} vs {_fmt(base_flops)})"
    choices = [
        answer,
        f"~{ratio / 2:.0f}× more (square root scaling applies)",
        "Same FLOPs — kernel size is unchanged",
        f"~{ratio * 2:.0f}× more (cubic scaling with spatial dims)",
    ]
    random.shuffle(choices)

    return {
        "challenge_id": _make_id("increase_image_size"),
        "question": (
            f"A Conv2D layer (C_in={C_in}, C_out={C_out}, kernel=3×3) operates on input spatial size {H1}×{W1} "
            f"({_fmt(base_flops)}). If the spatial size is doubled to {H2}×{W2}, "
            f"how do FLOPs change?"
        ),
        "choices": choices,
        "answer": answer,
        "answer_index": choices.index(answer),
        "explanation": (
            f"FLOPs ∝ H_out × W_out (linear in each dimension). Doubling both H and W → 4× more FLOPs.\n"
            f"Base: 2×{C_in}×{C_out}×9×{H1}×{W1} = {_fmt(base_flops)}\n"
            f"New:  2×{C_in}×{C_out}×9×{H2}×{W2} = {_fmt(new_flops)}\n"
            f"Ratio: {round(ratio, 1)}×"
        ),
        "difficulty": "intermediate",
        "computation": {
            "formula": "2 × C_in × C_out × K² × H × W",
            "base": {"H": H1, "W": W1, "flops_mflops": round(base_flops, 2)},
            "modified": {"H": H2, "W": W2, "flops_mflops": round(new_flops, 2)},
            "ratio": round(ratio, 2),
        },
    }


# ---------------------------------------------------------------------------
# Challenge 3: Add residual blocks
# ---------------------------------------------------------------------------


def _add_residual_blocks_challenge() -> dict[str, Any]:
    C, H, W = 128, 28, 28
    base_blocks = 2
    new_blocks = 4
    base_flops = _residual_block_flops(C, C, H, W) * base_blocks
    new_flops = _residual_block_flops(C, C, H, W) * new_blocks
    delta = new_flops - base_flops

    answer = f"+{_fmt(delta)} additional (total: {_fmt(new_flops)})"
    choices = [
        answer,
        f"+{_fmt(delta * 2)} additional (doubling blocks compounds non-linearly)",
        "No change — residual blocks have zero FLOPs (they only add tensors)",
        f"+{_fmt(delta / 2)} additional (shared parameters reduce cost)",
    ]
    random.shuffle(choices)

    return {
        "challenge_id": _make_id("add_residual_blocks"),
        "question": (
            f"A ResNet stage has {base_blocks} Residual Blocks (C={C}, H={H}×{W}) costing {_fmt(base_flops)}. "
            f"2 more identical blocks are added (total: {new_blocks} blocks). "
            f"How much does FLOPs increase?"
        ),
        "choices": choices,
        "answer": answer,
        "answer_index": choices.index(answer),
        "explanation": (
            f"Each residual block costs 2×C²×K²×H×W (two conv layers).\n"
            f"Per block: {_fmt(_residual_block_flops(C, C, H, W))}\n"
            f"Adding {new_blocks - base_blocks} more blocks: +{_fmt(delta)}\n"
            f"FLOPs scale linearly with block count (no cross-block interaction at inference)."
        ),
        "difficulty": "intermediate",
        "computation": {
            "per_block_mflops": round(_residual_block_flops(C, C, H, W), 2),
            "base_blocks": base_blocks,
            "new_blocks": new_blocks,
            "base_total_mflops": round(base_flops, 2),
            "new_total_mflops": round(new_flops, 2),
            "delta_mflops": round(delta, 2),
        },
    }


# ---------------------------------------------------------------------------
# Challenge 4: Increase transformer heads
# ---------------------------------------------------------------------------


def _increase_transformer_heads_challenge() -> dict[str, Any]:
    D, N = 768, 197  # ViT-B/16 defaults
    H1, H2 = 12, 24

    flops1 = _mha_flops(N, D)
    flops2 = _mha_flops(N, D)  # MHA FLOPs don't change with num_heads (same QKV projections)

    answer = "FLOPs stay the same — only memory footprint and parallelism change."
    choices = [
        answer,
        f"FLOPs double — {H2} heads compute twice as many attention matrices as {H1} heads.",
        "FLOPs halve — more heads mean each head processes less data.",
        "Complexity class changes from O(N²D) to O(N²D²).",
    ]
    random.shuffle(choices)

    return {
        "challenge_id": _make_id("increase_transformer_heads"),
        "question": (
            f"A ViT-B encoder (D={D}, N={N} tokens) uses {H1} attention heads ({_fmt(flops1)} per block). "
            f"If heads are increased to {H2} (while keeping D={D} unchanged), how do FLOPs change?"
        ),
        "choices": choices,
        "answer": answer,
        "answer_index": choices.index(answer),
        "explanation": (
            f"MHA FLOPs = QKV projections (3·2·N·D²) + Scores (2·N²·D) + AV (2·N²·D) + Output proj (2·N·D²).\n"
            f"None of these terms include num_heads when D is fixed. Heads split D into D/H sub-spaces but total "
            f"compute is identical.\n"
            f"More heads → finer attention patterns + parallel computation, NOT more FLOPs.\n"
            f"Total: {_fmt(flops1)} regardless of {H1} or {H2} heads."
        ),
        "difficulty": "advanced",
        "computation": {
            "formula": "3·2·N·D² + 2·N²·D + 2·N²·D + 2·N·D²",
            "N": N,
            "D": D,
            "h1_flops_mflops": round(flops1, 2),
            "h2_flops_mflops": round(flops2, 2),
            "delta": 0,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CHALLENGE_BUILDERS = [
    _double_channels_challenge,
    _increase_image_size_challenge,
    _add_residual_blocks_challenge,
    _increase_transformer_heads_challenge,
]

_DIFFICULTY_MAP = {
    "beginner": [_double_channels_challenge],
    "intermediate": [_increase_image_size_challenge, _add_residual_blocks_challenge],
    "advanced": [_increase_transformer_heads_challenge],
}


def get_flops_challenge(
    difficulty: str = "intermediate",
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Return a deterministic FLOPs challenge computed by the backend engine.

    Args:
        difficulty: beginner | intermediate | advanced
        seed: Optional random seed
    """
    rng = random.Random(seed)
    pool = _DIFFICULTY_MAP.get(difficulty.lower(), _CHALLENGE_BUILDERS)
    builder = rng.choice(pool)
    rng2 = random.Random(seed)
    random.seed(seed)
    result = builder()
    random.seed(None)
    return result
