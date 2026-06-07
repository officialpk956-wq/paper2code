"""
core/assessment/tensor_challenges.py

Deterministic tensor shape challenges backed by TensorTracker calculations.

Question types:
  Beginner:     Conv output shape given kernel/stride/padding
  Intermediate: Pooling output shape
  Advanced:     Multi-stage shape propagation

Every challenge returns:
  {
    "challenge_id": str,
    "question": str,
    "choices": list[str],
    "answer": str,
    "answer_index": int,
    "explanation": str,
    "difficulty": str,
    "computation": dict   ← the actual TensorTracker/formula result
  }
"""
from __future__ import annotations
import hashlib
import random
from typing import Dict, Any, List, Tuple


def _make_id(label: str) -> str:
    return "tensor_" + hashlib.md5(label.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Core formula — same math as TensorTracker._compute_output_shape
# ---------------------------------------------------------------------------

def _conv_output(H: int, W: int, kernel: int, stride: int, padding: int) -> Tuple[int, int]:
    """Exact formula: floor((H + 2P - K) / S) + 1"""
    out_h = (H + 2 * padding - kernel) // stride + 1
    out_w = (W + 2 * padding - kernel) // stride + 1
    return out_h, out_w


def _pool_output(H: int, W: int, kernel: int, stride: int) -> Tuple[int, int]:
    """MaxPool / AvgPool: floor((H - K) / S) + 1"""
    out_h = (H - kernel) // stride + 1
    out_w = (W - kernel) // stride + 1
    return out_h, out_w


def _patch_embed_output(H: int, W: int, C: int, patch_size: int, embed_dim: int) -> Tuple[int, int]:
    """PatchEmbedding: (H//P) * (W//P) patches, each embed_dim wide"""
    n_patches = (H // patch_size) * (W // patch_size)
    return n_patches, embed_dim


# ---------------------------------------------------------------------------
# Beginner: Conv output shape
# ---------------------------------------------------------------------------

CONV_BEGINNER: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("conv_basic_3x3_padding1"),
        "question": (
            "A Conv2D layer with kernel_size=3, stride=1, padding=1 receives an input of shape (B, 64, 112, 112). "
            "What is the output spatial shape?"
        ),
        "computation": {
            "formula": "floor((H + 2P - K) / S) + 1",
            "values": {"H": 112, "W": 112, "K": 3, "S": 1, "P": 1},
            "result": _conv_output(112, 112, 3, 1, 1),
        },
        "difficulty": "beginner",
    },
    {
        "challenge_id": _make_id("conv_stride2_no_pad"),
        "question": (
            "A Conv2D layer with kernel_size=7, stride=2, padding=3 receives input (B, 3, 224, 224). "
            "What is the output spatial shape (H_out × W_out)?"
        ),
        "computation": {
            "formula": "floor((H + 2P - K) / S) + 1",
            "values": {"H": 224, "W": 224, "K": 7, "S": 2, "P": 3},
            "result": _conv_output(224, 224, 7, 2, 3),
        },
        "difficulty": "beginner",
    },
    {
        "challenge_id": _make_id("conv_1x1_stride1"),
        "question": (
            "A 1×1 Conv2D layer with stride=1, padding=0 receives input (B, 256, 56, 56). "
            "What is the output spatial shape?"
        ),
        "computation": {
            "formula": "floor((H + 2P - K) / S) + 1",
            "values": {"H": 56, "W": 56, "K": 1, "S": 1, "P": 0},
            "result": _conv_output(56, 56, 1, 1, 0),
        },
        "difficulty": "beginner",
    },
]


def _build_conv_challenge(spec: Dict[str, Any]) -> Dict[str, Any]:
    H_out, W_out = spec["computation"]["result"]
    answer = f"{H_out} × {W_out}"

    # Generate 3 plausible wrong answers
    wrongs = set()
    v = spec["computation"]["values"]
    # Common mistakes: forgetting padding, wrong formula
    w1_h, w1_w = _conv_output(v["H"], v["W"], v["K"], v["S"], 0)
    w2_h, w2_w = v["H"] // v["S"], v["W"] // v["S"]
    w3_h, w3_w = (v["H"] - v["K"]) // v["S"] + 1, (v["W"] - v["K"]) // v["S"] + 1

    for wh, ww in [(w1_h, w1_w), (w2_h, w2_w), (w3_h, w3_w)]:
        wrong = f"{wh} × {ww}"
        if wrong != answer:
            wrongs.add(wrong)

    choices = [answer] + list(wrongs)[:3]
    random.shuffle(choices)
    answer_index = choices.index(answer)

    expl = (
        f"Using the formula: H_out = floor((H + 2P - K) / S) + 1\n"
        f"= floor(({v['H']} + 2×{v['P']} - {v['K']}) / {v['S']}) + 1 = {H_out}\n"
        f"Spatial output: {H_out} × {W_out}"
    )

    return {
        "challenge_id": spec["challenge_id"],
        "question": spec["question"],
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "explanation": expl,
        "difficulty": spec["difficulty"],
        "computation": spec["computation"],
    }


# ---------------------------------------------------------------------------
# Intermediate: Pooling output shape
# ---------------------------------------------------------------------------

POOL_INTERMEDIATE: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("maxpool_2x2_stride2"),
        "question": (
            "A MaxPool2D layer with kernel_size=2, stride=2 receives input (B, 64, 112, 112). "
            "What is the output spatial shape?"
        ),
        "computation": {
            "formula": "floor((H - K) / S) + 1",
            "values": {"H": 112, "W": 112, "K": 2, "S": 2},
            "result": _pool_output(112, 112, 2, 2),
        },
        "difficulty": "intermediate",
    },
    {
        "challenge_id": _make_id("maxpool_3x3_stride2"),
        "question": (
            "A MaxPool2D layer with kernel_size=3, stride=2 receives input (B, 64, 224, 224). "
            "What is the output spatial shape?"
        ),
        "computation": {
            "formula": "floor((H - K) / S) + 1",
            "values": {"H": 224, "W": 224, "K": 3, "S": 2},
            "result": _pool_output(224, 224, 3, 2),
        },
        "difficulty": "intermediate",
    },
    {
        "challenge_id": _make_id("avgpool_global"),
        "question": (
            "A GlobalAveragePool2D layer receives input (B, 512, 7, 7). "
            "What is the output shape?"
        ),
        "computation": {
            "formula": "GlobalAvgPool: (B, C, H, W) → (B, C)",
            "values": {"B": "B", "C": 512, "H": 7, "W": 7},
            "result": ("B", 512),
        },
        "difficulty": "intermediate",
    },
]


def _build_pool_challenge(spec: Dict[str, Any]) -> Dict[str, Any]:
    result = spec["computation"]["result"]
    if isinstance(result, tuple) and result[0] == "B":
        answer = f"(B, {result[1]})"
        wrongs = ["(B, 512, 1, 1)", "(B, 1, 512)", "(B, 512, 7, 7)"]
        choices = [answer] + [w for w in wrongs if w != answer][:3]
    else:
        H_out, W_out = result
        answer = f"{H_out} × {W_out}"
        v = spec["computation"]["values"]
        wrongs = [
            f"{v['H'] // v['S']} × {v['W'] // v['S']}",
            f"{H_out + 1} × {W_out + 1}",
            f"{H_out * 2} × {W_out * 2}",
        ]
        choices = [answer] + [w for w in wrongs if w != answer][:3]

    random.shuffle(choices)
    answer_index = choices.index(answer)

    v = spec["computation"]["values"]
    if "K" in v:
        expl = (
            f"Pool formula: H_out = floor((H - K) / S) + 1\n"
            f"= floor(({v['H']} - {v['K']}) / {v['S']}) + 1 = {result[0]}"
        )
    else:
        expl = "GlobalAveragePool reduces each (H, W) spatial map to a single scalar → output is (B, C)."

    return {
        "challenge_id": spec["challenge_id"],
        "question": spec["question"],
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "explanation": expl,
        "difficulty": spec["difficulty"],
        "computation": spec["computation"],
    }


# ---------------------------------------------------------------------------
# Advanced: Multi-stage shape propagation
# ---------------------------------------------------------------------------

MULTISTAGE_ADVANCED: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("resnet_stem_propagation"),
        "question": (
            "An input image (B, 3, 224, 224) passes through the ResNet stem:\n"
            "  1. Conv2D: kernel=7, stride=2, padding=3\n"
            "  2. MaxPool2D: kernel=3, stride=2\n"
            "What is the shape after step 2?"
        ),
        "stages": [
            ("Conv2D", "conv", {"H": 224, "W": 224, "K": 7, "S": 2, "P": 3}),
            ("MaxPool2D", "pool", {"K": 3, "S": 2}),
        ],
        "difficulty": "advanced",
    },
    {
        "challenge_id": _make_id("vit_patch_embed_propagation"),
        "question": (
            "An input (B, 3, 224, 224) passes through ViT's Patch Embedding:\n"
            "  PatchEmbedding: patch_size=16, embed_dim=768\n"
            "What is the output shape (batch-agnostic)?"
        ),
        "stages": [
            ("PatchEmbedding", "patch", {"H": 224, "W": 224, "C": 3, "P": 16, "D": 768}),
        ],
        "difficulty": "advanced",
    },
    {
        "challenge_id": _make_id("three_stage_conv_pool"),
        "question": (
            "An input (B, 3, 64, 64) passes through:\n"
            "  1. Conv2D: kernel=3, stride=1, padding=1  → channels=32\n"
            "  2. MaxPool2D: kernel=2, stride=2\n"
            "  3. Conv2D: kernel=3, stride=1, padding=1  → channels=64\n"
            "What is the output spatial shape (H × W) after step 3?"
        ),
        "stages": [
            ("Conv2D", "conv", {"H": 64, "W": 64, "K": 3, "S": 1, "P": 1}),
            ("MaxPool2D", "pool", {"K": 2, "S": 2}),
            ("Conv2D", "conv", {"K": 3, "S": 1, "P": 1}),  # H/W inherited
        ],
        "difficulty": "advanced",
    },
]


def _propagate_stages(stages) -> Tuple[int, int]:
    """Propagate H, W through a sequence of conv/pool/patch operations."""
    H, W = None, None
    for name, op, v in stages:
        if op == "conv":
            if H is None:
                H, W = v["H"], v["W"]
            H, W = _conv_output(H, W, v["K"], v["S"], v["P"])
        elif op == "pool":
            H, W = _pool_output(H, W, v["K"], v["S"])
        elif op == "patch":
            n_patches = (v["H"] // v["P"]) * (v["W"] // v["P"])
            return n_patches, v["D"]  # type: ignore
    return H, W


def _build_multistage_challenge(spec: Dict[str, Any]) -> Dict[str, Any]:
    result = _propagate_stages(spec["stages"])

    if spec["challenge_id"] == _make_id("vit_patch_embed_propagation"):
        n_patches, embed_dim = result
        answer = f"(B, {n_patches}, {embed_dim})"
        wrongs = [
            f"(B, {n_patches * 4}, {embed_dim})",
            f"(B, {n_patches}, {embed_dim // 2})",
            "(B, 3, 224, 224)",
        ]
        expl = (
            f"Patch Embedding splits image into (224/16)² = 196 patches, each embedded to dim {embed_dim}.\n"
            f"Output shape: (B, 196, 768)"
        )
    else:
        H_out, W_out = result
        answer = f"{H_out} × {W_out}"
        wrongs = [
            f"{H_out * 2} × {W_out * 2}",
            f"{H_out + 1} × {W_out + 1}",
            f"{H_out - 1} × {W_out - 1}",
        ]
        expl = "Propagated each stage using: Conv → floor((H+2P-K)/S)+1, Pool → floor((H-K)/S)+1"

    choices = [answer] + [w for w in wrongs if w != answer][:3]
    random.shuffle(choices)
    answer_index = choices.index(answer)

    return {
        "challenge_id": spec["challenge_id"],
        "question": spec["question"],
        "choices": choices,
        "answer": answer,
        "answer_index": answer_index,
        "explanation": expl,
        "difficulty": spec["difficulty"],
        "computation": {"stages": [(s[0], s[1]) for s in spec["stages"]], "result": result},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tensor_challenge(
    difficulty: str = "beginner",
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Return a deterministic tensor shape challenge.

    Args:
        difficulty: beginner | intermediate | advanced
        seed: Optional random seed for reproducibility
    """
    rng = random.Random(seed)
    d = difficulty.lower()

    if d == "beginner":
        spec = rng.choice(CONV_BEGINNER)
        return _build_conv_challenge(spec)
    elif d == "intermediate":
        spec = rng.choice(POOL_INTERMEDIATE)
        return _build_pool_challenge(spec)
    else:
        spec = rng.choice(MULTISTAGE_ADVANCED)
        return _build_multistage_challenge(spec)
