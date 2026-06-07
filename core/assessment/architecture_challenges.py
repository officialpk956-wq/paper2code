"""
core/assessment/architecture_challenges.py

Deterministic architecture mutation challenges.
Supported architectures: ResNet, DenseNet, U-Net, Transformer, ViT

Each challenge returns:
  {
    "challenge_id": str,
    "question": str,
    "choices": list[str],          # multiple-choice options
    "answer": str,                  # correct choice or value
    "answer_index": int,            # 0-based index into choices
    "explanation": str,
    "difficulty": str,
    "architecture": str,
    "mutation": str
  }
"""
from __future__ import annotations
import random
import hashlib
import json
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_id(arch: str, mutation: str) -> str:
    raw = f"{arch}::{mutation}"
    return "arch_" + hashlib.md5(raw.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# ResNet Challenges
# ---------------------------------------------------------------------------

RESNET_CHALLENGES: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("ResNet", "remove_skip"),
        "mutation": "remove_skip_connection",
        "question": (
            "In a ResNet-18 Residual Block, the skip connection is removed. "
            "The block contains two Conv3×3 layers. What is the primary consequence of removing the skip connection?"
        ),
        "choices": [
            "Gradient flow is unaffected because batch normalization compensates.",
            "The vanishing gradient problem returns — deep layers receive near-zero gradients during backprop.",
            "The model becomes faster but maintains the same accuracy.",
            "The output shape of the block changes from (B, C, H, W) to (B, 2C, H, W)."
        ],
        "answer": "The vanishing gradient problem returns — deep layers receive near-zero gradients during backprop.",
        "answer_index": 1,
        "explanation": (
            "Skip connections in ResNet allow gradients to bypass the conv layers entirely via the identity path. "
            "Without them, the effective depth causes exponential gradient decay through chain-rule multiplication, "
            "which is the classical vanishing gradient problem. The output shape does NOT change — the skip adds (not concatenates) tensors."
        ),
        "difficulty": "intermediate",
        "architecture": "ResNet",
    },
    {
        "challenge_id": _make_id("ResNet", "skip_projection"),
        "mutation": "skip_projection",
        "question": (
            "ResNet-50 uses a 1×1 projection convolution in the skip connection when the number of channels "
            "changes between stages. If this projection conv is removed and channel counts still change, what happens?"
        ),
        "choices": [
            "Nothing — the addition still works because channels are broadcast automatically.",
            "A dimension mismatch error occurs: the skip tensor (B, C_in, H, W) cannot be added to the main path output (B, C_out, H, W).",
            "The model doubles its parameter count to compensate.",
            "The pooling layer absorbs the mismatch at runtime."
        ],
        "answer": "A dimension mismatch error occurs: the skip tensor (B, C_in, H, W) cannot be added to the main path output (B, C_out, H, W).",
        "answer_index": 1,
        "explanation": (
            "Tensor addition requires identical shapes. When C_in ≠ C_out, the skip must be projected via a 1×1 conv "
            "to match C_out. Removing this projection causes a runtime tensor shape error."
        ),
        "difficulty": "advanced",
        "architecture": "ResNet",
    },
]


# ---------------------------------------------------------------------------
# DenseNet Challenges
# ---------------------------------------------------------------------------

DENSENET_CHALLENGES: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("DenseNet", "remove_dense_connection"),
        "mutation": "remove_dense_connection",
        "question": (
            "In DenseNet-121, each layer receives input from ALL preceding layers within a dense block. "
            "If all dense connections are removed (keeping only sequential flow), how does the input to layer L change?"
        ),
        "choices": [
            "Input channels increase by growth_rate per layer → removing dense connections reduces input channels to just growth_rate.",
            "The input stays the same because the first layer still feeds into all subsequent layers.",
            "Each layer's input becomes 2× larger since skip connections double the channel count.",
            "Output channels double because the dense block concatenates in both directions."
        ],
        "answer": "Input channels increase by growth_rate per layer → removing dense connections reduces input channels to just growth_rate.",
        "answer_index": 0,
        "explanation": (
            "With dense connections, layer L receives concatenated outputs from layers 0..L-1, so its input size is "
            "k₀ + (L-1)×k channels where k is the growth rate. Without dense connections, each layer only receives "
            "the previous layer's output (just k channels). This drastically reduces parameter count and feature reuse."
        ),
        "difficulty": "advanced",
        "architecture": "DenseNet",
    },
    {
        "challenge_id": _make_id("DenseNet", "transition_layer"),
        "mutation": "remove_transition_layer",
        "question": (
            "DenseNet uses Transition Layers (1×1 Conv + 2×2 AvgPool) between dense blocks. "
            "If the Transition Layer is removed, what happens to spatial resolution across blocks?"
        ),
        "choices": [
            "Resolution stays constant — only the dense connections change spatial size.",
            "Resolution doubles because pooling is removed.",
            "Resolution remains constant across all blocks — no downsampling occurs, feature maps grow unbounded in channel count.",
            "The model crashes because batch normalization fails without pooling."
        ],
        "answer": "Resolution remains constant across all blocks — no downsampling occurs, feature maps grow unbounded in channel count.",
        "answer_index": 2,
        "explanation": (
            "Transition layers serve two roles: channel compression (1×1 conv) and spatial downsampling (AvgPool). "
            "Without them, spatial resolution never decreases and channel count grows by growth_rate per layer across all blocks, "
            "leading to extremely wide feature maps and memory explosion."
        ),
        "difficulty": "intermediate",
        "architecture": "DenseNet",
    },
]


# ---------------------------------------------------------------------------
# U-Net Challenges
# ---------------------------------------------------------------------------

UNET_CHALLENGES: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("UNet", "remove_skip_path"),
        "mutation": "remove_encoder_decoder_skip",
        "question": (
            "U-Net concatenates encoder feature maps to decoder feature maps via skip paths at each resolution level. "
            "If ALL skip paths are removed, what is the primary consequence for segmentation quality?"
        ),
        "choices": [
            "No effect — the decoder can reconstruct fine details from the bottleneck alone.",
            "The model can no longer localize — fine spatial detail from the encoder is lost, and the decoder only sees coarse semantic information from the bottleneck.",
            "The model's parameter count doubles because decoder convolutions must compensate.",
            "Pooling operations fail because there's no skip tensor to add."
        ],
        "answer": "The model can no longer localize — fine spatial detail from the encoder is lost, and the decoder only sees coarse semantic information from the bottleneck.",
        "answer_index": 1,
        "explanation": (
            "U-Net's skip connections carry high-resolution spatial information (edges, textures) from encoder stages directly to the decoder. "
            "The bottleneck alone contains semantically rich but spatially coarse features. "
            "Without skip paths, the decoder must reconstruct pixel-level boundaries from only low-resolution representations — "
            "this severely degrades segmentation precision, especially at object boundaries."
        ),
        "difficulty": "intermediate",
        "architecture": "U-Net",
    },
    {
        "challenge_id": _make_id("UNet", "channel_concat"),
        "mutation": "channel_concatenation",
        "question": (
            "In U-Net, decoder layer at resolution 64×64 receives an upsampled tensor of shape (B, 256, 64, 64) "
            "and a skip tensor of shape (B, 256, 64, 64). After concatenation, what is the input channel count "
            "to the following 3×3 Conv2D?"
        ),
        "choices": ["128", "256", "512", "1024"],
        "answer": "512",
        "answer_index": 2,
        "explanation": (
            "U-Net concatenates tensors along the channel dimension (dim=1). "
            "upsampled (B, 256, 64, 64) + skip (B, 256, 64, 64) → (B, 512, 64, 64). "
            "The following Conv2D receives 512 input channels."
        ),
        "difficulty": "beginner",
        "architecture": "U-Net",
    },
]


# ---------------------------------------------------------------------------
# Transformer Challenges
# ---------------------------------------------------------------------------

TRANSFORMER_CHALLENGES: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("Transformer", "remove_attention_block"),
        "mutation": "remove_attention_block",
        "question": (
            "A Transformer encoder has 6 attention blocks. If 3 blocks are removed (leaving 3), "
            "how does the computational complexity of the attention mechanism scale with sequence length N?"
        ),
        "choices": [
            "O(N) — each remaining block processes tokens sequentially.",
            "O(N²) — unchanged. Each block independently computes N×N attention scores.",
            "O(N²/2) — halved because only 3 blocks remain.",
            "O(N log N) — the reduced depth changes the computational class."
        ],
        "answer": "O(N²) — unchanged. Each block independently computes N×N attention scores.",
        "answer_index": 1,
        "explanation": (
            "Self-attention complexity O(N²·D) comes from the Q·Kᵀ matrix multiplication within each block. "
            "Reducing the number of blocks reduces the constant factor (3× fewer FLOPs) but does NOT change the "
            "asymptotic complexity class. Even a single attention block is O(N²)."
        ),
        "difficulty": "intermediate",
        "architecture": "Transformer",
    },
    {
        "challenge_id": _make_id("Transformer", "remove_feedforward"),
        "mutation": "remove_feedforward",
        "question": (
            "A standard Transformer encoder block contains: LayerNorm → MHSA → Residual → LayerNorm → FFN → Residual. "
            "If the Feed-Forward Network (FFN) sublayer is removed, what capability is most affected?"
        ),
        "choices": [
            "Token mixing — the model can no longer compute relationships between tokens.",
            "Position-wise nonlinear feature transformation — each token's representation is processed only linearly through attention projections.",
            "Gradient flow — residual connections fail without the FFN.",
            "Sequence length handling — the model can only process sequences up to 512 tokens."
        ],
        "answer": "Position-wise nonlinear feature transformation — each token's representation is processed only linearly through attention projections.",
        "answer_index": 1,
        "explanation": (
            "The FFN applies two linear transformations with a nonlinear activation (GELU/ReLU) to each token independently. "
            "This is where most position-wise nonlinear capacity lives. Attention projections are linear; the FFN provides "
            "the model's ability to represent complex non-linear functions per token."
        ),
        "difficulty": "advanced",
        "architecture": "Transformer",
    },
]


# ---------------------------------------------------------------------------
# ViT Challenges
# ---------------------------------------------------------------------------

VIT_CHALLENGES: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("ViT", "remove_patch_embedding"),
        "mutation": "remove_patch_embedding",
        "question": (
            "In Vision Transformer (ViT-B/16), the Patch Embedding converts an image of shape (B, 3, 224, 224) "
            "into tokens. If Patch Embedding is removed and the raw image is fed directly to the first attention block, "
            "what happens?"
        ),
        "choices": [
            "The attention block processes pixel-level tokens — 50176 tokens per image, causing quadratic memory explosion.",
            "The attention block reshapes the image automatically and computation proceeds normally.",
            "The model outputs NaN because batch normalization fails on raw image pixels.",
            "Only 196 tokens are processed because the attention block truncates automatically."
        ],
        "answer": "The attention block processes pixel-level tokens — 50176 tokens per image, causing quadratic memory explosion.",
        "answer_index": 0,
        "explanation": (
            "Patch Embedding converts (B, 3, 224, 224) into (B, 196, 768) using a Conv2D with kernel=16, stride=16. "
            "Without it, a flattened 224×224×3 = 150,528 vector would be fed in. "
            "If treated as sequence length N=150528, attention cost is N² = ~22.6 billion elements — completely infeasible. "
            "The patch embedding is critical for reducing sequence length to a manageable 196 tokens."
        ),
        "difficulty": "advanced",
        "architecture": "ViT",
    },
    {
        "challenge_id": _make_id("ViT", "patch_size_change"),
        "mutation": "patch_size_change",
        "question": (
            "ViT-B/16 uses patch_size=16 on a 224×224 image, producing 196 patches. "
            "If patch_size is changed to 32, how many patches are produced?"
        ),
        "choices": ["49", "98", "196", "784"],
        "answer": "49",
        "answer_index": 0,
        "explanation": (
            "Number of patches = (H / patch_size) × (W / patch_size) = (224/32) × (224/32) = 7 × 7 = 49 patches. "
            "Larger patches mean fewer tokens (faster, less precise). Smaller patches mean more tokens (slower, finer detail)."
        ),
        "difficulty": "beginner",
        "architecture": "ViT",
    },
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURE_CHALLENGES: Dict[str, List[Dict[str, Any]]] = {
    "resnet": RESNET_CHALLENGES,
    "densenet": DENSENET_CHALLENGES,
    "unet": UNET_CHALLENGES,
    "u-net": UNET_CHALLENGES,
    "transformer": TRANSFORMER_CHALLENGES,
    "vit": VIT_CHALLENGES,
}


def get_architecture_challenge(
    architecture: str = "ResNet",
    difficulty: str | None = None,
    seed: int | None = None,
) -> Dict[str, Any]:
    """
    Return a deterministic architecture-mutation challenge.

    Args:
        architecture: One of ResNet, DenseNet, U-Net, Transformer, ViT
        difficulty: Optional filter: beginner | intermediate | advanced
        seed: Optional random seed for reproducibility
    """
    arch_key = architecture.lower()
    pool = ARCHITECTURE_CHALLENGES.get(arch_key, [])
    if not pool:
        # Fall back to a cross-architecture challenge
        pool = RESNET_CHALLENGES

    if difficulty:
        filtered = [c for c in pool if c["difficulty"] == difficulty.lower()]
        if filtered:
            pool = filtered

    rng = random.Random(seed)
    challenge = rng.choice(pool)
    return dict(challenge)  # defensive copy
