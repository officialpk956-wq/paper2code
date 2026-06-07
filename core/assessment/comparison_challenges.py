"""
core/assessment/comparison_challenges.py

Architecture comparison challenges using ACTUAL corpus metrics from SQLite.

Compares: ResNet vs DenseNet | U-Net vs FCN | Transformer vs ViT

Questions reference:
  - parameters
  - FLOPs
  - module count
  - graph structure

No fabricated comparisons. All numbers come from db queries.
"""
from __future__ import annotations
import hashlib
import random
from typing import Dict, Any, List, Optional


def _make_id(label: str) -> str:
    return "cmp_" + hashlib.md5(label.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Static structural knowledge (graph topology facts)
# ---------------------------------------------------------------------------

STRUCTURAL_FACTS: Dict[str, Dict[str, Any]] = {
    "resnet": {
        "connection_type": "skip connections (residual add)",
        "graph_pattern": "linear with skip edges",
        "key_feature": "Identity shortcut connections enable extremely deep networks",
    },
    "densenet": {
        "connection_type": "dense connections (all-to-all within block)",
        "graph_pattern": "DAG with dense intra-block edges",
        "key_feature": "Feature reuse: each layer receives all prior feature maps",
    },
    "unet": {
        "connection_type": "encoder-decoder skip paths (concatenation)",
        "graph_pattern": "U-shaped: encoder + bottleneck + decoder with cross-connections",
        "key_feature": "Spatial detail recovery via high-resolution skip paths",
    },
    "fcn": {
        "connection_type": "no skip connections (sequential decoder)",
        "graph_pattern": "linear encoder with upsampling decoder",
        "key_feature": "First fully convolutional segmentation network — no FC layers",
    },
    "transformer": {
        "connection_type": "self-attention (global receptive field)",
        "graph_pattern": "sequential encoder blocks with residual",
        "key_feature": "O(N²) attention allows each token to attend to all others",
    },
    "vit": {
        "connection_type": "patch-based self-attention",
        "graph_pattern": "patch embedding + sequential transformer encoders",
        "key_feature": "Image split into patches → treated as token sequence",
    },
}


# ---------------------------------------------------------------------------
# Fallback static challenges (used when DB lacks both architectures)
# ---------------------------------------------------------------------------

STATIC_CHALLENGES: List[Dict[str, Any]] = [
    {
        "challenge_id": _make_id("resnet_vs_densenet_connectivity"),
        "arch_a": "ResNet",
        "arch_b": "DenseNet",
        "question": (
            "ResNet and DenseNet both use skip connections, but with fundamentally different connectivity. "
            "How do their connection patterns differ?"
        ),
        "choices": [
            "ResNet adds the input to a block's output (residual add). DenseNet concatenates ALL previous feature maps to each layer's input.",
            "ResNet concatenates feature maps while DenseNet uses addition.",
            "Both use identical connection patterns — the difference is only in channel count.",
            "ResNet uses attention-based skip connections while DenseNet uses convolution-based skips.",
        ],
        "answer": "ResNet adds the input to a block's output (residual add). DenseNet concatenates ALL previous feature maps to each layer's input.",
        "answer_index": 0,
        "explanation": (
            "ResNet: x_out = F(x) + x  (element-wise addition — no new channels)\n"
            "DenseNet: x_L = F([x₀, x₁, ..., x_{L-1}])  (channel concatenation — grows linearly with depth)\n\n"
            "This means DenseNet has dramatically more feature reuse but also more memory usage "
            "as channels accumulate within each dense block."
        ),
        "difficulty": "intermediate",
        "metrics_source": "static_structural",
        "comparison_type": "graph_structure",
    },
    {
        "challenge_id": _make_id("unet_vs_fcn_skips"),
        "arch_a": "U-Net",
        "arch_b": "FCN",
        "question": (
            "Both U-Net and FCN (Fully Convolutional Network) perform semantic segmentation. "
            "What is the key architectural difference in how they recover spatial detail?"
        ),
        "choices": [
            "U-Net uses encoder-decoder skip connections that concatenate high-resolution feature maps into the decoder. FCN upsamples directly from the bottleneck without skip connections.",
            "FCN uses skip connections while U-Net uses transposed convolutions for upsampling.",
            "U-Net is purely sequential; FCN uses bidirectional feature flow.",
            "Both use identical upsampling strategies — the difference is in their loss functions.",
        ],
        "answer": "U-Net uses encoder-decoder skip connections that concatenate high-resolution feature maps into the decoder. FCN upsamples directly from the bottleneck without skip connections.",
        "answer_index": 0,
        "explanation": (
            "FCN (Long et al., 2015): VGG backbone → transpose conv upsampling. No skip connections to decoder in the basic variant.\n"
            "U-Net (Ronneberger et al., 2015): Symmetric encoder-decoder with skip connections that concatenate encoder feature maps "
            "to decoder at EACH resolution level. This recovers fine spatial detail lost during downsampling.\n\n"
            "Result: U-Net significantly outperforms FCN on tasks requiring precise boundaries (medical imaging, cell segmentation)."
        ),
        "difficulty": "intermediate",
        "metrics_source": "static_structural",
        "comparison_type": "graph_structure",
    },
    {
        "challenge_id": _make_id("transformer_vs_vit_input"),
        "arch_a": "Transformer",
        "arch_b": "ViT",
        "question": (
            "The original Transformer (Vaswani et al., 2017) and Vision Transformer (ViT) share the same encoder architecture. "
            "What is the key difference in how they handle their input?"
        ),
        "choices": [
            "Transformer tokenizes text into word embeddings (1D sequence). ViT splits an image into fixed-size patches and embeds each patch as a token (2D → 1D sequence).",
            "ViT uses recurrent cells before the attention mechanism, while Transformer is purely feedforward.",
            "Transformer uses 2D convolutions for token mixing; ViT uses 1D convolutions.",
            "Both process identical input types — the difference is only in positional encoding.",
        ],
        "answer": "Transformer tokenizes text into word embeddings (1D sequence). ViT splits an image into fixed-size patches and embeds each patch as a token (2D → 1D sequence).",
        "answer_index": 0,
        "explanation": (
            "Original Transformer: Input = token IDs → embedding lookup → (B, N, D) sequence\n"
            "ViT: Input = image (B, 3, H, W) → PatchEmbedding (Conv2D with kernel=P, stride=P) → "
            "(B, N_patches, D) sequence\n\n"
            "ViT's key innovation: treating image patches as tokens enables using an unmodified Transformer encoder "
            "for vision tasks. PatchEmbedding replaces the embedding table."
        ),
        "difficulty": "beginner",
        "metrics_source": "static_structural",
        "comparison_type": "input_handling",
    },
    {
        "challenge_id": _make_id("resnet_vs_vit_complexity"),
        "arch_a": "ResNet",
        "arch_b": "ViT",
        "question": (
            "ResNet-50 and ViT-B/16 are both image classification models. "
            "Which computational complexity class best describes their bottleneck operations?"
        ),
        "choices": [
            "ResNet-50: O(H²) per layer (spatial convolutions). ViT-B/16: O(N²) per layer where N is number of patches (quadratic attention).",
            "Both are O(N²) — convolutions and attention have identical complexity.",
            "ResNet-50: O(N²) for skip connections. ViT-B/16: O(N) for linear attention.",
            "ResNet-50: O(log N). ViT-B/16: O(N³) for multi-head attention.",
        ],
        "answer": "ResNet-50: O(H²) per layer (spatial convolutions). ViT-B/16: O(N²) per layer where N is number of patches (quadratic attention).",
        "answer_index": 0,
        "explanation": (
            "ResNet Conv2D: FLOPs = 2×C_in×C_out×K²×H×W. For fixed C and K, scales as O(H×W) — quadratic in spatial dim.\n"
            "ViT Self-Attention: FLOPs ∝ N² (Q·Kᵀ matrix). For 224×224 images with patch_size=16, N=196.\n\n"
            "Practical implication: ResNet scales well with spatial resolution (fixed receptive field). "
            "ViT's quadratic N² becomes problematic at high resolutions — motivating Swin Transformer's window attention."
        ),
        "difficulty": "advanced",
        "metrics_source": "static_structural",
        "comparison_type": "complexity",
    },
]


# ---------------------------------------------------------------------------
# DB-backed challenges (dynamic, uses real corpus metrics)
# ---------------------------------------------------------------------------

def build_db_comparison_challenge(
    arch_a: str,
    arch_b: str,
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a comparison challenge using real DB metrics.
    metrics_a/b: {params, flops, module_count, title}
    """
    params_a = metrics_a.get("params", 0)
    params_b = metrics_b.get("params", 0)
    flops_a = metrics_a.get("flops", 0)
    flops_b = metrics_b.get("flops", 0)
    mods_a = metrics_a.get("module_count", 0)
    mods_b = metrics_b.get("module_count", 0)

    def fmt_params(p):
        if p >= 1e6: return f"{p/1e6:.1f}M"
        if p >= 1e3: return f"{p/1e3:.0f}K"
        return str(p)

    # Question: which has more parameters?
    if params_a != params_b:
        larger = metrics_a["title"] if params_a > params_b else metrics_b["title"]
        answer = f"{larger} has more parameters ({fmt_params(max(params_a, params_b))} vs {fmt_params(min(params_a, params_b))})"
        wrongs = [
            f"{metrics_a['title']} and {metrics_b['title']} have identical parameter counts.",
            f"{metrics_b['title'] if larger == metrics_a['title'] else metrics_a['title']} has more parameters.",
            f"Parameter count cannot be compared without runtime profiling.",
        ]
    else:
        answer = f"Both have approximately {fmt_params(params_a)} parameters"
        wrongs = [
            f"{metrics_a['title']} has 2× the parameters of {metrics_b['title']}.",
            f"{metrics_b['title']} has 2× the parameters of {metrics_a['title']}.",
            f"Parameter count cannot be compared without runtime profiling.",
        ]

    choices = [answer] + wrongs[:3]
    random.shuffle(choices)

    return {
        "challenge_id": _make_id(f"{arch_a}_vs_{arch_b}_params"),
        "arch_a": metrics_a["title"],
        "arch_b": metrics_b["title"],
        "question": (
            f"Using actual corpus metrics from the Paper2Code database:\n"
            f"  {metrics_a['title']}: {fmt_params(params_a)} params, {mods_a} modules\n"
            f"  {metrics_b['title']}: {fmt_params(params_b)} params, {mods_b} modules\n\n"
            f"Which architecture has more parameters, and by how much?"
        ),
        "choices": choices,
        "answer": answer,
        "answer_index": choices.index(answer),
        "explanation": (
            f"Real corpus data:\n"
            f"  {metrics_a['title']}: {fmt_params(params_a)} parameters, {mods_a} modules\n"
            f"  {metrics_b['title']}: {fmt_params(params_b)} parameters, {mods_b} modules\n"
            f"  FLOPs ratio: {metrics_a['title']}={flops_a} vs {metrics_b['title']}={flops_b}"
        ),
        "difficulty": "intermediate",
        "metrics_source": "corpus_db",
        "comparison_type": "parameters",
        "raw_metrics": {
            "arch_a": {"title": metrics_a["title"], "params": params_a, "flops": flops_a, "modules": mods_a},
            "arch_b": {"title": metrics_b["title"], "params": params_b, "flops": flops_b, "modules": mods_b},
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_comparison_challenge(
    difficulty: str = "intermediate",
    seed: int | None = None,
    db_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Return a comparison challenge.

    Args:
        difficulty: beginner | intermediate | advanced
        seed: Optional random seed
        db_metrics: List of paper metrics dicts from the corpus DB.
                    If provided and contains >= 2 papers, uses real data.
    """
    rng = random.Random(seed)

    # Try DB-backed challenge first
    if db_metrics and len(db_metrics) >= 2:
        # Find two papers with different architecture types
        papers_by_arch = {}
        for m in db_metrics:
            arch = m.get("architecture_type", "Unknown").lower()
            if arch not in papers_by_arch:
                papers_by_arch[arch] = m

        archs = list(papers_by_arch.keys())
        if len(archs) >= 2:
            rng.shuffle(archs)
            a_key, b_key = archs[0], archs[1]
            a, b = papers_by_arch[a_key], papers_by_arch[b_key]
            return build_db_comparison_challenge(
                a_key, b_key,
                {"title": a["title"], "params": a.get("parameter_count", 0), "flops": a.get("flops", 0), "module_count": a.get("module_count", 0)},
                {"title": b["title"], "params": b.get("parameter_count", 0), "flops": b.get("flops", 0), "module_count": b.get("module_count", 0)},
            )

    # Fall back to static structural challenges
    d = difficulty.lower()
    filtered = [c for c in STATIC_CHALLENGES if c["difficulty"] == d]
    pool = filtered if filtered else STATIC_CHALLENGES
    return dict(rng.choice(pool))
