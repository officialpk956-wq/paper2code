"""
Metrics estimation for neural network architectures.

F3: Compute Metrics — FLOPs score and parameter estimation from semantic_params.
F6: Memory Estimation — Activation memory per layer based on compute roles.
"""

from typing import Dict, List, Any
from src.architecture_graph import ArchitectureGraph, GraphNode


# F3: Compute Metrics
FLOPS_SCORES = {"very high": 4, "high": 2, "medium": 1, "low": 0}


def estimate_metrics_from_graph(graph: ArchitectureGraph) -> Dict[str, Any]:
    """
    Estimate compute metrics for a graph.

    Returns dict with:
    - total_flops_score: Sum of FLOPs weights across nodes
    - total_params_estimate: Rough parameter count estimate
    - breakdown: Per-node detail rows
    - depth: Number of nodes
    """
    total_score = 0
    param_estimate = 0
    breakdown = []

    for node in graph.nodes:
        sp = node.semantic_params or {}
        flops_level = sp.get("flops", "low")
        score = FLOPS_SCORES.get(flops_level, 0)
        total_score += score

        node_params = _estimate_node_params(node)
        param_estimate += node_params

        breakdown.append({
            "node": node.label,
            "flops_level": flops_level,
            "param_estimate": node_params,
        })

    return {
        "total_flops_score": total_score,
        "total_params_estimate": param_estimate,
        "breakdown": breakdown,
        "depth": len(graph.nodes),
    }


def _extract_numeric_value(val: Any, default: int = 0) -> int:
    """
    Extract numeric value from param, handling strings like '3 → 64'.
    Returns the last number found (output size), or default if parsing fails.
    """
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if not val:
        return default

    # Convert to string and extract all numbers
    s = str(val).strip()
    import re
    numbers = re.findall(r'\d+', s)

    # Return last number (output size for ranges like "3 → 64"), or first if only one
    if numbers:
        return int(numbers[-1])
    return default


def _estimate_node_params(node: GraphNode) -> int:
    """
    Rough parameter count estimation from node properties.

    Rules by type:
    - Conv2D: channels * kernel_size^2 * channels
    - Linear: hidden_size * hidden_size
    - Attention/Transformer: 4 * d^2
    - Norm: 2 * channels
    - Residual: 9 * channels^2 * 2
    - Others: 0
    """
    t = (node.type or "").lower()
    p = node.params or {}

    ch = _extract_numeric_value(p.get("channels", p.get("filters", 64)), 64)
    k = _extract_numeric_value(p.get("kernel_size", 3), 3)
    hs = _extract_numeric_value(p.get("hidden_size", 512), 512)

    if t in ("conv2d", "conv1d"):
        return ch * k * k * ch
    elif t == "linear":
        return hs * hs
    elif t in ("multiheadattention", "transformerblock"):
        d = _extract_numeric_value(p.get("d_model"), _extract_numeric_value(p.get("heads"), 8) * 64)
        return 4 * d * d
    elif t in ("batchnorm2d", "layernorm"):
        return 2 * ch
    elif t == "residualblock":
        return 9 * ch * ch * 2
    else:
        return 0


# F6: Memory Estimation
BYTES_PER_FLOAT = 4


def estimate_activation_memory(
    graph: ArchitectureGraph, batch_size: int = 1, input_spatial: int = 224
) -> List[Dict[str, Any]]:
    """
    Estimate activation memory per layer.

    Args:
        graph: ArchitectureGraph to analyze
        batch_size: Batch size for activation tensors
        input_spatial: Input spatial dimension (H or W, assumes square)

    Returns:
        List of dicts with per-node memory estimates (in MB)
    """
    results = []
    spatial = input_spatial

    for node in graph.nodes:
        sp = node.semantic_params or {}
        role = sp.get("compute_role", "unknown")
        feature_map = sp.get("feature_map", "")
        ch = _get_channels(node)

        # Update spatial resolution based on feature_map signal
        if feature_map == "downsampling":
            spatial = max(1, spatial // 2)
        elif feature_map == "upsampling":
            spatial = min(spatial * 2, 2048)

        # Estimate activation size based on compute role
        if role in ("feature_extraction", "normalization", "residual", "activation"):
            # Spatial feature map: B * C * H * W * 4 bytes
            mem_bytes = batch_size * ch * spatial * spatial * BYTES_PER_FLOAT
        elif role == "attention":
            # Attention map: B * heads * seq_len^2
            heads = (node.params or {}).get("heads", 8)
            try:
                seq = int(sp.get("tokens", spatial))
            except (ValueError, TypeError):
                seq = 196  # ViT default
            mem_bytes = batch_size * heads * seq * seq * BYTES_PER_FLOAT
        elif role in ("projection", "pooling"):
            # Flat vector: B * C * 4 bytes
            mem_bytes = batch_size * ch * BYTES_PER_FLOAT
        else:
            mem_bytes = 0

        results.append({
            "node": node.label,
            "role": role,
            "channels": ch,
            "spatial": spatial,
            "mem_mb": round(mem_bytes / (1024 ** 2), 3),
        })

    return results


def _get_channels(node: GraphNode) -> int:
    """Extract channel count from node params with fallbacks."""
    p = node.params or {}
    return p.get("channels", p.get("filters", p.get("out_channels", 64)))
