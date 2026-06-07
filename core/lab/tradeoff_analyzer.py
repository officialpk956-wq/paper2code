"""
core/lab/tradeoff_analyzer.py — Tradeoff Scatter Data (Phase 11)

Generates scatter-plot ready data for the Tradeoff Explorer:
  - X axis: FLOPs score (compute cost)
  - Y axis: Parameter count (model size)
  - Bubble size: Activation memory (MB)
  - Color/label: Architecture variant name

Provides:
  1. tradeoff_scatter(base_graph, architecture)
       → Generate a 3×3 grid of mutations to populate the scatter plot
  2. get_efficiency_frontiers(points)
       → Identify Pareto-optimal points (min params for given FLOPs)
"""

import dataclasses
from typing import Any, Dict, List

from core.architecture_graph import ArchitectureGraph
from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
from core.lab.mutator import (
    increase_depth, decrease_depth,
    increase_width, decrease_width,
    add_residual, remove_residual,
    add_attention, change_patch_size, change_hidden_dim,
    _clone_graph,
)


# ---------------------------------------------------------------------------
# Scatter generation
# ---------------------------------------------------------------------------

def _graph_point(
    graph: ArchitectureGraph,
    label: str,
    group: str,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Extract a scatter data point from a graph."""
    metrics = estimate_metrics_from_graph(graph)
    mem_rows = estimate_activation_memory(graph, batch_size=batch_size, input_spatial=224)
    mem_mb = round(sum(r["mem_mb"] for r in mem_rows), 2)

    return {
        "label": label,
        "group": group,
        "flops_score": metrics["total_flops_score"],
        "params": metrics["total_params_estimate"],
        "depth": metrics["depth"],
        "memory_mb": mem_mb,
    }


def tradeoff_scatter(
    base_graph: ArchitectureGraph,
    architecture: str = "Custom",
) -> List[Dict[str, Any]]:
    """
    Build a list of scatter data points representing the tradeoff landscape.

    Always includes:
      - Baseline
      - Depth variants: +1, +2, −1
      - Width variants: ×1.5, ×0.5
      - Attention injected
      - Residual added / removed

    Returns list of dicts for Chart.js bubble dataset.
    """
    points = []

    # Baseline
    points.append(_graph_point(base_graph, f"{architecture} (Baseline)", "baseline"))

    # Depth variants
    try:
        points.append(_graph_point(increase_depth(base_graph, 1), f"{architecture} +1 Block", "depth"))
    except Exception:
        pass
    try:
        points.append(_graph_point(increase_depth(base_graph, 2), f"{architecture} +2 Blocks", "depth"))
    except Exception:
        pass
    try:
        points.append(_graph_point(decrease_depth(base_graph, 1), f"{architecture} −1 Block", "depth"))
    except Exception:
        pass

    # Width variants
    try:
        points.append(_graph_point(increase_width(base_graph, 1.5), f"{architecture} Width ×1.5", "width"))
    except Exception:
        pass
    try:
        points.append(_graph_point(decrease_width(base_graph, 0.5), f"{architecture} Width ×0.5", "width"))
    except Exception:
        pass

    # Structural variants
    try:
        points.append(_graph_point(add_residual(base_graph), f"{architecture} +Residual", "structure"))
    except Exception:
        pass
    try:
        points.append(_graph_point(remove_residual(base_graph), f"{architecture} −Residual", "structure"))
    except Exception:
        pass
    try:
        points.append(_graph_point(add_attention(base_graph), f"{architecture} +Attention", "attention"))
    except Exception:
        pass

    return points


def get_efficiency_frontiers(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return Pareto-optimal points: those where no other point has both
    lower params and lower FLOPs score.

    These represent the "efficiency frontier" — the best accuracy-compute tradeoff.
    """
    if not points:
        return []

    frontier = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            # q dominates p if q has strictly lower or equal flops AND params
            if (q["flops_score"] <= p["flops_score"] and
                    q["params"] <= p["params"] and
                    (q["flops_score"] < p["flops_score"] or q["params"] < p["params"])):
                dominated = True
                break
        if not dominated:
            frontier.append({**p, "is_frontier": True})
        else:
            frontier.append({**p, "is_frontier": False})

    return frontier


# ---------------------------------------------------------------------------
# Quick tradeoff summary (textual)
# ---------------------------------------------------------------------------

TRADEOFF_SUMMARIES = {
    "increase_depth": {
        "compute": "high",
        "memory": "medium",
        "expressivity": "high",
        "inference_speed": "slow",
        "training_stability": "requires residuals for deep networks",
        "recommendation": "Use when underfitting. Combine with residual connections for stability.",
    },
    "decrease_depth": {
        "compute": "low",
        "memory": "low",
        "expressivity": "low",
        "inference_speed": "fast",
        "training_stability": "stable",
        "recommendation": "Use for edge/mobile deployment. Risk of underfitting on complex tasks.",
    },
    "increase_width": {
        "compute": "high",
        "memory": "high",
        "expressivity": "high",
        "inference_speed": "slow",
        "training_stability": "stable",
        "recommendation": "Effective for complex feature spaces. Memory scales quadratically.",
    },
    "decrease_width": {
        "compute": "low",
        "memory": "low",
        "expressivity": "medium",
        "inference_speed": "fast",
        "training_stability": "stable",
        "recommendation": "MobileNet-style efficiency. Good accuracy-to-compute tradeoff.",
    },
    "add_residual": {
        "compute": "minimal",
        "memory": "low",
        "expressivity": "high",
        "inference_speed": "same",
        "training_stability": "excellent",
        "recommendation": "Always add residuals for networks deeper than 10 layers. Near-zero cost.",
    },
    "remove_residual": {
        "compute": "minimal",
        "memory": "low",
        "expressivity": "lower",
        "inference_speed": "same",
        "training_stability": "poor for deep nets",
        "recommendation": "Only remove for shallow (<6 layer) architectures. Risks gradient vanishing.",
    },
    "add_attention": {
        "compute": "very high (O(N²))",
        "memory": "high",
        "expressivity": "very high",
        "inference_speed": "slow",
        "training_stability": "stable",
        "recommendation": "Adds global context. Use for tasks requiring long-range dependencies.",
    },
    "change_patch_size": {
        "compute": "varies (quadratic with 1/patch_size)",
        "memory": "high for small patches",
        "expressivity": "high for small patches",
        "inference_speed": "slow for small patches",
        "training_stability": "stable",
        "recommendation": "Smaller patches = finer granularity but O(N²) cost. 16 is standard for ViT.",
    },
    "change_hidden_dim": {
        "compute": "quadratic with dim (d²)",
        "memory": "high",
        "expressivity": "high",
        "inference_speed": "slow",
        "training_stability": "stable",
        "recommendation": "Critical for transformer capacity. d=768 (ViT-B), d=1024 (ViT-L).",
    },
}


def get_tradeoff_summary(mutation_type: str) -> Dict[str, Any]:
    """Return the structured tradeoff summary for a mutation type."""
    return TRADEOFF_SUMMARIES.get(mutation_type, {
        "compute": "unknown",
        "memory": "unknown",
        "expressivity": "unknown",
        "inference_speed": "unknown",
        "training_stability": "unknown",
        "recommendation": "No tradeoff data available for this mutation type.",
    })
