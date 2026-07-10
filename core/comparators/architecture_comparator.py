"""Deterministic architecture comparison engine.

This module provides rule-based analysis and comparison of neural network
architectures represented as ArchitectureGraph objects.

No LLMs, no heuristics - pure deterministic reasoning.
"""

from typing import Any

from core.architecture_graph import ArchitectureGraph


def summarize_compute(graph: ArchitectureGraph) -> dict[str, Any]:
    """
    Analyze computational characteristics of an architecture.

    Identifies compute-intensive nodes based on semantic_params["flops"].

    Args:
        graph: ArchitectureGraph to analyze

    Returns:
        Dictionary with:
        - high_flops_nodes: List of node labels with high/very high FLOPs
        - total_high_flops: Count of compute-heavy nodes
        - primary_bottleneck: Label of most significant bottleneck (or None)
    """
    high_flops_nodes = []
    very_high_nodes = []

    for node in graph.nodes:
        flops = str(node.semantic_params.get("flops", "")).lower()
        if flops == "very high":
            very_high_nodes.append(node.label)
            high_flops_nodes.append(node.label)
        elif flops == "high":
            high_flops_nodes.append(node.label)

    # Primary bottleneck: prefer "very high" over "high"
    primary_bottleneck = None
    if very_high_nodes:
        primary_bottleneck = very_high_nodes[0]
    elif high_flops_nodes:
        primary_bottleneck = high_flops_nodes[0]

    return {
        "high_flops_nodes": high_flops_nodes,
        "total_high_flops": len(high_flops_nodes),
        "primary_bottleneck": primary_bottleneck,
    }


def summarize_spatial_behavior(graph: ArchitectureGraph) -> dict[str, str]:
    """
    Analyze spatial locality preservation in an architecture.

    Rules:
    - CNNs (Conv2D-based) → high spatial preservation
    - U-Net (encoder-decoder + skip connections) → high with restoration
    - Vision Transformers (token-based) → low spatial preservation

    Args:
        graph: ArchitectureGraph to analyze

    Returns:
        Dictionary with:
        - spatial_preservation: "high" | "medium" | "low"
        - reason: Human-readable explanation
    """
    # Check for skip connections (U-Net indicator)
    has_skip_connections = any(
        str(node.semantic_params.get("skip_connection", "")).lower() == "yes"
        for node in graph.nodes
    )

    # Check for token-based processing (ViT indicator)
    has_tokens = any("tokens" in node.semantic_params for node in graph.nodes)

    # Check for convolutional layers
    has_conv = any(str(node.type).lower() == "conv2d" for node in graph.nodes)

    # Determine spatial preservation
    if has_tokens and not has_conv:
        # Pure transformer - weak spatial locality
        return {
            "spatial_preservation": "low",
            "reason": "Token-based architecture with weak spatial inductive bias",
        }
    elif has_skip_connections:
        # U-Net style - preserves and restores
        return {
            "spatial_preservation": "high",
            "reason": "Skip connections preserve and restore spatial detail across scales",
        }
    elif has_conv:
        # CNN-based - strong spatial locality
        return {
            "spatial_preservation": "high",
            "reason": "Convolutional layers maintain strong spatial locality",
        }
    else:
        # Fallback
        return {
            "spatial_preservation": "medium",
            "reason": "Mixed or unclear spatial processing strategy",
        }


def summarize_scaling_behavior(graph: ArchitectureGraph) -> dict[str, str]:
    """
    Analyze how architecture complexity scales with input size.

    Rules:
    - Quadratic attention (O(n²)) → poor scaling
    - CNN-based → good scaling (linear in spatial dimensions)
    - Mixed or unclear → moderate

    Args:
        graph: ArchitectureGraph to analyze

    Returns:
        Dictionary with:
        - scaling: "good" | "moderate" | "poor"
        - reason: Human-readable explanation
    """
    # Check for quadratic attention
    has_quadratic_attention = any(
        str(node.semantic_params.get("attention", "")).lower() == "quadratic"
        for node in graph.nodes
    )

    # Check for CNN dominance
    conv_nodes = [n for n in graph.nodes if str(n.type).lower() == "conv2d"]
    total_nodes = len(graph.nodes)
    is_cnn_dominated = len(conv_nodes) > 0 and len(conv_nodes) >= total_nodes * 0.3

    if has_quadratic_attention:
        return {
            "scaling": "poor",
            "reason": "Quadratic attention complexity (O(n²)) scales poorly with input size",
        }
    elif is_cnn_dominated:
        return {
            "scaling": "good",
            "reason": "Convolutional architecture scales linearly with spatial dimensions",
        }
    else:
        return {
            "scaling": "moderate",
            "reason": "Mixed architecture with balanced computational complexity",
        }


def compare_graphs(graph_a: ArchitectureGraph, graph_b: ArchitectureGraph) -> dict[str, Any]:
    """
    Compare two architectures across compute, spatial, and scaling dimensions.

    Args:
        graph_a: First architecture to compare
        graph_b: Second architecture to compare

    Returns:
        Dictionary with structured comparison across:
        - compute: Computational characteristics
        - spatial: Spatial preservation behavior
        - scaling: Input size scaling properties
        - summary: High-level comparison insights
    """
    # Get summaries for both graphs
    compute_a = summarize_compute(graph_a)
    compute_b = summarize_compute(graph_b)

    spatial_a = summarize_spatial_behavior(graph_a)
    spatial_b = summarize_spatial_behavior(graph_b)

    scaling_a = summarize_scaling_behavior(graph_a)
    scaling_b = summarize_scaling_behavior(graph_b)

    # Generate comparative insights
    summary = []

    # Compute comparison
    if compute_a["total_high_flops"] > compute_b["total_high_flops"]:
        summary.append(
            f"{graph_a.name} is more compute-intensive ({compute_a['total_high_flops']} vs {compute_b['total_high_flops']} high-FLOPs nodes)"
        )
    elif compute_a["total_high_flops"] < compute_b["total_high_flops"]:
        summary.append(
            f"{graph_b.name} is more compute-intensive ({compute_b['total_high_flops']} vs {compute_a['total_high_flops']} high-FLOPs nodes)"
        )
    else:
        summary.append(
            f"Both architectures have similar compute intensity ({compute_a['total_high_flops']} high-FLOPs nodes each)"
        )

    # Spatial comparison
    spatial_order = {"high": 3, "medium": 2, "low": 1}
    spatial_a_val = spatial_order.get(spatial_a["spatial_preservation"], 2)
    spatial_b_val = spatial_order.get(spatial_b["spatial_preservation"], 2)

    if spatial_a_val > spatial_b_val:
        summary.append(f"{graph_a.name} preserves spatial structure better")
    elif spatial_a_val < spatial_b_val:
        summary.append(f"{graph_b.name} preserves spatial structure better")
    else:
        summary.append("Both architectures have similar spatial preservation")

    # Scaling comparison
    scaling_order = {"good": 3, "moderate": 2, "poor": 1}
    scaling_a_val = scaling_order.get(scaling_a["scaling"], 2)
    scaling_b_val = scaling_order.get(scaling_b["scaling"], 2)

    if scaling_a_val > scaling_b_val:
        summary.append(f"{graph_a.name} scales better with input size")
    elif scaling_a_val < scaling_b_val:
        summary.append(f"{graph_b.name} scales better with input size")
    else:
        summary.append("Both architectures have similar scaling behavior")

    return {
        "graph_a": {
            "name": graph_a.name,
            "compute": compute_a,
            "spatial": spatial_a,
            "scaling": scaling_a,
        },
        "graph_b": {
            "name": graph_b.name,
            "compute": compute_b,
            "spatial": spatial_b,
            "scaling": scaling_b,
        },
        "summary": summary,
    }
