"""Rule-based semantic explainer for architecture graphs."""

from src.architecture_graph import GraphNode, ArchitectureGraph


def explain_node(node: GraphNode) -> str:
    """
    Generate a human-readable explanation for a node.
    Combines educational KAG-driven explanation with technical metrics.
    """
    from src.rag.semantic_explainer import SemanticExplainer
    
    # 1. Educational Explanation (The "Why")
    educational = SemanticExplainer.explain(
        node.type, 
        node.semantic_params.get("semantic_role") or node.semantic_params.get("compute_role"),
        node.params
    )
    
    # 2. Technical Context (The "What")
    technical = []
    
    # Analyze semantic parameters
    if node.semantic_params:
        # FLOPS / Compute intensity
        flops_val = node.semantic_params.get("flops")
        if flops_val == "high":
            technical.append("• **High computational cost**")
        elif flops_val == "very high":
            technical.append("• **Very high cost** (Quadratic attention complexity)")
            
        # Skip connections
        if node.semantic_params.get("skip_connection") == "yes":
            technical.append("• **Skip connection enabled**")
            
        # Spatial info
        fm = node.semantic_params.get("feature_map")
        if fm == "downsampling":
            technical.append("• **Downsampling layer** (Reduces resolution)")
        elif fm == "upsampling":
            technical.append("• **Upsampling layer** (Increases resolution)")

    # Combine
    result = educational
    if technical:
        result += "\n\n" + "\n".join(technical)
        
    return result


def explain_graph(graph: ArchitectureGraph) -> str:
    """
    Generate a human-readable summary explanation for an entire graph.
    
    Covers:
    - Architecture name and scale
    - Number of nodes and composite blocks
    - Computational bottlenecks
    - Overall design pattern
    
    Args:
        graph: ArchitectureGraph to explain
        
    Returns:
        str: Multi-line explanation paragraph
    """
    
    lines = []
    
    # Header
    lines.append(f"**{graph.name}**")
    
    # Scale metrics
    num_nodes = len(graph.nodes)
    num_edges = len(graph.edges)
    composite_blocks = sum(1 for n in graph.nodes if n.is_composite())
    
    lines.append(f"\nArchitecture size: {num_nodes} nodes, {num_edges} connections.")
    
    if composite_blocks > 0:
        lines.append(f"Contains {composite_blocks} composite block(s) with internal structure.")
    
    # Analyze computation hotspots
    high_flops_nodes = [n for n in graph.nodes if n.semantic_params.get("flops") in ["high", "very high"]]
    
    if high_flops_nodes:
        high_flops_names = [n.label for n in high_flops_nodes]
        if len(high_flops_names) == 1:
            lines.append(f"\nPrimary computational bottleneck: {high_flops_names[0]}.")
        else:
            lines.append(f"\nKey computation-heavy stages: {', '.join(high_flops_names)}.")
    
    # Detect architecture pattern
    if "Residual" in graph.name or any("Residual" in n.label for n in graph.nodes):
        lines.append("\nDesign pattern: Deep residual network with identity shortcuts for improved gradient flow.")
    elif "U-Net" in graph.name:
        lines.append("\nDesign pattern: Encoder-decoder with skip connections for feature restoration.")
    elif "Transformer" in graph.name or "ViT" in graph.name:
        lines.append("\nDesign pattern: Transformer-based architecture using self-attention for global reasoning.")
    
    return "".join(lines)
