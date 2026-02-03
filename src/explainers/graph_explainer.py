"""Rule-based semantic explainer for architecture graphs."""

from src.architecture_graph import GraphNode, ArchitectureGraph


def explain_node(node: GraphNode) -> str:
    """
    Generate a human-readable explanation for a node based on its properties.
    
    Uses rule-based logic:
    - semantic_params for compute/structure info
    - node.type for operation category
    - node.description for semantic purpose
    - node.label as fallback
    
    Args:
        node: GraphNode to explain
        
    Returns:
        str: Human-readable explanation
    """
    
    lines = []
    
    # Add description if available
    if node.description:
        lines.append(node.description)
    
    # Analyze semantic parameters
    if node.semantic_params:
        semantic_info = []
        
        # FLOPS / Compute intensity
        if "flops" in node.semantic_params:
            flops_val = node.semantic_params["flops"]
            if flops_val == "high":
                semantic_info.append("This block is compute-heavy")
            elif flops_val == "very high":
                semantic_info.append("This block performs very expensive computations (quadratic in complexity)")
            elif flops_val == "medium":
                semantic_info.append("This block has moderate computational cost")
        
        # Compute role
        if "compute_role" in node.semantic_params:
            role = node.semantic_params["compute_role"]
            semantic_info.append(f"Role: {role}")
        
        # Feature map / spatial info
        if "feature_map" in node.semantic_params:
            feature_map = node.semantic_params["feature_map"]
            if feature_map == "downsampling":
                semantic_info.append("Reduces spatial resolution to capture contextual features")
            elif feature_map == "upsampling":
                semantic_info.append("Recovers spatial resolution while maintaining semantic information")
            elif feature_map == "varies":
                semantic_info.append("Spatial resolution varies depending on stage parameters")
            elif feature_map != "constant":
                semantic_info.append(f"Spatial resolution: {feature_map}")
        
        # Tokens (Vision Transformer)
        if "tokens" in node.semantic_params:
            tokens = node.semantic_params["tokens"]
            if tokens == "constant":
                semantic_info.append("Processes tokens without changing count")
            elif tokens == "196":
                semantic_info.append("Processes 196 tokens (14×14 patches)")
            else:
                semantic_info.append(f"Token count: {tokens}")
        
        # Skip connections
        if "skip_connection" in node.semantic_params:
            if node.semantic_params["skip_connection"] == "yes":
                semantic_info.append("Uses skip connections to preserve earlier features")
        
        # Attention mechanism
        if "attention" in node.semantic_params:
            attention = node.semantic_params["attention"]
            semantic_info.append(f"Attention complexity: {attention}")
        
        # Receptive field
        if "receptive_field" in node.semantic_params:
            rf = node.semantic_params["receptive_field"]
            semantic_info.append(f"Receptive field: {rf}")
        
        # Patch size
        if "patch_size" in node.semantic_params:
            patch = node.semantic_params["patch_size"]
            semantic_info.append(f"Patch size: {patch}")
        
        if semantic_info:
            lines.extend(semantic_info)
    
    # Fallback: use label if no other info
    if not lines:
        lines.append(f"{node.label} ({node.type})")
    
    return "\n".join(lines)


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
