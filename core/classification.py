from core.architecture_graph import ArchitectureGraph

def classify_architecture(graph: ArchitectureGraph) -> str:
    """
    Deterministically classify the architecture graph into one of:
    CNN, Transformer, Encoder-Decoder, Hybrid, Unknown.
    """
    types = [node.type.lower() for node in graph.nodes]
    
    has_conv = any("conv" in t for t in types)
    has_attention = any("attention" in t or "transformer" in t or "selfatt" in t for t in types)
    has_upsample = any("upsample" in t or "convtranspose" in t or "upconv" in t for t in types)
    
    if has_conv and has_attention:
        return "Hybrid"
    elif has_conv and has_upsample:
        return "Encoder-Decoder"
    elif has_conv and not has_attention:
        return "CNN"
    elif has_attention and not has_conv:
        return "Transformer"
    else:
        return "Unknown"
