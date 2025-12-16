from src.diagram_base import create_graph



def draw_vit(schema):
    dot = create_graph("Vision Transformer")

    dot.node("input", "Image")
    dot.node("patch", "Patch Embedding")
    dot.node("tokens", "Tokens + CLS")
    dot.edge("input", "patch")
    dot.edge("patch", "tokens")

    prev = "tokens"

    for i, stage in enumerate(schema["stages"]):
        node = f"enc{i}"
        dot.node(node, f"Transformer x{stage['repeats']}")
        dot.edge(prev, node)
        prev = node

    dot.node("head", "MLP Head")
    dot.edge(prev, "head")

    return dot
