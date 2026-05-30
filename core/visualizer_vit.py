from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge

def build_vit_graph() -> ArchitectureGraph:
    graph = ArchitectureGraph(name="Vision Transformer")

    patch = GraphNode(
        id="patch",
        type="PatchEmbedding",
        label="Patch Embedding",
        params={"patch": "16×16"},
        block="Embedding",
        description="Converts image patches into token embeddings",
        semantic_params={
            "tokens": "196",
            "patch_size": "16×16"
        }
    )
    graph.add_node(patch)

    prev = patch

    # -------- Transformer Encoders --------
    for i in range(1, 5):
        block_id = f"encoder_{i}"

        block = GraphNode(
            id=block_id,
            type="TransformerEncoder",
            label=f"Encoder Layer {i}",
            params={"heads": 12},
            block="Transformer",
            description="Applies self-attention to model global relationships",
            semantic_params={
                "tokens": "constant",
                "flops": "very high",
                "attention": "quadratic"
            }
        )

        attn = GraphNode(f"{block_id}_attn", "MHSA", "Multi-Head Attention")
        mlp  = GraphNode(f"{block_id}_mlp", "MLP", "Feed Forward")

        block.internal_nodes = [attn, mlp]
        block.internal_edges = [
            GraphEdge(attn.id, mlp.id, "flow"),
            GraphEdge(block_id, attn.id, "residual"),
            GraphEdge(mlp.id, block_id, "residual"),
        ]

        graph.add_node(block)
        graph.add_edge(prev.id, block.id, "flow")
        prev = block

    # -------- Head --------
    graph.add_node(GraphNode(
        id="head",
        type="MLPHead",
        label="Classification Head",
        params={"classes": 1000},
        block="Head",
        description="Maps the final token representation to class predictions"
    ))

    graph.add_edge(prev.id, "head", "flow")

    return graph
