from core.architecture_graph import ArchitectureGraph, GraphEdge, GraphNode


def build_vit_graph(
    hidden_size: int = 768, num_heads: int = 12, depth: int = 4
) -> ArchitectureGraph:
    graph = ArchitectureGraph(name="Vision Transformer")

    patch = GraphNode(
        id="patch",
        type="PatchEmbedding",
        label="Patch Embedding",
        params={"patch": "16×16", "embed_dim": hidden_size},
        block="Embedding",
        description="Converts image patches into token embeddings",
        semantic_params={"tokens": "196", "patch_size": "16×16"},
    )
    graph.add_node(patch)

    prev = patch

    # -------- Transformer Encoders --------
    for i in range(1, depth + 1):
        block_id = f"encoder_{i}"

        block = GraphNode(
            id=block_id,
            type="TransformerEncoder",
            label=f"Encoder Layer {i}",
            params={"heads": num_heads, "d_model": hidden_size},
            block="Transformer",
            description="Applies self-attention to model global relationships",
            semantic_params={"tokens": "constant", "flops": "very high", "attention": "quadratic"},
        )

        attn = GraphNode(
            f"{block_id}_attn", "MHSA", "Multi-Head Attention", params={"num_heads": num_heads}
        )
        mlp = GraphNode(f"{block_id}_mlp", "MLP", "Feed Forward")

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
    graph.add_node(
        GraphNode(
            id="head",
            type="MLPHead",
            label="Classification Head",
            params={"classes": 1000, "in_features": hidden_size},
            block="Head",
            description="Maps the final token representation to class predictions",
        )
    )

    graph.add_edge(prev.id, "head", "flow")

    return graph


def build_transformer_graph(
    hidden_size: int = 512,
    num_heads: int = 8,
    depth: int = 6,
    vocab_size: int = 32000,
    seq_len: int = 128,
) -> ArchitectureGraph:
    """Encoder-stack Transformer over tokens: same blocks as the ViT, but a token embedding in
    front and a vocabulary projection at the end instead of patches and an image classifier."""
    graph = build_vit_graph(hidden_size=hidden_size, num_heads=num_heads, depth=depth)
    graph.name = "Transformer"

    graph.nodes[0] = GraphNode(
        id="tok_embed",
        type="Embedding",
        label="Token Embedding",
        params={"embed_dim": hidden_size, "vocab_size": vocab_size, "seq_len": seq_len},
        block="Embedding",
        description="Maps token ids to vectors and adds positional information",
    )
    for edge in graph.edges:
        if edge.source == "patch":
            edge.source = "tok_embed"

    head = graph.nodes[-1]
    head.label = "Output Projection"
    head.params = {"classes": vocab_size, "in_features": hidden_size}
    head.description = "Projects each token representation onto the vocabulary"
    return graph
