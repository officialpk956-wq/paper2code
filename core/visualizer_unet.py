from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge

def build_unet_graph() -> ArchitectureGraph:
    graph = ArchitectureGraph(name="U-Net")

    encoder_blocks = []
    prev = None

    # ---------------- Encoder ----------------
    for i in range(1, 4):
        block_id = f"enc_{i}"

        block = GraphNode(
            id=block_id,
            type="EncoderBlock",
            label=f"Encoder Block {i}",
            params={"filters": 64 * 2**(i-1)},
            block="Encoder",
            description="Downsamples spatial resolution to capture contextual information",
            semantic_params={
                "feature_map": "downsampling",
                "flops": "medium"
            }
        )

        # Internal structure
        conv1 = GraphNode(f"{block_id}_conv1", "Conv2D", "Conv 3×3")
        conv2 = GraphNode(f"{block_id}_conv2", "Conv2D", "Conv 3×3")
        pool  = GraphNode(f"{block_id}_pool",  "MaxPool", "MaxPool")

        block.internal_nodes = [conv1, conv2, pool]
        block.internal_edges = [
            GraphEdge(conv1.id, conv2.id, "flow"),
            GraphEdge(conv2.id, pool.id, "flow"),
        ]

        graph.add_node(block)

        if prev:
            graph.add_edge(prev.id, block.id, edge_type="flow")

        prev = block
        encoder_blocks.append(block)

    # ---------------- Bottleneck ----------------
    bottleneck = GraphNode(
        id="bottleneck",
        type="Bottleneck",
        label="Bottleneck",
        params={"filters": 1024},
        block="Bottleneck",
        description="Connects encoder and decoder at the lowest resolution",
        semantic_params={
            "feature_map": "downsampling",
            "flops": "medium"
        }
    )

    bottleneck.internal_nodes = [
        GraphNode("bottleneck_conv1", "Conv2D", "Conv 3×3"),
        GraphNode("bottleneck_conv2", "Conv2D", "Conv 3×3"),
    ]
    bottleneck.internal_edges = [
        GraphEdge("bottleneck_conv1", "bottleneck_conv2", "flow")
    ]

    graph.add_node(bottleneck)
    graph.add_edge(prev.id, bottleneck.id, "flow")

    # ---------------- Decoder ----------------
    prev = bottleneck

    for enc in reversed(encoder_blocks):
        block_id = f"dec_{enc.id}"

        block = GraphNode(
            id=block_id,
            type="DecoderBlock",
            label=f"Decoder Block {enc.id[-1]}",
            params={"skip": enc.id},
            block="Decoder",
            description="Upsamples features while recovering spatial detail using skip connections",
            semantic_params={
                "feature_map": "upsampling",
                "skip_connection": "yes"
            }
        )

        up   = GraphNode(f"{block_id}_up", "UpConv", "UpConv")
        cat  = GraphNode(f"{block_id}_cat", "Concat", "Concat")
        conv = GraphNode(f"{block_id}_conv", "Conv2D", "Conv 3×3")

        block.internal_nodes = [up, cat, conv]
        block.internal_edges = [
            GraphEdge(up.id, cat.id, "flow"),
            GraphEdge(enc.id, cat.id, "skip"),
            GraphEdge(cat.id, conv.id, "flow"),
        ]

        graph.add_node(block)
        graph.add_edge(prev.id, block.id, "flow")
        prev = block

    # ---------------- Output ----------------
    graph.add_node(GraphNode(
        id="out",
        type="Conv2D",
        label="Output Layer",
        params={"channels": 1},
        block="Head"
    ))

    graph.add_edge(prev.id, "out", "flow")

    return graph
