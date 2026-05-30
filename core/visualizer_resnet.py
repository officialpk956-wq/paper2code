from core.architecture_graph import ArchitectureGraph, GraphNode


def build_residual_block_internal(block_name: str) -> ArchitectureGraph:
    g = ArchitectureGraph(name=f"{block_name}_internal")

    conv1 = GraphNode(
        id=f"{block_name}_conv1",
        type="Conv2D",
        label="Conv 3×3",
        params={"channels": "C → C"}
    )

    conv2 = GraphNode(
        id=f"{block_name}_conv2",
        type="Conv2D",
        label="Conv 3×3",
        params={"channels": "C → C"}
    )

    add = GraphNode(
        id=f"{block_name}_add",
        type="Add",
        label="Residual Add"
    )

    g.add_node(conv1)
    g.add_node(conv2)
    g.add_node(add)

    g.add_edge(conv1.id, conv2.id)
    g.add_edge(conv2.id, add.id)
    g.add_edge(block_name, add.id, edge_type="residual")

    return g


def build_resnet18_graph() -> ArchitectureGraph:
    graph = ArchitectureGraph(name="ResNet-18")

    # Stem
    graph.add_node(GraphNode(
        id="conv1",
        type="Conv2D",
        label="Conv 7×7",
        params={"stride": 2, "channels": "3 → 64"},
        block="Stem",
        description="Extracts low-level visual features from the input image",
        semantic_params={
            "feature_map": "112×112",
            "receptive_field": "7×7",
            "flops": "high"
        }
    ))

    graph.add_node(GraphNode(
        id="maxpool",
        type="MaxPool",
        label="MaxPool",
        params={"kernel": "3×3", "stride": 2},
        block="Stem"
    ))

    graph.add_edge("conv1", "maxpool")

    previous = "maxpool"

    # Residual stages
    for stage in range(1, 5):
        block_id = f"res_stage_{stage}"

        block_node = GraphNode(
            id=block_id,
            type="ResidualBlock",
            label=f"Residual Block (Stage {stage})",
            params={"blocks": 2},
            block="Backbone",
            description="Improves gradient flow using identity shortcut connections",
            semantic_params={
                "feature_map": "varies",
                "flops": "medium",
                "compute_role": "stabilization + depth"
            },
            internal_graph=build_residual_block_internal(block_id)
        )

        graph.add_node(block_node)
        graph.add_edge(previous, block_id)

        previous = block_id

    # Head
    graph.add_node(GraphNode(
        id="fc",
        type="Linear",
        label="Fully Connected",
        params={"out_features": 1000},
        block="Head",
        description="Maps learned features to output classes"
    ))

    graph.add_edge(previous, "fc")

    return graph
