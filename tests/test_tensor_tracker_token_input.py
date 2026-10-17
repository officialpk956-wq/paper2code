"""A graph that starts at an embedding table gets token-id input, not an image."""

from core.architecture_graph import ArchitectureGraph, GraphNode
from core.rag.tensor_tracker import TensorTracker


def _graph(*types: str) -> ArchitectureGraph:
    g = ArchitectureGraph(name="t")
    for i, t in enumerate(types, 1):
        g.add_node(GraphNode(id=f"layer_{i}", type=t, label=t, params={"embed_dim": 512}))
        if i > 1:
            g.add_edge(f"layer_{i - 1}", f"layer_{i}")
    return g


def test_embedding_first_graph_propagates_from_token_ids():
    g = _graph("token_embedding", "positional_embedding")
    TensorTracker().propagate_shapes(g)
    assert g.nodes[0].input_shape == ("B", 128)
    assert g.nodes[-1].output_shape == ("B", 128, 512)


def test_vision_first_graph_still_gets_image_input():
    g = _graph("conv2d")
    TensorTracker().propagate_shapes(g)
    assert g.nodes[0].input_shape == ("B", 3, 224, 224)
