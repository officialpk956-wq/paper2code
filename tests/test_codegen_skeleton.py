"""
Regression tests for core.codegen._generate_skeleton -- the final
LLM-generation fallback for architectures that don't map to a known
family. Had zero test coverage before the Phase 2 second audit, which
found and fixed two bugs that made it crash on essentially any realistic
graph:

1. forward() referenced self.{node.id} for every node, but __init__ only
   defined that attribute when _node_to_layer() recognized the node's
   type -- and _node_to_layer's MAP doesn't cover residualblock,
   bottleneckblock, concat, or identity, all types used elsewhere in this
   same pipeline. Any node of an unmapped type crashed with
   AttributeError on the first line of forward().
2. _node_to_layer's in_hs computation used hasattr(node, "input_shape") to
   guard a subscript, but GraphNode.input_shape is a dataclass field that
   defaults to None -- hasattr is True regardless of the value, so
   node.input_shape[-1] crashed with TypeError for any node without
   tensor_tracker-populated shape info (the common case for
   skeleton-fallback graphs).
"""

import torch

from core.architecture_graph import ArchitectureGraph, GraphNode
from core.codegen import _generate_skeleton, _node_to_layer


def _exec_skeleton(code: str, class_name: str):
    namespace: dict = {}
    exec(compile(code, "<skeleton-test>", "exec"), namespace)
    return namespace[class_name]


def test_skeleton_handles_node_type_with_no_input_shape():
    """Regression test for bug 2: input_shape=None must not crash _node_to_layer."""
    graph = ArchitectureGraph(name="NoShapeModel")
    graph.add_node(GraphNode(id="layer_0", type="linear", label="head", params={}))

    # input_shape defaults to None; _node_to_layer must not crash computing in_hs.
    layer_str = _node_to_layer(graph.nodes[0])
    assert layer_str == "nn.Linear(512, 512)"


def test_skeleton_handles_unrecognized_layer_types_without_crashing():
    """
    Regression test for bug 1: a node whose type isn't in _node_to_layer's
    MAP (e.g. a diffusion model's timestep_embedding, or residualblock/
    concat/identity -- all real types used elsewhere in this pipeline)
    must not produce a forward() call to an attribute that was never
    defined in __init__.
    """
    graph = ArchitectureGraph(name="DDPM")
    graph.add_node(GraphNode(id="layer_0", type="conv2d", label="stem", params={"channels": 3}))
    graph.add_node(
        GraphNode(id="layer_1", type="timestep_embedding", label="time_embed", params={})
    )
    graph.add_node(GraphNode(id="layer_2", type="residualblock", label="res", params={}))
    graph.add_node(GraphNode(id="layer_3", type="relu", label="act", params={}))

    code = _generate_skeleton(graph)

    # The unrecognized nodes must not get a self.layer_N attribute
    # reference in __init__ that was never defined.
    assert "self.layer_1 = " not in code
    assert "self.layer_2 = " not in code
    # ...but forward() must still be well-formed Python (compiles + runs).
    model_cls = _exec_skeleton(code, "DDPM")
    model = model_cls()
    output = model(torch.randn(1, 3, 224, 224))
    assert output.shape[0] == 1


def test_skeleton_produces_runnable_code_for_a_fully_recognized_graph():
    """
    When every node type is recognized and channel counts are internally
    consistent, the generated skeleton must compile, instantiate, and run
    a real forward pass end to end -- not just avoid crashing on
    generation.
    """
    graph = ArchitectureGraph(name="SimpleCNN")
    graph.add_node(GraphNode(id="layer_0", type="conv2d", label="conv", params={"channels": 3}))
    graph.add_node(GraphNode(id="layer_1", type="relu", label="act", params={}))
    graph.add_node(GraphNode(id="layer_2", type="maxpool2d", label="pool", params={}))

    code = _generate_skeleton(graph)
    model_cls = _exec_skeleton(code, "SimpleCNN")
    model = model_cls()

    output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 3, 112, 112)  # conv (same-pad) -> relu -> maxpool halves spatial dims


def test_skeleton_sanitizes_class_name_from_graph_name():
    graph = ArchitectureGraph(name="1st-Weird Name!")
    graph.add_node(GraphNode(id="layer_0", type="relu", label="act", params={}))

    code = _generate_skeleton(graph)
    assert "class _1stWeirdName(nn.Module):" in code
