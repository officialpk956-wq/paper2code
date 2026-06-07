from core.rag.tensor_tracker import TensorTracker
from core.architecture_graph import ArchitectureGraph, GraphNode
import json

graph = ArchitectureGraph(name="MobileNetV2")
n1 = GraphNode(id="n1", type="Conv2d", label="Conv 3x3", params={"channels": 32, "stride": 2})
n2 = GraphNode(id="n2", type="InvertedResidual", label="Inverted Residual 1", params={"channels": 16})

graph.add_node(n1)
graph.add_node(n2)
graph.add_edge("n1", "n2", "flow")

tracker = TensorTracker()
tracker.propagate_shapes(graph)

for ev in tracker.flops_events:
    print(ev['node_type'], ev['params_M'])
