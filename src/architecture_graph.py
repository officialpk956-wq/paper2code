from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str = "flow"   # flow | skip | residual


@dataclass
class ArchitectureGraph:
    name: str
    nodes: List["GraphNode"] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: "GraphNode"):
        self.nodes.append(node)

    def add_edge(self, source: str, target: str, edge_type="flow"):
        self.edges.append(GraphEdge(source, target, edge_type=edge_type))


@dataclass
class GraphNode:
    id: str
    type: str
    label: str
    params: Dict[str, any] = field(default_factory=dict)
    block: Optional[str] = None
    description: Optional[str] = None
    semantic_params: Dict[str, any] = field(default_factory=dict)

    # 🔥 NEW (core of Phase 3.7)
    internal_graph: Optional[ArchitectureGraph] = None

    def is_composite(self) -> bool:
        return self.internal_graph is not None
