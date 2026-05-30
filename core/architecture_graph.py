from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str = "flow"   # flow | skip | residual
    tensor_shape: Optional[tuple] = None  # Tracks the shape flowing through this edge


@dataclass
class ArchitectureGraph:
    name: str
    nodes: List["GraphNode"] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: "GraphNode"):
        self.nodes.append(node)

    def add_edge(self, source: str, target: str, edge_type="flow"):
        self.edges.append(GraphEdge(source, target, edge_type=edge_type))


@dataclass
class GraphNode:
    id: str
    type: str
    label: str
    params: Dict[str, Any] = field(default_factory=dict)
    block: Optional[str] = None
    description: Optional[str] = None
    semantic_params: Dict[str, Any] = field(default_factory=dict)
    
    # Tensor flow tracking
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None

    # 🔥 NEW (core of Phase 3.7)
    internal_graph: Optional["ArchitectureGraph"] = None

    def is_composite(self) -> bool:
        return self.internal_graph is not None

    def get_patch_info(self) -> Dict[str, Any]:
        """Helper to extract patch-related metadata if available."""
        if self.type != "patchembedding":
            return {}
        return {
            "patch_size": self.semantic_params.get("patch_size") or self.params.get("patch_size"),
            "num_patches": self.semantic_params.get("num_patches") or self.params.get("num_patches"),
            "embed_dim": self.semantic_params.get("embed_dim") or self.params.get("embed_dim") or self.params.get("embedding_dim"),
        }

    def get_explanation(self) -> str:
        """Get educational explanation for this node."""
        from core.rag.semantic_explainer import SemanticExplainer
        return SemanticExplainer.explain(
            self.type, 
            self.semantic_params.get("semantic_role") or self.semantic_params.get("compute_role"),
            self.params
        )
