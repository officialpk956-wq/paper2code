from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str = "flow"  # flow | skip | residual
    tensor_shape: tuple | None = None  # Tracks the shape flowing through this edge


@dataclass
class ArchitectureGraph:
    name: str
    nodes: list["GraphNode"] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: "GraphNode"):
        self.nodes.append(node)

    def add_edge(self, source: str, target: str, edge_type="flow"):
        self.edges.append(GraphEdge(source, target, edge_type=edge_type))


@dataclass
class GraphNode:
    id: str
    type: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    block: str | None = None
    description: str | None = None
    semantic_params: dict[str, Any] = field(default_factory=dict)

    # Tensor flow tracking
    input_shape: tuple | None = None
    output_shape: tuple | None = None

    # 🔥 NEW (core of Phase 3.7)
    internal_graph: Optional["ArchitectureGraph"] = None

    def is_composite(self) -> bool:
        return self.internal_graph is not None

    def get_patch_info(self) -> dict[str, Any]:
        """Helper to extract patch-related metadata if available."""
        if self.type != "patchembedding":
            return {}
        return {
            "patch_size": self.semantic_params.get("patch_size") or self.params.get("patch_size"),
            "num_patches": self.semantic_params.get("num_patches")
            or self.params.get("num_patches"),
            "embed_dim": self.semantic_params.get("embed_dim")
            or self.params.get("embed_dim")
            or self.params.get("embedding_dim"),
        }

    def get_explanation(self) -> str:
        """Get educational explanation for this node."""
        from core.rag.semantic_explainer import SemanticExplainer

        return SemanticExplainer.explain(
            self.type,
            self.semantic_params.get("semantic_role") or self.semantic_params.get("compute_role"),
            self.params,
        )
