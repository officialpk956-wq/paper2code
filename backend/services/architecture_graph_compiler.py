"""
Architecture Graph Compiler for Phase 14B.

Converts an ArchitectureBlueprint (produced by Phase 14A) into an
ExecutableGraph — a validated, shape-annotated, FLOPs-annotated graph
suitable for export to JSON / Mermaid / Graphviz DOT.

Reuses:
  - TensorTracker  (core/rag/tensor_tracker.py)  — shape inference
  - FLOPsEngine values are already embedded in blueprint.tensor_flow
  - ArchitectureBlueprint data model from Phase 14A
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EGNode:
    id: str
    name: str
    type: str
    shape: List[int]
    params: int
    flops: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EGEdge:
    source: str
    target: str
    edge_type: str  # sequential | residual | concat | cross_attention

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    valid: bool
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ValidationReport":
        return cls(
            valid=d.get("valid", True),
            errors=d.get("errors", []),
            warnings=d.get("warnings", []),
        )


@dataclass
class ExecutableGraph:
    id: str
    nodes: List[EGNode]
    edges: List[EGEdge]
    input_spec: List[int]
    output_spec: List[int]
    parameter_count: int
    flops: float
    validation_status: str  # "valid" | "warnings" | "invalid"
    validation_report: ValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "input_spec": self.input_spec,
            "output_spec": self.output_spec,
            "parameter_count": self.parameter_count,
            "flops": self.flops,
            "validation_status": self.validation_status,
            "validation_report": self.validation_report.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ExecutableGraph":
        return cls(
            id=d["id"],
            nodes=[EGNode(**n) for n in d.get("nodes", [])],
            edges=[EGEdge(**e) for e in d.get("edges", [])],
            input_spec=d.get("input_spec", []),
            output_spec=d.get("output_spec", []),
            parameter_count=d.get("parameter_count", 0),
            flops=d.get("flops", 0.0),
            validation_status=d.get("validation_status", "valid"),
            validation_report=ValidationReport.from_dict(
                d.get("validation_report", {"valid": True, "errors": [], "warnings": []})
            ),
        )


# ---------------------------------------------------------------------------
# Type mappings
# ---------------------------------------------------------------------------

# Blueprint component.type → ArchitectureGraph node.type (for TensorTracker)
_COMP_TO_GRAPH_TYPE: Dict[str, str] = {
    "conv":            "conv2d",
    "attention":       "multiheadattention",
    "embedding":       "patchembedding",
    "pooling":         "maxpool2d",
    "mlp":             "feedforward",
    "normalization":   "layernorm",
    "activation":      "relu",
    "skip_connection": "residual_add",
    "upsample":        "upsample",
    "downsample":      "conv2d",
    "head":            "linear",
}

# Blueprint connection.connection_type → ArchitectureGraph edge.edge_type
_CONN_TO_EDGE: Dict[str, str] = {
    "sequential":    "flow",
    "residual":      "residual",
    "concat":        "skip",
    "cross_attention": "flow",
}


# ---------------------------------------------------------------------------
# Shape inference via TensorTracker
# ---------------------------------------------------------------------------

def _run_tracker(
    components: List[Dict],
    connections: List[Dict],
    input_shape: List[int],
) -> Dict[str, List[int]]:
    """
    Build a minimal ArchitectureGraph and run TensorTracker to infer shapes.
    Returns a mapping {component_id: output_shape_list}.
    Falls back gracefully if TensorTracker raises.
    """
    from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge as AGEdge
    from core.rag.tensor_tracker import TensorTracker

    graph_nodes = []
    for comp in components:
        meta = comp.get("metadata", {}) or {}
        # Prefer the FLOPs-engine-level type stored in metadata; fall back to mapping
        node_type = (
            meta.get("flops_type")
            or _COMP_TO_GRAPH_TYPE.get(comp.get("type", ""), comp.get("type", ""))
        )
        graph_nodes.append(GraphNode(
            id=comp["id"],
            type=node_type,
            label=comp.get("name", comp["id"]),
            params={**meta},
            input_shape=None,
            output_shape=None,
        ))

    graph_edges = []
    for conn in connections:
        edge_type = _CONN_TO_EDGE.get(conn.get("connection_type", "sequential"), "flow")
        graph_edges.append(AGEdge(
            source=conn["source"],
            target=conn["target"],
            edge_type=edge_type,
        ))

    ag = ArchitectureGraph(
        name="executable-graph",
        nodes=graph_nodes,
        edges=graph_edges,
        metadata={},
    )

    tracker = TensorTracker()
    initial: Optional[tuple] = tuple(input_shape) if input_shape else None

    try:
        tracker.propagate_shapes(ag, initial_shape=initial)
    except Exception:
        pass  # Partial results in node.output_shape are still usable

    result: Dict[str, List[int]] = {}
    for node in ag.nodes:
        if node.output_shape and isinstance(node.output_shape, (list, tuple)):
            try:
                result[node.id] = [
                    int(d) if isinstance(d, (int, float)) else 1
                    for d in node.output_shape
                ]
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _has_cycle(nodes: List[EGNode], edges: List[EGEdge]) -> bool:
    """Detect a directed cycle using DFS with a recursion-stack set."""
    adj: Dict[str, List[str]] = {n.id: [] for n in nodes}
    for e in edges:
        if e.source in adj:
            adj[e.source].append(e.target)

    visited: set = set()
    in_stack: set = set()

    def dfs(nid: str) -> bool:
        if nid in in_stack:
            return True
        if nid in visited:
            return False
        visited.add(nid)
        in_stack.add(nid)
        for child in adj.get(nid, []):
            if dfs(child):
                return True
        in_stack.discard(nid)
        return False

    for n in nodes:
        if n.id not in visited:
            if dfs(n.id):
                return True
    return False


def validate_graph(graph: ExecutableGraph) -> ValidationReport:
    """
    Check the graph for structural validity:
      - Directed cycles
      - Dangling edge references
      - Disconnected nodes
      - Missing input / output nodes
      - Sequential shape mismatches
    """
    errors: List[str] = []
    warnings: List[str] = []

    node_ids = {n.id for n in graph.nodes}

    # 1. Cycle detection
    if _has_cycle(graph.nodes, graph.edges):
        errors.append("Graph contains a directed cycle")

    # 2. Dangling edges
    for e in graph.edges:
        if e.source not in node_ids:
            errors.append(f"Edge references unknown source node: {e.source!r}")
        if e.target not in node_ids:
            errors.append(f"Edge references unknown target node: {e.target!r}")

    # 3. Disconnected nodes (not reachable via any edge)
    if len(graph.nodes) > 1:
        connected: set = set()
        for e in graph.edges:
            connected.add(e.source)
            connected.add(e.target)
        for n in graph.nodes:
            if n.id not in connected:
                warnings.append(f"Node {n.id!r} is not connected to any edge")

    # 4. Missing input / output
    incoming = {e.target for e in graph.edges}
    outgoing = {e.source for e in graph.edges}
    has_input_node = any(n.id not in incoming for n in graph.nodes)
    has_output_node = any(n.id not in outgoing for n in graph.nodes)
    if graph.nodes and not has_input_node:
        errors.append("No input node found — all nodes have incoming edges")
    if graph.nodes and not has_output_node:
        errors.append("No output node found — all nodes have outgoing edges")

    # 5. Shape consistency on sequential edges
    shape_by_id: Dict[str, List[int]] = {n.id: n.shape for n in graph.nodes}
    in_shape_by_id: Dict[str, Optional[List[int]]] = {}
    for n in graph.nodes:
        raw = n.metadata.get("input_shape")
        if isinstance(raw, (list, tuple)) and raw:
            in_shape_by_id[n.id] = list(raw)

    for e in graph.edges:
        if e.edge_type != "sequential":
            continue
        src_out = shape_by_id.get(e.source)
        tgt_in = in_shape_by_id.get(e.target)
        if src_out and tgt_in and list(src_out) != list(tgt_in):
            warnings.append(
                f"Shape mismatch on {e.source!r}→{e.target!r}: "
                f"out={src_out} in={tgt_in}"
            )

    valid = len(errors) == 0
    return ValidationReport(valid=valid, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Main compilation function
# ---------------------------------------------------------------------------

def compile_blueprint(blueprint_dict: Dict[str, Any]) -> ExecutableGraph:
    """
    Convert an ArchitectureBlueprint dict (Phase 14A) to an ExecutableGraph.

    Shape inference is done by passing through TensorTracker; values from the
    blueprint's tensor_flow are used as the authoritative FLOPs / params source.
    """
    components: List[Dict] = blueprint_dict.get("components", [])
    connections: List[Dict] = blueprint_dict.get("connections", [])
    tensor_flow: List[Dict] = blueprint_dict.get("tensor_flow", [])
    input_shape: List[int] = blueprint_dict.get("input_shape", [])
    output_shape: List[int] = blueprint_dict.get("output_shape", [])
    bp_id: str = blueprint_dict.get("id", "bp-0")

    # --- Shape lookup from blueprint tensor_flow (authoritative FLOPs source)
    flow_by_id: Dict[str, Dict] = {f["component_id"]: f for f in tensor_flow}

    # --- Shape inference via TensorTracker (reuse, not duplicate)
    tracker_shapes = _run_tracker(components, connections, input_shape)

    # --- Build EGNodes
    eg_nodes: List[EGNode] = []
    for comp in components:
        cid = comp["id"]
        flow = flow_by_id.get(cid, {})

        # Prefer TensorTracker shapes (shape-validated); fall back to tensor_flow / component
        out_shape: List[int] = (
            tracker_shapes.get(cid)
            or flow.get("output_shape")
            or comp.get("shape", [])
            or []
        )
        in_shape: Optional[List[int]] = (
            flow.get("input_shape")
            or None
        )

        params_m: float = float(flow.get("params_M", 0.0))
        flops_m: float = float(flow.get("flops_mflops", 0.0))

        eg_nodes.append(EGNode(
            id=cid,
            name=comp.get("name", cid),
            type=comp.get("type", ""),
            shape=out_shape,
            params=int(params_m * 1_000_000),
            flops=flops_m,
            metadata={
                **(comp.get("metadata") or {}),
                "input_shape": in_shape,
            },
        ))

    # --- Build EGEdges
    eg_edges: List[EGEdge] = [
        EGEdge(
            source=conn["source"],
            target=conn["target"],
            edge_type=conn.get("connection_type", "sequential"),
        )
        for conn in connections
    ]

    # --- Validate
    proto = ExecutableGraph(
        id=bp_id.replace("bp-", "eg-"),
        nodes=eg_nodes,
        edges=eg_edges,
        input_spec=input_shape,
        output_spec=output_shape,
        parameter_count=0,
        flops=0.0,
        validation_status="valid",
        validation_report=ValidationReport(valid=True, errors=[], warnings=[]),
    )
    report = validate_graph(proto)

    status = "valid"
    if not report.valid:
        status = "invalid"
    elif report.warnings:
        status = "warnings"

    total_params = sum(n.params for n in eg_nodes)
    total_flops = round(sum(n.flops for n in eg_nodes), 4)

    return ExecutableGraph(
        id=proto.id,
        nodes=eg_nodes,
        edges=eg_edges,
        input_spec=input_shape,
        output_spec=output_shape,
        parameter_count=total_params,
        flops=total_flops,
        validation_status=status,
        validation_report=report,
    )


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def _safe_id(raw: str) -> str:
    """Sanitize a node id for use in Mermaid and DOT syntax."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw)


def _shape_label(shape: List[int]) -> str:
    """Human-readable shape string, skipping batch dimension."""
    if not shape:
        return "?"
    dims = shape[1:] if len(shape) > 1 else shape
    return "×".join(str(d) for d in dims)


def export_graph_json(graph: ExecutableGraph) -> str:
    """Serialise the entire ExecutableGraph as indented JSON."""
    return json.dumps(graph.to_dict(), indent=2)


def export_graph_mermaid(graph: ExecutableGraph) -> str:
    """Generate a Mermaid flowchart string for the executable graph."""
    lines = ["flowchart TD"]
    for n in graph.nodes:
        safe = _safe_id(n.id)
        label = f'{n.name}\\n[{_shape_label(n.shape)}]'
        lines.append(f'  {safe}["{label}"]')
    for e in graph.edges:
        src = _safe_id(e.source)
        tgt = _safe_id(e.target)
        if e.edge_type in ("residual", "concat", "cross_attention"):
            arrow = "-.->"
        else:
            arrow = "-->"
        lines.append(f"  {src} {arrow} {tgt}")
    return "\n".join(lines)


def export_graph_dot(graph: ExecutableGraph) -> str:
    """Generate a Graphviz DOT string for the executable graph."""
    lines = [
        "digraph ExecutableGraph {",
        "  rankdir=TB;",
        '  node [shape=box fontname="monospace" style=filled fillcolor="#1e2030"];',
    ]
    for n in graph.nodes:
        safe = _safe_id(n.id)
        label = f'{n.name}\\n[{_shape_label(n.shape)}]'
        lines.append(f'  {safe} [label="{label}"];')
    for e in graph.edges:
        src = _safe_id(e.source)
        tgt = _safe_id(e.target)
        if e.edge_type in ("residual", "concat", "cross_attention"):
            style = " [style=dashed]"
        else:
            style = ""
        lines.append(f"  {src} -> {tgt}{style};")
    lines.append("}")
    return "\n".join(lines)
