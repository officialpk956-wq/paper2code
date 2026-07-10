"""
Concrete VisualizationAgent implementation.

Wraps existing Graphviz rendering and comparison styling from app.py.
No new logic — extraction and structuring only.
"""

import graphviz

from core.agents.types import (
    ComparisonContext,
    NodeVisuals,
    VisualizationMode,
    VisualizationOptions,
    VisualRepresentation,
)
from core.agents.visualization_agent import VisualizationAgent
from core.architecture_graph import ArchitectureGraph, GraphNode

# ---------------------------------------------------------------------------
# Constants (extracted verbatim from app.py)
# ---------------------------------------------------------------------------

_FLOPS_COLORS: dict[str, dict[str, str]] = {
    "high": {"color": "#FF4444", "penwidth": "2.5"},
    "very high": {"color": "#FF0000", "penwidth": "3"},
    "medium": {"color": "#FFA500", "penwidth": "2"},
}

_ROLE_COLORS: dict[str, str] = {
    "patch_embedding": "#9b59b6",  # Purple
    "attention": "#e67e22",  # Orange
    "residual": "#3498db",  # Blue
    "normalization": "#95a5a6",  # Gray
    "projection": "#2ecc71",  # Green
}

_EDGE_STYLES: dict[str, dict[str, str]] = {
    "flow": {"style": "solid", "color": "black", "penwidth": "1.5"},
    "skip": {"style": "dashed", "color": "blue", "penwidth": "1.5"},
    "residual": {"style": "dashed", "color": "red", "penwidth": "1.5"},
}

_DEFAULT_GRAPH_ATTRS: dict[str, str] = {"rankdir": "TB"}


# ---------------------------------------------------------------------------
# DefaultVisualizationAgent
# ---------------------------------------------------------------------------


class DefaultVisualizationAgent(VisualizationAgent):
    """
    Concrete VisualizationAgent.

    Wraps Graphviz rendering and semantic-based styling.
    Extracts logic from app.py — no new behaviour introduced.
    """

    def render(
        self,
        graph: ArchitectureGraph,
        mode: VisualizationMode = "single",
        comparison_ctx: ComparisonContext | None = None,
        options: VisualizationOptions | None = None,
    ) -> VisualRepresentation:
        """
        Render ArchitectureGraph to VisualRepresentation.

        Single mode  → FLOPs-based coloring only.
        Compare mode → Priority-ordered comparison highlighting.
        """
        if mode == "compare" and comparison_ctx is None:
            raise ValueError("comparison_ctx is required when mode='compare'")

        opts = options or {}
        rankdir = opts.get("rankdir", "TB")
        include_params = opts.get("include_params", True)
        expand_composite = opts.get("expand_composite", False)

        node_annotations: dict[str, NodeVisuals] = {}
        applied_cues: list = []

        dot = graphviz.Digraph(
            comment=graph.name,
            graph_attr={"rankdir": rankdir},
        )

        # ----------------------------------------------------------------
        # Nodes
        # ----------------------------------------------------------------
        for node in graph.nodes:
            sp = node.semantic_params or {}
            role = sp.get("semantic_role") or sp.get("compute_role", "unknown")
            tooltip = f"Role: {role}\nInput: {node.input_shape}\nOutput: {node.output_shape}\n\n{node.get_explanation()}"

            label = _build_label(node, mode, include_params)
            node_attrs: dict[str, str] = {
                "label": label,
                "shape": "box",
                "style": "rounded",
                "tooltip": tooltip.strip(),
            }

            if mode == "compare" and comparison_ctx is not None:
                visuals, cue = _apply_compare_styling(node, comparison_ctx)
                if cue and cue not in applied_cues:
                    applied_cues.append(cue)
            else:
                visuals, cue = _apply_single_styling(node)
                if cue and cue not in applied_cues:
                    applied_cues.append(cue)

            # Merge visuals into node_attrs
            if "label_suffix" in visuals:
                node_attrs["label"] = node_attrs["label"] + visuals["label_suffix"]
            for key in ("color", "penwidth", "fillcolor", "style"):
                if key in visuals:
                    node_attrs[key] = str(visuals[key])

            node_annotations[node.id] = visuals
            dot.node(node.id, **node_attrs)

        # ----------------------------------------------------------------
        # Edges
        # ----------------------------------------------------------------
        for edge in graph.edges:
            style = dict(_EDGE_STYLES.get(edge.edge_type, _EDGE_STYLES["flow"]))

            # Show tensor shape on edge if available
            if edge.tensor_shape:
                style["label"] = str(edge.tensor_shape)
                style["fontsize"] = "10"
                style["fontcolor"] = "#666666"

            dot.edge(edge.source, edge.target, **style)

        return {
            "graphviz_dot": dot.source,
            "node_annotations": node_annotations,
            "visual_cues": applied_cues,
            "comparison_mode": mode == "compare",
        }

    def get_visual_cues(self) -> set[str]:
        return {
            "flops_color",
            "bottleneck_badge",
            "compute_highlight",
            "scaling_highlight",
            "spatial_highlight",
            "ghost_overlay",
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_label(node: GraphNode, mode: str, include_params: bool) -> str:
    """Build node label string with tensor shapes and parameters."""
    header = f"{node.label}\n({node.type})"

    # Show shapes prominently
    shape_str = ""
    if node.input_shape and node.output_shape:
        shape_str = f"\n{node.input_shape} → {node.output_shape}"
    elif node.output_shape:
        shape_str = f"\nOutput: {node.output_shape}"

    label = f"{header}{shape_str}"

    if include_params:
        # Filter out internal repeat metadata for cleaner view
        for k, v in node.params.items():
            if not k.startswith("_"):
                label += f"\n{k}: {v}"

    # Show semantic params in single mode only (reduces clutter in compare)
    if mode != "compare" and node.semantic_params:
        for k, v in node.semantic_params.items():
            if not k.startswith("_") and k not in ["compute_role", "flops", "semantic_role"]:
                label += f"\n{k}: {v}"

    return label


def _apply_single_styling(node: GraphNode):
    """
    Apply semantic role or FLOPs-based coloring for single mode.
    """
    sp = node.semantic_params or {}
    role = sp.get("semantic_role") or sp.get("compute_role")
    flops = sp.get("flops")

    # Priority 1: Semantic Role color
    if role in _ROLE_COLORS:
        return {"color": _ROLE_COLORS[role], "penwidth": 3.0, "style": "rounded,bold"}, "role_color"

    # Priority 2: FLOPs-based color
    if flops in _FLOPS_COLORS:
        entry = _FLOPS_COLORS[flops]
        visuals: NodeVisuals = {
            "color": entry["color"],
            "penwidth": float(entry["penwidth"]),
        }
        return visuals, "flops_color"

    return {}, None


def _apply_compare_styling(node: GraphNode, ctx: ComparisonContext):
    """
    Apply priority-ordered comparison highlighting (extracted from app.py).

    Priority:
    1. bottleneck_badge   — node.id == bottleneck_node_id
    2. compute_highlight  — dominant_compute == current AND flops high
    3. scaling_highlight  — scaling_issue == current AND attention quadratic
    4. spatial_highlight  — dominant_spatial == current AND skip_connection yes
    5. ghost_overlay      — all remaining nodes

    Returns (NodeVisuals, cue_name).
    """
    current = ctx.get("current_arch")
    sp = node.semantic_params or {}

    # 1. Bottleneck badge
    if node.id == ctx.get("bottleneck_node_id"):
        visuals: NodeVisuals = {
            "color": "#CC0000",
            "penwidth": 4.0,
            "label_suffix": "\nCOMPUTE BOTTLENECK",
        }
        return visuals, "bottleneck_badge"

    # 2. Compute highlight
    if ctx.get("dominant_compute") == current:
        if sp.get("flops") in ("high", "very high"):
            visuals = {
                "color": "#FF6666",
                "penwidth": 3.0,
            }
            return visuals, "compute_highlight"

    # 3. Scaling warning
    if ctx.get("scaling_issue") == current:
        if sp.get("attention") == "quadratic":
            visuals = {
                "color": "#FFA500",
                "penwidth": 3.0,
                "label_suffix": "\nQuadratic Scaling",
            }
            return visuals, "scaling_highlight"

    # 4. Spatial highlight
    if ctx.get("dominant_spatial") == current:
        if sp.get("skip_connection") == "yes":
            visuals = {
                "color": "#4169E1",
                "penwidth": 2.5,
            }
            return visuals, "spatial_highlight"

    # 5. Ghost overlay (all remaining nodes in compare mode)
    visuals = {
        "color": "#CCCCCC",
        "penwidth": 1.0,
        "style": "rounded,filled",
        "fillcolor": "#F8F8F8",
    }
    return visuals, "ghost_overlay"
