"""
Visualization Agent interface.

Responsibility: Render ArchitectureGraphs into visual representations.
Deterministic, no graph construction, no reasoning, no semantic inference.
"""

from abc import ABC, abstractmethod
from typing import Optional, Set

from src.architecture_graph import ArchitectureGraph
from src.agents.types import (
    VisualizationMode,
    ComparisonContext,
    VisualizationOptions,
    VisualRepresentation,
)


class VisualizationAgent(ABC):
    """
    Abstract agent for rendering architecture graphs.

    Contract:
    ─────────
    INPUT:  ArchitectureGraph + optional comparison context
    OUTPUT: VisualRepresentation (Graphviz DOT + annotations)

    GUARANTEE: Deterministic (same input → same visuals)

    MUST DO:
    ✓ Render valid ArchitectureGraph to visual output
    ✓ Apply visual cues in priority order
    ✓ Produce deterministic output
    ✓ Support both single and comparison modes
    ✓ Consume comparison context without modifying it

    MUST NOT DO:
    ✗ Construct new graphs
    ✗ Infer semantic parameters
    ✗ Generate explanations or text
    ✗ Make architectural judgments
    ✗ Modify input graphs
    ✗ Compute summaries or comparisons
    ✗ Use visualization logic as reasoning
    """

    @abstractmethod
    def render(
        self,
        graph: ArchitectureGraph,
        mode: VisualizationMode = "single",
        comparison_ctx: Optional[ComparisonContext] = None,
        options: Optional[VisualizationOptions] = None,
    ) -> VisualRepresentation:
        """
        Render graph with semantic-aware visual cues.

        Args:
            graph: ArchitectureGraph to render

            mode: Visualization mode
                - "single": Standard single-graph visualization
                - "compare": Side-by-side comparison (requires comparison_ctx)

            comparison_ctx: Required if mode="compare"
                Contains: dominant_compute, dominant_spatial, scaling_issue, bottleneck_node_id
                Used to apply visual highlights in priority order.

            options: Visual configuration (optional)
                Controls: expand_composite, include_params, theme, rankdir

        Returns:
            VisualRepresentation containing:
            - graphviz_dot: Graphviz DOT language string
            - node_annotations: Dict mapping node_id → NodeVisuals
            - visual_cues: List of applied visual cue names
            - comparison_mode: Boolean indicating if comparison mode was used

        Raises:
            VisualizationError: If graph is invalid or comparison_ctx missing in compare mode

        Visual Cues (Priority Order in Compare Mode):
        ──────────────────────────────────────────
        1. bottleneck_badge (🔥)
           - Applied to: node matching bottleneck_node_id
           - Style: Dark red (#CC0000), thick border (4.0)

        2. compute_highlight (🔴)
           - Applied to: high-FLOPs nodes in dominant_compute architecture
           - Style: Light red (#FF6666), border (3.0)

        3. scaling_highlight (🟠)
           - Applied to: quadratic attention in scaling_issue architecture
           - Style: Orange (#FFA500), border (3.0), label_suffix: "⚠ Quadratic Scaling"

        4. spatial_highlight (🔵)
           - Applied to: skip connections in dominant_spatial architecture
           - Style: Blue (#4169E1), border (2.5)

        5. ghost_overlay (⚪)
           - Applied to: all non-highlighted nodes in compare mode
           - Style: Grey (#CCCCCC), thin border (1.0), filled (#F8F8F8)

        Single Mode:
        ────────────
        - Use semantic_params["flops"] for standard FLOPs coloring
        - No ghost overlay
        - No priority hierarchy

        Determinism Guarantee:
            Given the same graph, mode, comparison_ctx, and options,
            this method will always produce an identical VisualRepresentation.
        """
        raise NotImplementedError

    @abstractmethod
    def get_visual_cues(self) -> Set[str]:
        """
        Get list of supported visual cues.

        Returns:
            Set of string identifiers for visual cues this agent can apply.
            Example: {"bottleneck_badge", "compute_highlight", "scaling_highlight",
                     "spatial_highlight", "ghost_overlay", "tooltip"}

        Used for documentation and validation.
        """
        raise NotImplementedError
