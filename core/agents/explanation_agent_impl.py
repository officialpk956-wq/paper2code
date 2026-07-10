"""
Concrete ExplanationAgent implementation.

Wraps existing explanation functions from explainers and comparators modules.
No new logic — extraction and structuring only.
"""

from core.agents.explanation_agent import ExplanationAgent
from core.agents.types import ComparisonResult, TemplateLibrary, VisualMetadata
from core.architecture_graph import ArchitectureGraph, GraphNode
from core.comparators import explain_architecture_comparison as _explain_arch_comparison
from core.explainers import explain_graph as _explain_graph
from core.explainers import explain_node as _explain_node

# ---------------------------------------------------------------------------
# Template Library (extracted from existing explanation logic)
# ---------------------------------------------------------------------------

_EXPLANATION_TEMPLATES: TemplateLibrary = {
    "semantic_templates": {
        ("flops", "high"): "This block is compute-heavy",
        ("flops", "very high"): "This block performs very expensive computations",
        ("flops", "medium"): "This block has moderate computational cost",
        ("attention", "quadratic"): "Scales poorly with sequence/spatial length (O(n²))",
        (
            "feature_map",
            "downsampling",
        ): "Reduces spatial resolution to capture contextual features",
        (
            "feature_map",
            "upsampling",
        ): "Recovers spatial resolution while maintaining semantic information",
        ("skip_connection", "yes"): "Uses skip connections to preserve earlier features",
        ("compute_role", "feature_extraction"): "Feature extraction and filtering",
        ("compute_role", "pooling"): "Spatial downsampling and feature aggregation",
        ("compute_role", "reconstruction"): "Spatial upsampling and detail recovery",
        ("compute_role", "attention"): "Global context aggregation via attention",
        ("compute_role", "normalization"): "Feature normalization for stable training",
        ("compute_role", "activation"): "Non-linear transformation for expressiveness",
    },
    "visual_references": {
        "bottleneck_badge": "🔥 COMPUTE BOTTLENECK — Primary computational hotspot",
        "compute_highlight": "🔴 High-FLOPs nodes driving compute differences",
        "scaling_highlight": "🟠 Quadratic scaling bottleneck (attention blocks)",
        "spatial_highlight": "🔵 Skip connections enabling spatial preservation",
        "ghost_overlay": "⚪ Shared structure with no significant difference",
    },
    "comparison_templates": {
        "dominant_compute": "requires more computation ({count} vs {other} high-FLOPs nodes)",
        "dominant_spatial": "preserves spatial structure better",
        "scaling_issue": "scales poorly with increasing input size (quadratic complexity)",
        "bottleneck": "primary computational bottleneck: {bottleneck}",
    },
    "tie_templates": {
        "compute": "Both architectures have similar compute intensity",
        "spatial": "Both architectures have similar spatial preservation",
        "scaling": "Both architectures have similar scaling behavior",
    },
}


# ---------------------------------------------------------------------------
# DefaultExplanationAgent
# ---------------------------------------------------------------------------


class DefaultExplanationAgent(ExplanationAgent):
    """
    Concrete ExplanationAgent.

    Wraps existing explanation functions from explainers and comparators.
    Extracts logic — no new reasoning introduced.
    """

    def explain_node(self, node: GraphNode) -> str:
        """
        Explain a single node using semantic_params.

        Returns:
            Markdown string with node explanation
        """
        return _explain_node(node)

    def explain_graph(self, graph: ArchitectureGraph) -> str:
        """
        Explain overall architecture, including repeated structures.

        Scans for repeat metadata (_repeat_group, _repeat_total) in semantic_params
        and documents any repeated blocks found in the architecture.

        Returns:
            Markdown string describing architecture purpose and design
        """
        base_explanation = _explain_graph(graph)

        # Scan for repeated structures in the graph
        # CRITICAL: Read from params, as config extractor places repeat metadata there
        repeat_groups: dict[str, int] = {}
        for node in graph.nodes:
            if not node.params:
                continue

            group = node.params.get("_repeat_group")
            total = node.params.get("_repeat_total", 1)

            # Only count groups where _repeat_total > 1 (actual repetition)
            if group and total > 1:
                if group not in repeat_groups:
                    repeat_groups[group] = total

        # Add repeated patterns section if any found
        if repeat_groups:
            patterns_section = "\n### Repeated Patterns\n\n"

            # Sort groups deterministically (alphabetical for stable output)
            for group in sorted(repeat_groups.keys()):
                count = repeat_groups[group]
                # Format: "conv2d_block" → "conv2d block" (add spacing)
                group_name = group.replace("_block", " block") if "_block" in group else group
                patterns_section += f"- {group_name} repeated {count} times\n"

            return base_explanation + patterns_section

        return base_explanation

    def explain_comparison(
        self,
        graph_a: ArchitectureGraph,
        graph_b: ArchitectureGraph,
        comparison_result: ComparisonResult,
        visual_metadata: VisualMetadata,
    ) -> str:
        """
        Explain architectural differences.

        Returns:
            Markdown string narrating differences using comparison_result
        """
        # Get base comparison explanation
        base_explanation = _explain_arch_comparison(graph_a, graph_b)

        # Append visual references if available
        visual_notes = []

        if visual_metadata.get("bottleneck_a"):
            visual_notes.append(
                f"\n**Visual Highlight (A):** {_EXPLANATION_TEMPLATES['visual_references'].get('bottleneck_badge', '')} — {visual_metadata['bottleneck_a']}"
            )

        if visual_metadata.get("bottleneck_b"):
            visual_notes.append(
                f"\n**Visual Highlight (B):** {_EXPLANATION_TEMPLATES['visual_references'].get('bottleneck_badge', '')} — {visual_metadata['bottleneck_b']}"
            )

        if visual_metadata.get("visual_cues_applied"):
            cues_str = ", ".join(visual_metadata["visual_cues_applied"])
            visual_notes.append(f"\n**Applied Visual Cues:** {cues_str}")

        if visual_notes:
            base_explanation += "\n" + "".join(visual_notes)

        return base_explanation

    def get_explanation_templates(self) -> TemplateLibrary:
        """
        Expose explanation templates for validation and debugging.

        Returns:
            TemplateLibrary with semantic, visual, comparison, and tie templates
        """
        return _EXPLANATION_TEMPLATES.copy()
