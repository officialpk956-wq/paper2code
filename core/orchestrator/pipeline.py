"""
Paper2Code Orchestration Pipeline.

Pure wiring layer connecting agents and analysis functions.
NO reasoning logic, NO duplication, FULLY deterministic.

Agents:
- ParsingAgent: config → graph
- VisualizationAgent: graph → visuals
- ExplanationAgent: graph/comparison → explanation

Pure functions:
- summarize_compute/spatial/scaling: graph → summary
- compare_graphs: graphs → comparison result
"""

from typing import Any, Dict, Optional

from core.agents import (
    ParsingAgentImpl,
    DefaultVisualizationAgent,
    DefaultExplanationAgent,
    ParsingSource,
    VisualizationOptions,
)
from core.rag.tensor_tracker import TensorTracker, TensorMismatchError
from core.rag.knowledge_graph import KnowledgeGraph
from core.rag.diff_engine import GraphDiffEngine
from core.comparators import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
    compare_graphs,
)
from core.rag import ConfigExtractor, preprocess_text


class Paper2CodePipeline:
    """
    Pure orchestration layer.

    Wires agents and analysis functions without introducing new logic.
    All reasoning comes from agents and pure functions, never from pipeline.
    """

    # Maximum layers per architecture (safety guard for UI/performance)
    MAX_LAYERS = 75

    # Maximum cache size (prevent unbounded growth)
    MAX_CACHE_SIZE = 100

    def __init__(
        self,
        parsing_agent: Optional[ParsingAgentImpl] = None,
        visualization_agent: Optional[DefaultVisualizationAgent] = None,
        explanation_agent: Optional[DefaultExplanationAgent] = None,
    ):
        """Initialize pipeline with agents (or create defaults)."""
        self.parser = parsing_agent or ParsingAgentImpl()
        self.visualizer = visualization_agent or DefaultVisualizationAgent()
        self.explainer = explanation_agent or DefaultExplanationAgent()
        self.tensor_tracker = TensorTracker()
        self.knowledge_graph = KnowledgeGraph()
        self.diff_engine = GraphDiffEngine()

        # Optional caching for text-based inputs
        self._text_cache: Dict[str, Dict[str, Any]] = {}

    def _apply_layer_safety_cap(self, config: ParsingSource) -> Dict[str, Any]:
        """
        Apply safety guard: cap layers at MAX_LAYERS.

        Returns config dict with optional _truncated flag if capped.
        """
        if not isinstance(config, dict):
            return config

        config_dict = dict(config)  # Copy to avoid mutation
        layers = config_dict.get("layers", [])

        if len(layers) > self.MAX_LAYERS:
            config_dict["layers"] = layers[: self.MAX_LAYERS]
            config_dict["_truncated"] = True
            config_dict["_original_layer_count"] = len(layers)

        return config_dict

    def run_single(
        self,
        config: ParsingSource,
        vis_options: Optional[VisualizationOptions] = None,
    ) -> Dict[str, Any]:
        """
        Run single architecture analysis.

        Flow: apply safety cap → parse → visualize (single) → explain graph

        Returns:
            {
                "graph": ArchitectureGraph,
                "visual": VisualRepresentation,
                "explanation": str,
                "metadata": dict (with truncation info if applicable)
            }
        """
        # Apply safety guard (cap layers at MAX_LAYERS)
        capped_config = self._apply_layer_safety_cap(config)
        was_truncated = capped_config.get("_truncated", False)

        graph = self.parser.parse(capped_config)
        
        # 1.5 Tensor Validation (Phase 1 Compiler Step)
        try:
            self.tensor_tracker.propagate_shapes(graph)
        except TensorMismatchError as e:
            if not hasattr(graph, "metadata"):
                graph.metadata = {}
            graph.metadata["validation_warning"] = str(e)

        # 1.6 KAG Symbolic Reasoning (Phase 2 — Motif + Topology)
        kag_motifs: list = []
        kag_anomalies: list = []
        kag_semantic_roles: dict = {}
        try:
            kag_motifs = self.knowledge_graph.detect_motifs(graph)
            kag_anomalies = self.knowledge_graph.verify_topology(graph)
            # Collect semantic roles for every node
            for node in graph.nodes:
                role = self.knowledge_graph.get_semantic_role(node.type.lower())
                if role:
                    kag_semantic_roles[node.id] = role
                    # Back-annotate so explain_node picks it up
                    if "semantic_role" not in node.semantic_params:
                        node.semantic_params["semantic_role"] = role
        except Exception:
            pass  # KAG is best-effort

        visual = self.visualizer.render(graph, mode="single", options=vis_options)
        explanation = self.explainer.explain_graph(graph)

        # Always include metadata for consistency
        metadata = {
            "truncated": was_truncated,
            "layer_count": len(graph.nodes),
        }

        if was_truncated:
            metadata["original_layer_count"] = capped_config.get("_original_layer_count")

        result = {
            "graph": graph,
            "visual": visual,
            "explanation": explanation,
            "metadata": metadata,
            "kag_motifs": kag_motifs,
            "kag_anomalies": kag_anomalies,
            "kag_semantic_roles": kag_semantic_roles,
        }

        return result

    def run_comparison(
        self,
        config_a: ParsingSource,
        config_b: ParsingSource,
        vis_options: Optional[VisualizationOptions] = None,
    ) -> Dict[str, Any]:
        """
        Run comparative architecture analysis.

        Flow:
        1. Parse both configs
        2. Analyze each (summarize_*)
        3. Compare
        4. Derive comparison contexts (direct from summaries)
        5. Visualize both (compare mode)
        6. Extract visual metadata (from viz outputs only)
        7. Explain comparison

        Returns:
            {
                "graph_a": ArchitectureGraph,
                "graph_b": ArchitectureGraph,
                "comparison_result": dict,
                "visual_a": VisualRepresentation,
                "visual_b": VisualRepresentation,
                "visual_metadata": dict,
                "explanation": str
            }
        """
        # Apply safety guard to both configs (cap layers at MAX_LAYERS)
        capped_config_a = self._apply_layer_safety_cap(config_a)
        capped_config_b = self._apply_layer_safety_cap(config_b)

        # Parse configs
        graph_a = self.parser.parse(capped_config_a)
        graph_b = self.parser.parse(capped_config_b)

        # Analyze each graph (pure functions)
        compute_a = summarize_compute(graph_a)
        spatial_a = summarize_spatial_behavior(graph_a)
        scaling_a = summarize_scaling_behavior(graph_a)

        compute_b = summarize_compute(graph_b)
        spatial_b = summarize_spatial_behavior(graph_b)
        scaling_b = summarize_scaling_behavior(graph_b)

        # Compare graphs using the new Diff Engine
        comparison_result = self.diff_engine.compare(graph_a, graph_b)

        # Derive comparison contexts (direct from summaries, no new logic)
        dominant_compute = (
            "A" if compute_a["total_high_flops"] > compute_b["total_high_flops"]
            else "B" if compute_b["total_high_flops"] > compute_a["total_high_flops"]
            else None
        )

        spatial_levels = {"high": 3, "medium": 2, "low": 1}
        spatial_a_val = spatial_levels.get(spatial_a["spatial_preservation"], 2)
        spatial_b_val = spatial_levels.get(spatial_b["spatial_preservation"], 2)
        dominant_spatial = (
            "A" if spatial_a_val > spatial_b_val
            else "B" if spatial_b_val > spatial_a_val
            else None
        )

        scaling_issue = (
            "A" if scaling_a["scaling"] == "poor" and scaling_b["scaling"] != "poor"
            else "B" if scaling_b["scaling"] == "poor" and scaling_a["scaling"] != "poor"
            else None
        )

        # Build comparison contexts (all fields from summaries only)
        ctx_a = {
            "mode": "compare",
            "current_arch": "A",
            "dominant_compute": dominant_compute,
            "dominant_spatial": dominant_spatial,
            "scaling_issue": scaling_issue,
            "bottleneck_node_id": compute_a.get("primary_bottleneck"),
        }

        ctx_b = {
            "mode": "compare",
            "current_arch": "B",
            "dominant_compute": dominant_compute,
            "dominant_spatial": dominant_spatial,
            "scaling_issue": scaling_issue,
            "bottleneck_node_id": compute_b.get("primary_bottleneck"),
        }

        # Visualize (agent decides styling)
        visual_a = self.visualizer.render(
            graph=graph_a,
            mode="compare",
            comparison_ctx=ctx_a,
            options=vis_options,
        )

        visual_b = self.visualizer.render(
            graph=graph_b,
            mode="compare",
            comparison_ctx=ctx_b,
            options=vis_options,
        )

        # Extract visual metadata (from viz outputs only, no recomputation)
        visual_metadata = {
            "highlighted_nodes_a": dict(visual_a.get("node_annotations", {})),
            "highlighted_nodes_b": dict(visual_b.get("node_annotations", {})),
            "visual_cues": list(set(visual_a.get("visual_cues", []) + visual_b.get("visual_cues", []))),
        }

        # Build comparison result for explainer
        comparison_result_typed = {
            "compute_summary_a": {
                "total_high_flops": compute_a["total_high_flops"],
                "primary_bottleneck": compute_a.get("primary_bottleneck"),
            },
            "compute_summary_b": {
                "total_high_flops": compute_b["total_high_flops"],
                "primary_bottleneck": compute_b.get("primary_bottleneck"),
            },
            "spatial_summary_a": spatial_a,
            "spatial_summary_b": spatial_b,
            "scaling_summary_a": scaling_a,
            "scaling_summary_b": scaling_b,
        }

        # Explain (agent adds all reasoning)
        explanation = self.explainer.explain_comparison(
            graph_a=graph_a,
            graph_b=graph_b,
            comparison_result=comparison_result_typed,
            visual_metadata=visual_metadata,
        )

        # Prepare metadata about potential truncations
        truncated_a = capped_config_a.get("_truncated", False)
        truncated_b = capped_config_b.get("_truncated", False)

        metadata = {
            "truncated_a": truncated_a,
            "truncated_b": truncated_b,
            "layer_count_a": len(graph_a.nodes),
            "layer_count_b": len(graph_b.nodes),
        }

        if truncated_a:
            metadata["original_layer_count_a"] = capped_config_a.get("_original_layer_count")
        if truncated_b:
            metadata["original_layer_count_b"] = capped_config_b.get("_original_layer_count")

        return {
            "graph_a": graph_a,
            "graph_b": graph_b,
            "comparison_result": comparison_result,
            "visual_a": visual_a,
            "visual_b": visual_b,
            "visual_metadata": visual_metadata,
            "explanation": explanation,
            "metadata": metadata,
        }

    def run_from_text(
        self,
        text: str,
        vis_options: Optional[VisualizationOptions] = None,
    ) -> Dict[str, Any]:
        """
        Run single architecture analysis from raw text.
        """
        # Normalize text for stable cache key
        cache_key = preprocess_text(text)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        source: ParsingSource = {"type": "text", "content": text}
        result = self.run_single(source, vis_options)

        if len(self._text_cache) >= self.MAX_CACHE_SIZE:
            self._text_cache.clear()

        self._text_cache[cache_key] = result
        return result

    def run_comparison_from_text(
        self,
        text_a: str,
        text_b: str,
        vis_options: Optional[VisualizationOptions] = None,
    ) -> Dict[str, Any]:
        """
        Run comparative architecture analysis from raw text.
        """
        source_a: ParsingSource = {"type": "text", "content": text_a}
        source_b: ParsingSource = {"type": "text", "content": text_b}
        return self.run_comparison(source_a, source_b, vis_options)
