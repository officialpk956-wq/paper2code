"""
Explanation Agent interface.

Responsibility: Narrate existing reasoning in natural language.
Deterministic, template-based, no invention, no recommendations.
"""

from abc import ABC, abstractmethod
from typing import Dict

from src.architecture_graph import ArchitectureGraph, GraphNode
from src.agents.types import ComparisonResult, VisualMetadata, TemplateLibrary


class ExplanationAgent(ABC):
    """
    Abstract agent for generating natural-language explanations.

    Contract:
    ─────────
    INPUT:  ArchitectureGraph + comparison results + visual metadata
    OUTPUT: Markdown explanation

    GUARANTEE: Deterministic, template-based narration

    Core Constraint: Must not invent facts. Only reorganize what is known.

    MUST DO:
    ✓ Narrate existing facts from semantic_params and comparison results
    ✓ Use pre-defined explanation templates (no free-form generation)
    ✓ Reference visual cues accurately
    ✓ Produce deterministic output
    ✓ Clearly state when architectures are similar/tied
    ✓ Link explanations to semantic reasoning

    MUST NOT DO:
    ✗ Invent facts not in semantic_params or comparison results
    ✗ Override comparison conclusions
    ✗ Generate novel architectural insights
    ✗ Make recommendations beyond "suitable for X task"
    ✗ Use LLMs or generative models
    ✗ Create visualization objects
    ✗ Perform comparisons (use results provided)
    ✗ Have side effects
    """

    @abstractmethod
    def explain_node(self, node: GraphNode) -> str:
        """
        Explain a single node in human-readable language.

        Args:
            node: GraphNode to explain
                Expected fields: label, type, description, semantic_params

        Returns:
            Markdown string explaining the node's role and properties.

        Uses:
        - node.description (from Parsing Agent)
        - node.semantic_params (from Parsing Agent)

        Constraints:
        - Must reference only information in the node
        - Must not compare to other nodes
        - Must not make architectural judgments
        - Must use pre-defined templates
        """
        raise NotImplementedError

    @abstractmethod
    def explain_graph(self, graph: ArchitectureGraph) -> str:
        """
        Explain the overall architecture.

        Args:
            graph: ArchitectureGraph to explain

        Returns:
            Markdown string describing architecture purpose and design.

        Uses:
        - graph.nodes and semantic_params
        - Node descriptions
        - Structural patterns (e.g., "encoder → decoder")

        Constraints:
        - Must not compare to other architectures
        - Must not make recommendations
        - Must narrate only what semantic_params reveal
        """
        raise NotImplementedError

    @abstractmethod
    def explain_comparison(
        self,
        graph_a: ArchitectureGraph,
        graph_b: ArchitectureGraph,
        comparison_result: ComparisonResult,
        visual_metadata: VisualMetadata,
    ) -> str:
        """
        Explain architectural differences.

        Args:
            graph_a: First architecture
            graph_b: Second architecture
            comparison_result: Output from summarize_compute/spatial/scaling functions
            visual_metadata: Visual highlighting metadata from Visualization Agent

        Returns:
            Markdown string narrating why one architecture differs.

        Uses:
        - summarize_compute() results (compute differences)
        - summarize_spatial_behavior() results
        - summarize_scaling_behavior() results
        - compare_graphs() results (if available)
        - VisualMetadata (bottleneck labels, highlights)

        Constraints:
        - Must use facts only from comparison_result
        - Must not override comparison conclusions
        - Must reference visual cues via VisualMetadata
        - Must handle ties/similarities explicitly
        - Must use pre-defined templates

        Typical Structure:
        1. Identify primary difference drivers from comparison_result
        2. Reference visual metadata (bottleneck, highlights)
        3. Use pre-defined templates
        4. Link to semantic reasoning ("because {node} has {param}")
        5. If tie: explicitly report similarity
        """
        raise NotImplementedError

    @abstractmethod
    def get_explanation_templates(self) -> TemplateLibrary:
        """
        Expose explanation templates (for debugging and validation).

        Returns:
            TemplateLibrary containing:
            - semantic_templates: Fact → explanation phrase mapping
            - visual_references: Visual cue → reference text mapping
            - comparison_templates: Comparison-specific templates
            - tie_templates: Templates for similar architectures

        Used for:
        - Understanding what explanations are possible
        - Validating determinism
        - Auditing for hallucinations
        - Documentation
        """
        raise NotImplementedError
