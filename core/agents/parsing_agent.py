"""
Parsing Agent interface.

Responsibility: Convert raw architecture descriptions into ArchitectureGraph.
Deterministic, no visualization, no reasoning, no comparison.
"""

from abc import ABC, abstractmethod

from core.agents.types import ParsingSource
from core.architecture_graph import ArchitectureGraph


class ParsingAgent(ABC):
    """
    Abstract agent for parsing architecture specifications.

    Contract:
    ─────────
    INPUT:  Raw architecture specification (config dict, paper excerpt, or symbolic notation)
    OUTPUT: ArchitectureGraph with semantic_params attached

    GUARANTEE: Deterministic (same input → same output)

    MUST DO:
    ✓ Parse well-formed architecture specifications
    ✓ Attach semantic_params to every node
    ✓ Validate that edges reference existing nodes
    ✓ Produce deterministic output
    ✓ Include descriptions for every node

    MUST NOT DO:
    ✗ Create visualization objects
    ✗ Invent semantic params beyond type-based defaults
    ✗ Make architectural judgments ("this is inefficient")
    ✗ Perform comparisons
    ✗ Reference external architectures as examples
    ✗ Have side effects
    """

    @abstractmethod
    def parse(self, source: ParsingSource, format_hint: str = "auto") -> ArchitectureGraph:
        """
        Parse architecture specification into graph structure.

        Args:
            source: Architecture specification in one of:
                - ConfigDict: {"name": "ResNet-18", "layers": [...]}
                - PaperExcerpt: {"type": "markdown", "content": "..."}
                - SymbolicDesc: {"type": "symbolic", "spec": "Conv2D(64)→..."}
                - Dict: Direct dict (for flexibility)

            format_hint: Optional format specification to guide parsing.
                Values: "yaml", "json", "symbolic", "auto" (default)
                Used for disambiguation if source format is ambiguous.

        Returns:
            ArchitectureGraph with:
            - nodes: List[GraphNode] with id, label, type, params, description
            - edges: List[GraphEdge] with source, target, edge_type
            - semantic_params: Attached to each node

        Raises:
            ParsingError: If source is malformed, ambiguous, or invalid.

        Determinism Guarantee:
            Given the same source and format_hint, this method will always
            produce an identical ArchitectureGraph.
        """
        raise NotImplementedError
