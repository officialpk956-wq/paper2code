"""
Concrete implementation of the ParsingAgent.

Responsibility: Convert raw architecture descriptions into ArchitectureGraph.
Handles ConfigDict, PaperExcerpt, and SymbolicDesc input formats.
Integrates the RAG layer (ConfigExtractor, SymbolicParser) for text/symbolic processing.
"""

from core.agents.config_parser import ConfigParsingAgent, ParsingError
from core.agents.parsing_agent import ParsingAgent
from core.agents.types import ParsingSource
from core.architecture_graph import ArchitectureGraph

# RAG dependencies are imported locally to avoid circular imports with src.agents.__init__


class ParsingAgentImpl(ParsingAgent):
    """
    Concrete ParsingAgent implementation.

    Acts as a facade:
    1. Determines the source format.
    2. Routes text/symbolic formats to the RAG layer to get a ConfigDict.
    3. Delegates ConfigDict parsing to the deterministic ConfigParsingAgent.
    """

    def __init__(self, use_llm: bool = True):
        from core.rag.config_extractor import ConfigExtractor

        self._config_parser = ConfigParsingAgent()
        self._extractor = ConfigExtractor(use_llm=use_llm)

    def parse(self, source: ParsingSource, format_hint: str = "auto") -> ArchitectureGraph:
        """
        Parse architecture specification into graph structure.
        """
        if not isinstance(source, dict):
            raise ParsingError(f"Expected dict, got {type(source).__name__}")

        # Route based on format hint or structural duck-typing
        config_dict = None

        if format_hint == "config" or ("layers" in source and "name" in source):
            # Already a ConfigDict
            config_dict = source

        elif format_hint == "symbolic" or source.get("type") == "symbolic":
            # Symbolic description (e.g., "Conv2D(64)→ReLU→...")
            spec = source.get("spec")
            if not spec:
                raise ParsingError("SymbolicDesc must have a 'spec' field")
            from core.rag.symbolic_parser import parse_symbolic

            config_dict = parse_symbolic(spec)

        elif format_hint in ("text", "markdown", "pdf_extract") or source.get("type") in (
            "text",
            "markdown",
            "pdf_extract",
        ):
            # Raw text excerpt from paper
            content = source.get("content")
            if not content:
                raise ParsingError("PaperExcerpt must have a 'content' field")

            # Use the RAG extraction layer (handles full PDFs vs snippets automatically)
            config_dict = self._extractor.extract_from_text(content)

        else:
            raise ParsingError("Ambiguous or unsupported parsing source format.")

        # Delegate final graph construction to the deterministic config parser
        return self._config_parser.parse(config_dict, format_hint="config")
