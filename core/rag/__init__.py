"""
RAG (Retrieval-Augmented Generation) layer for config extraction.

Converts raw architecture descriptions into ConfigDict format.
Uses LLM with temperature=0 for deterministic extraction.

Modules:
  config_extractor   - Main extraction pipeline (LLM + rule-based)
  normalizer         - Config normalization and validation
  symbolic_parser    - Symbolic notation parser (Conv(64)->ReLU->...)
  section_splitter   - Section-aware PDF text focusing (R2)
  retriever          - BM25 retrieval for large context reduction (R2)
  knowledge_graph    - KAG symbolic architecture reasoning engine
  semantic_explainer - Deterministic educational explanations
"""

from core.rag.config_extractor import (
    ConfigExtractor,
    extract_table_layers,
    preprocess_text,
)
from core.rag.knowledge_graph import KnowledgeGraph
from core.rag.normalizer import normalize_config
from core.rag.retriever import retrieve_and_merge, retrieve_top_chunks
from core.rag.section_splitter import (
    chunk_for_retrieval,
    get_architecture_text,
    score_chunks_by_density,
    split_into_sections,
)
from core.rag.semantic_explainer import SemanticExplainer
from core.rag.symbolic_parser import parse_symbolic

__all__ = [
    # Core extraction
    "ConfigExtractor",
    "preprocess_text",
    "extract_table_layers",
    # Normalization
    "normalize_config",
    # Symbolic input
    "parse_symbolic",
    # Section splitting
    "get_architecture_text",
    "split_into_sections",
    "chunk_for_retrieval",
    "score_chunks_by_density",
    # BM25 retrieval
    "retrieve_top_chunks",
    "retrieve_and_merge",
    # KAG reasoning
    "KnowledgeGraph",
    "SemanticExplainer",
]
