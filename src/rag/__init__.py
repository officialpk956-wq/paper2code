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

from src.rag.config_extractor import (
    ConfigExtractor,
    preprocess_text,
    extract_table_layers,
)
from src.rag.normalizer import normalize_config
from src.rag.symbolic_parser import parse_symbolic
from src.rag.section_splitter import (
    get_architecture_text,
    split_into_sections,
    chunk_for_retrieval,
    score_chunks_by_density,
)
from src.rag.retriever import retrieve_top_chunks, retrieve_and_merge
from src.rag.knowledge_graph import KnowledgeGraph
from src.rag.semantic_explainer import SemanticExplainer

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
