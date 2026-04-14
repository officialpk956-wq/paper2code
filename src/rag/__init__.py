"""
RAG (Retrieval-Augmented Generation) layer for config extraction.

Converts raw architecture descriptions into ConfigDict format.
Uses LLM with temperature=0 for deterministic extraction.
"""

from src.rag.config_extractor import ConfigExtractor, preprocess_text
from src.rag.normalizer import normalize_config

__all__ = ["ConfigExtractor", "preprocess_text", "normalize_config"]
