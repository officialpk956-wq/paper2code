"""
BM25-based retrieval layer for architecture-aware chunk selection.

Improvement: Retrieval-Augmented Context Selection.
When paper sections are too large for the LLM context window, this module
uses BM25 scoring to surface the chunks with the highest concentration
of architectural keywords.

No external dependencies beyond standard library — BM25 is implemented
from scratch using TF-IDF weighting to keep the project lightweight.
"""

import math
import re
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# Architecture-focused query terms (used as the "query" for BM25 scoring)
# ---------------------------------------------------------------------------

_ARCH_QUERY_TERMS = [
    # Layer types
    "conv",
    "convolution",
    "convolutional",
    "linear",
    "dense",
    "attention",
    "transformer",
    "embedding",
    "pooling",
    "dropout",
    "batchnorm",
    "layernorm",
    "activation",
    "relu",
    "gelu",
    "softmax",
    "residual",
    "skip",
    "upsample",
    "downsample",
    # Structural terms
    "encoder",
    "decoder",
    "head",
    "neck",
    "backbone",
    "stem",
    "block",
    "layer",
    "module",
    "stage",
    # Parameter terms
    "channels",
    "filters",
    "kernel",
    "stride",
    "padding",
    "heads",
    "hidden",
    "dimension",
    "embedding",
    "patch",
    "token",
    # Quantitative signals
    "64",
    "128",
    "256",
    "512",
    "768",
    "1024",
    "2048",
]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and strips non-alphanumeric characters.
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return tokens


# ---------------------------------------------------------------------------
# BM25 Implementation
# ---------------------------------------------------------------------------


class BM25:
    """
    BM25 (Okapi BM25) scoring for chunk retrieval.

    Parameters follow the standard defaults:
        k1=1.5 (term saturation)
        b=0.75 (length normalization)
    """

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self._tokenized: list[list[str]] = [_tokenize(doc) for doc in corpus]
        self._doc_lengths = [len(doc) for doc in self._tokenized]
        self._avg_doc_len = sum(self._doc_lengths) / self.corpus_size if self.corpus_size else 1
        self._idf: dict = self._compute_idf()
        self._tf: list[Counter] = [Counter(doc) for doc in self._tokenized]

    def _compute_idf(self) -> dict:
        """Compute inverse document frequency for each unique term."""
        df: dict = defaultdict(int)
        for doc in self._tokenized:
            for term in set(doc):
                df[term] += 1

        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        """Compute BM25 score for a single document against a query."""
        score = 0.0
        doc_len = self._doc_lengths[doc_idx]
        tf_counter = self._tf[doc_idx]

        for term in query_terms:
            if term not in tf_counter:
                continue

            idf = self._idf.get(term, 0.0)
            tf = tf_counter[term]
            norm = (
                tf
                * (self.k1 + 1)
                / (tf + self.k1 * (1 - self.b + self.b * doc_len / self._avg_doc_len))
            )
            score += idf * norm

        return score

    def get_top_k(
        self,
        query_terms: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Retrieve top-k document indices with their BM25 scores.

        Returns:
            List of (doc_idx, score) tuples, sorted descending by score.
        """
        scores = [(i, self.score(query_terms, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_top_chunks(
    chunks: list[str],
    top_k: int = 5,
    query_terms: list[str] | None = None,
) -> list[str]:
    """
    Retrieve the most architecturally relevant chunks using BM25.

    Args:
        chunks: List of text chunks to search over.
        top_k: Number of top chunks to return.
        query_terms: Custom query terms. Defaults to _ARCH_QUERY_TERMS.

    Returns:
        Top-k chunks ordered by BM25 score (highest first).
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        return chunks

    query = query_terms or _ARCH_QUERY_TERMS

    bm25 = BM25(chunks)
    ranked = bm25.get_top_k(query, top_k=top_k)

    # Return chunks in their original order (preserve reading flow)
    selected_indices = sorted(idx for idx, _ in ranked)
    return [chunks[i] for i in selected_indices]


def retrieve_and_merge(
    chunks: list[str],
    top_k: int = 5,
    max_chars: int = 8_000,
    query_terms: list[str] | None = None,
) -> str:
    """
    Retrieve top-k chunks and merge into a single context string.

    Args:
        chunks: List of text chunks.
        top_k: Number of top chunks to retrieve.
        max_chars: Hard limit on output characters.
        query_terms: Optional custom BM25 query.

    Returns:
        Merged text from the most relevant chunks, within max_chars.
    """
    top_chunks = retrieve_top_chunks(chunks, top_k=top_k, query_terms=query_terms)
    merged = "\n\n---\n\n".join(top_chunks)
    return merged[:max_chars]
