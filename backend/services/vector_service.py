"""
Semantic vector search for papers using Qdrant + sentence-transformers.

All methods are safe to call even when Qdrant is not configured:
they log a warning and return empty results rather than raising.
"""

import logging
import os

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")  # e.g. https://xxx.qdrant.io
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "papers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_DIM = 384  # bge-small-en output dimension

_qdrant_client = None
_embedder = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    if not QDRANT_URL:
        return None
    try:
        from qdrant_client import QdrantClient

        _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
        _ensure_collection(_qdrant_client)
        return _qdrant_client
    except Exception as e:
        logger.warning("Qdrant unavailable: %s", e)
        return None


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        return _embedder
    except Exception as e:
        logger.warning("Embedder unavailable: %s", e)
        return None


def _ensure_collection(client):
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def embed_text(text: str) -> list[float] | None:
    """Return embedding vector or None if embedder unavailable."""
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        return embedder.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.warning("embed_text failed: %s", e)
        return None


def index_paper(paper_id: int, title: str, abstract: str, authors: str = "") -> bool:
    """
    Embed and store a paper in Qdrant.
    Returns True on success, False if Qdrant is unavailable.
    """
    client = _get_qdrant()
    if client is None:
        return False
    text = f"{title}\n{authors}\n{abstract}"
    vector = embed_text(text)
    if vector is None:
        return False
    try:
        from qdrant_client.models import PointStruct

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=paper_id,
                    vector=vector,
                    payload={"title": title, "abstract": abstract[:500]},
                )
            ],
        )
        return True
    except Exception as e:
        logger.exception("index_paper failed for paper %s: %s", paper_id, e)
        return False


def semantic_search(query: str, limit: int = 10) -> list[int]:
    """
    Return list of paper IDs ranked by semantic similarity.
    Returns [] if Qdrant unavailable (caller falls back to SQL search).
    """
    client = _get_qdrant()
    if client is None:
        return []
    vector = embed_text(query)
    if vector is None:
        return []
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=limit,
            score_threshold=0.35,
        )
        return [int(r.id) for r in results]
    except Exception as e:
        logger.exception("semantic_search failed: %s", e)
        return []


def delete_paper(paper_id: int) -> bool:
    """Remove a paper from the vector index."""
    client = _get_qdrant()
    if client is None:
        return False
    try:
        from qdrant_client.models import PointIdsList

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[paper_id]),
        )
        return True
    except Exception as e:
        logger.exception("delete_paper vector failed: %s", e)
        return False
