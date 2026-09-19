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
CHUNKS_COLLECTION = os.getenv("QDRANT_CHUNKS_COLLECTION", "paper_chunks")
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
        _ensure_collection(_qdrant_client, COLLECTION_NAME)
        _ensure_collection(_qdrant_client, CHUNKS_COLLECTION)
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


def _ensure_collection(client, collection_name: str):
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
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
        # QdrantClient.search() was removed from the installed client version
        # (qdrant-client >=1.10 replaced it with the Query API) -- this was
        # never caught before because QDRANT_URL was always empty until the
        # Qdrant service existed, so this call never actually ran.
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=limit,
            score_threshold=0.35,
        )
        return [int(point.id) for point in response.points]
    except Exception as e:
        logger.exception("semantic_search failed: %s", e)
        return []


def index_chunk(
    chunk_id: int, paper_id: int, text: str, section: str = "other", page: int | None = None
) -> bool:
    """Embed and store a paper chunk in the chunks collection."""
    client = _get_qdrant()
    if client is None:
        return False
    vector = embed_text(text)
    if vector is None:
        return False
    try:
        from qdrant_client.models import PointStruct

        client.upsert(
            collection_name=CHUNKS_COLLECTION,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        "paper_id": paper_id,
                        "section": section,
                        "page": page,
                        "text": text[:1000],
                    },
                )
            ],
        )
        return True
    except Exception as e:
        logger.exception("index_chunk failed for chunk %s: %s", chunk_id, e)
        return False


def search_chunks(
    query: str,
    limit: int = 10,
    paper_id: int | None = None,
    section: str | None = None,
) -> list[dict]:
    """
    Dense search over indexed chunks. Returns [{chunk_id, score, paper_id, section, page, text}, ...].
    Returns [] if Qdrant unavailable.
    """
    client = _get_qdrant()
    if client is None:
        return []
    vector = embed_text(query)
    if vector is None:
        return []
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = []
        if paper_id is not None:
            conditions.append(FieldCondition(key="paper_id", match=MatchValue(value=paper_id)))
        if section is not None:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
        query_filter = Filter(must=conditions) if conditions else None

        response = client.query_points(
            collection_name=CHUNKS_COLLECTION,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=0.3,
        )
        return [
            {"chunk_id": int(p.id), "score": p.score, **(p.payload or {})} for p in response.points
        ]
    except Exception as e:
        logger.exception("search_chunks failed: %s", e)
        return []


# Family-characteristic vocabulary used to bias hybrid retrieval toward a
# paper's own architecture family. Kept small and deterministic -- extend
# alongside core.classification.infer_family_from_name's family list.
_FAMILY_TERMS = {
    "resnet": ["residual", "skip connection", "shortcut", "bottleneck"],
    "unet": ["skip connection", "downsample", "upsample", "concat"],
    "vit": ["patch embedding", "cls token", "position embedding"],
    "swin": ["window attention", "patch merging", "shifted window"],
    "transformer": ["self-attention", "encoder", "decoder", "positional encoding"],
    "bert_gpt": ["masked language", "autoregressive", "token embedding"],
    "gan": ["generator", "discriminator", "adversarial"],
    "diffusion": ["denoising", "timestep", "noise schedule"],
    "ldm": ["latent", "noise scheduler", "diffusion"],
    "mae": ["masking", "mask ratio", "reconstruction"],
    "efficientnet": ["mbconv", "squeeze-and-excitation", "compound scaling"],
    "mobilenet": ["depthwise", "inverted residual", "pointwise"],
    "densenet": ["dense block", "growth rate", "transition layer"],
    "yolo": ["anchor box", "detection head", "feature pyramid"],
}


def _resolve_family(query: str, family: str | None) -> str | None:
    if family is not None:
        return family
    from core.rag.knowledge_graph import KnowledgeGraph

    return KnowledgeGraph().infer_family_from_concept(query)


def _expand_query_terms(query: str) -> list[str]:
    """Return KAG surface-form expansions without risking retrieval failure."""
    try:
        from core.rag.knowledge_graph import KnowledgeGraph

        return KnowledgeGraph().expand_query_terms(query)
    except Exception as exc:
        logger.warning("KAG query expansion failed: %s", exc)
        return []


def _bm25_and_family_scores(
    query: str, texts: list[str], family: str | None
) -> tuple[list[float], list[float]]:
    """Shared scoring core for hybrid_search_chunks and hybrid_rank_texts."""
    from core.rag.retriever import BM25, _tokenize

    bm25 = BM25(texts)
    query_terms = _tokenize(query)
    expanded_terms = _tokenize(" ".join(_expand_query_terms(query)))
    raw_bm25 = [
        bm25.score(query_terms, i) + 0.5 * bm25.score(expanded_terms, i) for i in range(len(texts))
    ]
    max_bm25 = max(raw_bm25) or 1.0
    bm25_norm = [s / max_bm25 for s in raw_bm25]

    family_terms = _FAMILY_TERMS.get((family or "").lower(), [])
    family_bonus = [
        0.1 if family_terms and any(t in text.lower() for t in family_terms) else 0.0
        for text in texts
    ]
    return bm25_norm, family_bonus


def _mmr_select(
    indices: list[int], scores: list[float], vectors, k: int, lambda_: float = 0.7
) -> list[int]:
    """Greedily select relevant candidates while penalizing near duplicates."""
    if not indices or k <= 0:
        return []

    ranked = sorted(indices, key=lambda index: (-scores[index], index))
    if vectors is None or lambda_ == 1.0:
        return ranked[:k]

    selected: list[int] = []
    remaining = set(indices)
    try:
        while remaining and len(selected) < k:

            def mmr_score(candidate: int) -> tuple[float, float, int]:
                max_similarity = max(
                    (float(vectors[candidate] @ vectors[chosen]) for chosen in selected),
                    default=0.0,
                )
                return (
                    lambda_ * scores[candidate] - (1.0 - lambda_) * max_similarity,
                    scores[candidate],
                    -candidate,
                )

            chosen = max(remaining, key=mmr_score)
            selected.append(chosen)
            remaining.remove(chosen)
    except Exception as exc:
        logger.warning("MMR selection failed: %s", exc)
        return ranked[:k]
    return selected


def hybrid_search_chunks(
    query: str,
    chunks: list[dict],
    limit: int = 10,
    paper_id: int | None = None,
    family: str | None = None,
) -> list[dict]:
    """
    Combine dense (Qdrant) search with BM25 keyword scoring over the same
    chunk pool, with an optional boost for the paper's architecture family.

    `chunks` is the candidate pool to score: [{"id": int, "text": str}, ...]
    (e.g. all PaperChunk rows for one paper, already indexed via
    index_chunk). Falls back to dense-only ranking if `chunks` is empty or
    Qdrant is unavailable.

    If `family` isn't given, it's inferred from the query text via the
    knowledge graph (KAG) -- e.g. a query mentioning "skip connections"
    resolves to the "resnet" family without the caller needing to know that.
    """
    family = _resolve_family(query, family)
    dense_scores = {
        r["chunk_id"]: r["score"]
        for r in search_chunks(query, limit=max(limit * 3, limit), paper_id=paper_id)
    }

    if not chunks:
        ranked = sorted(dense_scores.items(), key=lambda x: -x[1])[:limit]
        return [{"chunk_id": cid, "score": score} for cid, score in ranked]

    texts = [str(c.get("text") or "") for c in chunks]
    bm25_norm, family_bonus = _bm25_and_family_scores(query, texts, family)

    combined = []
    for i, c in enumerate(chunks):
        cid = c["id"]
        dense = dense_scores.get(cid, 0.0)
        combined.append((cid, 0.5 * dense + 0.4 * bm25_norm[i] + family_bonus[i]))

    combined.sort(key=lambda x: -x[1])
    return [{"chunk_id": cid, "score": score} for cid, score in combined[:limit]]


def hybrid_rank_texts(
    query: str,
    texts: list[str],
    top_k: int = 6,
    family: str | None = None,
    diversity: float = 0.7,
) -> list[str]:
    """
    Rank arbitrary texts by relevance to `query`, combining in-memory dense
    similarity (embeddings computed ad-hoc -- no Qdrant needed) with BM25
    and a KAG-inferred family boost.

    For retrieval *before* chunks exist in Qdrant -- e.g. at paper
    extraction time, when chunks are only embedded/persisted afterward
    (see backend.tasks.paper_tasks). Falls back to returning `texts`
    unchanged if the sentence-transformers embedder is unavailable, so
    this never hard-fails extraction over a missing/slow model download.
    """
    if not texts:
        return []
    if len(texts) <= top_k:
        return texts

    family = _resolve_family(query, family)
    dense_scores = [0.0] * len(texts)
    text_vecs = None
    embedder = _get_embedder()
    if embedder is not None:
        try:
            query_vec = embedder.encode(query, normalize_embeddings=True)
            text_vecs = embedder.encode(texts, normalize_embeddings=True)
            dense_scores = (text_vecs @ query_vec).tolist()
        except Exception as e:
            logger.warning("hybrid_rank_texts dense scoring failed: %s", e)

    bm25_norm, family_bonus = _bm25_and_family_scores(query, texts, family)

    scored = [
        (i, 0.5 * dense_scores[i] + 0.4 * bm25_norm[i] + family_bonus[i]) for i in range(len(texts))
    ]
    selection_scores = [score for _, score in scored]
    top_indices = _mmr_select(
        list(range(len(texts))), selection_scores, text_vecs, top_k, diversity
    )
    top_indices.sort()  # preserve reading order for the extraction narrative
    return [texts[i] for i in top_indices]


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
