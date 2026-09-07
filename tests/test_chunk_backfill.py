"""Regression tests for the operator-triggered PaperChunk embedding backfill."""

import os
from unittest.mock import MagicMock

import pytest

from backend.models import PaperChunk
from backend.scripts import backfill_chunk_embeddings
from backend.services import vector_service


def _chunk(paper_id: int, text: str) -> PaperChunk:
    return PaperChunk(paper_id=paper_id, section="methods", page=1, text=text)


def test_backfill_returns_zero_without_qdrant(db_session, monkeypatch):
    db_session.add(_chunk(1, "A chunk that must not be iterated."))
    db_session.commit()

    index_chunk = MagicMock(return_value=True)
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "index_chunk", index_chunk)
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "_get_qdrant", lambda: None)

    assert backfill_chunk_embeddings.backfill(db_session) == {
        "scanned": 0,
        "indexed": 0,
        "failed": 0,
        "skipped": 0,
    }
    index_chunk.assert_not_called()


def test_backfill_indexes_nonempty_chunks_and_skips_empty_text(db_session, monkeypatch):
    chunks = [_chunk(1, "first chunk"), _chunk(1, "  "), _chunk(1, "third chunk")]
    db_session.add_all(chunks)
    db_session.commit()
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "_get_qdrant", lambda: object())
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "index_chunk", lambda *args: True)

    assert backfill_chunk_embeddings.backfill(db_session, batch_size=2) == {
        "scanned": 3,
        "indexed": 2,
        "failed": 0,
        "skipped": 1,
    }
    db_session.refresh(chunks[0])
    db_session.refresh(chunks[1])
    db_session.refresh(chunks[2])
    assert chunks[0].embedding_id is not None
    assert chunks[1].embedding_id is None
    assert chunks[2].embedding_id is not None


def test_backfill_keeps_failed_rows_unembedded(db_session, monkeypatch):
    chunks = [_chunk(1, "first chunk"), _chunk(1, "second chunk")]
    db_session.add_all(chunks)
    db_session.commit()
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "_get_qdrant", lambda: object())
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "index_chunk", lambda *args: False)

    result = backfill_chunk_embeddings.backfill(db_session)

    assert result["failed"] == 2
    db_session.refresh(chunks[0])
    db_session.refresh(chunks[1])
    assert chunks[0].embedding_id is None
    assert chunks[1].embedding_id is None


def test_backfill_dry_run_writes_no_embedding_ids(db_session, monkeypatch):
    chunks = [_chunk(1, "first chunk"), _chunk(1, "second chunk")]
    db_session.add_all(chunks)
    db_session.commit()
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "_get_qdrant", lambda: object())
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "index_chunk", lambda *args: True)

    result = backfill_chunk_embeddings.backfill(db_session, dry_run=True)

    assert result == {"scanned": 2, "indexed": 2, "failed": 0, "skipped": 0}
    db_session.refresh(chunks[0])
    db_session.refresh(chunks[1])
    assert chunks[0].embedding_id is None
    assert chunks[1].embedding_id is None


def test_backfill_is_rerunnable(db_session, monkeypatch):
    db_session.add_all([_chunk(1, "first chunk"), _chunk(1, "second chunk")])
    db_session.commit()
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "_get_qdrant", lambda: object())
    monkeypatch.setattr(backfill_chunk_embeddings.vector_service, "index_chunk", lambda *args: True)

    backfill_chunk_embeddings.backfill(db_session)
    assert backfill_chunk_embeddings.backfill(db_session)["indexed"] == 0


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("QDRANT_URL"), reason="requires a real Qdrant instance (QDRANT_URL)"
)
def test_live_backfill_indexes_and_searches_chunks(db_session):
    chunks = [_chunk(99101, "residual shortcut path for the live backfill check"), _chunk(99101, "unrelated token sequence")]
    db_session.add_all(chunks)
    db_session.commit()
    point_ids = []
    try:
        result = backfill_chunk_embeddings.backfill(db_session)
        assert result["indexed"] == 2
        point_ids = [chunk.id for chunk in chunks]
        matches = vector_service.search_chunks("residual shortcut path", paper_id=99101)
        assert any(match["chunk_id"] == chunks[0].id for match in matches)
    finally:
        client = vector_service._get_qdrant()
        if client is not None and point_ids:
            from qdrant_client.models import PointIdsList

            client.delete(
                collection_name=vector_service.CHUNKS_COLLECTION,
                points_selector=PointIdsList(points=point_ids),
            )
        for chunk in chunks:
            db_session.delete(chunk)
        db_session.commit()
