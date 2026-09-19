"""Usage: python -m backend.scripts.backfill_chunk_embeddings [options]

Embed PaperChunk rows that were ingested before chunk indexing was enabled.
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

from backend.database import SessionLocal
from backend.models import PaperChunk
from backend.services import vector_service

log = logging.getLogger(__name__)


def backfill(db, batch_size: int = 100, paper_id: int | None = None, dry_run: bool = False) -> dict:
    """Embed PaperChunk rows that have no embedding_id yet.

    Returns {"scanned": int, "indexed": int, "failed": int, "skipped": int}.
    """
    result = {"scanned": 0, "indexed": 0, "failed": 0, "skipped": 0}
    if vector_service._get_qdrant() is None:
        log.error("Qdrant is unavailable; chunk embedding backfill was not started")
        return result

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    query = db.query(PaperChunk).filter(PaperChunk.embedding_id.is_(None))
    if paper_id is not None:
        query = query.filter(PaperChunk.paper_id == paper_id)

    last_chunk_id = 0
    while True:
        batch = (
            query.filter(PaperChunk.id > last_chunk_id)
            .order_by(PaperChunk.id)
            .limit(batch_size)
            .all()
        )
        if not batch:
            break

        last_chunk_id = batch[-1].id
        for chunk in batch:
            result["scanned"] += 1
            if not chunk.text or not chunk.text.strip():
                result["skipped"] += 1
                continue

            if dry_run:
                result["indexed"] += 1
                continue

            if vector_service.index_chunk(
                chunk.id, chunk.paper_id, chunk.text, chunk.section, chunk.page
            ):
                chunk.embedding_id = str(chunk.id)
                result["indexed"] += 1
            else:
                result["failed"] += 1

        if not dry_run:
            db.commit()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed PaperChunk rows that do not yet have Qdrant embeddings."
    )
    parser.add_argument("--paper-id", type=int, help="Backfill only this paper ID")
    parser.add_argument("--batch-size", type=int, default=100, help="Rows per transaction")
    parser.add_argument("--dry-run", action="store_true", help="Report work without writing")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        print(
            backfill(
                db,
                paper_id=args.paper_id,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
            )
        )
        return 0
    except Exception as exc:
        log.exception("Chunk embedding backfill failed")
        print(f"Chunk embedding backfill failed: {exc}", file=sys.stderr)
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
