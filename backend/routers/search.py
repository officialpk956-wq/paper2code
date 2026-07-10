"""
backend/routers/search.py

Unified full-text search across papers and problems.

PostgreSQL: uses tsvector/tsquery (GIN-indexed when available).
SQLite (dev/test): falls back to ILIKE / LIKE pattern matching.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Paper, Problem

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Search"])


def _is_postgres(db: Session) -> bool:
    try:
        return db.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def _search_papers(db: Session, q: str, limit: int) -> list:
    base = db.query(Paper).filter(Paper.visibility == "public")
    if _is_postgres(db):
        vec = func.to_tsvector(
            "english",
            func.coalesce(Paper.title, "") + " " + func.coalesce(Paper.abstract, ""),
        )
        tsq = func.plainto_tsquery("english", q)
        return base.filter(vec.op("@@")(tsq)).limit(limit).all()
    # SQLite fallback
    pat = f"%{q}%"
    return base.filter(or_(Paper.title.ilike(pat), Paper.abstract.ilike(pat))).limit(limit).all()


def _search_problems(db: Session, q: str, limit: int) -> list:
    base = db.query(Problem).filter(Problem.is_retired == False)
    if _is_postgres(db):
        vec = func.to_tsvector(
            "english",
            func.coalesce(Problem.title, "") + " " + func.coalesce(Problem.description, ""),
        )
        tsq = func.plainto_tsquery("english", q)
        return base.filter(vec.op("@@")(tsq)).limit(limit).all()
    pat = f"%{q}%"
    return (
        base.filter(
            or_(
                Problem.title.ilike(pat),
                Problem.description.ilike(pat),
                Problem.category.ilike(pat),
            )
        )
        .limit(limit)
        .all()
    )


@router.get("/search")
def search(
    q: str = Query(..., min_length=2, max_length=200, description="Search query"),
    types: str = Query("papers,problems", description="Comma-separated: papers,problems"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Full-text search across papers and/or problems.

    Returns a unified list of results sorted by type (papers first), each with:
    - type: "paper" | "problem"
    - id, title, snippet (truncated description/abstract), url hint
    """
    requested = {t.strip().lower() for t in types.split(",")}
    results = []

    if "papers" in requested:
        papers = _search_papers(db, q, limit)
        for p in papers:
            arch = (p.architecture_graph or {}).get("classification", "Unknown")
            results.append(
                {
                    "type": "paper",
                    "id": p.id,
                    "title": p.title,
                    "snippet": (p.abstract or "")[:200],
                    "tags": [arch] if arch != "Unknown" else [],
                    "url": f"/papers/{p.id}",
                }
            )

    if "problems" in requested:
        problems = _search_problems(db, q, limit)
        for p in problems:
            results.append(
                {
                    "type": "problem",
                    "id": p.id,
                    "title": p.title or "",
                    "snippet": (p.description or "")[:200],
                    "tags": [p.difficulty or "", p.category or ""],
                    "url": f"/dojo/{p.id}",
                }
            )

    return {
        "query": q,
        "total": len(results),
        "results": results[:limit],
    }
