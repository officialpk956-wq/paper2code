"""
backend/routers/leaderboard.py

Cross-user leaderboard ranked by points.
Supports period filtering: all | weekly | monthly.
"""

import logging
import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func, desc

from backend.database import get_db
from backend.models import User, DojoSubmission

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Leaderboard"])

_VALID_PERIODS = {"all", "weekly", "monthly"}


def _period_start(period: str) -> datetime.datetime | None:
    now = datetime.datetime.utcnow()
    if period == "weekly":
        return now - datetime.timedelta(days=7)
    if period == "monthly":
        return now - datetime.timedelta(days=30)
    return None


def _solved_count_subquery():
    """Correlated subquery: count of distinct problems passed by each user."""
    return (
        select(func.count(func.distinct(DojoSubmission.problem_id)))
        .where(
            DojoSubmission.user_id == User.id,
            DojoSubmission.passed == True,
        )
        .correlate(User)
        .scalar_subquery()
    )


@router.get("/leaderboard")
def get_leaderboard(
    period: str = Query("all", description="all | weekly | monthly"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    if period not in _VALID_PERIODS:
        raise HTTPException(400, f"period must be one of: {', '.join(_VALID_PERIODS)}")

    since = _period_start(period)
    solved_sq = _solved_count_subquery()

    query = db.query(
        User.id,
        User.name,
        User.avatar_url,
        User.points,
        User.streak,
        solved_sq.label("problems_solved"),
    )

    # For period filters, restrict to users active within the window
    if since is not None:
        query = query.filter(
            User.last_active.isnot(None),
            User.last_active >= since,
        )

    rows = (
        query
        .filter(User.points > 0)           # exclude zero-point ghosts
        .order_by(desc(User.points))
        .limit(limit)
        .all()
    )

    leaders = [
        {
            "rank": i + 1,
            "user_id": r.id,
            "name": r.name,
            "avatar_url": r.avatar_url,
            "points": r.points,
            "xp_level": (r.points or 0) // 100 + 1,
            "streak": r.streak,
            "problems_solved": r.problems_solved or 0,
        }
        for i, r in enumerate(rows)
    ]

    return {
        "period": period,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_ranked": len(leaders),
        "leaders": leaders,
    }
