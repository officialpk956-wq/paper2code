"""
backend/routers/leaderboard.py

Cross-user leaderboard ranked by points.
Supports period filtering: all | weekly | monthly.
"""

import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import asc, desc, func, select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import DojoSubmission, Problem, User

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


def _category_user_ids(db: Session, category: str) -> set[int]:
    """Return IDs of users who passed at least one problem in `category`."""
    rows = (
        db.query(func.distinct(DojoSubmission.user_id))
        .join(Problem, Problem.id == DojoSubmission.problem_id)
        .filter(
            DojoSubmission.passed == True,
            Problem.category.ilike(f"%{category}%"),
        )
        .all()
    )
    return {r[0] for r in rows}


@router.get("/leaderboard")
def get_leaderboard(
    period: str = Query("all", description="all | weekly | monthly"),
    category: str = Query("", description="Filter to users who solved problems in this category"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    if period not in _VALID_PERIODS:
        raise HTTPException(400, f"period must be one of: {', '.join(_VALID_PERIODS)}")

    import json

    from backend.redis_config import cache_redis

    cache_key = f"leaderboard:{period}:{category}:{limit}"
    if cache_redis:
        cached = cache_redis.get(cache_key)
        if cached:
            return json.loads(cached)

    since = _period_start(period)
    solved_sq = _solved_count_subquery()

    points_field = User.weekly_points if period == "weekly" else User.points
    order_fields = [desc(User.weekly_points), asc(User.id)] if period == "weekly" else [desc(User.points), asc(User.id)]

    query = db.query(
        User.id,
        User.name,
        User.avatar_url,
        points_field.label("points"),
        User.streak,
        solved_sq.label("problems_solved"),
    )

    if since is not None:
        query = query.filter(
            User.last_active.isnot(None),
            User.last_active >= since,
        )

    if category and isinstance(category, str):
        cat_user_ids = _category_user_ids(db, category)
        if not cat_user_ids:
            return {
                "period": period,
                "category": category,
                "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "total_ranked": 0,
                "leaders": [],
            }
        query = query.filter(User.id.in_(cat_user_ids))

    rows = query.filter(points_field > 0).order_by(*order_fields).limit(limit).all()

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

    result = {
        "period": period,
        "category": category or None,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_ranked": len(leaders),
        "leaders": leaders,
    }

    if cache_redis:
        cache_redis.setex(cache_key, 60, json.dumps(result))

    return result
