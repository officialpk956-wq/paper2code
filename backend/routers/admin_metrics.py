import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import (
    DojoSubmission,
    LeaderboardArchive,
    Paper,
    Problem,
    Task,
    UsageLog,
    User,
    XPEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin Metrics"])


# ---------------------------------------------------------------------------
# Admin dependency — 403 for non-admins
# ---------------------------------------------------------------------------


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ---------------------------------------------------------------------------
# GET /api/admin/stats
# ---------------------------------------------------------------------------


@router.get("/stats", include_in_schema=False)
def admin_stats(
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    total_users = db.query(func.count(User.id)).scalar() or 0
    verified_users = (
        db.query(func.count(User.id)).filter(User.is_email_verified == True).scalar() or 0
    )
    users_today = db.query(func.count(User.id)).filter(User.created_at >= today).scalar() or 0
    papers_today = db.query(func.count(Paper.id)).filter(Paper.created_at >= today).scalar() or 0
    submissions_today = (
        db.query(func.count(DojoSubmission.id)).filter(DojoSubmission.created_at >= today).scalar()
        or 0
    )
    llm_cost_today = (
        db.query(func.sum(UsageLog.cost_usd)).filter(UsageLog.created_at >= today).scalar() or 0
    )
    llm_calls_today = (
        db.query(func.count(UsageLog.id)).filter(UsageLog.created_at >= today).scalar() or 0
    )
    tasks_running = db.query(func.count(Task.id)).filter(Task.status == "running").scalar() or 0
    tasks_failed_today = (
        db.query(func.count(Task.id))
        .filter(Task.status == "failed", Task.created_at >= today)
        .scalar()
        or 0
    )

    return {
        "users": {
            "total": total_users,
            "verified": verified_users,
            "new_today": users_today,
            "verification_rate": round(verified_users / max(total_users, 1) * 100, 1),
        },
        "content": {
            "papers_today": papers_today,
            "submissions_today": submissions_today,
        },
        "llm": {
            "calls_today": llm_calls_today,
            "cost_today_usd": round(float(llm_cost_today), 4),
            "avg_cost_per_call": round(float(llm_cost_today) / max(llm_calls_today, 1), 5),
        },
        "tasks": {
            "running": tasks_running,
            "failed_today": tasks_failed_today,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /api/admin/costs
# ---------------------------------------------------------------------------


@router.get("/costs", include_in_schema=False)
def admin_costs(
    days: int = 7,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    since = datetime.now(UTC) - timedelta(days=days)

    by_action = (
        db.query(
            UsageLog.action,
            func.count(UsageLog.id).label("calls"),
            func.sum(UsageLog.cost_usd).label("total_cost"),
        )
        .filter(UsageLog.created_at >= since)
        .group_by(UsageLog.action)
        .all()
    )

    top_users = (
        db.query(
            UsageLog.user_id,
            func.sum(UsageLog.cost_usd).label("total_cost"),
            func.count(UsageLog.id).label("calls"),
        )
        .filter(
            UsageLog.created_at >= since,
            UsageLog.user_id.isnot(None),
        )
        .group_by(UsageLog.user_id)
        .order_by(func.sum(UsageLog.cost_usd).desc())
        .limit(20)
        .all()
    )

    return {
        "period_days": days,
        "by_action": [
            {
                "action": r.action,
                "calls": r.calls,
                "total_cost_usd": round(float(r.total_cost or 0), 4),
            }
            for r in by_action
        ],
        "top_users_by_cost": [
            {
                "user_id": r.user_id,
                "calls": r.calls,
                "total_cost_usd": round(float(r.total_cost or 0), 4),
            }
            for r in top_users
        ],
    }


# ---------------------------------------------------------------------------
# GET /api/admin/users
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PATCH /api/admin/users/{user_id}
# ---------------------------------------------------------------------------


class AdminUserUpdate(BaseModel):
    is_admin: bool | None = None
    is_email_verified: bool | None = None


# ---------------------------------------------------------------------------
# Admin Problem CRUD
# ---------------------------------------------------------------------------


class AdminProblemCreate(BaseModel):
    id: str
    slug: str
    title: str
    difficulty: str
    category: str
    description: str
    python_template: str = ""
    test_cases: list[Any] = []
    hints: list[Any] = []
    explanation: list[Any] = []
    tags: list[Any] = []
    related_architectures: list[Any] = []
    related_papers: list[Any] = []
    related_math: list[Any] = []
    learning_points: list[Any] = []
    estimated_time: int | None = None
    visualization_url: str | None = None
    time_limit_ms: int | None = None  # None → global default (10 000 ms)


class AdminProblemUpdate(BaseModel):
    slug: str | None = None
    title: str | None = None
    difficulty: str | None = None
    category: str | None = None
    description: str | None = None
    python_template: str | None = None
    test_cases: list[Any] | None = None
    hints: list[Any] | None = None
    explanation: list[Any] | None = None
    tags: list[Any] | None = None
    related_architectures: list[Any] | None = None
    related_papers: list[Any] | None = None
    related_math: list[Any] | None = None
    learning_points: list[Any] | None = None
    estimated_time: int | None = None
    visualization_url: str | None = None
    time_limit_ms: int | None = None


def _problem_to_dict(p: Problem) -> dict:
    return {
        "id": p.id,
        "slug": p.slug,
        "title": p.title,
        "difficulty": p.difficulty,
        "category": p.category,
        "description": p.description,
        "estimated_time": p.estimated_time,
        "is_retired": p.is_retired,
        "version": p.version,
        "time_limit_ms": p.time_limit_ms,
        "acceptance_rate": float(p.acceptance_rate) if p.acceptance_rate is not None else None,
        "tags": p.tags,
        "test_cases": p.test_cases,
    }


# ---------------------------------------------------------------------------
# GET /api/admin/users/{user_id}  (individual detail)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DELETE /api/admin/users/{user_id}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Paper moderation: GET /api/admin/papers, DELETE, POST flag
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GET /api/admin/papers/moderation-queue
# ---------------------------------------------------------------------------


class PaperFlagRequest(BaseModel):
    reason: str = "policy_violation"


# ---------------------------------------------------------------------------
# GET /api/admin/xp-events  — audit trail of all XP events
# ---------------------------------------------------------------------------


@router.get("/xp-events", include_in_schema=False)
def admin_xp_events(
    page: int = 1,
    limit: int = 100,
    user_id: int | None = None,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    query = db.query(XPEvent)
    if user_id is not None:
        query = query.filter(XPEvent.user_id == user_id)
    total = query.count()
    events = query.order_by(XPEvent.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "events": [
            {
                "id": e.id,
                "user_id": e.user_id,
                "action": e.action,
                "amount": e.amount,
                "entity_id": e.entity_id,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


# ---------------------------------------------------------------------------
# Leaderboard archive: GET /api/admin/leaderboard/archive
# ---------------------------------------------------------------------------


@router.get("/leaderboard/archive", include_in_schema=False)
def admin_leaderboard_archive(
    weeks: int = 4,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    since = datetime.now(UTC) - timedelta(weeks=weeks)
    rows = (
        db.query(LeaderboardArchive)
        .filter(LeaderboardArchive.week_start >= since)
        .order_by(LeaderboardArchive.week_start.desc(), LeaderboardArchive.rank)
        .limit(500)
        .all()
    )
    return {
        "weeks": weeks,
        "entries": [
            {
                "week_start": r.week_start.isoformat(),
                "user_id": r.user_id,
                "weekly_points": r.weekly_points,
                "rank": r.rank,
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Admin Paper Challenges CRUD
# ---------------------------------------------------------------------------


class AdminPaperChallengeCreate(BaseModel):
    title: str
    description: str | None = None
    order_idx: int = 0


class AdminPaperChallengePartCreate(BaseModel):
    title: str
    description_md: str
    paper_section_md: str | None = None
    setup_code: str | None = None
    starter_code: str
    solution_code: str | None = None
    test_code: str
    unlock_requires_part_id: int | None = None
    xp_reward: int = 50
    order_idx: int = 0


class AdminPaperChallengePublish(BaseModel):
    is_published: bool
