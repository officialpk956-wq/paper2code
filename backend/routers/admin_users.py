import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import (
    DojoSubmission,
    Paper,
    Problem,
    Task,
    User,
    XPEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin Users"])


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


# ---------------------------------------------------------------------------
# GET /api/admin/costs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GET /api/admin/users
# ---------------------------------------------------------------------------


@router.get("/users", include_in_schema=False)
def admin_list_users(
    page: int = 1,
    limit: int = 50,
    q: str | None = None,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if q:
        query = query.filter(User.email.ilike(f"%{q}%"))
    total = query.count()
    users = query.order_by(User.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "users": [
            {
                "id": u.id,
                "email": u.email,
                "name": u.name,
                "is_admin": u.is_admin,
                "is_email_verified": u.is_email_verified,
                "streak": u.streak,
                "points": u.points,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ],
    }


# ---------------------------------------------------------------------------
# PATCH /api/admin/users/{user_id}
# ---------------------------------------------------------------------------


class AdminUserUpdate(BaseModel):
    is_admin: bool | None = None
    is_email_verified: bool | None = None


@router.patch("/users/{user_id}", include_in_schema=False)
def admin_update_user(
    user_id: int,
    body: AdminUserUpdate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own admin account")
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    if body.is_email_verified is not None:
        user.is_email_verified = body.is_email_verified
    db.commit()
    return {
        "id": user.id,
        "email": user.email,
        "is_admin": user.is_admin,
        "is_email_verified": user.is_email_verified,
    }


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


@router.get("/users/{user_id}", include_in_schema=False)
def admin_get_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    papers_count = (
        db.query(func.count(Task.id))
        .filter(Task.user_id == user_id, Task.type == "paper.codegen")
        .scalar()
        or 0
    )
    submissions_count = (
        db.query(func.count(DojoSubmission.id)).filter_by(user_id=user_id).scalar() or 0
    )
    recent_xp = (
        db.query(XPEvent)
        .filter_by(user_id=user_id)
        .order_by(XPEvent.created_at.desc())
        .limit(20)
        .all()
    )
    recent_papers = (
        db.query(Paper)
        .filter_by(uploaded_by=user_id)
        .order_by(Paper.created_at.desc())
        .limit(20)
        .all()
    )
    recent_submissions = (
        db.query(DojoSubmission)
        .filter_by(user_id=user_id)
        .order_by(DojoSubmission.created_at.desc())
        .limit(20)
        .all()
    )
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url,
        "is_admin": user.is_admin,
        "is_email_verified": user.is_email_verified,
        "streak": user.streak,
        "points": user.points,
        "weekly_points": user.weekly_points,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_active": user.last_active.isoformat() if user.last_active else None,
        "stats": {
            "papers_uploaded": papers_count,
            "dojo_submissions": submissions_count,
        },
        "xp_events": [
            {
                "id": e.id,
                "action": e.action,
                "amount": e.amount,
                "entity_id": e.entity_id,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in recent_xp
        ],
        "papers": [
            {
                "id": p.id,
                "title": p.title,
                "visibility": p.visibility,
                "is_flagged": p.is_flagged,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in recent_papers
        ],
        "submissions": [
            {
                "id": s.id,
                "problem_id": s.problem_id,
                "passed": s.passed,
                "time_ms": s.time_ms,
                "is_best": s.is_best,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in recent_submissions
        ],
    }


# ---------------------------------------------------------------------------
# DELETE /api/admin/users/{user_id}
# ---------------------------------------------------------------------------


@router.delete("/users/{user_id}", include_in_schema=False, status_code=200)
def admin_delete_user(
    user_id: int,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"deleted": True, "user_id": user_id}


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


# ---------------------------------------------------------------------------
# Leaderboard archive: GET /api/admin/leaderboard/archive
# ---------------------------------------------------------------------------


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
