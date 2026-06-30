import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import User, Paper, DojoSubmission, Task, UsageLog, Problem, LeaderboardArchive, XPEvent

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
    q: Optional[str] = None,
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
    is_admin: Optional[bool] = None
    is_email_verified: Optional[bool] = None


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
    return {"id": user.id, "email": user.email, "is_admin": user.is_admin, "is_email_verified": user.is_email_verified}


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
    test_cases: List[Any] = []
    hints: List[Any] = []
    explanation: List[Any] = []
    tags: List[Any] = []
    related_architectures: List[Any] = []
    related_papers: List[Any] = []
    related_math: List[Any] = []
    learning_points: List[Any] = []
    estimated_time: Optional[int] = None
    visualization_url: Optional[str] = None
    time_limit_ms: Optional[int] = None   # None → global default (10 000 ms)


class AdminProblemUpdate(BaseModel):
    slug: Optional[str] = None
    title: Optional[str] = None
    difficulty: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    python_template: Optional[str] = None
    test_cases: Optional[List[Any]] = None
    hints: Optional[List[Any]] = None
    explanation: Optional[List[Any]] = None
    tags: Optional[List[Any]] = None
    related_architectures: Optional[List[Any]] = None
    related_papers: Optional[List[Any]] = None
    related_math: Optional[List[Any]] = None
    learning_points: Optional[List[Any]] = None
    estimated_time: Optional[int] = None
    visualization_url: Optional[str] = None
    time_limit_ms: Optional[int] = None


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
    papers_count = db.query(func.count(Task.id)).filter(
        Task.user_id == user_id, Task.type == "paper.codegen"
    ).scalar() or 0
    submissions_count = db.query(func.count(DojoSubmission.id)).filter_by(
        user_id=user_id
    ).scalar() or 0
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

from backend.models import PaperChallenge, PaperChallengePart

class AdminPaperChallengeCreate(BaseModel):
    title: str
    description: Optional[str] = None
    order_idx: int = 0

class AdminPaperChallengePartCreate(BaseModel):
    title: str
    description_md: str
    paper_section_md: Optional[str] = None
    setup_code: Optional[str] = None
    starter_code: str
    solution_code: Optional[str] = None
    test_code: str
    unlock_requires_part_id: Optional[int] = None
    xp_reward: int = 50
    order_idx: int = 0

class AdminPaperChallengePublish(BaseModel):
    is_published: bool




