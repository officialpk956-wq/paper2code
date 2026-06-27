import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import User, Paper, DojoSubmission, Task, UsageLog, Problem, LeaderboardArchive

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


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
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    total_users        = db.query(func.count(User.id)).scalar() or 0
    verified_users     = db.query(func.count(User.id)).filter(User.is_email_verified == True).scalar() or 0
    users_today        = db.query(func.count(User.id)).filter(User.created_at >= today).scalar() or 0
    papers_today       = db.query(func.count(Paper.id)).filter(Paper.created_at >= today).scalar() or 0
    submissions_today  = db.query(func.count(DojoSubmission.id)).filter(DojoSubmission.created_at >= today).scalar() or 0
    llm_cost_today     = db.query(func.sum(UsageLog.cost_usd)).filter(UsageLog.created_at >= today).scalar() or 0
    llm_calls_today    = db.query(func.count(UsageLog.id)).filter(UsageLog.created_at >= today).scalar() or 0
    tasks_running      = db.query(func.count(Task.id)).filter(Task.status == "running").scalar() or 0
    tasks_failed_today = db.query(func.count(Task.id)).filter(
        Task.status == "failed", Task.created_at >= today
    ).scalar() or 0

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    since = datetime.now(timezone.utc) - timedelta(days=days)

    by_action = db.query(
        UsageLog.action,
        func.count(UsageLog.id).label("calls"),
        func.sum(UsageLog.cost_usd).label("total_cost"),
    ).filter(UsageLog.created_at >= since).group_by(UsageLog.action).all()

    top_users = db.query(
        UsageLog.user_id,
        func.sum(UsageLog.cost_usd).label("total_cost"),
        func.count(UsageLog.id).label("calls"),
    ).filter(
        UsageLog.created_at >= since,
        UsageLog.user_id.isnot(None),
    ).group_by(UsageLog.user_id).order_by(
        func.sum(UsageLog.cost_usd).desc()
    ).limit(20).all()

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


@router.get("/problems", include_in_schema=False)
def admin_list_problems(
    include_retired: bool = True,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    query = db.query(Problem)
    if not include_retired:
        query = query.filter(Problem.is_retired == False)
    problems = query.order_by(Problem.id).all()
    return {"total": len(problems), "problems": [_problem_to_dict(p) for p in problems]}


@router.post("/problems", include_in_schema=False, status_code=201)
def admin_create_problem(
    body: AdminProblemCreate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    if db.query(Problem).filter_by(id=body.id).first():
        raise HTTPException(status_code=409, detail=f"Problem id '{body.id}' already exists")
    if db.query(Problem).filter_by(slug=body.slug).first():
        raise HTTPException(status_code=409, detail=f"Problem slug '{body.slug}' already exists")

    problem = Problem(
        id=body.id,
        slug=body.slug,
        title=body.title,
        difficulty=body.difficulty,
        category=body.category,
        description=body.description,
        python_template=body.python_template,
        test_cases=body.test_cases,
        hints=body.hints,
        explanation=body.explanation,
        tags=body.tags,
        related_architectures=body.related_architectures,
        related_papers=body.related_papers,
        related_math=body.related_math,
        learning_points=body.learning_points,
        estimated_time=body.estimated_time,
        visualization_url=body.visualization_url,
        time_limit_ms=body.time_limit_ms,
        version=1,
        is_retired=False,
    )
    db.add(problem)
    db.commit()
    db.refresh(problem)
    return _problem_to_dict(problem)


@router.put("/problems/{problem_id}", include_in_schema=False)
def admin_update_problem(
    problem_id: str,
    body: AdminProblemUpdate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    problem = db.query(Problem).filter_by(id=problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    updates = body.model_dump(exclude_none=True)
    # Auto-bump version when test_cases are changed
    if "test_cases" in updates:
        problem.version = (problem.version or 1) + 1
    for field, value in updates.items():
        setattr(problem, field, value)
    db.commit()
    db.refresh(problem)
    return _problem_to_dict(problem)


@router.patch("/problems/{problem_id}/retire", include_in_schema=False)
def admin_retire_problem(
    problem_id: str,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    problem = db.query(Problem).filter_by(id=problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    problem.is_retired = True
    db.commit()
    return {"id": problem.id, "is_retired": True}


@router.patch("/problems/{problem_id}/restore", include_in_schema=False)
def admin_restore_problem(
    problem_id: str,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    problem = db.query(Problem).filter_by(id=problem_id).first()
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")
    problem.is_retired = False
    db.commit()
    return {"id": problem.id, "is_retired": False}


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

@router.get("/papers", include_in_schema=False)
def admin_list_papers(
    page: int = 1,
    limit: int = 50,
    flagged_only: bool = False,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    query = db.query(Paper)
    if flagged_only:
        query = query.filter(Paper.is_flagged == True)
    total = query.count()
    # Flagged papers first, then newest
    papers = (
        query.order_by(Paper.is_flagged.desc(), Paper.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "papers": [
            {
                "id": p.id,
                "title": p.title,
                "visibility": p.visibility,
                "is_flagged": p.is_flagged,
                "flag_reason": p.flag_reason,
                "uploaded_by": p.uploaded_by,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in papers
        ],
    }


class PaperFlagRequest(BaseModel):
    reason: str = "policy_violation"


@router.post("/papers/{paper_id}/flag", include_in_schema=False)
def admin_flag_paper(
    paper_id: int,
    body: PaperFlagRequest,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    paper.is_flagged = True
    paper.flag_reason = body.reason
    db.commit()
    return {"paper_id": paper_id, "is_flagged": True, "flag_reason": body.reason}


@router.delete("/papers/{paper_id}", include_in_schema=False, status_code=200)
def admin_delete_paper(
    paper_id: int,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    # Decrement uploader's storage quota before deleting
    if paper.uploaded_by and paper.file_size_bytes:
        owner = db.query(User).filter_by(id=paper.uploaded_by).first()
        if owner:
            owner.storage_bytes_used = max(0, (owner.storage_bytes_used or 0) - paper.file_size_bytes)
    # Clean up R2/storage before deleting DB record
    try:
        from backend.services import storage_service
        if paper.r2_key:
            storage_service.cleanup(f"r2://{paper.r2_key}")
    except Exception:
        pass
    db.delete(paper)
    db.commit()
    return {"deleted": True, "paper_id": paper_id}


# ---------------------------------------------------------------------------
# Leaderboard archive: GET /api/admin/leaderboard/archive
# ---------------------------------------------------------------------------

@router.get("/leaderboard/archive", include_in_schema=False)
def admin_leaderboard_archive(
    weeks: int = 4,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    since = datetime.now(timezone.utc) - timedelta(weeks=weeks)
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
