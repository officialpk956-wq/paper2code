import logging
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import User, Paper, DojoSubmission, Task, UsageLog, Problem, LeaderboardArchive, XPEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


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



# ---------------------------------------------------------------------------
# PATCH /api/admin/users/{user_id}
# ---------------------------------------------------------------------------

class AdminUserUpdate(BaseModel):
    is_admin: Optional[bool] = None
    is_email_verified: Optional[bool] = None




# ---------------------------------------------------------------------------
# Admin Problem CRUD
# ---------------------------------------------------------------------------

class AdminProblemCreate(BaseModel):
    id: str = Field(..., max_length=256)
    slug: str = Field(..., max_length=256)
    title: str = Field(..., max_length=500)
    difficulty: str = Field(..., max_length=50)
    category: str = Field(..., max_length=100)
    description: str = Field(..., max_length=50000)
    python_template: str = Field(default="", max_length=65536)
    test_cases: List[Any] = []
    hints: List[Any] = []
    explanation: List[Any] = []
    tags: List[Any] = []
    related_architectures: List[Any] = []
    related_papers: List[Any] = []
    related_math: List[Any] = []
    learning_points: List[Any] = []
    estimated_time: Optional[int] = None
    visualization_url: Optional[str] = Field(default=None, max_length=2048)
    time_limit_ms: Optional[int] = None   # None → global default (10 000 ms)


class AdminProblemUpdate(BaseModel):
    slug: Optional[str] = Field(default=None, max_length=256)
    title: Optional[str] = Field(default=None, max_length=500)
    difficulty: Optional[str] = Field(default=None, max_length=50)
    category: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = Field(default=None, max_length=50000)
    python_template: Optional[str] = Field(default=None, max_length=65536)
    test_cases: Optional[List[Any]] = None
    hints: Optional[List[Any]] = None
    explanation: Optional[List[Any]] = None
    tags: Optional[List[Any]] = None
    related_architectures: Optional[List[Any]] = None
    related_papers: Optional[List[Any]] = None
    related_math: Optional[List[Any]] = None
    learning_points: Optional[List[Any]] = None
    estimated_time: Optional[int] = None
    visualization_url: Optional[str] = Field(default=None, max_length=2048)
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



# ---------------------------------------------------------------------------
# DELETE /api/admin/users/{user_id}
# ---------------------------------------------------------------------------



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


# ---------------------------------------------------------------------------
# GET /api/admin/papers/moderation-queue
# ---------------------------------------------------------------------------

@router.get("/papers/moderation-queue", include_in_schema=False)
def admin_moderation_queue(
    page: int = 1,
    limit: int = 50,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    query = db.query(Paper).filter(Paper.is_flagged == True)
    total = query.count()
    papers = (
        query.order_by(Paper.created_at.desc())
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
                "flag_reason": p.flag_reason,
                "uploaded_by": p.uploaded_by,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in papers
        ],
    }


class PaperFlagRequest(BaseModel):
    reason: str = Field(default="policy_violation", max_length=500)


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
    title: str = Field(..., max_length=500)
    description: Optional[str] = Field(default=None, max_length=50000)
    order_idx: int = 0

class AdminPaperChallengePartCreate(BaseModel):
    title: str = Field(..., max_length=500)
    description_md: str = Field(..., max_length=50000)
    paper_section_md: Optional[str] = Field(default=None, max_length=50000)
    setup_code: Optional[str] = Field(default=None, max_length=65536)
    starter_code: str = Field(..., max_length=65536)
    solution_code: Optional[str] = Field(default=None, max_length=65536)
    test_code: str = Field(..., max_length=65536)
    unlock_requires_part_id: Optional[int] = None
    xp_reward: int = 50
    order_idx: int = 0

class AdminPaperChallengePublish(BaseModel):
    is_published: bool

@router.post("/papers/{paper_id}/challenges", include_in_schema=False, status_code=201)
def admin_create_paper_challenge(
    paper_id: int,
    body: AdminPaperChallengeCreate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    challenge = PaperChallenge(
        paper_id=paper_id,
        title=body.title,
        description=body.description,
        order_idx=body.order_idx,
    )
    db.add(challenge)
    db.commit()
    db.refresh(challenge)
    return {"id": challenge.id, "title": challenge.title}

@router.post("/challenges/{challenge_id}/parts", include_in_schema=False, status_code=201)
def admin_create_paper_challenge_part(
    challenge_id: int,
    body: AdminPaperChallengePartCreate,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    challenge = db.query(PaperChallenge).filter_by(id=challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    part = PaperChallengePart(
        challenge_id=challenge_id,
        title=body.title,
        description_md=body.description_md,
        paper_section_md=body.paper_section_md,
        setup_code=body.setup_code,
        starter_code=body.starter_code,
        solution_code=body.solution_code,
        test_code=body.test_code,
        unlock_requires_part_id=body.unlock_requires_part_id,
        xp_reward=body.xp_reward,
        order_idx=body.order_idx,
    )
    db.add(part)
    db.commit()
    db.refresh(part)
    return {"id": part.id, "title": part.title}

@router.patch("/challenges/{challenge_id}", include_in_schema=False)
def admin_publish_challenge(
    challenge_id: int,
    body: AdminPaperChallengePublish,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    challenge = db.query(PaperChallenge).filter_by(id=challenge_id).first()
    if not challenge:
        raise HTTPException(status_code=404, detail="Challenge not found")

    challenge.is_published = body.is_published
    db.commit()
    return {"id": challenge.id, "is_published": challenge.is_published}

@router.delete("/parts/{part_id}", include_in_schema=False, status_code=200)
def admin_delete_part(
    part_id: int,
    admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    part = db.query(PaperChallengePart).filter_by(id=part_id).first()
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")
    db.delete(part)
    db.commit()
    return {"deleted": True, "part_id": part_id}
