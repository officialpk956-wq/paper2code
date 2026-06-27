from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
import time
from collections import defaultdict

from backend.database import get_db
from backend.models import (
    User,
    DojoSubmission,
    Problem,
    XPEvent,
    Paper,
    Achievement,
    UserAchievement,
    LearnerProgress,
    AssessmentAttempt,
)
from backend.modules.auth.dependencies import get_current_user
from backend.modules.auth.schemas import UpdateProfileRequest
from core.agents.learning_path_agent import generate_learning_path

router = APIRouter(prefix="/api/users", tags=["Users"])
me_router = APIRouter(prefix="/api/me", tags=["Profile"])


# ---------------------------------------------------------------------------
# Public profile
# ---------------------------------------------------------------------------

@router.get("/{user_id}")
def get_public_profile(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Leaderboard rank: count users with strictly more points
    higher = (
        db.query(func.count(User.id))
        .filter(User.points > user.points)
        .scalar()
    ) or 0
    rank = higher + 1

    # Problems solved (distinct problems with a best passing submission)
    problems_solved = (
        db.query(func.count(func.distinct(DojoSubmission.problem_id)))
        .filter(
            DojoSubmission.user_id == user_id,
            DojoSubmission.passed == True,
            DojoSubmission.is_best == True,
        )
        .scalar()
    ) or 0

    # Earned achievements
    achievements = (
        db.query(Achievement)
        .join(UserAchievement, UserAchievement.achievement_id == Achievement.id)
        .filter(UserAchievement.user_id == user_id)
        .order_by(UserAchievement.earned_at.desc())
        .all()
    )
    earned = [{"slug": a.slug, "title": a.title, "description": a.description} for a in achievements]

    return {
        "id":              user.id,
        "name":            user.name,
        "avatar_url":      user.avatar_url,
        "points":          user.points,
        "weekly_points":   user.weekly_points,
        "xp_level":        user.xp_level,
        "streak":          user.streak,
        "rank":            rank,
        "problems_solved": problems_solved,
        "achievements":    earned,
    }


# ---------------------------------------------------------------------------
# /api/me endpoints
# ---------------------------------------------------------------------------

@me_router.patch("")
def update_me(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(id=current_user.id).first()
    if body.name is not None:
        user.name = body.name
    if body.avatar_url is not None:
        user.avatar_url = body.avatar_url
    db.commit()
    db.refresh(user)
    return {
        "id":         user.id,
        "name":       user.name,
        "avatar_url": user.avatar_url,
    }


@me_router.get("/xp-history")
def get_xp_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    offset = (page - 1) * page_size
    total = (
        db.query(func.count(XPEvent.id))
        .filter(XPEvent.user_id == current_user.id)
        .scalar()
    ) or 0
    events = (
        db.query(XPEvent)
        .filter(XPEvent.user_id == current_user.id)
        .order_by(XPEvent.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "events": [
            {
                "id":         e.id,
                "action":     e.action,
                "amount":     e.amount,
                "entity_id":  e.entity_id,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
    }


@me_router.get("/learning-path")
def get_learning_path(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a personalized learning path. Cached in memory for 1 hour."""
    # Simple in-process cache
    cache_key = f"lp_{current_user.id}"
    cache = getattr(get_learning_path, "_cache", {})
    cached = cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < 3600:
        return cached["data"]
        
    # Completed architectures
    completed = [
        r.entity_id for r in
        db.query(LearnerProgress)
        .filter_by(learner_id=str(current_user.id), status="completed")
        .all()
    ]
    
    # Weak topics
    attempts = db.query(AssessmentAttempt).filter_by(
        learner_id=str(current_user.id)
    ).all()
    arch_results = defaultdict(list)
    for a in attempts:
        if a.architecture:
            arch_results[a.architecture].append(a.is_correct)
    weak = [k for k, v in arch_results.items() if v and sum(v)/len(v) < 0.5]
    
    # Solved problems
    solved = [
        s.problem_id for s in
        db.query(DojoSubmission)
        .filter_by(user_id=current_user.id, passed=True, is_best=True)
        .all()
    ]
    
    # Available papers (public only)
    papers = db.query(Paper).filter(Paper.visibility == "public").limit(50).all()
    paper_list = [{"id": p.id, "title": p.title} for p in papers]
    
    # Available unsolved problems
    probs = db.query(Problem).filter(
        Problem.is_retired == False,
        ~Problem.id.in_(solved)
    ).limit(30).all()
    prob_list = [{"id": p.id, "title": p.title, "difficulty": p.difficulty} for p in probs]
    
    result = generate_learning_path(completed, weak, solved, paper_list, prob_list)
    
    if not hasattr(get_learning_path, "_cache"):
        get_learning_path._cache = {}
    get_learning_path._cache[cache_key] = {"ts": time.time(), "data": result}
    return result


@me_router.get("/papers")
def get_my_papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    offset = (page - 1) * page_size
    total = (
        db.query(func.count(Paper.id))
        .filter(Paper.uploaded_by == current_user.id)
        .scalar()
    ) or 0
    papers = (
        db.query(Paper)
        .filter(Paper.uploaded_by == current_user.id)
        .order_by(Paper.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "papers": [
            {
                "id":              p.id,
                "title":           p.title,
                "authors":         p.authors,
                "visibility":      p.visibility,
                "is_flagged":      p.is_flagged,
                "created_at":      p.created_at.isoformat() if p.created_at else None,
                "file_size_bytes": p.file_size_bytes,
            }
            for p in papers
        ],
    }

@me_router.get("/notification-preferences")
def get_notification_prefs(current_user: User = Depends(get_current_user)):
    return {
        "email_drip_opt_out": getattr(current_user, "email_drip_opt_out", False),
        "email_digest": getattr(current_user, "email_digest", False),
    }

from pydantic import BaseModel
from typing import Optional

class NotificationPrefsUpdate(BaseModel):
    email_drip_opt_out: Optional[bool] = None
    email_digest: Optional[bool] = None

@me_router.patch("/notification-preferences")
def patch_notification_prefs(
    prefs: NotificationPrefsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if prefs.email_drip_opt_out is not None:
        current_user.email_drip_opt_out = prefs.email_drip_opt_out
    if prefs.email_digest is not None:
        current_user.email_digest = prefs.email_digest
    db.commit()
    db.refresh(current_user)
    return {
        "email_drip_opt_out": current_user.email_drip_opt_out,
        "email_digest": current_user.email_digest,
    }

@me_router.get("/achievements")
def get_my_achievements(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    user_achievements = db.query(UserAchievement, Achievement).join(
        Achievement, UserAchievement.achievement_id == Achievement.id
    ).filter(
        UserAchievement.user_id == current_user.id
    ).all()
    
    earned = []
    for ua, ach in user_achievements:
        earned.append({
            "slug": ach.slug,
            "title": ach.title,
            "description": ach.description,
            "earned_at": ua.earned_at
        })
    return {"earned": earned}
