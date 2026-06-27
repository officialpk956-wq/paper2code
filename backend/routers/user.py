from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import (
    User,
    DojoSubmission,
    Problem,
    XPEvent,
    Paper,
    Achievement,
    UserAchievement,
)
from backend.modules.auth.dependencies import get_current_user
from backend.modules.auth.schemas import UpdateProfileRequest

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

@me_router.get("/notification-prefs")
def get_notification_prefs(current_user: User = Depends(get_current_user)):
    return {
        "email_drip_opt_out": getattr(current_user, "email_drip_opt_out", False)
    }

from pydantic import BaseModel
class NotificationPrefsUpdate(BaseModel):
    email_drip_opt_out: bool

@me_router.patch("/notification-prefs")
def patch_notification_prefs(
    prefs: NotificationPrefsUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    current_user.email_drip_opt_out = prefs.email_drip_opt_out
    db.commit()
    db.refresh(current_user)
    return {
        "email_drip_opt_out": current_user.email_drip_opt_out
    }
