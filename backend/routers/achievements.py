"""
backend/routers/achievements.py

GET /api/achievements       — full catalogue (public)
GET /api/achievements/my    — user's earned achievements (JWT required)
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import Achievement, User, UserAchievement

router = APIRouter(prefix="/api/achievements", tags=["Achievements"])


@router.get("")
def list_achievements(db: Session = Depends(get_db)):
    all_ach = db.query(Achievement).order_by(Achievement.category, Achievement.xp_reward).all()
    return [
        {
            "id": a.id,
            "slug": a.slug,
            "title": a.title,
            "description": a.description,
            "icon": a.icon,
            "xp_reward": a.xp_reward,
            "category": a.category,
        }
        for a in all_ach
    ]


@router.get("/my")
def my_achievements(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(UserAchievement, Achievement)
        .join(Achievement, UserAchievement.achievement_id == Achievement.id)
        .filter(UserAchievement.user_id == current_user.id)
        .order_by(UserAchievement.earned_at.desc())
        .all()
    )
    return {
        "count": len(rows),
        "achievements": [
            {
                "slug": ach.slug,
                "title": ach.title,
                "icon": ach.icon,
                "xp_reward": ach.xp_reward,
                "category": ach.category,
                "earned_at": ua.earned_at.isoformat() if ua.earned_at else None,
                "payload": ua.payload,
            }
            for ua, ach in rows
        ],
    }
