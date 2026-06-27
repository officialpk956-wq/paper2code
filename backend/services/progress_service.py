from datetime import datetime, timezone, date
from sqlalchemy.orm import Session
from backend.models import User

XP_AWARDS = {
    "dojo.solved.easy":        50,
    "dojo.solved.medium":     100,
    "dojo.solved.hard":       250,
    "dojo.attempt":             5,
    "topic.completed":         50,
    "paper.uploaded":          25,
    "assessment.completed":    75,
    "domain.completed":       500,
}

def update_user_activity(db: Session, user_id: int) -> None:
    """Call on any learning action to maintain streak."""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        return
    today = date.today()
    last = user.last_active.date() if user.last_active else None
    if last == today:
        return  # already active today, no change
    if last and (today - last).days == 1:
        user.streak = (user.streak or 0) + 1   # consecutive day
    elif last and (today - last).days > 1:
        user.streak = 1   # streak broken, reset to 1
    else:
        user.streak = 1   # first ever activity
    user.last_active = datetime.now(timezone.utc)
    try:
        db.commit()
    except Exception:
        db.rollback()

def award_xp(db: Session, user_id: int, event: str) -> int:
    """Award XP for an event. Returns new total points."""
    amount = XP_AWARDS.get(event, 0)
    if not amount:
        return 0
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        return 0
    user.points = (user.points or 0) + amount
    try:
        db.commit()
        return user.points
    except Exception:
        db.rollback()
        return 0
