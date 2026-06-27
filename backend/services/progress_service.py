from datetime import datetime, timezone, date
from sqlalchemy.orm import Session
from backend.models import User

XP_AWARDS = {
    "dojo.solved.easy":        50,
    "dojo.solved.medium":     100,
    "dojo.solved.hard":       300,
    "dojo.attempt":             5,
    "topic.completed":         50,
    "paper.uploaded":          25,
    "assessment.completed":    75,
    "domain.completed":       500,
}

# Architecture slugs grouped by domain — used for domain completion detection.
# entity_type="architecture", entity_id=slug (matches content-index slugs).
DOMAIN_ARCHITECTURES: dict[str, list[str]] = {
    "transformers": ["transformer", "bert", "gpt", "vit", "clip", "llama"],
    "cnns":         ["alexnet", "vgg", "resnet", "inception", "mobilenet", "efficientnet"],
    "diffusion":    ["ddpm", "ddim", "stable-diffusion", "ldm"],
    "generative":   ["gan", "vae", "ae"],
    "rl":           ["ppo", "dqn", "a3c"],
    "graph":        ["gcn", "gat", "graphsage"],
}

# Reverse lookup: slug → domain
_SLUG_TO_DOMAIN: dict[str, str] = {
    slug: domain
    for domain, slugs in DOMAIN_ARCHITECTURES.items()
    for slug in slugs
}


def update_user_activity(db: Session, user_id: int) -> None:
    """Call on any learning action to maintain streak and trigger streak achievements."""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        return
    today = datetime.now(timezone.utc).date()
    last = user.last_active.date() if user.last_active else None
    if last == today:
        return  # already active today, no change
    if last and (today - last).days == 1:
        user.streak = (user.streak or 0) + 1
    else:
        user.streak = 1  # first ever, or streak broken
    user.last_active = datetime.now(timezone.utc)
    streak = user.streak
    try:
        db.commit()
    except Exception:
        db.rollback()
        return
    # Check streak achievements after commit
    try:
        from backend.services.achievement_service import check_and_award
        check_and_award(db, user_id, "streak.updated", {"streak": streak})
    except Exception:
        pass


def award_xp(db: Session, user_id: int, event: str, entity_id: str | None = None) -> int:
    """
    Award XP for an event, log to XPEvent, and return the new total points.
    entity_id is an optional string referencing the triggering resource.
    """
    amount = XP_AWARDS.get(event, 0)
    if not amount:
        return 0
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        return 0
    user.points        = (user.points or 0) + amount
    user.weekly_points = (user.weekly_points or 0) + amount
    from backend.models import XPEvent
    db.add(XPEvent(user_id=user_id, action=event, amount=amount, entity_id=entity_id))
    try:
        db.commit()
        return user.points
    except Exception:
        db.rollback()
        return 0


def check_domain_completion(db: Session, user_id: int, entity_id: str) -> str | None:
    """
    After an architecture is completed, check if the entire domain is now done.
    Returns the domain slug if newly completed, else None.
    Does NOT award XP — caller is responsible for that.
    """
    domain = _SLUG_TO_DOMAIN.get(entity_id)
    if not domain:
        return None

    required = set(DOMAIN_ARCHITECTURES[domain])
    from backend.models import LearnerProgress
    completed = {
        r.entity_id
        for r in db.query(LearnerProgress).filter_by(
            learner_id=str(user_id),
            entity_type="architecture",
            status="completed",
        ).all()
        if r.entity_id in required
    }
    return domain if completed >= required else None
