"""
backend/services/achievement_service.py

Achievement catalogue + event-triggered award logic.

Usage:
    from backend.services.achievement_service import check_and_award, seed_achievements
    check_and_award(db, user_id, "paper.uploaded")
    check_and_award(db, user_id, "dojo.solved.hard", {"problem_id": "..."})
"""

import logging

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.models import Achievement, DojoSubmission, Task, User, UserAchievement

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalogue — seeded on startup (idempotent)
# ---------------------------------------------------------------------------

ACHIEVEMENTS = [
    {
        "slug": "first-paper",
        "title": "First Upload",
        "description": "Upload your first research paper.",
        "icon": "📄",
        "xp_reward": 50,
        "category": "Learning",
    },
    {
        "slug": "paper-master",
        "title": "Paper Master",
        "description": "Upload 10 research papers.",
        "icon": "📚",
        "xp_reward": 200,
        "category": "Learning",
    },
    {
        "slug": "first-solve",
        "title": "First Blood",
        "description": "Solve your first Dojo problem.",
        "icon": "⚔️",
        "xp_reward": 50,
        "category": "Learning",
    },
    {
        "slug": "easy-streak-3",
        "title": "Warm Up",
        "description": "Solve 3 Easy problems.",
        "icon": "🔥",
        "xp_reward": 100,
        "category": "Learning",
    },
    {
        "slug": "hard-conqueror",
        "title": "Hard Conqueror",
        "description": "Solve a Hard difficulty problem.",
        "icon": "🏔️",
        "xp_reward": 300,
        "category": "Learning",
    },
    {
        "slug": "week-streak-7",
        "title": "Week Warrior",
        "description": "Maintain a 7-day activity streak.",
        "icon": "📅",
        "xp_reward": 200,
        "category": "Consistency",
    },
    {
        "slug": "month-streak-30",
        "title": "Unstoppable",
        "description": "Maintain a 30-day activity streak.",
        "icon": "🌟",
        "xp_reward": 1000,
        "category": "Consistency",
    },
    {
        "slug": "ai-curious",
        "title": "AI Curious",
        "description": "Ask the AI Tutor 10 questions.",
        "icon": "🤖",
        "xp_reward": 100,
        "category": "Learning",
    },
    {
        "slug": "leaderboard-top-10",
        "title": "Top 10",
        "description": "Reach the top 10 on the all-time leaderboard.",
        "icon": "🏆",
        "xp_reward": 300,
        "category": "Community",
    },
    {
        "slug": "early-adopter",
        "title": "Early Adopter",
        "description": "One of the first 100 users to join Paper2Code.",
        "icon": "🌱",
        "xp_reward": 500,
        "category": "Community",
    },
]


def seed_achievements(db: Session) -> int:
    """Insert missing achievements (idempotent). Returns count of new rows."""
    added = 0
    for spec in ACHIEVEMENTS:
        exists = db.query(Achievement).filter_by(slug=spec["slug"]).first()
        if not exists:
            db.add(Achievement(**spec))
            added += 1
    if added:
        db.commit()
    return added


# ---------------------------------------------------------------------------
# Award logic
# ---------------------------------------------------------------------------


def _award(db: Session, user_id: int, slug: str, payload: dict | None = None) -> bool:
    """
    Award achievement `slug` to `user_id`.
    Returns True if newly awarded, False if already had it or slug not found.
    """
    ach = db.query(Achievement).filter_by(slug=slug).first()
    if not ach:
        return False

    ua = UserAchievement(user_id=user_id, achievement_id=ach.id, payload=payload)
    db.add(ua)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        return False  # already earned

    # Award XP
    try:
        from backend.services.progress_service import award_xp

        award_xp(db, user_id, "achievement.earned")
        user = db.get(User, user_id)
        if user:
            user.points = (user.points or 0) + ach.xp_reward
    except Exception:
        pass

    # Create in-app notification
    try:
        from backend.models import Notification

        notif = Notification(
            user_id=user_id,
            type="achievement.unlocked",
            title=f"Achievement unlocked: {ach.title}",
            body=ach.description,
            payload={"slug": slug, "xp_reward": ach.xp_reward, "icon": ach.icon},
        )
        db.add(notif)
    except Exception:
        pass

    db.commit()
    log.info("Achievement awarded: user_id=%d slug=%s xp=%d", user_id, slug, ach.xp_reward)
    return True


def check_and_award(
    db: Session, user_id: int, event: str, context: dict | None = None
) -> list[str]:
    """
    Check all achievements that can be triggered by `event`.
    Returns list of newly-awarded slugs.
    """
    awarded = []

    if event == "paper.uploaded":
        if _award(db, user_id, "first-paper"):
            awarded.append("first-paper")
        paper_count = (
            db.query(Task)
            .filter(
                Task.user_id == user_id,
                Task.type == "paper.codegen",
                Task.status == "completed",
            )
            .count()
        )
        if paper_count >= 10 and _award(db, user_id, "paper-master"):
            awarded.append("paper-master")

    elif event in ("dojo.solved.easy", "dojo.solved.medium", "dojo.solved.hard"):
        if _award(db, user_id, "first-solve", context):
            awarded.append("first-solve")

        if event == "dojo.solved.hard" and _award(db, user_id, "hard-conqueror", context):
            awarded.append("hard-conqueror")

        easy_solves = (
            db.query(DojoSubmission)
            .filter(
                DojoSubmission.user_id == user_id,
                DojoSubmission.passed == True,
            )
            .join(DojoSubmission.problem)
            .filter_by(difficulty="Easy")
            .count()
        )
        if easy_solves >= 3 and _award(db, user_id, "easy-streak-3"):
            awarded.append("easy-streak-3")

    elif event == "streak.updated":
        streak = (context or {}).get("streak", 0)
        if streak >= 7 and _award(db, user_id, "week-streak-7"):
            awarded.append("week-streak-7")
        if streak >= 30 and _award(db, user_id, "month-streak-30"):
            awarded.append("month-streak-30")

    elif event == "tutor.query":
        tutor_count = (context or {}).get("queries_today", 0)
        if tutor_count >= 10 and _award(db, user_id, "ai-curious"):
            awarded.append("ai-curious")

    elif event == "user.registered":
        total_users = db.query(User).count()
        if total_users <= 100 and _award(db, user_id, "early-adopter"):
            awarded.append("early-adopter")

    elif event == "leaderboard.top10":
        if _award(db, user_id, "leaderboard-top-10", context):
            awarded.append("leaderboard-top-10")

    return awarded
