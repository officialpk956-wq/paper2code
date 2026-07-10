"""
backend/tasks/growth_tasks.py

Celery Beat periodic tasks for user retention:
  - send_onboarding_drips   : daily 09:00 UTC — Day 1/3/7 email sequences
  - send_streak_at_risk     : daily 18:00 UTC — warn users whose streak is at risk
"""

import datetime
import logging

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.services.email_service import (
    send_drip_email_sync,
    send_streak_at_risk_email_sync,
    send_weekly_digest_email_sync,
)

log = logging.getLogger(__name__)

_DRIP_DAYS = (1, 3, 7)


# ---------------------------------------------------------------------------
# Onboarding drip (testable inner function)
# ---------------------------------------------------------------------------


def _do_onboarding_drips(db) -> dict:
    from backend.models import EmailDripLog, User

    today = datetime.datetime.utcnow().date()
    sent = 0
    errors = 0

    for day in _DRIP_DAYS:
        target_date = today - datetime.timedelta(days=day)
        window_start = datetime.datetime.combine(target_date, datetime.time.min)
        window_end = datetime.datetime.combine(
            target_date + datetime.timedelta(days=1), datetime.time.min
        )
        users = (
            db.query(User)
            .filter(
                User.email.isnot(None),
                User.created_at >= window_start,
                User.created_at < window_end,
            )
            .all()
        )
        for user in users:
            already_sent = db.query(EmailDripLog).filter_by(user_id=user.id, drip_day=day).first()
            if already_sent:
                continue
            try:
                send_drip_email_sync(user.email, user.name or user.email, day)
                db.add(EmailDripLog(user_id=user.id, drip_day=day))
                db.commit()
                sent += 1
            except Exception as exc:
                db.rollback()
                log.error("Drip day=%d user=%d error: %s", day, user.id, exc)
                errors += 1

    return {"sent": sent, "errors": errors}


@celery_app.task(name="backend.tasks.growth_tasks.send_onboarding_drips")
def send_onboarding_drips():
    db = SessionLocal()
    try:
        return _do_onboarding_drips(db)
    except Exception as exc:
        log.error("send_onboarding_drips error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Streak-at-risk  (testable inner function)
# ---------------------------------------------------------------------------


def _do_streak_at_risk(db) -> dict:
    from backend.models import User

    today = datetime.datetime.utcnow().date()
    yesterday = today - datetime.timedelta(days=1)

    # Users with streak > 0 whose last_active was yesterday (not yet active today)
    at_risk = (
        db.query(User)
        .filter(
            User.streak > 0,
            User.email.isnot(None),
            User.last_active >= datetime.datetime.combine(yesterday, datetime.time.min),
            User.last_active < datetime.datetime.combine(today, datetime.time.min),
        )
        .all()
    )

    sent = 0
    errors = 0
    for user in at_risk:
        try:
            send_streak_at_risk_email_sync(user.email, user.name or user.email, user.streak)
            sent += 1
        except Exception as exc:
            log.error("streak_at_risk user=%d error: %s", user.id, exc)
            errors += 1

    return {"notified": sent, "errors": errors, "at_risk_count": len(at_risk)}


@celery_app.task(name="backend.tasks.growth_tasks.send_streak_at_risk")
def send_streak_at_risk():
    db = SessionLocal()
    try:
        return _do_streak_at_risk(db)
    except Exception as exc:
        log.error("send_streak_at_risk error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Weekly leaderboard reset  (Mon 00:00 UTC)
# ---------------------------------------------------------------------------


def _do_weekly_leaderboard_reset(db) -> dict:
    from backend.models import LeaderboardArchive, User

    now = datetime.datetime.utcnow()
    week_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    users = db.query(User).filter(User.weekly_points > 0).order_by(User.weekly_points.desc()).all()
    archived = 0
    for rank, user in enumerate(users, start=1):
        entry = LeaderboardArchive(
            week_start=week_start,
            user_id=user.id,
            weekly_points=user.weekly_points,
            rank=rank,
        )
        db.add(entry)
        user.weekly_points = 0
        archived += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        log.error("weekly_leaderboard_reset commit error: %s", exc)
        return {"error": str(exc)}

    # Award leaderboard-top-10 achievement to all-time top 10 users
    lb_achievements = 0
    try:
        from backend.services.achievement_service import check_and_award

        top_users = (
            db.query(User).filter(User.points > 0).order_by(User.points.desc()).limit(10).all()
        )
        for user in top_users:
            newly = check_and_award(db, user.id, "leaderboard.top10")
            lb_achievements += len(newly)
    except Exception as exc:
        log.error("leaderboard achievement check failed: %s", exc)

    log.info(
        "Weekly leaderboard reset: archived %d entries, %d lb achievements",
        archived,
        lb_achievements,
    )
    return {
        "archived": archived,
        "week_start": week_start.isoformat(),
        "lb_achievements_awarded": lb_achievements,
    }


@celery_app.task(name="backend.tasks.growth_tasks.weekly_leaderboard_reset")
def weekly_leaderboard_reset():
    db = SessionLocal()
    try:
        return _do_weekly_leaderboard_reset(db)
    except Exception as exc:
        log.error("weekly_leaderboard_reset error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Monthly usage quota reset  (1st of month 00:01 UTC)
# ---------------------------------------------------------------------------


def _do_monthly_quota_reset(db) -> dict:
    from backend.models import UsageLog

    # Clear usage logs older than 60 days to prevent unbounded table growth
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=60)
    deleted = (
        db.query(UsageLog).filter(UsageLog.created_at < cutoff).delete(synchronize_session=False)
    )
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        log.error("monthly_quota_reset commit error: %s", exc)
        return {"error": str(exc)}

    log.info("Monthly quota reset: removed %d old UsageLog entries", deleted)
    return {"deleted_usage_logs": deleted}


@celery_app.task(name="backend.tasks.growth_tasks.monthly_quota_reset")
def monthly_quota_reset():
    db = SessionLocal()
    try:
        return _do_monthly_quota_reset(db)
    except Exception as exc:
        log.error("monthly_quota_reset error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Weekly digest emails  (Sun 08:00 UTC)
# ---------------------------------------------------------------------------


def _do_weekly_digest(db) -> dict:
    from sqlalchemy import func

    from backend.models import DojoSubmission, User

    # Stats window: past 7 days
    since = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    users = db.query(User).filter(User.email.isnot(None)).all()

    sent = 0
    errors = 0
    for user in users:
        problems_solved = (
            db.query(func.count(DojoSubmission.id))
            .filter(
                DojoSubmission.user_id == user.id,
                DojoSubmission.passed == True,
                DojoSubmission.created_at >= since,
            )
            .scalar()
            or 0
        )
        # Only send if user did something this week
        if problems_solved == 0 and (user.weekly_points or 0) == 0:
            continue
        stats = {
            "problems_solved": problems_solved,
            "xp_earned": user.weekly_points or 0,
            "rank_change": 0,
        }
        try:
            send_weekly_digest_email_sync(user.email, user.name or user.email, stats)
            sent += 1
        except Exception as exc:
            log.error("digest user=%d error: %s", user.id, exc)
            errors += 1

    return {"sent": sent, "errors": errors}


@celery_app.task(name="backend.tasks.growth_tasks.weekly_digest")
def weekly_digest():
    db = SessionLocal()
    try:
        return _do_weekly_digest(db)
    except Exception as exc:
        log.error("weekly_digest error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()
