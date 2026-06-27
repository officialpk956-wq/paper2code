"""
backend/tasks/growth_tasks.py

Celery Beat periodic tasks for user retention:
  - send_onboarding_drips   : daily 09:00 UTC — Day 1/3/7 email sequences
  - send_streak_at_risk     : daily 18:00 UTC — warn users whose streak is at risk
"""

import os
import logging
import datetime

from backend.celery_app import celery_app
from backend.database import SessionLocal

log = logging.getLogger(__name__)

_DRIP_DAYS = (1, 3, 7)


# ---------------------------------------------------------------------------
# Onboarding drip (testable inner function)
# ---------------------------------------------------------------------------

def _do_onboarding_drips(db) -> dict:
    from backend.models import User, EmailDripLog
    from backend.services.email_service import send_drip_email_sync

    today = datetime.date.today()
    sent = 0
    errors = 0

    for day in _DRIP_DAYS:
        target_date = today - datetime.timedelta(days=day)
        # Users created on target_date who haven't received this drip yet
        users = (
            db.query(User)
            .filter(
                User.email.isnot(None),
                # date(created_at) == target_date
                User.created_at >= datetime.datetime.combine(target_date, datetime.time.min),
                User.created_at <  datetime.datetime.combine(target_date + datetime.timedelta(days=1), datetime.time.min),
            )
            .all()
        )
        for user in users:
            already_sent = db.query(EmailDripLog).filter_by(
                user_id=user.id, drip_day=day
            ).first()
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
    from backend.services.email_service import send_streak_at_risk_email_sync

    today     = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    # Users with streak > 0 whose last_active was yesterday (not yet active today)
    at_risk = (
        db.query(User)
        .filter(
            User.streak > 0,
            User.email.isnot(None),
            User.last_active >= datetime.datetime.combine(yesterday, datetime.time.min),
            User.last_active <  datetime.datetime.combine(today,     datetime.time.min),
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
