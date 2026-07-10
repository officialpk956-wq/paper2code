"""
backend/tasks/scheduled_tasks.py

Celery Beat periodic tasks:
  - cleanup_zombie_tasks    — hourly: mark stuck "running" tasks as failed
  - daily_db_backup         — daily at 03:00 UTC: pg_dump → R2
"""

import datetime
import logging
import os
import subprocess
from datetime import timedelta

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.models import Task

log = logging.getLogger(__name__)

ZOMBIE_THRESHOLD_HOURS = int(os.getenv("ZOMBIE_THRESHOLD_HOURS", "2"))
DATABASE_URL = os.getenv("DATABASE_URL", "")
TUTOR_SESSION_RETENTION_DAYS = int(os.getenv("TUTOR_SESSION_RETENTION_DAYS", "90"))
XP_EVENT_RETENTION_DAYS = int(os.getenv("XP_EVENT_RETENTION_DAYS", "365"))


def _do_cleanup_zombie_tasks(db) -> dict:
    """Business logic for zombie cleanup — accepts any SQLAlchemy session."""
    from sqlalchemy import and_, or_

    cutoff = datetime.datetime.utcnow() - timedelta(hours=ZOMBIE_THRESHOLD_HOURS)
    zombies = (
        db.query(Task)
        .filter(
            Task.status == "running",
            or_(
                and_(Task.updated_at.isnot(None), Task.updated_at < cutoff),
                and_(Task.updated_at.is_(None), Task.created_at < cutoff),
            ),
        )
        .all()
    )

    count = 0
    for t in zombies:
        t.status = "failed"
        t.error = f"Zombie: task stuck in 'running' for >{ZOMBIE_THRESHOLD_HOURS}h"
        count += 1
    db.commit()
    log.info("cleanup_zombie_tasks: marked %d zombie task(s) as failed", count)
    return {"cleaned": count}


@celery_app.task(name="backend.tasks.scheduled_tasks.cleanup_zombie_tasks")
def cleanup_zombie_tasks():
    """Find tasks stuck in 'running' and mark them failed. Runs every hour via Beat."""
    db = SessionLocal()
    try:
        return _do_cleanup_zombie_tasks(db)
    except Exception as exc:
        log.error("cleanup_zombie_tasks error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


def _do_daily_db_backup() -> dict:
    """Business logic for DB backup — separated for testability."""
    if not DATABASE_URL.startswith("postgresql"):
        log.info("daily_db_backup: not PostgreSQL — skipping")
        return {"skipped": True, "reason": "not_postgres"}

    from backend.services.storage_service import R2_AVAILABLE

    if not R2_AVAILABLE:
        log.info("daily_db_backup: R2 not configured — skipping")
        return {"skipped": True, "reason": "r2_not_configured"}

    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"backup_{stamp}.dump"
    try:
        import urllib.parse

        parsed = urllib.parse.urlparse(DATABASE_URL)
        env = os.environ.copy()
        if parsed.password:
            env["PGPASSWORD"] = parsed.password
        pg_args = ["pg_dump"]
        if parsed.hostname:
            pg_args += ["--host", parsed.hostname]
        if parsed.port:
            pg_args += ["--port", str(parsed.port)]
        if parsed.username:
            pg_args += ["--username", parsed.username]
        db_name = parsed.path.lstrip("/")
        if db_name:
            pg_args += ["--dbname", db_name]
        pg_args += ["--format=c", "--no-password"]
        result = subprocess.run(
            pg_args,
            capture_output=True,
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))

        dump_bytes = result.stdout
        from backend.services.storage_service import store_pdf as _store

        ref = _store(dump_bytes, filename)
        log.info("daily_db_backup: uploaded %s (%d bytes) → %s", filename, len(dump_bytes), ref)
        return {"success": True, "ref": ref, "size_bytes": len(dump_bytes), "stamp": stamp}
    except Exception as exc:
        log.error("daily_db_backup failed: %s", exc)
        return {"success": False, "error": str(exc)}


@celery_app.task(name="backend.tasks.scheduled_tasks.daily_db_backup")
def daily_db_backup():
    """Dump PostgreSQL DB → R2 as a dated .dump. Skipped on SQLite/no R2. Runs at 03:00 UTC."""
    return _do_daily_db_backup()


# ---------------------------------------------------------------------------
# Prune old tutor sessions  (Tue 02:00 UTC)
# ---------------------------------------------------------------------------


def _do_prune_tutor_sessions(db) -> dict:
    from sqlalchemy import and_, or_

    from backend.models import TutorSessionRecord

    cutoff = datetime.datetime.utcnow() - timedelta(days=TUTOR_SESSION_RETENTION_DAYS)
    deleted = (
        db.query(TutorSessionRecord)
        .filter(
            or_(
                TutorSessionRecord.last_active_at < cutoff,
                and_(
                    TutorSessionRecord.last_active_at.is_(None),
                    TutorSessionRecord.created_at < cutoff,
                ),
            )
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    log.info("prune_tutor_sessions: deleted %d old session records", deleted)
    return {"deleted": deleted, "cutoff_days": TUTOR_SESSION_RETENTION_DAYS}


@celery_app.task(name="backend.tasks.scheduled_tasks.prune_old_tutor_sessions")
def prune_old_tutor_sessions():
    """Delete TutorSessionRecord rows older than TUTOR_SESSION_RETENTION_DAYS. Runs Tue 02:00 UTC."""
    db = SessionLocal()
    try:
        return _do_prune_tutor_sessions(db)
    except Exception as exc:
        log.error("prune_tutor_sessions error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Prune old XP events  (2nd of month 01:00 UTC)
# ---------------------------------------------------------------------------


def _do_prune_xp_events(db) -> dict:
    from backend.models import XPEvent

    cutoff = datetime.datetime.utcnow() - timedelta(days=XP_EVENT_RETENTION_DAYS)
    deleted = (
        db.query(XPEvent).filter(XPEvent.created_at < cutoff).delete(synchronize_session=False)
    )
    db.commit()
    log.info(
        "prune_xp_events: deleted %d XP events older than %d days", deleted, XP_EVENT_RETENTION_DAYS
    )
    return {"deleted": deleted, "cutoff_days": XP_EVENT_RETENTION_DAYS}


@celery_app.task(name="backend.tasks.scheduled_tasks.prune_old_xp_events")
def prune_old_xp_events():
    """Delete XPEvent rows older than XP_EVENT_RETENTION_DAYS. Runs 2nd of month 01:00 UTC."""
    db = SessionLocal()
    try:
        return _do_prune_xp_events(db)
    except Exception as exc:
        log.error("prune_xp_events error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Nightly acceptance-rate reconciliation  (04:00 UTC)
# ---------------------------------------------------------------------------


def _do_recalc_acceptance_rates(db) -> dict:
    from sqlalchemy import func

    from backend.models import DojoSubmission, Problem

    problem_ids = [row[0] for row in db.query(DojoSubmission.problem_id).distinct().all()]
    updated = 0
    for pid in problem_ids:
        total = (
            db.query(func.count(func.distinct(DojoSubmission.user_id)))
            .filter(DojoSubmission.problem_id == pid)
            .scalar()
        ) or 0
        passed = (
            db.query(func.count(func.distinct(DojoSubmission.user_id)))
            .filter(DojoSubmission.problem_id == pid, DojoSubmission.passed == True)
            .scalar()
        ) or 0
        rate = round(passed / total, 4) if total > 0 else 0.0
        problem = db.query(Problem).filter_by(id=pid).first()
        if problem is not None and float(problem.acceptance_rate or 0) != rate:
            problem.acceptance_rate = rate
            updated += 1
    if updated:
        db.commit()
    log.info("recalc_acceptance_rates: reconciled %d / %d problems", updated, len(problem_ids))
    return {"problems_checked": len(problem_ids), "problems_updated": updated}


@celery_app.task(name="backend.tasks.scheduled_tasks.recalc_all_acceptance_rates")
def recalc_all_acceptance_rates():
    """Reconcile Problem.acceptance_rate for all problems with submissions. Runs 04:00 UTC."""
    db = SessionLocal()
    try:
        return _do_recalc_acceptance_rates(db)
    except Exception as exc:
        log.error("recalc_acceptance_rates error: %s", exc)
        return {"error": str(exc)}
    finally:
        db.close()
