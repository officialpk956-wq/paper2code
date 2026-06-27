"""
backend/tasks/scheduled_tasks.py

Celery Beat periodic tasks:
  - cleanup_zombie_tasks    — hourly: mark stuck "running" tasks as failed
  - daily_db_backup         — daily at 03:00 UTC: pg_dump → R2
"""

import os
import logging
import datetime
import subprocess
from datetime import timedelta

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.models import Task

log = logging.getLogger(__name__)

ZOMBIE_THRESHOLD_HOURS = int(os.getenv("ZOMBIE_THRESHOLD_HOURS", "2"))
DATABASE_URL = os.getenv("DATABASE_URL", "")


def _do_cleanup_zombie_tasks(db) -> dict:
    """Business logic for zombie cleanup — accepts any SQLAlchemy session."""
    from sqlalchemy import or_, and_

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
        result = subprocess.run(
            ["pg_dump", "--dbname", DATABASE_URL, "--format=c", "--no-password"],
            capture_output=True,
            timeout=300,
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
