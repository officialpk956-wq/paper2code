import os
from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "p2c",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=[
        "backend.tasks.paper_tasks",
        "backend.tasks.dojo_tasks",
        "backend.tasks.scheduled_tasks",
        "backend.tasks.growth_tasks",
    ],
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.task_track_started = True
celery_app.conf.worker_prefetch_multiplier = 1  # fair dispatch for long tasks

celery_app.conf.beat_schedule = {
    "cleanup-zombie-tasks-hourly": {
        "task": "backend.tasks.scheduled_tasks.cleanup_zombie_tasks",
        "schedule": crontab(minute=0),           # top of every hour
    },
    "daily-db-backup": {
        "task": "backend.tasks.scheduled_tasks.daily_db_backup",
        "schedule": crontab(hour=3, minute=0),   # 03:00 UTC daily
    },
    "onboarding-drip-daily": {
        "task": "backend.tasks.growth_tasks.send_onboarding_drips",
        "schedule": crontab(hour=9, minute=0),   # 09:00 UTC daily
    },
    "streak-at-risk-daily": {
        "task": "backend.tasks.growth_tasks.send_streak_at_risk",
        "schedule": crontab(hour=18, minute=0),  # 18:00 UTC daily
    },
    "weekly-leaderboard-reset": {
        "task": "backend.tasks.growth_tasks.weekly_leaderboard_reset",
        "schedule": crontab(day_of_week=1, hour=0, minute=0),  # Mon 00:00 UTC
    },
    "monthly-quota-reset": {
        "task": "backend.tasks.growth_tasks.monthly_quota_reset",
        "schedule": crontab(day_of_month=1, hour=0, minute=1),  # 1st of month 00:01 UTC
    },
    "weekly-digest-emails": {
        "task": "backend.tasks.growth_tasks.weekly_digest",
        "schedule": crontab(day_of_week=0, hour=8, minute=0),  # Sun 08:00 UTC
    },
    "prune-old-tutor-sessions": {
        "task": "backend.tasks.scheduled_tasks.prune_old_tutor_sessions",
        "schedule": crontab(day_of_week=2, hour=2, minute=0),  # Tue 02:00 UTC
    },
    "prune-old-xp-events": {
        "task": "backend.tasks.scheduled_tasks.prune_old_xp_events",
        "schedule": crontab(day_of_month=2, hour=1, minute=0),  # 2nd of month 01:00 UTC
    },
    "recalc-acceptance-rates-nightly": {
        "task": "backend.tasks.scheduled_tasks.recalc_all_acceptance_rates",
        "schedule": crontab(hour=4, minute=0),  # 04:00 UTC nightly
    },
}
celery_app.conf.timezone = "UTC"
