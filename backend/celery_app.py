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
}
celery_app.conf.timezone = "UTC"
