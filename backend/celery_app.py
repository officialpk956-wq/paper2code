import os
import ssl

from celery import Celery
from celery.schedules import crontab
import redis

# Monkey-patch redis.from_url to bypass Kombu stripping query params, which
# causes redis-py to throw the ssl_cert_reqs ValueError on rediss:// URLs.
_orig_from_url = redis.from_url
def _patched_from_url(url, **kwargs):
    if isinstance(url, str) and url.startswith("rediss://") and "ssl_cert_reqs" not in url:
        url += ("&" if "?" in url else "?") + "ssl_cert_reqs=CERT_NONE"
    return _orig_from_url(url, **kwargs)
redis.from_url = _patched_from_url

if hasattr(redis.Redis, "from_url"):
    _orig_redis_from_url = redis.Redis.from_url
    def _patched_redis_from_url(cls, url, **kwargs):
        if isinstance(url, str) and url.startswith("rediss://") and "ssl_cert_reqs" not in url:
            url += ("&" if "?" in url else "?") + "ssl_cert_reqs=CERT_NONE"
        return _orig_redis_from_url(url, **kwargs)
    redis.Redis.from_url = classmethod(_patched_redis_from_url)

if hasattr(redis, "ConnectionPool") and hasattr(redis.ConnectionPool, "from_url"):
    _orig_pool_from_url = redis.ConnectionPool.from_url
    def _patched_pool_from_url(cls, url, **kwargs):
        if isinstance(url, str) and url.startswith("rediss://") and "ssl_cert_reqs" not in url:
            url += ("&" if "?" in url else "?") + "ssl_cert_reqs=CERT_NONE"
        return _orig_pool_from_url(url, **kwargs)
    redis.ConnectionPool.from_url = classmethod(_patched_pool_from_url)

# Initialize Sentry in the Celery worker process when DSN is configured.
# FastAPI server has its own init in server.py; this covers the worker side.
_sentry_dsn = os.getenv("SENTRY_DSN", "")
if _sentry_dsn:
    import sentry_sdk
    from sentry_sdk.integrations.celery import CeleryIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    sentry_sdk.init(
        dsn=_sentry_dsn,
        integrations=[CeleryIntegration(), SqlalchemyIntegration()],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.05")),
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("APP_VERSION", ""),
    )

_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Render's managed Redis uses TLS (rediss://). redis-py requires ssl_cert_reqs
# in the URL query string when using rediss://.
if _redis_url.startswith("rediss://") and "ssl_cert_reqs" not in _redis_url:
    _delimiter = "&" if "?" in _redis_url else "?"
    _redis_url += f"{_delimiter}ssl_cert_reqs=CERT_NONE"

# If the environment explicitly overrides the result backend, we must patch that too
if "CELERY_RESULT_BACKEND" in os.environ:
    _result_backend = os.environ["CELERY_RESULT_BACKEND"]
    if _result_backend.startswith("rediss://") and "ssl_cert_reqs" not in _result_backend:
        _delimiter = "&" if "?" in _result_backend else "?"
        os.environ["CELERY_RESULT_BACKEND"] = f"{_result_backend}{_delimiter}ssl_cert_reqs=CERT_NONE"

if "CELERY_BROKER_URL" in os.environ:
    _broker_url = os.environ["CELERY_BROKER_URL"]
    if _broker_url.startswith("rediss://") and "ssl_cert_reqs" not in _broker_url:
        _delimiter = "&" if "?" in _broker_url else "?"
        os.environ["CELERY_BROKER_URL"] = f"{_broker_url}{_delimiter}ssl_cert_reqs=CERT_NONE"

celery_app = Celery(
    "p2c",
    broker=_redis_url,
    backend=_redis_url,
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

if _redis_url.startswith("rediss://"):
    _ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE}
    celery_app.conf.broker_use_ssl = _ssl_opts
    celery_app.conf.redis_backend_use_ssl = _ssl_opts


celery_app.conf.beat_schedule = {
    "cleanup-zombie-tasks-hourly": {
        "task": "backend.tasks.scheduled_tasks.cleanup_zombie_tasks",
        "schedule": crontab(minute=0),  # top of every hour
    },
    "daily-db-backup": {
        "task": "backend.tasks.scheduled_tasks.daily_db_backup",
        "schedule": crontab(hour=3, minute=0),  # 03:00 UTC daily
    },
    "onboarding-drip-daily": {
        "task": "backend.tasks.growth_tasks.send_onboarding_drips",
        "schedule": crontab(hour=9, minute=0),  # 09:00 UTC daily
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
