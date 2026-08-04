import logging
import os

from dotenv import load_dotenv

load_dotenv()
import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.database import engine
from backend.models import Base

# Initialize Sentry if DSN is provided
_dsn = os.getenv("SENTRY_DSN", "")
if _dsn:
    sentry_sdk.init(
        dsn=_dsn,
        integrations=[FastApiIntegration(), SqlalchemyIntegration(), CeleryIntegration()],
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.05")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.01")),
        environment=os.getenv("ENVIRONMENT", "development"),
        release=os.getenv("APP_VERSION", ""),
    )

limiter = Limiter(key_func=get_remote_address)

from backend.logging_config import JSONFormatter, RequestIDMiddleware
from backend.middleware.alerting import SlackAlertingMiddleware
from backend.middleware.metrics import PrometheusMiddleware, metrics_endpoint
from backend.middleware.sentry_context import SentryUserContextMiddleware
from backend.routers import (
    achievements,
    admin,
    admin_metrics,
    admin_users,
    architectures,
    assessment,
    dojo,
    health,
    lab,
    leaderboard,
    learning,
    notifications,
    papers,
    papers_analysis,
    papers_pipeline,
    search,
    tasks,
    tutor,
    user,
)
from backend.routers.announcements import router as announcements_router
from backend.routers.model_viz import router as model_viz_router
from backend.routers.oauth import router as oauth_router
from backend.routers.paper_challenges import router as paper_challenges_router

# Configure JSON Logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Run strict configuration validation
from backend.modules.security.startup_validation import validate_production_security_config

validate_production_security_config()

# create_all is idempotent — safe to run on every startup (no Alembic configured)
Base.metadata.create_all(bind=engine)


def _heal_missing_columns() -> None:
    """create_all() creates missing TABLES but never ALTERs existing ones. With no
    Alembic-on-deploy, a column added to an existing model never reaches an
    already-provisioned DB → 'column does not exist' 500s at runtime. Heal it:
    ADD COLUMN (nullable, idempotent, best-effort) any model column the live table
    is missing. Never drops or alters existing columns."""
    from sqlalchemy import inspect as _inspect
    from sqlalchemy import text as _text

    try:
        insp = _inspect(engine)
        existing = set(insp.get_table_names())
    except Exception as e:  # DB unreachable — let the normal path surface it
        logger.warning("schema heal skipped (inspect failed): %s", e)
        return

    prep = engine.dialect.identifier_preparer
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue  # create_all already made this brand-new table in full
        try:
            db_cols = {c["name"] for c in insp.get_columns(table.name)}
        except Exception:
            continue
        for col in table.columns:
            if col.name in db_cols:
                continue
            try:
                ddl = col.type.compile(dialect=engine.dialect)
                stmt = (
                    f"ALTER TABLE {prep.quote(table.name)} "
                    f"ADD COLUMN {prep.quote(col.name)} {ddl}"  # nullable — safe on populated tables
                )
                with engine.begin() as conn:
                    conn.execute(_text(stmt))
                logger.warning(
                    "schema heal: added missing column %s.%s (%s)", table.name, col.name, ddl
                )
            except Exception as e:
                logger.error("schema heal: could not add %s.%s: %s", table.name, col.name, e)


_heal_missing_columns()

# Seed achievement catalogue (idempotent)
try:
    from backend.database import SessionLocal as _SL
    from backend.services.achievement_service import seed_achievements as _seed

    _db = _SL()
    _seed(_db)
    _db.close()
except Exception as _e:
    logger.warning("Achievement seeding skipped: %s", _e)

# Seed dojo problems so frontend slugs (ml-sigmoid, ...) resolve (idempotent)
try:
    from backend.database import SessionLocal as _SL2
    from backend.services.problem_seed_service import seed_dojo_problems as _seed_problems

    _db2 = _SL2()
    _seed_problems(_db2)
    _db2.close()
except Exception as _e:
    logger.warning("Dojo problem seeding skipped: %s", _e)

app = FastAPI(title="Paper2Code API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add Request ID and Security headers
app.add_middleware(RequestIDMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(SlackAlertingMiddleware)
app.add_middleware(SentryUserContextMiddleware)

from backend.middleware.trace_id import TraceIDMiddleware

app.add_middleware(TraceIDMiddleware)

# Strict CORS settings
from backend.modules.security.cors import (
    get_allow_credentials,
    get_allowed_headers,
    get_allowed_methods,
    get_allowed_origins,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=get_allow_credentials(),
    allow_methods=get_allowed_methods(),
    allow_headers=get_allowed_headers(),
)

from backend.middleware.security_headers import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)

# Mount Routers
from backend.modules.auth.api.v1 import router as new_auth_router

app.include_router(health.router)
app.include_router(new_auth_router)
app.include_router(papers_pipeline.router)
app.include_router(papers_analysis.router)
app.include_router(papers.router)
app.include_router(dojo.router)
app.include_router(learning.router)
app.include_router(tutor.router)
app.include_router(assessment.router)
app.include_router(lab.router)
app.include_router(tasks.router)
app.include_router(user.router)
app.include_router(user.me_router)
app.include_router(admin.router)
app.include_router(admin_users.router)
app.include_router(admin_metrics.router)
app.include_router(search.router)
app.include_router(leaderboard.router)
app.include_router(notifications.router)
app.include_router(achievements.router)
app.include_router(oauth_router)
app.include_router(announcements_router)
app.include_router(paper_challenges_router)

app.include_router(architectures.router)
app.include_router(model_viz_router)

# Prometheus metrics endpoint (restricted to internal IPs by Nginx in production)
from fastapi import Request as _Req


@app.get("/metrics", include_in_schema=False)
def get_metrics(_: _Req):
    return metrics_endpoint()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
