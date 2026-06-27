import logging
import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.database import engine
from backend.models import Base

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize Sentry if DSN is provided
_dsn = os.getenv("SENTRY_DSN", "")
if _dsn:
    sentry_sdk.init(
        dsn=_dsn,
        integrations=[FastApiIntegration(), SqlalchemyIntegration()],
        traces_sample_rate=0.05,
        environment=os.getenv("ENVIRONMENT", "development"),
    )

limiter = Limiter(key_func=get_remote_address)

from backend.routers import health, auth, papers, dojo, learning, lab, tasks, user, admin, search, leaderboard, notifications, achievements
from backend.routers.oauth import router as oauth_router
from backend.logging_config import RequestIDMiddleware, JSONFormatter

# Configure JSON Logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Run strict configuration validation
from backend.modules.security.startup_validation import validate_production_security_config
validate_production_security_config()

# Only use create_all for local dev/test. Production uses: alembic upgrade head
import os as _os
if _os.getenv("ENVIRONMENT", "development") != "production":
    Base.metadata.create_all(bind=engine)
    # Seed achievement catalogue (idempotent)
    try:
        from backend.database import SessionLocal as _SL
        from backend.services.achievement_service import seed_achievements as _seed
        _db = _SL()
        _seed(_db)
        _db.close()
    except Exception as _e:
        logger.warning("Achievement seeding skipped: %s", _e)

app = FastAPI(title="Paper2Code API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add Request ID and Security headers
app.add_middleware(RequestIDMiddleware)
from backend.modules.security.middleware.security_headers import SecurityHeadersMiddleware
app.add_middleware(SecurityHeadersMiddleware)

# Strict CORS settings
from backend.modules.security.cors import (
    get_allowed_origins,
    get_allowed_headers,
    get_allowed_methods,
    get_allow_credentials,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=get_allow_credentials(),
    allow_methods=get_allowed_methods(),
    allow_headers=get_allowed_headers(),
)

# Mount Routers
from backend.modules.auth.api.v1 import router as new_auth_router
app.include_router(health.router)
app.include_router(new_auth_router)
app.include_router(papers.router)
app.include_router(dojo.router)
app.include_router(learning.router)
app.include_router(lab.router)
app.include_router(tasks.router)
app.include_router(user.router)
app.include_router(admin.router)
app.include_router(search.router)
app.include_router(leaderboard.router)
app.include_router(notifications.router)
app.include_router(achievements.router)
app.include_router(oauth_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
