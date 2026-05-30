"""
backend/database.py

SQLAlchemy 2.x database engine, session factory, and FastAPI dependency
injection helper.

Usage:
    from backend.database import engine, SessionLocal, get_db
    from backend.models import Base

    # Create all tables (dev/test only — use Alembic in production)
    Base.metadata.create_all(bind=engine)

    # In a FastAPI route:
    @app.get("/items")
    def read_items(db: Session = Depends(get_db)):
        ...
"""

import os
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

# ---------------------------------------------------------------------------
# Connection URL
# ---------------------------------------------------------------------------
# Reads DATABASE_URL from environment.  Falls back to an in-process SQLite
# file so the application can start without a running PostgreSQL instance
# (useful for CI and local dev without Docker).
# In production always set DATABASE_URL to a PostgreSQL DSN, e.g.
#   postgresql+psycopg2://user:pass@localhost:5432/tensortonic
# ---------------------------------------------------------------------------

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite:///./tensortonic_dev.db",  # safe fallback — replaced in prod
)

# Postgres-specific connection pool settings are ignored for SQLite.
_connect_args: dict = (
    {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_engine(
    DATABASE_URL,
    # Echo SQL to stdout when SQLALCHEMY_ECHO is set (dev debugging only)
    echo=os.getenv("SQLALCHEMY_ECHO", "false").lower() == "true",
    # Pool settings (harmless for SQLite, meaningful for Postgres)
    pool_pre_ping=True,          # verify connections before checkout
    pool_recycle=1800,           # recycle connections every 30 min
    connect_args=_connect_args,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# ---------------------------------------------------------------------------
# FastAPI dependency injection
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """
    Yield a SQLAlchemy Session and guarantee cleanup on request teardown.

    Use as a FastAPI Depends() argument:

        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Health-check helper
# ---------------------------------------------------------------------------

def ping_db() -> dict:
    """
    Execute a lightweight round-trip query to verify the database is reachable.

    Returns a dict with keys:
        ok      (bool)   — True if reachable
        dialect (str)    — e.g. "postgresql", "sqlite"
        url     (str)    — redacted connection URL (password masked)
        error   (str)    — empty string on success, message on failure
    """
    dialect = engine.dialect.name
    # Mask password in URL for safe logging
    safe_url = str(engine.url).split("@")[-1] if "@" in str(engine.url) else str(engine.url)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True, "dialect": dialect, "url": safe_url, "error": ""}
    except Exception as exc:
        return {"ok": False, "dialect": dialect, "url": safe_url, "error": str(exc)}
