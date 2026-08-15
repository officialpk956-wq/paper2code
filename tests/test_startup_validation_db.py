"""
tests/test_startup_validation_db.py

Unit tests for database URL startup validation in validate_production_security_config():
1. ENVIRONMENT=production with postgresql:// URL passes.
2. ENVIRONMENT=production with postgres:// URL passes.
3. ENVIRONMENT=production with postgresql+psycopg2:// URL passes.
4. ENVIRONMENT=production with sqlite:/// URL raises ValueError ("SECURITY FAILURE: Production database must be PostgreSQL").
5. ENVIRONMENT=production with unset/empty DATABASE_URL raises ValueError ("SECURITY FAILURE: DATABASE_URL environment variable is missing").
6. ENVIRONMENT=development with sqlite:/// URL passes (dev allowed to use SQLite).
"""

import os
import pytest
from backend.modules.security.startup_validation import validate_production_security_config


@pytest.fixture(autouse=True)
def setup_valid_production_env(monkeypatch):
    """Set up baseline valid production configuration for non-DB checks."""
    monkeypatch.setenv("SECRET_KEY", "super_secret_jwt_key_for_testing_32chars_long!")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://paper2code.com,https://www.paper2code.com")
    monkeypatch.setenv("CONTENT_SECURITY_POLICY", "default-src 'self'")
    monkeypatch.setenv("REDIS_REQUIRED", "false")


def test_production_postgresql_scheme_passes(monkeypatch):
    """Scenario 1: postgresql:// scheme is accepted in production."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql://p2c:p2cdev@localhost:5432/p2c")

    # Should not raise
    validate_production_security_config()


def test_production_postgres_alias_scheme_passes(monkeypatch):
    """Scenario 2: postgres:// scheme (Render/Heroku style) is accepted in production."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgres://p2c:p2cdev@localhost:5432/p2c")

    validate_production_security_config()


def test_production_postgresql_psycopg2_scheme_passes(monkeypatch):
    """Scenario 2b: postgresql+psycopg2:// driver scheme is accepted in production."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://p2c:p2cdev@localhost:5432/p2c")

    validate_production_security_config()


def test_production_sqlite_rejected(monkeypatch):
    """Scenario 3: sqlite:/// URL in production raises ValueError."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./tensortonic_dev.db")

    with pytest.raises(ValueError) as exc_info:
        validate_production_security_config()

    err = str(exc_info.value)
    assert "SECURITY FAILURE" in err
    assert "Production database must be PostgreSQL" in err
    assert "SQLite is forbidden in production" in err


def test_production_missing_database_url_rejected(monkeypatch):
    """Scenario 4: Missing or empty DATABASE_URL in production raises ValueError."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValueError) as exc_info:
        validate_production_security_config()

    err = str(exc_info.value)
    assert "SECURITY FAILURE" in err
    assert "DATABASE_URL environment variable is missing" in err


def test_production_empty_string_database_url_rejected(monkeypatch):
    """Scenario 4b: Empty/whitespace-only DATABASE_URL in production raises ValueError."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DATABASE_URL", "   ")

    with pytest.raises(ValueError) as exc_info:
        validate_production_security_config()

    err = str(exc_info.value)
    assert "SECURITY FAILURE" in err
    assert "DATABASE_URL environment variable is missing" in err


def test_development_sqlite_allowed(monkeypatch):
    """Scenario 5: SQLite is fully permitted when ENVIRONMENT=development."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./tensortonic_dev.db")

    # Should not raise in development
    validate_production_security_config()
