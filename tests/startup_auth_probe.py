"""Run in a new interpreter so model imports cannot mask startup ordering."""

import os
import sqlite3
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ.update(
    DATABASE_URL=sys.argv[1],
    ENVIRONMENT="development",
    ALLOWLIST_EMAILS="",
    SECRET_KEY="startup-regression-test-only-32-characters",
    REDIS_URL="redis://127.0.0.1:1/0",
    RESEND_API_KEY="",
    SENTRY_DSN="",
    POSTHOG_API_KEY="",
    SLACK_WEBHOOK_URL="",
)

import sqlalchemy
from sqlalchemy.pool import StaticPool

original_create_engine = sqlalchemy.create_engine

# Compare existing records before application startup can change the schema.
existing_rows = {}
database_file = sys.argv[1].removeprefix("sqlite:///")
if database_file != ":memory:" and Path(database_file).exists():
    with sqlite3.connect(database_file) as database:
        tables_before = {
            row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, columns in (
            ("users", "id, email, name, hashed_password"),
            ("repair_sentinel", "value"),
        ):
            if table in tables_before:
                existing_rows[(table, columns)] = database.execute(
                    f"SELECT {columns} FROM {table}"
                ).fetchall()


def create_engine(url, **kwargs):
    if url == "sqlite:///:memory:":
        kwargs["poolclass"] = StaticPool
    return original_create_engine(url, **kwargs)


with patch.object(sqlalchemy, "create_engine", create_engine):
    from backend.server import app

from backend.database import engine
from backend.modules.auth.services.email_service import EmailService
from fastapi.testclient import TestClient

tables = sqlalchemy.inspect(engine).get_table_names()
print("verification_tokens present:", "verification_tokens" in tables, flush=True)
with engine.begin() as connection:
    for (table, columns), rows in existing_rows.items():
        current = connection.execute(sqlalchemy.text(f"SELECT {columns} FROM {table}")).fetchall()
        assert all(row in current for row in rows), f"Existing {table} records lost or changed"
    print("existing rows preserved:", sum(map(len, existing_rows.values())), flush=True)
    connection.execute(sqlalchemy.text("CREATE TABLE IF NOT EXISTS repair_sentinel (value TEXT)"))
    if not connection.execute(sqlalchemy.text("SELECT value FROM repair_sentinel")).first():
        connection.execute(sqlalchemy.text("INSERT INTO repair_sentinel VALUES ('preserve-me')"))
    assert (
        connection.execute(sqlalchemy.text("SELECT value FROM repair_sentinel")).scalar()
        == "preserve-me"
    )

email = f"startup-{uuid.uuid4().hex}@example.com"
with (
    patch.object(EmailService, "send_verification_email") as send,
    TestClient(app, raise_server_exceptions=False) as client,
):
    response = client.post(
        "/api/auth/register",
        json={"email": email, "name": "Startup", "password": "StartupPass2026!"},
    )
    print("register:", response.status_code, response.text[:120], flush=True)
    assert response.status_code == 201, response.text
    token = send.call_args.args[1]
    response = client.post("/api/auth/verify-email", json={"token": token})
    assert response.status_code == 200, response.text
    response = client.post(
        "/api/auth/login", data={"username": email, "password": "StartupPass2026!"}
    )
    assert response.status_code == 200, response.text
    auth = response.json()
    response = client.get(
        "/api/auth/me", headers={"Authorization": f"Bearer {auth['access_token']}"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["email"] == email
    response = client.post("/api/auth/refresh", json={"refresh_token": auth["refresh_token"]})
    assert response.status_code == 200, response.text
    print("verify=200 login=200 me=200 refresh=200 sentinel=preserved", flush=True)
