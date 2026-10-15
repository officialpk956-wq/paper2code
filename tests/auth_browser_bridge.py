"""Local browser-to-ASGI test bridge. JSON over stdin; no network listener or email."""

import datetime
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ.update(
    DATABASE_URL="sqlite:///:memory:",
    ENVIRONMENT="development",
    ALLOWLIST_EMAILS="",
    SECRET_KEY="browser-repair-test-only-secret-32-characters",
    REDIS_URL="redis://127.0.0.1:1/0",
    RESEND_API_KEY="",
    SENTRY_DSN="",
    POSTHOG_API_KEY="",
    SLACK_WEBHOOK_URL="",
)

import jwt
import pyotp
import sqlalchemy
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

original_create_engine = sqlalchemy.create_engine


def create_engine(url, **kwargs):
    kwargs["poolclass"] = StaticPool
    return original_create_engine(url, **kwargs)


with patch.object(sqlalchemy, "create_engine", create_engine):
    from backend.server import app

from backend.database import engine, get_db
from backend.models import EmailVerificationToken, User
from backend.modules.auth.config import ALGORITHM, SECRET_KEY
from backend.modules.auth.middleware import rate_limit
from backend.modules.auth.models import AuditLog, VerificationToken
from backend.modules.auth.repositories.verification_repository import hash_token
from backend.modules.auth.services.email_service import EmailService
from backend.modules.auth.services.mfa_service import hash_backup_code
from backend.services import email_service as legacy_email

db = sessionmaker(bind=engine)()
app.dependency_overrides[get_db] = lambda: db
rate_limit._redis_client = None
user_id = None
PASSWORD = "BrowserRepair2026!"
BACKUP = "aabbccddee"
verification_sender = EmailService.send_verification_email


def handle(message, client, send):
    global user_id
    action = message["action"]
    if action == "probe_issuer":
        issued = handle({"action": "seed"}, client, send)
        before = handle({"action": "state"}, client, send)
        get_result = client.get(
            "/api/auth/verify-email", params={"token": issued["token"]}, follow_redirects=False
        )
        post_result = client.post("/api/auth/verify-email", json={"token": issued["token"]})
        return {
            "routes": handle({"action": "routes"}, client, send),
            "signup_token_store": "verification_tokens SHA256",
            "GET": get_result.status_code,
            "POST": post_result.status_code,
            "before": before,
            "after": handle({"action": "state"}, client, send),
        }
    if action == "seed":
        rate_limit._in_memory_store.clear()
        email = f"repair-{uuid.uuid4().hex}@example.com"
        response = client.post(
            "/api/auth/register", json={"email": email, "name": "Repair", "password": PASSWORD}
        )
        assert response.status_code == 201, response.text
        user = db.query(User).filter_by(email=email).one()
        user_id = user.id
        token = send.call_args.args[1]
        assert db.query(VerificationToken).filter_by(token_hash=hash_token(token)).one()
        assert db.query(EmailVerificationToken).filter_by(token=token).first() is None
        if message.get("expired"):
            row = db.query(VerificationToken).filter_by(token_hash=hash_token(token)).one()
            row.expires_at = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        if message.get("mfa"):
            user.mfa_enabled = True
            user.totp_secret = pyotp.random_base32()
            user.backup_codes = [hash_backup_code(BACKUP)]
        db.commit()
        return {"email": email, "password": PASSWORD, "token": token, "backup": BACKUP}
    if action == "state":
        db.expire_all()
        user = db.get(User, user_id)
        return {
            "verified": user.is_email_verified,
            "verified_at": str(user.email_verified_at),
            "events": db.query(AuditLog)
            .filter_by(user_id=user_id, action="email_verified")
            .count(),
            "backups": len(user.backup_codes or []),
        }
    if action == "code":
        return {"code": pyotp.TOTP(db.get(User, user_id).totp_secret).now()}
    if action == "expired_mfa":
        return {
            "token": jwt.encode(
                {
                    "sub": str(user_id),
                    "type": "mfa_temp",
                    "exp": datetime.datetime.utcnow() - datetime.timedelta(minutes=1),
                },
                SECRET_KEY,
                algorithm=ALGORITHM,
            )
        }
    if action == "routes":
        return [
            {
                "path": r.path,
                "methods": sorted(r.methods),
                "handler": r.endpoint.__module__ + "." + r.endpoint.__name__,
            }
            for r in app.routes
            if getattr(r, "path", "").endswith("/verify-email")
        ]
    if action == "senders":
        with (
            patch("backend.modules.auth.services.email_service._send") as first,
            patch.object(legacy_email, "send_email_sync") as second,
        ):
            verification_sender("fixture@example.com", "fixture-token")
            legacy_email.send_verification_email_sync("fixture@example.com", "fixture-token")
            return [first.call_args.kwargs["html"], second.call_args.args[2]]
    if action == "request":
        response = client.request(
            message["method"],
            message["path"],
            content=message.get("body"),
            headers=message.get("headers", {}),
            follow_redirects=False,
        )
        return {"status": response.status_code, "body": response.text}
    raise ValueError("Unknown test action")


with patch.object(EmailService, "send_verification_email") as send, TestClient(app) as client:
    print("REPAIR_JSON:" + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            value = handle(json.loads(line), client, send)
            print("REPAIR_JSON:" + json.dumps(value), flush=True)
        except Exception as exc:
            print("REPAIR_JSON:" + json.dumps({"bridge_error": str(exc)}), flush=True)
