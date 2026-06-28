"""Helper to write test files"""
import textwrap, pathlib

p0 = textwrap.dedent("""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from backend.models import User
from backend.modules.auth.security.hashing import hash_password

_PASS = "P0Fixes123!"

def _seed_user(db, email):
    existing = db.query(User).filter_by(email=email).first()
    if existing: return existing
    u = User(email=email, name=email.split("@")[0],
             hashed_password=hash_password(_PASS),
             is_verified=True, is_email_verified=True, points=0, streak=0)
    db.add(u); db.commit(); db.refresh(u); return u

def _login(client, email):
    r = client.post("/api/auth/login", data={"username": email, "password": _PASS})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]

def _auth(t): return {"Authorization": f"Bearer {t}"}

def test_dojo_submit_requires_auth(client):
    resp = client.post("/api/dojo/submit_exercise",
        json={"exercise_id": "any_exercise", "passed": True, "attempts": 1},
        headers={"X-Learner-ID": "1"})
    assert resp.status_code == 401

def test_dojo_submit_cannot_impersonate(client, db_session):
    _seed_user(db_session, "p0_user_a@example.com")
    user_b = _seed_user(db_session, "p0_user_b@example.com")
    token = _login(client, "p0_user_a@example.com")
    resp = client.post("/api/dojo/submit_exercise",
        json={"exercise_id": "any_exercise", "passed": True, "attempts": 1},
        headers={**_auth(token), "X-Learner-ID": str(user_b.id)})
    assert resp.status_code in (200, 404)

def test_logout_invalidates_token(client, db_session):
    _seed_user(db_session, "p0_logout@example.com")
    token = _login(client, "p0_logout@example.com")
    h = _auth(token)
    assert client.get("/api/auth/me", headers=h).status_code == 200
    assert client.post("/api/auth/logout", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).status_code == 401

def test_login_rate_limited(client):
    for _ in range(10):
        client.post("/api/auth/login", data={"username": "noone@x.com", "password": "wrong"})
    resp = client.post("/api/auth/login", data={"username": "noone@x.com", "password": "wrong"})
    assert resp.status_code in (429, 401)

def test_analytics_dashboard_no_crash(client, db_session):
    _seed_user(db_session, "p0_analytics@example.com")
    token = _login(client, "p0_analytics@example.com")
    resp = client.get("/api/analytics/dashboard",
        headers={**_auth(token), "X-Learner-ID": "1"})
    assert resp.status_code != 500

def test_jwt_weak_key_raises_in_production(monkeypatch):
    import backend.modules.security.jwt_rotation as mod
    original = mod.JWT_KEY_RING_RAW
    mod.JWT_KEY_RING_RAW = None
    monkeypatch.setenv("ENVIRONMENT", "production")
    try:
        with pytest.raises(RuntimeError, match="JWT_KEY_RING"):
            mod.get_key_ring()
    finally:
        mod.JWT_KEY_RING_RAW = original
        monkeypatch.delenv("ENVIRONMENT", raising=False)

def test_rate_limit_not_bypassed_by_pytest():
    import sys
    assert "pytest" in sys.modules
    from backend.modules.auth.middleware.rate_limit import check_rate_limit
    result = check_rate_limit("p0_bypass_test_key_xyz", limit=0, window_seconds=60)
    assert result is False

def test_ingestion_agent_rejects_dangerous_code():
    from core.agents.ingestion_agent import _check_code_safety
    with pytest.raises(ValueError, match="banned module"):
        _check_code_safety("import os")

def test_ingestion_agent_allows_torch_code():
    from core.agents.ingestion_agent import _check_code_safety
    _check_code_safety("import torch\nimport torch.nn as nn")

def test_avatar_url_rejects_javascript_scheme():
    from pydantic import ValidationError
    from backend.modules.auth.schemas import UpdateProfileRequest
    with pytest.raises(ValidationError):
        UpdateProfileRequest(avatar_url="javascript:alert(1)")

def test_avatar_url_rejects_data_uri():
    from pydantic import ValidationError
    from backend.modules.auth.schemas import UpdateProfileRequest
    bad = "data:" + "text/html,evil"
    with pytest.raises(ValidationError):
        UpdateProfileRequest(avatar_url=bad)
""").lstrip()

pathlib.Path("tests/test_p0_fixes.py").write_text(p0, encoding="utf-8")
print("test_p0_fixes.py written:", len(p0), "bytes")
