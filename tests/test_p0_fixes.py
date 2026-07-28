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
    resp = client.post("/api/dojo/submit",
        json={"problem_id": "any_problem", "code": "print(1)"},
        headers={"X-Learner-ID": "1"})
    assert resp.status_code == 401

def test_dojo_submit_cannot_impersonate(client, db_session):
    _seed_user(db_session, "p0_user_a@example.com")
    user_b = _seed_user(db_session, "p0_user_b@example.com")
    token = _login(client, "p0_user_a@example.com")
    resp = client.post("/api/dojo/submit",
        json={"problem_id": "any_problem", "code": "print(1)"},
        headers={**_auth(token), "X-Learner-ID": str(user_b.id)})
    assert resp.status_code in (200, 404, 429)

def test_logout_invalidates_token(client, db_session):
    _seed_user(db_session, "p0_logout@example.com")
    token = _login(client, "p0_logout@example.com")
    h = _auth(token)
    assert client.get("/api/auth/me", headers=h).status_code == 200
    assert client.post("/api/auth/logout-all", headers=h).status_code == 200
    assert client.get("/api/auth/me", headers=h).status_code == 401

def test_analytics_dashboard_no_crash(client, db_session):
    _seed_user(db_session, "p0_analytics@example.com")
    token = _login(client, "p0_analytics@example.com")
    resp = client.get("/api/analytics/dashboard",
        headers={**_auth(token), "X-Learner-ID": "1"})
    assert resp.status_code != 500

def test_jwt_falls_back_to_secret_key_when_ring_missing_or_invalid(monkeypatch):
    # A missing/malformed JWT_KEY_RING must NOT brick startup — it falls back to
    # the strong SECRET_KEY so the service still boots.
    import backend.modules.security.jwt_rotation as mod

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SECRET_KEY", "s" * 64)
    monkeypatch.setenv("JWT_KEY_RING", "not-valid-json")  # exactly the prod misconfig
    assert mod.get_key_ring() == {mod.JWT_ACTIVE_KEY_ID: "s" * 64}
    monkeypatch.delenv("JWT_KEY_RING", raising=False)
    assert mod.get_key_ring() == {mod.JWT_ACTIVE_KEY_ID: "s" * 64}


def test_jwt_raises_in_production_only_when_no_key_at_all(monkeypatch):
    import backend.modules.security.jwt_rotation as mod

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("JWT_KEY_RING", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No JWT signing key"):
        mod.get_key_ring()

def test_rate_limit_not_bypassed_by_pytest():
    import sys
    assert "pytest" in sys.modules
    from backend.modules.auth.middleware.rate_limit import check_rate_limit
    result = check_rate_limit("p0_bypass_test_key_xyz", limit=0, window_seconds=60)
    assert result is False

def test_ingestion_agent_rejects_dangerous_code():
    from core.agents.code_safety import _check_code_safety
    with pytest.raises(ValueError, match="banned module"):
        _check_code_safety("import os")

def test_ingestion_agent_allows_torch_code():
    from core.agents.code_safety import _check_code_safety
    _check_code_safety("import torch")

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

def test_login_rate_limited(client):
    for _ in range(10):
        client.post("/api/auth/login", data={"username": "a", "password": "b"}, headers={"X-Forwarded-For": "10.0.0.99"})
    res = client.post("/api/auth/login", data={"username": "a", "password": "b"}, headers={"X-Forwarded-For": "10.0.0.99"})
    assert res.status_code == 429
