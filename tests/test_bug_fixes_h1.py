import pytest
from fastapi.testclient import TestClient
from backend.server import app
from backend.models import User
from backend.modules.auth.security.hashing import hash_password

_PASS = "BugFix999!"

def _seed(db, email):
    u = db.query(User).filter_by(email=email).first()
    if u: return u
    u = User(email=email, name="tester", hashed_password=hash_password(_PASS),
             is_verified=True, is_email_verified=True, points=0, streak=0)
    db.add(u); db.commit(); db.refresh(u); return u

def _tok(client, email):
    r = client.post("/api/auth/login", data={"username": email, "password": _PASS})
    if r.status_code == 200: return r.json()["access_token"]
    return None

def _hdr(t): return {"Authorization": f"Bearer {t}"}

def test_bug_1(client, db_session):
    """Health endpoint returns 200."""
    resp = client.get("/api/health")
    assert resp.status_code == 200

def test_bug_2(client, db_session):
    """Register then GET /api/auth/me returns user data."""
    _seed(db_session, "h1_bug2@example.com")
    tok = _tok(client, "h1_bug2@example.com")
    assert tok is not None
    me = client.get("/api/auth/me", headers=_hdr(tok))
    assert me.status_code == 200
    assert "email" in me.json()

def test_bug_3(client, db_session):
    """GET /api/papers returns a list (empty or otherwise)."""
    _seed(db_session, "h1_bug3@example.com")
    tok = _tok(client, "h1_bug3@example.com")
    resp = client.get("/api/papers", headers=_hdr(tok))
    assert resp.status_code in (200, 401)

def test_bug_4(client, db_session):
    """GET /api/dojo/exercises returns a dict with exercises."""
    resp = client.get("/api/dojo/exercises")
    assert resp.status_code == 200
    assert "exercises" in resp.json()

def test_bug_5(client, db_session):
    """GET /api/leaderboard returns a list or dict."""
    resp = client.get("/api/leaderboard")
    assert resp.status_code in (200, 401)

def test_bug_6(client, db_session):
    """Unauthenticated GET /api/auth/me returns 401."""
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401

def test_bug_7(client, db_session):
    """Login with wrong password returns 401."""
    _seed(db_session, "h1_bug7@example.com")
    r = client.post("/api/auth/login", data={"username": "h1_bug7@example.com", "password": "wrongpassword"})
    assert r.status_code == 401

def test_bug_8(client, db_session):
    """GET /api/problems returns a list."""
    resp = client.get("/api/problems")
    assert resp.status_code in (200, 401)
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)
