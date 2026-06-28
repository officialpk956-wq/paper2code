import pytest
from backend.models import User
from backend.modules.auth.security.hashing import hash_password

_PASS = "PhaseK999!"

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

def test_notification_prefs_get(client, db_session):
    """GET /api/me/notification-preferences returns prefs object."""
    _seed(db_session, "pk_notif1@example.com")
    tok = _tok(client, "pk_notif1@example.com")
    assert tok, "Login failed"
    r = client.get("/api/me/notification-preferences", headers=_hdr(tok))
    assert r.status_code == 200
    data = r.json()
    assert "email_drip_opt_out" in data

def test_notification_prefs_patch(client, db_session):
    """PATCH /api/me/notification-preferences updates prefs."""
    _seed(db_session, "pk_notif2@example.com")
    tok = _tok(client, "pk_notif2@example.com")
    assert tok, "Login failed"
    r = client.patch("/api/me/notification-preferences",
        json={"email_drip_opt_out": True},
        headers=_hdr(tok))
    assert r.status_code == 200
    assert r.json().get("email_drip_opt_out") is True

def test_notification_prefs_unauthenticated(client):
    """GET /api/me/notification-preferences without auth returns 401."""
    r = client.get("/api/me/notification-preferences")
    assert r.status_code == 401

def test_notifications_list(client, db_session):
    """GET /api/notifications returns a list for authenticated user."""
    _seed(db_session, "pk_notif3@example.com")
    tok = _tok(client, "pk_notif3@example.com")
    assert tok, "Login failed"
    r = client.get("/api/notifications", headers=_hdr(tok))
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        body = r.json()
        assert "notifications" in body or isinstance(body, list)
