"""Tests for /api/admin/* endpoints."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.auth.security.hashing import hash_password

ADMIN_EMAIL = "admin_test@example.com"
USER_EMAIL  = "plain_user_test@example.com"
TEST_PASS   = "SecurePass123!"


def _create_user(db: Session, email: str, is_admin: bool = False) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        return existing
    user = User(
        email=email,
        name="Test User",
        hashed_password=hash_password(TEST_PASS),
        is_verified=True,
        is_email_verified=True,
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS})
    assert r.status_code == 200, f"Login failed for {email}: {r.text}"
    return r.json()["access_token"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_admin_stats_requires_auth(client: TestClient):
    r = client.get("/api/admin/stats")
    assert r.status_code == 401


def test_admin_stats_requires_admin_role(client: TestClient, db_session: Session):
    _create_user(db_session, USER_EMAIL, is_admin=False)
    token = _login(client, USER_EMAIL)
    r = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
    assert "Admin" in r.json()["detail"]


def test_admin_stats_succeeds_as_admin(client: TestClient, db_session: Session):
    _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    token = _login(client, ADMIN_EMAIL)
    r = client.get("/api/admin/stats", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert "users" in body
    assert "llm" in body
    assert "tasks" in body
    assert "content" in body
    assert "timestamp" in body
    assert body["users"]["total"] >= 1


def test_admin_costs_returns_breakdown(client: TestClient, db_session: Session):
    _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    token = _login(client, ADMIN_EMAIL)
    r = client.get("/api/admin/costs?days=7", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["by_action"], list)
    assert isinstance(body["top_users_by_cost"], list)
    assert body["period_days"] == 7


def test_admin_users_list_and_pagination(client: TestClient, db_session: Session):
    _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    _create_user(db_session, USER_EMAIL, is_admin=False)
    token = _login(client, ADMIN_EMAIL)

    r = client.get("/api/admin/users?page=1&limit=50", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert "total" in body
    assert isinstance(body["users"], list)
    assert body["total"] >= 2

    # Pagination: limit=1 should return exactly 1 user
    r2 = client.get("/api/admin/users?page=1&limit=1", headers={"Authorization": f"Bearer {token}"})
    assert r2.status_code == 200
    assert len(r2.json()["users"]) == 1


def test_admin_users_search(client: TestClient, db_session: Session):
    _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    _create_user(db_session, USER_EMAIL, is_admin=False)
    token = _login(client, ADMIN_EMAIL)

    r = client.get(f"/api/admin/users?q=plain_user_test", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    users = r.json()["users"]
    assert all("plain_user_test" in u["email"] for u in users)


def test_admin_cannot_modify_own_account(client: TestClient, db_session: Session):
    admin = _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    token = _login(client, ADMIN_EMAIL)
    r = client.patch(
        f"/api/admin/users/{admin.id}",
        json={"is_admin": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 400
    assert "own admin account" in r.json()["detail"]


def test_admin_update_user_requires_admin(client: TestClient, db_session: Session):
    plain = _create_user(db_session, USER_EMAIL, is_admin=False)
    token = _login(client, USER_EMAIL)
    r = client.patch(
        f"/api/admin/users/{plain.id}",
        json={"is_admin": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403


def test_admin_update_user_grant_admin(client: TestClient, db_session: Session):
    admin = _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    plain = _create_user(db_session, USER_EMAIL, is_admin=False)
    token = _login(client, ADMIN_EMAIL)

    r = client.patch(
        f"/api/admin/users/{plain.id}",
        json={"is_admin": True},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    assert r.json()["is_admin"] is True

    # Cleanup: revoke admin again
    client.patch(
        f"/api/admin/users/{plain.id}",
        json={"is_admin": False},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_admin_update_nonexistent_user(client: TestClient, db_session: Session):
    _create_user(db_session, ADMIN_EMAIL, is_admin=True)
    token = _login(client, ADMIN_EMAIL)
    r = client.patch(
        "/api/admin/users/999999",
        json={"is_admin": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404
