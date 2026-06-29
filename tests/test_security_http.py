import pytest
from fastapi.testclient import TestClient
from backend.server import app
from tests.conftest import generate_expired_token

client = TestClient(app)

# ── Unauthenticated access should return 401 ──────────────────────────────

PROTECTED_ENDPOINTS = [
    ("POST", "/api/papers/upload"),
    ("POST", "/api/dojo/submit"),
    ("GET",  "/api/tutor/sessions"),
    ("POST", "/api/tutor/feedback"),
    ("GET",  "/api/me/dojo-stats"),
    ("POST", "/api/progress/module/1"),
]

@pytest.mark.parametrize("method,path", PROTECTED_ENDPOINTS)
def test_endpoint_requires_auth(method, path):
    """Every protected endpoint must return 401 without a JWT."""
    response = client.request(method, path)
    assert response.status_code == 401, (
        f"{method} {path} returned {response.status_code} — "
        f"endpoint may be missing auth dependency"
    )

# ── Wrong role should return 403 ──────────────────────────────────────────

ADMIN_ENDPOINTS = [
    ("GET",  "/api/admin/users"),
    ("POST", "/api/admin/problems"),
    ("GET",  "/api/admin/stats"),
]

@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_endpoint_requires_admin_role(method, path, regular_user_headers):
    """Admin endpoints must return 403 for regular (non-admin) users."""
    response = client.request(method, path, headers=regular_user_headers)
    assert response.status_code in (403, 404, 405), (
        f"{method} {path} returned {response.status_code} for non-admin user"
    )

# ── Expired token should return 401 ──────────────────────────────────────

def test_expired_token_returns_401():
    """An expired JWT must be rejected, not granted access."""
    expired_token = generate_expired_token()  # create a JWT with past expiry
    response = client.post("/api/papers/upload", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == 401

# ── IDOR: user can't access other user's resources ───────────────────────

def test_user_cannot_access_other_users_submission(client, user_a_headers, user_b_submission_id):
    """User A must not be able to read User B's submission."""
    response = client.get(
        f"/api/dojo/submissions/{user_b_submission_id}",
        headers=user_a_headers
    )
    assert response.status_code in (403, 404)

# ── SQL injection probe ───────────────────────────────────────────────────

def test_search_handles_sql_injection(client, regular_user_headers):
    """Search endpoint must not expose SQL errors on injection attempts."""
    payloads = ["' OR '1'='1", "'; DROP TABLE users;--", "1 UNION SELECT * FROM users"]
    for payload in payloads:
        response = client.get(f"/api/papers?search={payload}", headers=regular_user_headers)
        assert response.status_code in (200, 400, 422), (
            f"SQL injection payload returned unexpected {response.status_code}"
        )
        assert "syntax error" not in response.text.lower()
        assert "postgresql" not in response.text.lower()
