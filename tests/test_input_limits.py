"""
tests/test_input_limits.py

Verifies that free-text Pydantic fields reject payloads that exceed max_length.
Every test expects HTTP 422 Unprocessable Entity from FastAPI's request validation.
"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from backend.server import app
from backend.dependencies import get_current_user
from backend.models import User


def _dummy_user() -> User:
    u = MagicMock(spec=User)
    u.id = 1
    u.email = "test@example.com"
    u.is_admin = False
    u.is_verified = True
    return u


@pytest.fixture(autouse=True)
def override_auth():
    """Inject a dummy authenticated user for all tests in this module."""
    app.dependency_overrides[get_current_user] = _dummy_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Tutor query > 10 000 chars → 422
# ---------------------------------------------------------------------------
def test_tutor_ask_oversized_query(client):
    oversized_query = "x" * 20_000
    response = client.post(
        "/api/tutor/ask",
        json={
            "query": oversized_query,
            "context_type": "general",
            "context_data": {},
        },
    )
    assert response.status_code == 422, (
        f"Expected 422 for oversized query, got {response.status_code}: {response.text}"
    )


# ---------------------------------------------------------------------------
# 2. Dojo code submission > 65536 chars → 422
# ---------------------------------------------------------------------------
def test_dojo_code_submission_oversized_code(client):
    oversized_code = "print('x')\n" * 9999  # well over 64 KB
    assert len(oversized_code) > 65536, "Test pre-condition: code must exceed 64 KB"

    response = client.post(
        "/api/dojo/code-submissions",
        json={
            "problem_id": "test-problem",
            "code": oversized_code,
        },
    )
    assert response.status_code == 422, (
        f"Expected 422 for oversized code, got {response.status_code}: {response.text}"
    )


# ---------------------------------------------------------------------------
# 3. Register with email > 254 chars → 422
# ---------------------------------------------------------------------------
def test_register_oversized_email(client):
    # 255 chars: local@<240-char domain>.com
    oversized_email = "a" * 240 + "@" + "b" * 10 + ".com"
    assert len(oversized_email) > 254

    response = client.post(
        "/api/auth/register",
        json={
            "email": oversized_email,
            "name": "Test User",
            "password": "SecurePassword123!",
        },
    )
    # EmailStr validation rejects malformed long emails as 422
    assert response.status_code == 422, (
        f"Expected 422 for oversized email, got {response.status_code}: {response.text}"
    )


# ---------------------------------------------------------------------------
# 4. Register with password > 128 chars → 422
# ---------------------------------------------------------------------------
def test_register_oversized_password(client):
    oversized_password = "A1!" + "x" * 130  # 133 chars, well over 128
    assert len(oversized_password) > 128

    response = client.post(
        "/api/auth/register",
        json={
            "email": "user@example.com",
            "name": "Test User",
            "password": oversized_password,
        },
    )
    assert response.status_code == 422, (
        f"Expected 422 for oversized password, got {response.status_code}: {response.text}"
    )
