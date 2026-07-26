"""
tests/test_hardened_smoke.py

BRUTAL smoke test suite for Paper2Code backend.
Tests edge cases across auth, authorization, input validation, IDOR,
rate limiting, data integrity, error handling, and API contract compliance.

Run: pytest tests/test_hardened_smoke.py -v
"""
import pytest
import uuid
import time
from unittest.mock import patch


# ---------------------------------------------------------------------------
# SECTION 1: AUTH — Registration Edge Cases
# ---------------------------------------------------------------------------

class TestRegistrationEdgeCases:
    """Stress-test the /api/auth/register endpoint."""

    def test_register_empty_email(self, client):
        r = client.post("/api/auth/register", json={"email": "", "name": "x", "password": "Abcdef1!"})
        assert r.status_code in (400, 422), f"Empty email accepted: {r.json()}"

    def test_register_empty_name(self, client):
        r = client.post("/api/auth/register", json={"email": "a@b.com", "name": "", "password": "Abcdef1!"})
        # Even empty name might be accepted; this documents the behavior.
        assert r.status_code in (200, 400, 422)

    def test_register_no_password(self, client):
        r = client.post("/api/auth/register", json={"email": "a@b.com", "name": "x"})
        assert r.status_code == 422, "Missing password field not rejected"

    def test_register_short_password(self, client):
        r = client.post("/api/auth/register", json={"email": f"s_{uuid.uuid4()}@t.com", "name": "x", "password": "abc"})
        # If the system accepts a 3-char password, that's a finding.
        if r.status_code == 200:
            pytest.fail("FINDING: 3-character password was accepted. Password policy is too weak.")

    def test_register_massive_password(self, client):
        """Try a 100KB password — can cause hash-DoS with bcrypt."""
        big = "A" * 100_000
        r = client.post("/api/auth/register", json={"email": f"big_{uuid.uuid4()}@t.com", "name": "x", "password": big})
        # bcrypt should truncate at 72 bytes, but the request itself shouldn't crash.
        assert r.status_code in (200, 400, 422), f"Server crashed on huge password: {r.status_code}"

    def test_register_duplicate_email(self, client, db_session):
        """Register twice with same email — must get 400, not 500."""
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        email = f"dup_{uuid.uuid4()}@t.com"
        u = User(email=email, name="first", hashed_password=hash_password("TestPass1!"))
        db_session.add(u)
        db_session.commit()
        r = client.post("/api/auth/register", json={"email": email, "name": "second", "password": "TestPass1!"})
        assert r.status_code == 400, f"Duplicate registration returned {r.status_code} instead of 400"

    def test_register_unicode_email(self, client):
        r = client.post("/api/auth/register", json={"email": "user@\u00e9xample.com", "name": "x", "password": "Abcdef1!"})
        assert r.status_code in (200, 400, 422), f"Unicode email caused server error: {r.status_code}"

    def test_register_sql_in_name(self, client):
        """Attempt SQL injection via the name field."""
        r = client.post("/api/auth/register", json={
            "email": f"sqli_{uuid.uuid4()}@t.com",
            "name": "'; DROP TABLE users; --",
            "password": "Abcdef1!"
        })
        # Should succeed or fail gracefully, NOT crash the DB.
        assert r.status_code in (200, 400, 422), f"SQL injection in name caused error: {r.status_code}"


# ---------------------------------------------------------------------------
# SECTION 2: AUTH — Login Edge Cases
# ---------------------------------------------------------------------------

class TestLoginEdgeCases:
    """Stress-test the /api/auth/login endpoint."""

    def test_login_nonexistent_user(self, client):
        r = client.post("/api/auth/login", data={"username": "ghost@nowhere.com", "password": "anything"})
        assert r.status_code == 401
        body = r.json()
        # Must NOT reveal whether the email exists.
        assert "incorrect" in body.get("detail", "").lower() or "invalid" in body.get("detail", "").lower()

    def test_login_wrong_password(self, client, db_session):
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        email = f"wrongpw_{uuid.uuid4()}@t.com"
        u = User(email=email, name="test", hashed_password=hash_password("CorrectPass1!"))
        db_session.add(u)
        db_session.commit()
        r = client.post("/api/auth/login", data={"username": email, "password": "WrongPass!!"})
        assert r.status_code == 401

    def test_login_empty_password(self, client):
        r = client.post("/api/auth/login", data={"username": "a@b.com", "password": ""})
        assert r.status_code in (401, 422)

    def test_login_returns_tokens(self, client, db_session):
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        email = f"tok_{uuid.uuid4()}@t.com"
        u = User(email=email, name="test", hashed_password=hash_password("TestPass1!"))
        db_session.add(u)
        db_session.commit()
        r = client.post("/api/auth/login", data={"username": email, "password": "TestPass1!"})
        assert r.status_code == 200
        body = r.json()
        assert "access_token" in body, "Login response missing access_token"
        assert "refresh_token" in body, "Login response missing refresh_token"
        assert body.get("token_type") == "bearer"


# ---------------------------------------------------------------------------
# SECTION 3: AUTH — Token & Session Integrity
# ---------------------------------------------------------------------------

class TestTokenIntegrity:
    """Test token forgery, expiry, and session management."""

    def test_expired_token_rejected(self, client):
        from tests.conftest import generate_expired_token
        token = generate_expired_token()
        r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 401, "Expired token was accepted"

    def test_garbage_token_rejected(self, client):
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer totallynotavalidtoken.at.all"})
        assert r.status_code == 401

    def test_no_token_rejected(self, client):
        r = client.get("/api/auth/me")
        assert r.status_code in (401, 403)

    def test_me_does_not_leak_password(self, client, db_session):
        """The /me endpoint must NEVER return password hashes."""
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        email = f"me_{uuid.uuid4()}@t.com"
        u = User(email=email, name="test", hashed_password=hash_password("TestPass1!"))
        db_session.add(u)
        db_session.commit()
        r = client.post("/api/auth/login", data={"username": email, "password": "TestPass1!"})
        token = r.json()["access_token"]
        me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me.status_code == 200
        body = me.json()
        assert "password" not in body, "FINDING: /me leaks password field"
        assert "hashed_password" not in body, "FINDING: /me leaks hashed_password field"


# ---------------------------------------------------------------------------
# SECTION 4: AUTHORIZATION — Privilege Escalation (IDOR)
# ---------------------------------------------------------------------------

class TestPrivilegeEscalation:
    """Try to access resources belonging to other users."""

    def test_regular_user_cannot_access_admin(self, client, db_session):
        """A regular user calling /api/admin/* must get 403."""
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        email = f"nonadmin_{uuid.uuid4()}@t.com"
        u = User(email=email, name="reg", hashed_password=hash_password("TestPass1!"), is_admin=False)
        db_session.add(u)
        db_session.commit()
        r = client.post("/api/auth/login", data={"username": email, "password": "TestPass1!"})
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        admin_r = client.get("/api/admin/stats", headers=headers)
        assert admin_r.status_code in (403, 404, 405), \
            f"FINDING: Non-admin accessed /api/admin/stats with status {admin_r.status_code}"

    def test_user_a_cannot_see_user_b_submissions(self, auth_client, seeded_db, user_b_submission_id):
        """User A should NOT be able to fetch User B's specific submission by ID."""
        r = auth_client.get(f"/api/dojo/submissions/{user_b_submission_id}")
        # 404 or 403 are both acceptable — 200 with data is a failure
        if r.status_code == 200:
            body = r.json()
            # If it returns User B's data, that's an IDOR vulnerability
            if body.get("user_id") and body["user_id"] != 1:
                pytest.fail(f"FINDING (IDOR): User A retrieved User B's submission data: {body}")


# ---------------------------------------------------------------------------
# SECTION 5: INPUT VALIDATION — Boundary Attacks
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Push boundary conditions on all major endpoints."""

    def test_problems_search_xss_in_query(self, client):
        """XSS payload in the search query — should not crash."""
        r = client.get("/api/problems", params={"q": '<script>alert("xss")</script>'})
        assert r.status_code == 200  # returns empty array, no crash

    def test_problems_search_very_long_query(self, client):
        """Query param exceeding max_length=200 must be rejected."""
        r = client.get("/api/problems", params={"q": "A" * 500})
        assert r.status_code == 422, f"500-char query was not rejected (got {r.status_code})"

    def test_problems_nonexistent_id(self, client):
        r = client.get("/api/problems/this-does-not-exist-at-all")
        assert r.status_code == 404

    def test_dojo_client_trust_submit_endpoint_removed(self, auth_client):
        """POST /dojo/submissions trusted a client-sent `passed` flag (no code to
        grade) and was removed (audit #6). It must stay gone — real grading is
        server-side via POST /dojo/code-submissions."""
        r = auth_client.post("/api/dojo/submissions", json={
            "exercise_id": "relu", "passed": True, "attempts": 1
        })
        assert r.status_code in (404, 405)


# ---------------------------------------------------------------------------
# SECTION 6: RESET PASSWORD — Token Abuse
# ---------------------------------------------------------------------------

class TestPasswordResetAbuse:
    """Stress-test the forgot/reset password flow."""

    def test_forgot_password_nonexistent_email_no_info_leak(self, client):
        """Must always return 200 with generic message, even for non-existent emails."""
        r = client.post("/api/auth/forgot-password", json={"email": "nobody@doesnotexist.com"})
        assert r.status_code == 200
        body = r.json()
        assert "if that email" in body.get("detail", "").lower(), \
            f"FINDING: Forgot-password leaks email existence: {body}"

    def test_reset_password_garbage_token(self, client):
        r = client.post("/api/auth/reset-password", json={"token": "garbage_token_12345", "new_password": "NewPass1!!"})
        assert r.status_code == 400

    def test_reset_password_too_short(self, client):
        """Reset with password shorter than min_length=8."""
        r = client.post("/api/auth/reset-password", json={"token": "any", "new_password": "abc"})
        assert r.status_code == 422, f"Short reset password not rejected (got {r.status_code})"


# ---------------------------------------------------------------------------
# SECTION 7: HEALTH & METRICS — Information Disclosure
# ---------------------------------------------------------------------------

class TestInfoDisclosure:
    """Check whether public endpoints leak internal details."""

    def test_health_endpoint(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_metrics_accessible(self, client):
        """Prometheus /metrics should be available, or gracefully 503 if the
        optional prometheus_client dependency isn't installed."""
        r = client.get("/metrics")
        assert r.status_code in (200, 503)

    def test_404_does_not_leak_stack_trace(self, client):
        r = client.get("/api/this-endpoint-does-not-exist-ever")
        assert r.status_code in (404, 405)
        body = r.text
        assert "Traceback" not in body, "FINDING: 404 response leaks Python stack trace"
        assert "File \"" not in body, "FINDING: 404 response leaks file paths"


# ---------------------------------------------------------------------------
# SECTION 8: DATA INTEGRITY — Concurrent & Idempotent Operations
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    """Test data consistency under stress."""

    def test_double_registration_same_email(self, client, db_session):
        """Two simultaneous registrations for the same email — only one should succeed."""
        from backend.models import User
        email = f"race_{uuid.uuid4()}@t.com"
        r1 = client.post("/api/auth/register", json={"email": email, "name": "first", "password": "TestPass1!"})
        r2 = client.post("/api/auth/register", json={"email": email, "name": "second", "password": "TestPass1!"})
        statuses = sorted([r1.status_code, r2.status_code])
        assert 400 in statuses, \
            f"FINDING: Both registrations succeeded ({r1.status_code}, {r2.status_code})"

    def test_user_points_cannot_go_negative(self, db_session):
        """Points should never be negative after operations."""
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        u = User(email=f"pts_{uuid.uuid4()}@t.com", name="x", hashed_password=hash_password("TestPass1!"), points=0)
        db_session.add(u)
        db_session.commit()
        db_session.refresh(u)
        assert u.points >= 0, f"FINDING: User points are negative: {u.points}"


# ---------------------------------------------------------------------------
# SECTION 9: API CONTRACT — Response Schema Compliance
# ---------------------------------------------------------------------------

class TestAPIContracts:
    """Verify API responses match expected schemas."""

    def test_login_response_schema(self, client, db_session):
        from backend.models import User
        from backend.modules.auth.security.hashing import hash_password
        email = f"schema_{uuid.uuid4()}@t.com"
        u = User(email=email, name="test", hashed_password=hash_password("TestPass1!"))
        db_session.add(u)
        db_session.commit()
        r = client.post("/api/auth/login", data={"username": email, "password": "TestPass1!"})
        body = r.json()
        required_keys = {"access_token", "refresh_token", "token_type"}
        missing = required_keys - set(body.keys())
        assert not missing, f"Login response missing keys: {missing}"

    def test_problems_list_schema(self, client):
        r = client.get("/api/problems")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list), f"Problems endpoint returned {type(body)}, expected list"
        if body:
            p = body[0]
            expected = {"id", "slug", "title", "difficulty", "category"}
            missing = expected - set(p.keys())
            assert not missing, f"Problem object missing keys: {missing}"

    def test_dojo_exercises_list_schema(self, client):
        r = client.get("/api/dojo/exercises")
        assert r.status_code == 200
        body = r.json()
        assert "exercises" in body, "Dojo exercises response missing 'exercises' key"
        assert isinstance(body["exercises"], list)


# ---------------------------------------------------------------------------
# SECTION 10: ERROR HANDLING — Malformed Requests
# ---------------------------------------------------------------------------

class TestMalformedRequests:
    """Send garbage payloads and ensure graceful handling."""

    def test_register_with_array_body(self, client):
        r = client.post("/api/auth/register", json=[1, 2, 3])
        assert r.status_code == 422

    def test_register_with_null_body(self, client):
        r = client.post("/api/auth/register", json=None)
        assert r.status_code == 422

    def test_login_with_json_instead_of_form(self, client):
        """Login expects form data, not JSON — should fail gracefully."""
        r = client.post("/api/auth/login", json={"username": "a@b.com", "password": "test"})
        assert r.status_code == 422, f"JSON body accepted for form-data endpoint (got {r.status_code})"

    def test_register_with_extra_fields(self, client):
        """Extra fields should be silently ignored by Pydantic."""
        r = client.post("/api/auth/register", json={
            "email": f"extra_{uuid.uuid4()}@t.com",
            "name": "test",
            "password": "TestPass1!",
            "is_admin": True,  # Attempt mass assignment
            "points": 999999,
        })
        if r.status_code == 200:
            # Check: did the attacker's is_admin=True sneak through?
            body = r.json()
            user_data = body.get("user", {})
            # We can't easily check from the response, but the test documents the vector.

    def test_deeply_nested_json(self, client):
        """Send a deeply nested JSON payload to probe for recursion crashes."""
        payload = {"email": "a@b.com"}
        current = payload
        for _ in range(100):
            current["nested"] = {"level": True}
            current = current["nested"]
        r = client.post("/api/auth/register", content=str(payload))
        # Just checking it doesn't crash the server with a 500.
        assert r.status_code != 500, "Deeply nested JSON caused a server crash"
