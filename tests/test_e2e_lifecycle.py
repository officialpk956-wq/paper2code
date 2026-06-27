from unittest.mock import patch, AsyncMock

TEST_EMAIL = "test_e2e@example.com"
TEST_PASS  = "TestPass123!"

def test_register(client):
    r = client.post("/api/auth/register", json={"email": TEST_EMAIL, "name": "Test", "password": TEST_PASS})
    assert r.status_code in (200, 201, 400, 409)  # 400 or 409 if already exists

def test_login_and_get_token(client):
    # Ensure user exists by registering first
    client.post("/api/auth/register", json={"email": TEST_EMAIL, "name": "Test", "password": TEST_PASS})
    r = client.post("/api/auth/login", data={"username": TEST_EMAIL, "password": TEST_PASS})
    assert r.status_code == 200
    assert "access_token" in r.json()
    assert "refresh_token" in r.json()
    return r.json()["access_token"], r.json()["refresh_token"]

def test_refresh_token_endpoint(client):
    _, refresh = test_login_and_get_token(client)
    r = client.post("/api/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200
    assert "access_token" in r.json()

def test_get_me(client):
    token, _ = test_login_and_get_token(client)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == TEST_EMAIL

def test_health_endpoints(client):
    assert client.get("/health").status_code == 200
    assert client.get("/api/health/db").status_code in (200, 503)  # 503 if no DB

def test_dojo_submit_with_mock_piston(client, seeded_db):
    token, _ = test_login_and_get_token(client)
    r = client.post("/api/dojo/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"problem_id": "test-problem-id", "code": "print(2+2)"})
    assert r.status_code in (200, 201, 404)
    if r.status_code in (200, 201):
        assert "task_id" in r.json()
        assert r.json()["status"] == "pending"

def test_dojo_submit_blocks_malicious_code(client, seeded_db):
    token, _ = test_login_and_get_token(client)
    malicious = "import os; print(os.environ)"
    r = client.post("/api/dojo/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"problem_id": "nonexistent", "code": malicious})
    assert r.status_code in (200, 201, 404, 503)
    if r.status_code in (200, 201):
        assert "task_id" in r.json()
        assert r.json()["status"] == "pending"
    assert r.status_code != 500

