import pytest
from backend.models import User, Paper, Task, DojoSubmission, LearnerProgress
from backend.modules.auth.security.hashing import hash_password

@pytest.fixture
def user_token_headers(client, db_session):
    u = db_session.query(User).filter_by(email="test_user@example.com").first()
    if not u:
        u = User(email="test_user@example.com", name="testuser", hashed_password=hash_password("Pass123!"), is_verified=True, is_email_verified=True)
        db_session.add(u)
        db_session.commit()
    r = client.post("/api/auth/login", data={"username": "test_user@example.com", "password": "Pass123!"})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def admin_token_headers(client, db_session):
    u = db_session.query(User).filter_by(email="admin_user@example.com").first()
    if not u:
        u = User(email="admin_user@example.com", name="adminuser", hashed_password=hash_password("Pass123!"), is_verified=True, is_email_verified=True, is_admin=True)
        db_session.add(u)
        db_session.commit()
    r = client.post("/api/auth/login", data={"username": "admin_user@example.com", "password": "Pass123!"})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_sec1_private_paper_graphs(client, db_session, user_token_headers, admin_token_headers):
    # Create private paper directly in db
    u = db_session.query(User).filter_by(email="admin_user@example.com").first()
    p = Paper(title="Private Paper", visibility="private", uploaded_by=u.id)
    db_session.add(p)
    db_session.commit()
    db_session.refresh(p)
    paper_id = p.id

    # Try to access as regular user (should fail)
    assert client.get(f"/api/papers/{paper_id}/executable-graph", headers=user_token_headers).status_code in (403, 404)
    assert client.get(f"/api/papers/{paper_id}/knowledge-graph", headers=user_token_headers).status_code in (403, 404)
    assert client.get(f"/api/papers/{paper_id}/graph-export", headers=user_token_headers).status_code in (403, 404)

def test_sec2_progress_idor(client, db_session, user_token_headers):
    # progress update shouldn't use X-Learner-ID header anymore
    res = client.post(
        "/api/progress/update", 
        json={"entity_type": "paper", "entity_id": "1", "status": "in_progress"},
        headers=user_token_headers
    )
    assert res.status_code == 200

def test_sec3_csrf_state(client):
    # State parameter should be verified
    res = client.post("/api/auth/oauth/google/exchange", json={"code": "abc", "state": "invalid"})
    # Since we bypass if Redis is down, we might get a 400 or a 401 if it attempts to hit google
    assert res.status_code in (400, 401, 503)

def test_sec5_task_privacy(client, db_session, user_token_headers):
    # Create task for another user
    t = Task(type="sync", user_id=999, status="pending")
    db_session.add(t)
    db_session.commit()

    res = client.get(f"/api/tasks/{t.id}", headers=user_token_headers)
    assert res.status_code == 403

def test_sec6_dojo_solution_auth(client, db_session, user_token_headers):
    # Fetch solution without passing
    res = client.get("/api/dojo/exercises/test-ex/solution", headers=user_token_headers)
    assert res.status_code == 403
