import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.models import User, Problem, DojoSubmission

from backend.services.auth_service import get_password_hash
import uuid

@pytest.fixture
def dojo_review_setup(db_session):
    u1_email = f"reviewer_{uuid.uuid4()}@example.com"
    u = User(
        email=u1_email,
        name="dojo reviewer",
        hashed_password=get_password_hash("pwd"),
    )
    db_session.add(u)
    db_session.commit()

    u2_email = f"other_{uuid.uuid4()}@example.com"
    u2 = User(
        email=u2_email,
        name="other dojo",
        hashed_password=get_password_hash("pwd"),
    )
    db_session.add(u2)
    db_session.commit()

    prob_id = f"prob_{uuid.uuid4()}"
    p = Problem(
        id=prob_id,
        slug=prob_id,
        title="Test Review",
        difficulty="Easy",
        category="Test",
        is_retired=False,
    )
    db_session.add(p)
    db_session.commit()

    sub1 = DojoSubmission(
        user_id=u.id,
        problem_id=p.id,
        code="def solve(): pass",
        passed=False,
        stdout="",
        stderr="",
        time_ms=100,
    )
    db_session.add(sub1)
    db_session.commit()

    return u, u2, p, sub1

def test_trigger_review_auth(client: TestClient, dojo_review_setup):
    u, u2, p, sub1 = dojo_review_setup

    # unauthorized
    resp = client.post(f"/api/submissions/{sub1.id}/review")
    assert resp.status_code == 401

    # login as u2 (not owner)
    resp = client.post("/api/auth/login", data={"username": u2.email, "password": "pwd"})
    token1 = resp.json()["access_token"]
    
    resp = client.post("/api/auth/login", data={"username": u.email, "password": "pwd"})
    token2 = resp.json()["access_token"]

    resp = client.post(f"/api/submissions/{sub1.id}/review", headers={"Authorization": f"Bearer {token1}"})
    assert resp.status_code == 403

    # login as u (owner)
    with patch("backend.tasks.review_tasks.review_submission_task.delay") as mock_delay:
        resp = client.post(f"/api/submissions/{sub1.id}/review", headers={"Authorization": f"Bearer {token2}"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"
        mock_delay.assert_called_once_with(sub1.id)

def test_get_review(client: TestClient, db_session, dojo_review_setup):
    u, u2, p, sub1 = dojo_review_setup
    resp = client.post("/api/auth/login", data={"username": u.email, "password": "pwd"})
    token1 = resp.json()["access_token"]

    # no review yet
    resp = client.get(f"/api/submissions/{sub1.id}/review", headers={"Authorization": f"Bearer {token1}"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    assert resp.json()["review_text"] is None

    # add review
    sub1.review_text = "Good job!"
    db_session.commit()

    resp = client.get(f"/api/submissions/{sub1.id}/review", headers={"Authorization": f"Bearer {token1}"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
    assert resp.json()["review_text"] == "Good job!"

@patch("backend.tasks.review_tasks.SessionLocal")
@patch("core.agents.code_review_agent.llm_complete")
def test_agent_generates_review(mock_llm, mock_session, dojo_review_setup, db_session):
    u, u2, p, sub1 = dojo_review_setup
    mock_llm.return_value = "This is a dummy AI review."
    mock_session.return_value = db_session

    sub1_id = sub1.id
    from backend.tasks.review_tasks import review_submission_task
    review_submission_task(sub1_id)

    sub1_refetched = db_session.query(DojoSubmission).filter_by(id=sub1_id).first()
    assert sub1_refetched.review_text == "This is a dummy AI review."
    mock_llm.assert_called_once()
