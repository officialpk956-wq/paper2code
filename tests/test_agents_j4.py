import pytest
from unittest.mock import patch
from backend.models import Paper, User, LearnerProgress, AssessmentAttempt, DojoSubmission, Problem
from backend.services.auth_service import get_password_hash
import uuid
import random

def get_token(client, db_session, user_email="test@test.com"):
    # Create user if not exists
    user = db_session.query(User).filter_by(email=user_email).first()
    if not user:
        user = User(
            name="Test User",
            email=user_email,
            hashed_password=get_password_hash("pwd"),
        )
        db_session.add(user)
        db_session.commit()
    resp = client.post("/api/auth/login", data={"username": user_email, "password": "pwd"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, user

@patch("core.agents.learning_path_agent.llm_complete")
def test_learning_path_success(mock_llm, client, db_session):
    headers, user = get_token(client, db_session)
    
    prog = LearnerProgress(learner_id=str(user.id), entity_type="architecture", entity_id="resnet", status="completed")
    db_session.add(prog)
    
    att = AssessmentAttempt(learner_id=str(user.id), assessment_type="quiz", is_correct=False, architecture="vit")
    db_session.add(att)
    
    db_session.commit()
    
    mock_llm.return_value = '```json\n{"steps": [{"step": 1, "type": "concept", "title": "test", "reason": "test"}], "reasoning": "strategy"}\n```'
    
    resp = client.get("/api/me/learning-path", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "steps" in data
    assert len(data["steps"]) == 1
    assert data["reasoning"] == "strategy"

@patch("core.agents.learning_path_agent.llm_complete")
def test_learning_path_unauthorized(mock_llm, client):
    resp = client.get("/api/me/learning-path")
    assert resp.status_code == 401

@patch("core.agents.learning_path_agent.llm_complete")
def test_learning_path_invalid_json(mock_llm, client, db_session):
    headers, user = get_token(client, db_session, f"test2_{uuid.uuid4()}@test.com")
    
    mock_llm.return_value = 'This is not json'
    resp = client.get("/api/me/learning-path", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["steps"] == []
    assert "Could not generate" in data["reasoning"]

@patch("core.agents.research_rag_agent.llm_complete")
def test_ask_paper_success(mock_llm, client, db_session):
    headers, user = get_token(client, db_session, f"test3_{uuid.uuid4()}@test.com")
    
    paper_id_1 = random.randint(10000, 99999)
    paper_id_2 = random.randint(10000, 99999)
    
    p1 = Paper(id=paper_id_1, title="Attention is All You Need", abstract="Transformer", visibility="public", uploaded_by=user.id)
    p2 = Paper(id=paper_id_2, title="BERT Pre-training", abstract="Bidirectional", visibility="public", uploaded_by=user.id)
    db_session.add_all([p1, p2])
    db_session.commit()

    mock_llm.return_value = "This is a great paper. References included."
    
    resp = client.post(
        f"/api/papers/{p1.id}/ask", 
        json={"question": "What is attention?"},
        headers=headers
    )
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "This is a great paper. References included."
    assert isinstance(data["referenced_papers"], list)

def test_ask_paper_not_found(client, db_session):
    headers, _ = get_token(client, db_session, f"test4_{uuid.uuid4()}@test.com")
    resp = client.post("/api/papers/999999/ask", json={"question": "hello?"}, headers=headers)
    assert resp.status_code == 404

def test_ask_paper_private_unauthorized(client, db_session):
    headers, user = get_token(client, db_session, f"test5_{uuid.uuid4()}@test.com")
    
    other_user = User(name="Other", email=f"other_{uuid.uuid4()}@test.com", hashed_password=get_password_hash("pwd"))
    db_session.add(other_user)
    db_session.commit()
    
    paper_id = random.randint(10000, 99999)
    p = Paper(id=paper_id, title="Private", visibility="private", uploaded_by=other_user.id)
    db_session.add(p)
    db_session.commit()
    
    # User 5 trying to ask about other_user's private paper
    resp = client.post(f"/api/papers/{p.id}/ask", json={"question": "hello?"}, headers=headers)
    assert resp.status_code == 403
