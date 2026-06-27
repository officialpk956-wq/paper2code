import pytest
from unittest.mock import MagicMock, patch
from core.agents.agentic_tutor import AgenticTutor
from backend.models import Paper, Problem, AssessmentAttempt, User
from backend.services.auth_service import get_password_hash
from fastapi.testclient import TestClient
from backend.server import app
import uuid
import random

class MockBlock:
    def __init__(self, type, text=None, name=None, input=None, id=None):
        self.type = type
        if text is not None:
            self.text = text
        if name is not None:
            self.name = name
        if input is not None:
            self.input = input
        if id is not None:
            self.id = id

class MockResponse:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason

@pytest.fixture
def tutor_setup(db_session):
    u = User(
        email=f"tutor_{uuid.uuid4()}@example.com",
        name="Tutor Learner",
        hashed_password=get_password_hash("pwd")
    )
    db_session.add(u)
    db_session.commit()

    paper_id = random.randint(10000, 99999)
    p = Paper(
        id=paper_id,
        title=f"Test Concept Paper {uuid.uuid4()}",
        authors="A. B",
        abstract="This paper introduces a test concept.",
        visibility="public",
        uploaded_by=u.id
    )
    db_session.add(p)
    
    prob = Problem(
        id=f"prob_concept_{uuid.uuid4()}",
        slug=f"prob-concept-{uuid.uuid4()}",
        title="Concept Problem",
        category="Test",
        difficulty="Hard",
        description="Learn about the test concept here.",
        is_retired=False
    )
    db_session.add(prob)
    
    # 2 attempts for user on 'resnet', 1 correct, 1 wrong (50%) - not weak
    a1 = AssessmentAttempt(learner_id=str(u.id), assessment_type="test", architecture="resnet", score=100, is_correct=True)
    a2 = AssessmentAttempt(learner_id=str(u.id), assessment_type="test", architecture="resnet", score=0, is_correct=False)
    # 2 attempts for user on 'vit', both wrong (0%) - weak
    a3 = AssessmentAttempt(learner_id=str(u.id), assessment_type="test", architecture="vit", score=0, is_correct=False)
    a4 = AssessmentAttempt(learner_id=str(u.id), assessment_type="test", architecture="vit", score=0, is_correct=False)
    db_session.add_all([a1, a2, a3, a4])
    
    db_session.commit()
    u.test_paper_id = paper_id
    return u

@patch("core.agents.agentic_tutor.Anthropic")
def test_agentic_tutor_no_tool(mock_anthropic, db_session):
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    mock_client.messages.create.return_value = MockResponse(
        content=[MockBlock(type="text", text="Direct answer without tools.")],
        stop_reason="end_turn"
    )
    
    tutor = AgenticTutor(db_session)
    resp, hist = tutor.ask("Hello", [], "Test", {}, 1)
    
    assert "answer" in resp
    assert resp["answer"] == "Direct answer without tools."
    assert len(hist) > 0
    assert hist[-1]["content"] == mock_client.messages.create.return_value.content

@patch("core.agents.agentic_tutor.Anthropic")
def test_agentic_tutor_with_tool(mock_anthropic, db_session, tutor_setup):
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # First call returns tool_use, second returns text
    mock_client.messages.create.side_effect = [
        MockResponse(
            content=[MockBlock(type="tool_use", name="lookup_paper_section", input={"paper_id": tutor_setup.test_paper_id}, id="tool1")],
            stop_reason="tool_use"
        ),
        MockResponse(
            content=[MockBlock(type="text", text="Paper says test concept.")],
            stop_reason="end_turn"
        )
    ]
    
    tutor = AgenticTutor(db_session)
    resp, hist = tutor.ask(f"What is paper {tutor_setup.test_paper_id}?", [], "Test", {}, 1)
    
    assert resp["answer"] == "Paper says test concept."
    assert mock_client.messages.create.call_count == 2

def test_tool_lookup_paper(db_session, tutor_setup):
    tutor = AgenticTutor(db_session)
    # Success
    res = tutor._execute_tool("lookup_paper_section", {"paper_id": tutor_setup.test_paper_id})
    assert "This paper introduces" in res
    # Not found
    res2 = tutor._execute_tool("lookup_paper_section", {"paper_id": 888})
    assert res2 == "Paper not found."

def test_tool_find_related_problem(db_session, tutor_setup):
    tutor = AgenticTutor(db_session)
    # Match
    res = tutor._execute_tool("find_related_problem", {"concept": "test concept"})
    assert "Concept Problem" in res
    assert "Hard" in res
    # No match
    res2 = tutor._execute_tool("find_related_problem", {"concept": "unknown concept"})
    assert res2 == "No related problem found."

def test_tool_get_user_weak_topics(db_session, tutor_setup):
    tutor = AgenticTutor(db_session)
    res = tutor._execute_tool("get_user_weak_topics", {"user_id": tutor_setup.id})
    assert "vit" in res
    assert "resnet" not in res
    
def test_tool_search_papers(db_session, tutor_setup):
    tutor = AgenticTutor(db_session)
    res = tutor._execute_tool("search_papers_by_concept", {"concept": "test concept"})
    assert "Test Concept Paper" in res

@patch("core.agents.agentic_tutor.AgenticTutor.ask")
def test_api_tutor_ask_200(mock_ask, client, tutor_setup, db_session):
    mock_ask.return_value = ({"answer": "Mocked answer", "reasoning_type": "Agentic"}, [])
    
    # Get token
    resp = client.post("/api/auth/login", data={"username": tutor_setup.email, "password": "pwd"})
    token = resp.json()["access_token"]
    
    req_data = {
        "context_type": "paper",
        "context_data": {"id": tutor_setup.test_paper_id},
        "query": "Tell me something"
    }
    
    ask_resp = client.post("/api/tutor/ask", json=req_data, headers={"Authorization": f"Bearer {token}"})
    print(ask_resp.json())
    assert ask_resp.status_code == 200
    assert "answer" in ask_resp.json()
    assert ask_resp.json()["answer"] == "Mocked answer"

def test_api_tutor_ask_401(client):
    req_data = {
        "context_type": "paper",
        "context_data": {"id": 999},
        "query": "Tell me something"
    }
    ask_resp = client.post("/api/tutor/ask", json=req_data)
    assert ask_resp.status_code == 401
