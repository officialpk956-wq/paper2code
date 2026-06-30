import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.server import app
from backend.models import User, UsageLog
from core.llm_client import BudgetExceededError, check_user_token_budget
from backend.routers.learning import _get_budget_callback
import os


def test_budget_exceeded(db_session):
    user = User(email="budget1@test.com", name="budget1", hashed_password="pw")
    db_session.add(user)
    db_session.commit()
    with patch("os.getenv", side_effect=lambda k, d=None: "100" if k == "LLM_DAILY_TOKEN_BUDGET_PER_USER" else os.getenv(k, d)):
        with patch("sqlalchemy.orm.query.Query.scalar", return_value=150):
            with pytest.raises(BudgetExceededError) as exc:
                check_user_token_budget(_get_budget_callback(db_session), user.id)
            assert "token budget exceeded" in str(exc.value).lower()

def test_budget_ok(db_session):
    user = User(email="budget2@test.com", name="budget2", hashed_password="pw")
    db_session.add(user)
    db_session.commit()
    with patch("os.getenv", side_effect=lambda k, d=None: "100" if k == "LLM_DAILY_TOKEN_BUDGET_PER_USER" else os.getenv(k, d)):
        with patch("sqlalchemy.orm.query.Query.scalar", return_value=50):
            check_user_token_budget(_get_budget_callback(db_session), user.id)

def test_budget_admin_bypass(db_session):
    user = User(email="budget3@test.com", name="budget3", hashed_password="pw", is_admin=True)
    db_session.add(user)
    db_session.commit()
    with patch("os.getenv", side_effect=lambda k, d=None: "100" if k == "LLM_DAILY_TOKEN_BUDGET_PER_USER" else os.getenv(k, d)):
        with patch("sqlalchemy.orm.query.Query.scalar", return_value=150):
            check_user_token_budget(_get_budget_callback(db_session), user.id)

def test_tutor_ask_429(client, regular_user_headers):
    with patch("backend.routers.learning.tutor_session_store.check_and_incr_rate", return_value=(True, 1, 30)):
        with patch("core.agents.agentic_tutor.AgenticTutor.ask", side_effect=BudgetExceededError("Daily LLM token budget exceeded. Try again tomorrow.")):
            resp = client.post(
                "/api/tutor/ask",
                headers=regular_user_headers,
                json={"context_type": "general", "context_data": {}, "query": "hello"}
            )
            assert resp.status_code == 429
            assert "budget exceeded" in resp.json()["detail"].lower()
