import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.server import app
from core.llm_client import BudgetExceededError
from backend.dependencies import get_current_user
from backend.models import User


def _make_dummy_user():
    u = MagicMock(spec=User)
    u.id = 1
    u.email = "test@example.com"
    u.is_admin = False
    return u


@patch('backend.routers.tutor.logger')
def test_tutor_ask_budget_exception_logging(mock_logger):
    """
    When AgenticTutor.ask raises BudgetExceededError,
    the handler must call logger.exception (not logger.error).
    """
    dummy_user = _make_dummy_user()
    app.dependency_overrides[get_current_user] = lambda: dummy_user

    # Patch AgenticTutor.ask to raise BudgetExceededError
    with patch('core.agents.agentic_tutor.AgenticTutor.ask',
               side_effect=BudgetExceededError("Daily budget exceeded")):
        client = TestClient(app)
        response = client.post(
            "/api/tutor/ask",
            json={
                "query": "what is attention?",
                "session_id": None,
                "context_type": "general",
                "context_data": {},
            },
        )

    app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 429, response.text

    # Must have used logger.exception, NOT logger.error
    mock_logger.exception.assert_called()
    # Ensure logger.error was NOT called for this path
    for call_args in mock_logger.error.call_args_list:
        assert "Tutor ask error" not in str(call_args), (
            "logger.error was called where logger.exception was expected"
        )
