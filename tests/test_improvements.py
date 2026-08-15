"""
Tests for the 8 product improvements:
  1. LiteLLM unified client
  2. Instructor structured outputs (learning path + RAG agents)
  3. Qdrant vector service (graceful degradation)
  4. Streaming tutor SSE endpoint
  5. Task status SSE endpoint (auth required)
  6. Email templates (_base_template + branded HTML)
  7. Ruff / pre-commit config files present
  8. .env.example and .gitignore hygiene
"""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── 1. LiteLLM client ────────────────────────────────────────────────────────

def test_llm_complete_delegates_to_litellm():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "hello world"
    mock_resp.usage = None
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        from core.llm_client import llm_complete
        result = llm_complete("test prompt")
    assert result == "hello world"
    mock_lit.assert_called_once()


def test_llm_complete_includes_fallback():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "ok"
    mock_resp.usage = None
    with patch("litellm.completion", return_value=mock_resp) as mock_lit:
        from core.llm_client import llm_complete, PRIMARY_MODEL, FALLBACK_MODEL
        llm_complete("test")
    call_kwargs = mock_lit.call_args[1]
    assert "fallbacks" in call_kwargs
    assert call_kwargs["fallbacks"] == (
        [FALLBACK_MODEL] if PRIMARY_MODEL != FALLBACK_MODEL else []
    )


def test_llm_complete_async_delegates_to_litellm():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "async ok"
    mock_resp.usage = None

    async def _run():
        with patch("litellm.acompletion", return_value=mock_resp):
            from core.llm_client import llm_complete_async
            return await llm_complete_async("async prompt")

    result = asyncio.run(_run())
    assert result == "async ok"


def test_llm_complete_auth_error_raises_runtime_error():
    from litellm import exceptions as litellm_exc
    with patch("litellm.completion", side_effect=litellm_exc.AuthenticationError("bad key", llm_provider="groq", model="x")):
        from core.llm_client import llm_complete
        with pytest.raises(RuntimeError, match="LLM auth failed"):
            llm_complete("test")


# ── 2. Instructor structured outputs ─────────────────────────────────────────

def test_generate_learning_path_returns_valid_schema():
    from core.agents.learning_path_agent import LearningPath, LearningStep
    mock_path = LearningPath(
        steps=[LearningStep(step=1, type="concept", title="Attention", reason="Core mechanism")],
        reasoning="Start with attention",
    )
    with patch("instructor.Instructor.chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_path
        from core.agents.learning_path_agent import generate_learning_path
        result = generate_learning_path([], [], [], [], [])
    assert isinstance(result, dict)
    assert "steps" in result
    assert "reasoning" in result


def test_generate_learning_path_fallback_on_error():
    import core.agents.learning_path_agent as lpa_mod
    orig_client = lpa_mod._client
    try:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("LLM down")
        lpa_mod._client = mock_client
        result = lpa_mod.generate_learning_path([], [], [], [], [])
        assert result["steps"] == []
        assert "reasoning" in result
    finally:
        lpa_mod._client = orig_client


def test_ask_about_paper_returns_valid_schema():
    from core.agents.research_rag_agent import PaperAnswer
    mock_answer = PaperAnswer(answer="ResNet uses skip connections.", referenced_papers=[2])
    with patch("instructor.Instructor.chat") as mock_chat:
        mock_chat.completions.create.return_value = mock_answer
        from core.agents.research_rag_agent import ask_about_paper
        paper = {"id": 1, "title": "ResNet", "abstract": "Deep residual learning"}
        result = ask_about_paper(paper, [], "What is ResNet?")
    assert isinstance(result, dict)
    assert "answer" in result
    assert "referenced_papers" in result


def test_ask_about_paper_fallback_on_error():
    import core.agents.research_rag_agent as rag_mod
    orig_client = rag_mod._client
    try:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("timeout")
        rag_mod._client = mock_client
        paper = {"id": 1, "title": "Test", "abstract": "Abstract"}
        result = rag_mod.ask_about_paper(paper, [], "What is this?")
        assert "answer" in result
        assert result["referenced_papers"] == []
    finally:
        rag_mod._client = orig_client


# ── 3. Qdrant vector service (graceful degradation) ──────────────────────────

def test_semantic_search_returns_empty_without_qdrant(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    from backend.services import vector_service
    vector_service._qdrant_client = None
    result = vector_service.semantic_search("transformers")
    assert result == []


def test_index_paper_returns_false_without_qdrant(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    from backend.services import vector_service
    vector_service._qdrant_client = None
    assert vector_service.index_paper(42, "ResNet", "Deep residual learning") is False


def test_delete_paper_returns_false_without_qdrant(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    from backend.services import vector_service
    vector_service._qdrant_client = None
    assert vector_service.delete_paper(42) is False


def test_embed_text_returns_none_when_embedder_fails(monkeypatch):
    from backend.services import vector_service
    vector_service._embedder = None
    with patch("sentence_transformers.SentenceTransformer", side_effect=OSError("no model")):
        result = vector_service.embed_text("hello")
    assert result is None


# ── 4. Streaming tutor SSE ────────────────────────────────────────────────────

def test_tutor_stream_requires_auth(client):
    resp = client.post("/api/tutor/stream", json={"query": "what is relu"})
    assert resp.status_code == 401


def test_tutor_stream_returns_sse_content_type(client, db_session):
    from backend.models import User
    from backend.modules.auth.security.hashing import hash_password
    email = "improv_stream@example.com"
    if not db_session.query(User).filter_by(email=email).first():
        db_session.add(User(
            email=email, name="Stream Tester",
            hashed_password=hash_password("StreamPass1!"),
            is_verified=True, is_email_verified=True, points=0, streak=0,
        ))
        db_session.commit()
    login = client.post("/api/auth/login", data={"username": email, "password": "StreamPass1!"})
    assert login.status_code == 200, f"login failed: {login.text}"
    token = login.json()["access_token"]
    resp = client.post(
        "/api/tutor/stream",
        json={"query": "explain attention", "context_type": "general", "context_data": {}},
        headers={"Authorization": f"Bearer {token}"},
    )
    # 200 = streaming started, 429 = rate limited — both are auth-passed responses
    assert resp.status_code in (200, 429)
    if resp.status_code == 200:
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ── 5. Task status SSE ───────────────────────────────────────────────────────

def test_task_stream_requires_auth(client):
    resp = client.get("/api/tasks/fake-task-id/stream")
    assert resp.status_code == 401


# ── 6. Email templates ───────────────────────────────────────────────────────

def test_base_template_contains_brand():
    from backend.services.email_service import _base_template
    html = _base_template("<p>Test content</p>")
    assert "Paper2Code" in html
    assert "Test content" in html
    assert "#7C3AED" in html   # brand purple present


def test_verification_email_contains_token_link():
    with patch("backend.services.email_service.send_email_sync", return_value=True) as mock_send:
        from backend.services.email_service import send_verification_email_sync
        send_verification_email_sync("user@test.com", "tok123")
    html = mock_send.call_args[0][2]
    assert "tok123" in html
    assert "Verify Email" in html
    assert "Paper2Code" in html  # wrapped in base template


def test_welcome_email_contains_cta():
    with patch("backend.services.email_service.send_email_sync", return_value=True) as mock_send:
        from backend.services.email_service import send_welcome_email_sync
        send_welcome_email_sync("user@test.com", name="Alice")
    html = mock_send.call_args[0][2]
    assert "Alice" in html
    assert "Upload" in html
    assert "Paper2Code" in html


def test_achievement_email_renders_achievement_name():
    with patch("backend.services.email_service.send_email_sync", return_value=True) as mock_send:
        from backend.services.email_service import send_achievement_unlocked_email_sync
        send_achievement_unlocked_email_sync(
            "user@test.com", "First Paper!", "Uploaded your first paper", name="Bob"
        )
    html = mock_send.call_args[0][2]
    assert "First Paper!" in html
    assert "Achievement unlocked" in html
    assert "Bob" in html


def test_paper_done_email_contains_paper_link():
    with patch("backend.services.email_service.send_email_sync", return_value=True) as mock_send:
        from backend.services.email_service import send_paper_done_email_sync
        send_paper_done_email_sync("user@test.com", "Attention Is All You Need", 42)
    html = mock_send.call_args[0][2]
    assert "Attention Is All You Need" in html
    assert "/papers/42" in html


def test_password_reset_email_contains_token():
    with patch("backend.services.email_service.send_email_sync", return_value=True) as mock_send:
        from backend.services.email_service import send_password_reset_email_sync
        send_password_reset_email_sync("user@test.com", "resetxyz")
    html = mock_send.call_args[0][2]
    assert "resetxyz" in html
    assert "Reset Password" in html


def test_mock_email_returns_true_without_api_key(monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    import backend.services.email_service as em
    orig_key = em.RESEND_API_KEY
    em.RESEND_API_KEY = ""
    try:
        result = em.send_email_sync("x@test.com", "hi", "<p>body</p>")
        assert result is True
    finally:
        em.RESEND_API_KEY = orig_key


# ── 7. Ruff + pre-commit config files present ────────────────────────────────

def test_ruff_toml_exists():
    assert os.path.exists(".ruff.toml"), ".ruff.toml missing from project root"


def test_pre_commit_config_exists():
    assert os.path.exists(".pre-commit-config.yaml"), ".pre-commit-config.yaml missing"


def test_ruff_toml_has_select_rules():
    with open(".ruff.toml") as f:
        content = f.read()
    assert "select" in content
    assert '"E"' in content or "'E'" in content


# ── 8. .env.example and .gitignore hygiene ──────────────────────────────────

def test_env_example_exists():
    assert os.path.exists(".env.example"), ".env.example missing"


def test_env_example_has_required_keys():
    with open(".env.example") as f:
        content = f.read()
    for key in ("DATABASE_URL", "SECRET_KEY", "GROQ_API_KEY", "RESEND_API_KEY"):
        assert key in content, f".env.example is missing {key}"


def test_gitignore_excludes_dotenv():
    with open(".gitignore") as f:
        content = f.read()
    assert ".env" in content, ".gitignore must exclude .env"
