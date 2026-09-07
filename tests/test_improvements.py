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


def test_llm_complete_includes_fallback_once_retries_are_exhausted():
    """
    Cross-provider fallback protection still exists, but (as of the Phase 2
    rate-limit fix -- see core/llm_client.py and
    tests/test_llm_client_retry.py) it's no longer passed to litellm on
    every single call. litellm's `fallbacks` param recovers silently
    *inside* one completion() call, which made a transient rate limit on
    the primary invisibly swap providers mid-pipeline (ConfigExtractor's
    multi-call extraction produced 20/8/20 layers for identical input
    because of this). Fallback is now reserved for the final attempt,
    after the primary has had real retries -- this test confirms that
    protection still exists, just later.
    """
    from litellm import exceptions as litellm_exc

    call_kwargs_seen = []

    def fake_completion(**kwargs):
        call_kwargs_seen.append(kwargs)
        if len(call_kwargs_seen) < 3:
            raise litellm_exc.RateLimitError(
                message="rate limited", llm_provider="groq", model=kwargs["model"]
            )
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        resp.usage = None
        return resp

    with (
        patch("litellm.completion", side_effect=fake_completion),
        patch("time.sleep"),
    ):
        from core.llm_client import llm_complete, PRIMARY_MODEL, FALLBACK_MODEL

        llm_complete("test")

    # The two retries on the primary must NOT carry fallbacks (that's the
    # whole point -- give the preferred model a real chance first).
    assert call_kwargs_seen[0]["fallbacks"] == []
    assert call_kwargs_seen[1]["fallbacks"] == []
    # Fallback protection genuinely exists once retries are exhausted.
    assert call_kwargs_seen[2]["fallbacks"] == (
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
    # QDRANT_URL is read once at module import and cached as a module-level
    # constant -- monkeypatch.delenv on os.environ can't un-cache it (and
    # would permanently poison it for the rest of the session if the env
    # var happened to be unset at whatever test imports this module first).
    # Patch the already-imported module attribute directly instead.
    from backend.services import vector_service
    monkeypatch.setattr(vector_service, "QDRANT_URL", "")
    vector_service._qdrant_client = None
    result = vector_service.semantic_search("transformers")
    assert result == []


def test_index_paper_returns_false_without_qdrant(monkeypatch):
    from backend.services import vector_service
    monkeypatch.setattr(vector_service, "QDRANT_URL", "")
    vector_service._qdrant_client = None
    assert vector_service.index_paper(42, "ResNet", "Deep residual learning") is False


def test_delete_paper_returns_false_without_qdrant(monkeypatch):
    from backend.services import vector_service
    monkeypatch.setattr(vector_service, "QDRANT_URL", "")
    vector_service._qdrant_client = None
    assert vector_service.delete_paper(42) is False


def test_embed_text_returns_none_when_embedder_fails(monkeypatch):
    from backend.services import vector_service
    vector_service._embedder = None
    with patch("sentence_transformers.SentenceTransformer", side_effect=OSError("no model")):
        result = vector_service.embed_text("hello")
    assert result is None


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("QDRANT_URL"), reason="requires a real Qdrant instance (QDRANT_URL)"
)
def test_semantic_search_finds_real_semantic_match_against_live_qdrant():
    """
    Regression test: semantic_search() called client.search(), a method
    removed from the installed qdrant-client version (>=1.10 replaced it
    with the Query API's query_points()). This went undetected because
    QDRANT_URL was empty in every prior environment, so the two tests
    above (both testing the "Qdrant unavailable" fallback) were the only
    coverage this function ever had -- the real query call never ran.
    Uses a genuinely different wording for the query vs. the indexed text
    so a pass proves real semantic (not keyword) matching.
    """
    from backend.services import vector_service

    vector_service._qdrant_client = None  # force a fresh client for this URL
    paper_id = 999_001
    try:
        assert vector_service.index_paper(
            paper_id,
            "Deep Residual Learning",
            "A paper about residual networks and skip connections for image classification.",
            "He et al.",
        )
        results = vector_service.semantic_search("residual connections for images", limit=5)
        assert paper_id in results
    finally:
        vector_service.delete_paper(paper_id)


def test_index_chunk_returns_false_without_qdrant(monkeypatch):
    from backend.services import vector_service
    monkeypatch.setattr(vector_service, "QDRANT_URL", "")
    vector_service._qdrant_client = None
    assert vector_service.index_chunk(1, 42, "some chunk text") is False


def test_search_chunks_returns_empty_without_qdrant(monkeypatch):
    from backend.services import vector_service
    monkeypatch.setattr(vector_service, "QDRANT_URL", "")
    vector_service._qdrant_client = None
    assert vector_service.search_chunks("query") == []


def test_hybrid_search_chunks_falls_back_to_bm25_and_family_without_qdrant(monkeypatch):
    """With Qdrant unavailable, dense score is always 0 -- ranking should
    still work from BM25 + KAG-inferred family alone."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    from backend.services import vector_service
    vector_service._qdrant_client = None

    chunks = [
        {"id": 1, "text": "A shortcut path adds the input back to the block output."},
        {"id": 2, "text": "We train with Adam and a batch size of 256."},
    ]
    results = vector_service.hybrid_search_chunks(
        "How do skip connections work?", chunks, limit=2, family=None
    )
    assert results[0]["chunk_id"] == 1


@pytest.mark.live
@pytest.mark.skipif(
    not os.getenv("QDRANT_URL"), reason="requires a real Qdrant instance (QDRANT_URL)"
)
def test_dense_and_hybrid_chunk_retrieval_against_live_qdrant():
    """
    Regression coverage for Phase 3 Half B (3.3/3.4/3.5):
      - index_chunk/search_chunks: a query sharing NO keywords with the
        target chunk's text must still surface it via dense similarity.
      - hybrid_search_chunks: combining dense + BM25 + KAG-inferred family
        must rank the correct chunk first, and the family bonus must be a
        real, isolated contribution (not just riding on dense agreement).
    """
    from backend.services import vector_service as vs

    vs._qdrant_client = None
    paper_id = 999_002
    chunks = [
        {"id": 101, "text": "The network adds the input tensor back to the block output, forming a shortcut path around two convolutional layers.", "section": "method", "page": 3},
        {"id": 102, "text": "We use the Adam optimizer with a learning rate of 1e-4 and a batch size of 256.", "section": "experiments", "page": 5},
    ]
    try:
        for c in chunks:
            assert vs.index_chunk(c["id"], paper_id, c["text"], c["section"], c["page"])

        dense = vs.search_chunks("residual connection identity mapping", limit=5, paper_id=paper_id)
        assert dense and dense[0]["chunk_id"] == 101

        with_family = vs.hybrid_search_chunks(
            "How do skip connections work in this network?", chunks, limit=2, paper_id=paper_id, family=None
        )
        without_family = vs.hybrid_search_chunks(
            "How do skip connections work in this network?", chunks, limit=2, paper_id=paper_id, family="__unknown__"
        )
        assert with_family[0]["chunk_id"] == 101
        w = {r["chunk_id"]: r["score"] for r in with_family}
        wo = {r["chunk_id"]: r["score"] for r in without_family}
        assert abs((w[101] - wo[101]) - 0.1) < 1e-6
    finally:
        client = vs._get_qdrant()
        if client is not None:
            from qdrant_client.models import PointIdsList
            client.delete(collection_name=vs.CHUNKS_COLLECTION, points_selector=PointIdsList(points=[c["id"] for c in chunks]))


@pytest.mark.live
def test_hybrid_rank_texts_surfaces_target_via_real_embeddings():
    """
    hybrid_rank_texts is the Qdrant-free ranking function ConfigExtractor's
    injected chunk_retriever uses at extraction time (before chunks are ever
    persisted to Qdrant). Uses a query sharing no keywords with the target
    text to prove real dense (not just BM25/family) matching.
    """
    from backend.services import vector_service as vs

    texts = [
        "The network adds the input tensor back to the output of the block before the activation, forming a shortcut path around two convolutional layers.",
        "We use the Adam optimizer with a learning rate of 1e-4 and a batch size of 256 for all experiments.",
        "Images are resized to 224x224 and normalized using ImageNet statistics before being fed to the model.",
        "The dataset contains 1.2 million training images across 1000 categories.",
    ]
    ranked = vs.hybrid_rank_texts("residual connection identity mapping", texts, top_k=1)
    assert ranked == [texts[0]]


def test_related_concepts_expands_skip_connection_to_residual_family():
    from core.rag.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    assert "residualblock" in kg.related_concepts("skip connections")
    assert kg.related_concepts("nonexistent_concept_xyz") == []


def test_infer_family_from_concept_resolves_resnet_from_skip_connections():
    from core.rag.knowledge_graph import KnowledgeGraph
    kg = KnowledgeGraph()
    assert kg.infer_family_from_concept("The skip connections help gradient flow.") == "resnet"
    assert kg.infer_family_from_concept("nothing architectural here") is None


def test_query_expansion_uses_surface_forms_and_is_bounded():
    from core.rag.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    expanded = kg.expand_query_terms("how do skip connections work")
    assert "residual connection" in expanded
    assert "skip connection" not in expanded
    assert "residualblock" not in expanded
    assert kg.expand_query_terms("the dataset has 1.2M images") == []

    many_concepts = (
        "skip connections self attention cross attention causal attention "
        "patch embedding layer normalization batch normalization"
    )
    assert len(kg.expand_query_terms(many_concepts, max_terms=6)) == 6


def test_kag_surface_forms_create_a_measurable_bm25_delta(monkeypatch):
    """Expansion must improve lexical BM25, not merely ride dense ranking."""
    from backend.services import vector_service as vs

    texts = [
        "A residual connection adds the input tensor to the block output. residual connection.",
        "Experiment metadata records the optimizer and learning-rate schedule.",
        "The visualization stores graph edges for a later diagram.",
    ]
    with_expansion, _ = vs._bm25_and_family_scores(
        "skip connections", texts, family="__unknown__"
    )
    monkeypatch.setattr(vs, "_expand_query_terms", lambda query: [])
    without_expansion, _ = vs._bm25_and_family_scores(
        "skip connections", texts, family="__unknown__"
    )

    assert without_expansion[0] == 0
    assert with_expansion[0] > 0

    monkeypatch.setattr(vs, "_expand_query_terms", lambda query: ["residual connection"])
    monkeypatch.setattr(vs, "search_chunks", lambda *args, **kwargs: [])
    ranked = vs.hybrid_search_chunks(
        "skip connections",
        [{"id": index + 1, "text": text} for index, text in enumerate(texts)],
        limit=3,
        family="__unknown__",
    )
    assert ranked[0]["chunk_id"] == 1


def test_mmr_select_falls_back_to_plain_ranking_without_vectors():
    from backend.services.vector_service import _mmr_select

    assert _mmr_select([0, 1, 2], [0.2, 0.9, 0.5], None, 2) == [1, 2]
    assert _mmr_select([0, 1], [0.2, 0.9], None, 3) == [1, 0]
    assert _mmr_select([], [], None, 3) == []


def test_hybrid_rank_mmr_diversifies_but_preserves_reading_order(monkeypatch):
    import numpy as np
    from backend.services import vector_service as vs

    duplicates = [f"Residual block {index} repeats the residual block design." for index in range(5)]
    texts = duplicates + [
        "The optimizer uses Adam with a learning rate of 1e-4.",
        "The dataset contains 1.2 million training images.",
    ]

    class FakeEmbedder:
        def encode(self, value, normalize_embeddings=True):
            if isinstance(value, str):
                return np.array([0.0, 0.0])
            return np.array(
                [[1.0, 0.0]] * 5 + [[0.0, 1.0], [0.0, -1.0]], dtype=float
            )

    monkeypatch.setattr(vs, "_get_embedder", lambda: FakeEmbedder())
    without_mmr = vs.hybrid_rank_texts("residual block", texts, top_k=3, diversity=1.0)
    diverse = vs.hybrid_rank_texts("residual block", texts, top_k=3, diversity=0.7)

    assert without_mmr == duplicates[:3]
    assert len([text for text in without_mmr if text in duplicates]) == 3
    assert len([text for text in diverse if text in duplicates]) <= 2
    assert diverse == [texts[index] for index in sorted(texts.index(text) for text in diverse)]


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
