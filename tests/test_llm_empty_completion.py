"""Regression: an empty LLM completion must retry, then raise -- never return ''."""
import types, pytest
from unittest.mock import patch
import core.llm_client as lc

def _resp(content):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))])

def test_empty_completion_retries_then_raises():
    lc._circuit_open = False; lc._failure_count = 0
    with patch("litellm.completion", return_value=_resp("")) as m, patch("time.sleep"):
        with pytest.raises(RuntimeError, match="empty completion"):
            lc.llm_complete("hi")
    # Derived from the configured policy, not a magic number: the retry
    # count is tunable and a hardcoded 3 silently went stale when the
    # rate-limit window was widened.
    expected = lc.RATE_LIMIT_RETRIES + 1
    assert m.call_count == expected, f"expected {expected} attempts, got {m.call_count}"

def test_empty_then_success_recovers():
    lc._circuit_open = False; lc._failure_count = 0
    with patch("litellm.completion", side_effect=[_resp(""), _resp('{"ok": 1}')]), patch("time.sleep"):
        assert lc.llm_complete("hi") == '{"ok": 1}'

def test_normal_completion_unaffected():
    lc._circuit_open = False; lc._failure_count = 0
    with patch("litellm.completion", return_value=_resp("hello")):
        assert lc.llm_complete("hi") == "hello"


# ── Fallback disable ─────────────────────────────────────────────────────────

def test_fallback_list_is_empty_when_no_fallback_model_configured(monkeypatch):
    """Unset LLM_FALLBACK_MODEL must disable fallback, not pass [''] to litellm."""
    monkeypatch.setattr(lc, "FALLBACK_MODEL", "")
    assert lc._fallback_list(True, "groq/model") == []


def test_fallback_list_used_on_final_attempt_when_configured(monkeypatch):
    monkeypatch.setattr(lc, "FALLBACK_MODEL", "gemini/flash")
    assert lc._fallback_list(True, "groq/model") == ["gemini/flash"]
    assert lc._fallback_list(False, "groq/model") == []
    assert lc._fallback_list(True, "gemini/flash") == []


def test_rate_limit_backoff_grows_exponentially():
    """A flat 8s retried twice could not ride out a Groq per-minute window.

    One transient 429 dropped a paper to rule-based extraction, which under
    --strict rejects a whole 10-paper run.
    """
    from core.llm_client import _backoff_seconds

    assert [_backoff_seconds(8, a) for a in range(4)] == [8, 16, 32, 64]
    assert sum(_backoff_seconds(8, a) for a in range(4)) >= 60, (
        "the retry window must outlast a per-minute rate limit")


def test_rate_limit_retry_policy_is_env_tunable(monkeypatch):
    import importlib

    monkeypatch.setenv("LLM_RATE_LIMIT_RETRIES", "6")
    monkeypatch.setenv("LLM_RATE_LIMIT_BACKOFF", "3")
    import core.llm_client as mod

    reloaded = importlib.reload(mod)
    try:
        assert reloaded.RATE_LIMIT_RETRIES == 6
        assert reloaded.RATE_LIMIT_BACKOFF_SECONDS == 3
    finally:
        monkeypatch.delenv("LLM_RATE_LIMIT_RETRIES", raising=False)
        monkeypatch.delenv("LLM_RATE_LIMIT_BACKOFF", raising=False)
        importlib.reload(mod)


def test_fallback_is_off_unless_explicitly_configured(monkeypatch):
    """Unsetting LLM_FALLBACK_MODEL must mean NO fallback, not a default Gemini."""
    import importlib
    import core.llm_client as mod

    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)
    # reload re-executes `from dotenv import load_dotenv`, so patch the source
    # module, or .env silently resurrects the variable for this check.
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    reloaded = importlib.reload(mod)
    try:
        assert reloaded.FALLBACK_MODEL == ""
        assert reloaded._fallback_list(True, reloaded.PRIMARY_MODEL) == []
    finally:
        importlib.reload(mod)
