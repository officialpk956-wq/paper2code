"""
Regression tests for core.llm_client's rate-limit retry behavior.

Found during Phase 2 live verification: a transient 429 on the primary
model fell straight through to litellm's cross-provider `fallbacks`,
silently swapping models mid-pipeline. For ConfigExtractor's multi-call
extraction pipeline this produced visibly inconsistent results between
otherwise-identical calls (20/8/20 layers for the same input text) --
some calls landed on Groq, some silently landed on Gemini instead.

llm_complete/llm_complete_async now retry the primary model a couple of
times with backoff before allowing the cross-provider fallback, so a
transient rate limit gets a real chance to resolve on the preferred model
first.
"""

from unittest.mock import MagicMock, patch

import pytest
from litellm import exceptions as litellm_exc


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=text))]
    return resp


def test_llm_complete_retries_primary_before_falling_back(monkeypatch):
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "_failure_count", 0)

    calls = []

    def fake_completion(model, messages, temperature, fallbacks):
        calls.append({"model": model, "fallbacks": fallbacks})
        if len(calls) < 3:
            raise litellm_exc.RateLimitError(
                message="rate limited", llm_provider="groq", model=model
            )
        return _mock_response("ok")

    with (
        patch("litellm.completion", side_effect=fake_completion),
        patch("time.sleep"),  # don't actually wait 8s per retry in tests
    ):
        result = llm_client.llm_complete("hi")

    assert result == "ok"
    assert len(calls) == 3
    # First two attempts must NOT allow the cross-provider fallback --
    # that's the whole point of retrying the primary first.
    assert calls[0]["fallbacks"] == []
    assert calls[1]["fallbacks"] == []
    # Only the final attempt (after retries exhausted) may use it.
    assert calls[2]["fallbacks"] == [llm_client.FALLBACK_MODEL]


def test_llm_complete_falls_back_after_exhausting_retries(monkeypatch):
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "_failure_count", 0)

    calls = []

    def fake_completion(model, messages, temperature, fallbacks):
        calls.append(model)
        raise litellm_exc.RateLimitError(
            message="still rate limited", llm_provider="groq", model=model
        )

    with (
        patch("litellm.completion", side_effect=fake_completion),
        patch("time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="circuit breaker tripped"):
            llm_client.llm_complete("hi")

    # 1 initial + 2 retries = 3 total attempts, all on the primary model.
    assert len(calls) == 3
