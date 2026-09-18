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

    def fake_completion(model, messages, temperature, fallbacks, **kwargs):
        calls.append({"model": model, "fallbacks": fallbacks})
        # Fail for every retry so the FINAL attempt is the one observed --
        # tying this to the policy instead of a fixed 2 failures, which stopped
        # reaching the fallback attempt once the retry window was widened.
        if len(calls) <= llm_client.RATE_LIMIT_RETRIES:
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
    # Derived from the configured policy: a hardcoded 3 went stale when the
    # rate-limit retry window was widened to outlast a per-minute limit.
    assert len(calls) == llm_client.RATE_LIMIT_RETRIES + 1
    # Every attempt before the last must NOT allow the cross-provider
    # fallback -- that's the whole point of retrying the primary first.
    for call in calls[:-1]:
        assert call["fallbacks"] == []
    # Only the final attempt (after retries exhausted) may use it.
    assert calls[-1]["fallbacks"] == [llm_client.FALLBACK_MODEL]


def test_llm_complete_falls_back_after_exhausting_retries(monkeypatch):
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "_failure_count", 0)

    calls = []

    def fake_completion(model, messages, temperature, fallbacks, **kwargs):
        calls.append(model)
        raise litellm_exc.RateLimitError(
            message="still rate limited", llm_provider="groq", model=model
        )

    with (
        patch("litellm.completion", side_effect=fake_completion),
        patch("time.sleep"),
    ):
        # The message used to say "circuit breaker tripped" for every exhausted
        # rate limit, whether or not the breaker had opened. It now says what
        # actually happened.
        with pytest.raises(RuntimeError, match="rate limited on .* after"):
            llm_client.llm_complete("hi")

    # 1 initial + RATE_LIMIT_RETRIES retries, all on the primary model.
    assert len(calls) == llm_client.RATE_LIMIT_RETRIES + 1


def test_exhausted_rate_limits_do_not_trip_the_circuit_breaker(monkeypatch):
    """A 429 means slow down, not provider down. Counting it toward the
    breaker turned one paper's burst into every following paper failing
    instantly -- observed 2026-09-17 with 3-sample consensus."""
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "_failure_count", 0)
    monkeypatch.setattr(llm_client, "FALLBACK_MODEL", "")

    def always_limited(model, messages, temperature, fallbacks, **kwargs):
        raise litellm_exc.RateLimitError(message="rate limited", llm_provider="groq", model=model)

    with patch("litellm.completion", side_effect=always_limited), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="rate limited on .* after"):
            llm_client.llm_complete("hi")

    assert llm_client._failure_count == 0, "rate limits must not feed the breaker"
    assert llm_client._circuit_open is False


def test_quota_wait_restarts_attempts_instead_of_raising(monkeypatch):
    """On a daily token budget a --strict benchmark should wait for quota,
    not give up in two minutes and fall back to rule-based (worth nothing)."""
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "_failure_count", 0)
    monkeypatch.setattr(llm_client, "FALLBACK_MODEL", "")
    monkeypatch.setattr(llm_client, "QUOTA_WAIT_SECONDS", 600)
    monkeypatch.setattr(llm_client, "QUOTA_WAIT_ROUNDS", 1)

    calls = []
    first_round = llm_client.RATE_LIMIT_RETRIES + 1

    def limited_then_ok(model, messages, temperature, fallbacks, **kwargs):
        calls.append(1)
        if len(calls) <= first_round:  # whole first round rate-limited
            raise litellm_exc.RateLimitError(message="rate limited", llm_provider="groq", model=model)
        return _mock_response("ok")

    sleeps = []
    with patch("litellm.completion", side_effect=limited_then_ok), \
         patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
        assert llm_client.llm_complete("hi") == "ok"

    assert 600 in sleeps, "must wait the configured quota interval"
    assert len(calls) == first_round + 1, "second round succeeds on its first attempt"


def test_quota_wait_off_by_default_still_raises(monkeypatch):
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "FALLBACK_MODEL", "")
    monkeypatch.setattr(llm_client, "QUOTA_WAIT_SECONDS", 0)

    def always(model, messages, temperature, fallbacks, **kwargs):
        raise litellm_exc.RateLimitError(message="rate limited", llm_provider="groq", model=model)

    with patch("litellm.completion", side_effect=always), patch("time.sleep"):
        with pytest.raises(RuntimeError, match="rate limited on"):
            llm_client.llm_complete("hi")


def test_transient_transport_error_is_retried_before_failing(monkeypatch):
    """`getaddrinfo failed` (DNS) surfaced as InternalServerError and went
    straight to the breaker with no retry, costing a paper."""
    import core.llm_client as llm_client

    monkeypatch.setattr(llm_client, "_circuit_open", False)
    monkeypatch.setattr(llm_client, "_failure_count", 0)
    monkeypatch.setattr(llm_client, "FALLBACK_MODEL", "")
    calls = []

    def dns_blip_then_ok(model, messages, temperature, fallbacks, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise litellm_exc.InternalServerError(
                message="GroqException - [Errno 11001] getaddrinfo failed",
                llm_provider="groq", model=model)
        return _mock_response("ok")

    with patch("litellm.completion", side_effect=dns_blip_then_ok), patch("time.sleep"):
        assert llm_client.llm_complete("hi") == "ok"
    assert len(calls) == 2
    assert llm_client._failure_count == 0, "a recovered blip must not feed the breaker"
