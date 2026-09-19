import logging
import os
from contextvars import ContextVar

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget config
# ---------------------------------------------------------------------------
class BudgetExceededError(Exception):
    pass


def check_user_token_budget(get_usage_callback, user_id: int | None) -> None:
    if user_id is None or get_usage_callback is None:
        return

    used = get_usage_callback(user_id)
    if used < 0:  # Admin bypass returns -1
        return

    import os

    budget = int(os.getenv("LLM_DAILY_TOKEN_BUDGET_PER_USER", "100000"))
    if used >= budget:
        raise BudgetExceededError("Daily LLM token budget exceeded. Try again tomorrow.")


# ---------------------------------------------------------------------------
# Model config — override via env vars
# ---------------------------------------------------------------------------
# The previous default, groq/llama-3.3-70b-versatile, no longer exists on
# Groq ("model_not_found"); only .env was keeping the system working.
PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "groq/openai/gpt-oss-120b")
# No fallback unless explicitly configured. The old default was a Gemini
# model, so *unsetting* the variable did not disable cross-provider fallback;
# it silently selected a different Gemini -- which is how a benchmark run
# meant to be Groq-pure came back provider-mixed and non-comparable.
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "").strip()
# Backward-compat alias (some callers do `if GROQ_API_KEY:`)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Simple circuit breaker state
_circuit_open = False
_circuit_open_until = 0
_failure_count = 0
_last_completion_model: ContextVar[str | None] = ContextVar("last_completion_model", default=None)
FAILURE_THRESHOLD = 5
CIRCUIT_OPEN_DURATION = 60  # seconds
# Architecture specs for deep models (U-Net, DenseNet) run to a few thousand
# tokens of layer list. The provider default truncated them mid-JSON.
# 4096 was not enough: reasoning models spend much of the budget before
# emitting output, and U-Net truncated mid-connections at 3219 chars. At
# 16384 it completes (recall 1.00, was a rule-based fallback).
MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "16384"))

# A flat 8s backoff retried twice (~16s of waiting) could not ride out a Groq
# per-minute token window. One transient 429 then dropped a single paper to
# rule-based extraction, which under --strict rejects an entire 10-paper
# benchmark run and costs a day of quota. Back off exponentially instead.
# temperature=0 alone did not make Groq's gpt-oss-120b deterministic: two
# benchmark runs with byte-identical prompts changed 5 of 6 papers' scores.
# A fixed seed is the provider's best-effort determinism lever. Set
# LLM_SEED="" to disable it.
# gpt-oss-120b is a reasoning model; its reasoning trace diverges between
# identical calls and drags the final answer with it. Optional knob to pin
# the effort level while measuring whether that narrows the variance.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip() or None


def _reasoning_kwargs(reasoning_effort: str | None = None) -> dict:
    effort = reasoning_effort or LLM_REASONING_EFFORT
    return {"reasoning_effort": effort} if effort else {}


_seed_env = os.getenv("LLM_SEED", "42")
LLM_SEED: int | None = int(_seed_env) if _seed_env.strip() else None

# Optional: after the backoff retries are exhausted on a 429, wait this long
# and start the attempts over, up to this many rounds. Off by default. For a
# --strict benchmark a rule-based fallback is worth nothing, so on a daily
# token budget waiting hours for quota beats giving up in two minutes.
QUOTA_WAIT_SECONDS = int(os.getenv("LLM_QUOTA_WAIT_SECONDS", "0"))
QUOTA_WAIT_ROUNDS = int(os.getenv("LLM_QUOTA_WAIT_ROUNDS", "0"))

RATE_LIMIT_RETRIES = int(os.getenv("LLM_RATE_LIMIT_RETRIES", "4"))
RATE_LIMIT_BACKOFF_SECONDS = int(os.getenv("LLM_RATE_LIMIT_BACKOFF", "8"))


def _backoff_seconds(base: int, attempt: int) -> int:
    """Exponential backoff for retry `attempt` (0-based): base, 2x, 4x, 8x..."""
    return base * (2**attempt)


def _fallback_list(use_fallback: bool, target: str) -> list[str]:
    """Cross-provider fallback targets for one attempt.

    Empty when LLM_FALLBACK_MODEL is unset: a rate limit then raises instead
    of silently switching providers. Benchmark runs want that -- a run whose
    papers were served by different models is not one measurement, and
    passing [""] here would have handed litellm a bogus model id rather than
    disabling fallback.
    """
    if not use_fallback or not FALLBACK_MODEL or target == FALLBACK_MODEL:
        return []
    return [FALLBACK_MODEL]


def get_last_completion_model() -> str | None:
    """Return the model that served this context's most recent completion."""
    return _last_completion_model.get()


def llm_complete(
    prompt: str,
    user_id: int | None = None,
    action: str = "llm.unknown",
    check_budget_callback=None,
    db_write_callback=None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Synchronous LLM completion with automatic provider fallback.

    ``reasoning_effort`` pins a reasoning model's effort for this call. It is
    the determinism lever for gpt-oss-120b on Groq: with temperature=0 and a
    fixed seed, "low" returned byte-identical completions for identical
    prompts where "medium" and unset did not (4 vs 11 layers, 3 vs 14).
    """
    import time

    from litellm import completion
    from litellm import exceptions as litellm_exc

    check_user_token_budget(check_budget_callback, user_id)

    global _circuit_open, _circuit_open_until, _failure_count

    now = time.time()
    if _circuit_open and now < _circuit_open_until:
        raise RuntimeError("LLM circuit breaker open — service temporarily unavailable")

    if _circuit_open and now >= _circuit_open_until:
        _circuit_open = False  # half-open: try again
        _failure_count = 0

    messages = [{"role": "user", "content": prompt}]
    target = model or PRIMARY_MODEL

    # A transient rate limit on the primary model used to fall through
    # immediately to litellm's cross-provider `fallbacks`, silently swapping
    # to a different (lower-fidelity, differently-behaved) model mid-pipeline.
    # For a multi-call pipeline (e.g. ConfigExtractor's extract+verify steps)
    # that produced visibly inconsistent results between calls -- some on
    # Groq, some silently rerouted to Gemini. Retry the primary a couple of
    # times with backoff first; only allow the cross-provider fallback on
    # the final attempt, once retrying the preferred model has been given a
    # real chance to succeed.
    max_rate_limit_retries = RATE_LIMIT_RETRIES
    rate_limit_backoff_seconds = RATE_LIMIT_BACKOFF_SECONDS

    resp = None
    quota_rounds_left = QUOTA_WAIT_ROUNDS if QUOTA_WAIT_SECONDS > 0 else 0
    _last_rate_limit: RuntimeError | None = None
    attempt = -1
    while True:
        attempt += 1
        if attempt > max_rate_limit_retries:
            # Backoff exhausted. Wait for quota if configured, else give up.
            if quota_rounds_left <= 0:
                raise _last_rate_limit or RuntimeError(f"rate limited on {target}")
            quota_rounds_left -= 1
            logger.warning(
                "Quota exhausted on %s; waiting %ds for it to return (%d round(s) left)",
                target,
                QUOTA_WAIT_SECONDS,
                quota_rounds_left,
            )
            time.sleep(QUOTA_WAIT_SECONDS)
            attempt = 0
        use_fallback = attempt == max_rate_limit_retries
        try:
            resp = completion(
                model=target,
                messages=messages,
                temperature=0,
                seed=LLM_SEED,
                **_reasoning_kwargs(reasoning_effort),
                # Without an explicit ceiling the provider default applied, and
                # architecture specs for deep models overran it: unet's response
                # was cut mid-token at 1084 chars ('"kernel_size": ') leaving
                # unparseable JSON, which then silently became a rule-based
                # fallback. Deep nets (U-Net, DenseNet) legitimately need a few
                # thousand tokens of layer list.
                max_tokens=MAX_COMPLETION_TOKENS,
                fallbacks=_fallback_list(use_fallback, target),
            )
            text = resp.choices[0].message.content or ""
            # An empty completion is a failure, not a success. Returning ""
            # here used to surface downstream as "LLM did not return valid
            # JSON", which ConfigExtractor's broad except then turned into a
            # silent rule-based fallback -- a wrong spec reported as a result.
            # Observed on densenet121 and unet: focused text was fine (~6.8k
            # chars), the model simply returned nothing. Retry like a rate
            # limit rather than propagating the empty string.
            if not text.strip():
                if not use_fallback:
                    delay = _backoff_seconds(rate_limit_backoff_seconds, attempt)
                    logger.warning(
                        "Empty completion from %s (attempt %d/%d) -- retrying in %ds",
                        target,
                        attempt + 1,
                        max_rate_limit_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"LLM returned an empty completion for {target} after "
                    f"{max_rate_limit_retries + 1} attempts"
                )
            _failure_count = 0  # success resets counter
            _last_completion_model.set(getattr(resp, "model", target))
            break
        except litellm_exc.RateLimitError as e:
            if not use_fallback:
                delay = _backoff_seconds(rate_limit_backoff_seconds, attempt)
                logger.warning(
                    "Rate limited on %s (attempt %d/%d) -- retrying same model in %ds",
                    target,
                    attempt + 1,
                    max_rate_limit_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            # A 429 is "slow down", not "the provider is down". It used to
            # count toward the breaker, so a burst of rate limits on one paper
            # opened the circuit and every following paper then failed
            # instantly without a single request -- backoff turned into a
            # cascade. It also said "circuit breaker tripped" whether or not
            # the breaker had actually opened. Rate limits now exhaust their
            # retries and raise; only genuine API failures feed the breaker.
            _last_rate_limit = RuntimeError(
                f"rate limited on {target} after {max_rate_limit_retries + 1} attempts: {e}"
            )
            _last_rate_limit.__cause__ = e
            continue  # loop head decides: wait for quota, or raise
        except (
            litellm_exc.APIConnectionError,
            litellm_exc.APIError,
            litellm_exc.InternalServerError,
        ) as e:
            # Transient transport failures (DNS: "getaddrinfo failed",
            # connection resets, 5xx) get a short retry before they count
            # as a real failure. One DNS blip took out a paper on 2026-09-17.
            if not use_fallback:
                delay = _backoff_seconds(rate_limit_backoff_seconds, attempt)
                logger.warning(
                    "Transport error on %s (attempt %d/%d) -- retrying in %ds: %s",
                    target,
                    attempt + 1,
                    max_rate_limit_retries,
                    delay,
                    type(e).__name__,
                )
                time.sleep(delay)
                continue
            _failure_count += 1
            if _failure_count >= FAILURE_THRESHOLD:
                _circuit_open = True
                _circuit_open_until = now + CIRCUIT_OPEN_DURATION
            raise RuntimeError(f"LLM circuit breaker tripped for {target}: {e}") from e
        except litellm_exc.AuthenticationError as e:
            raise RuntimeError(f"LLM auth failed for {target}: {e}") from e
        except Exception as e:
            logger.error("llm_complete failed: %s", e)
            raise
    _log_usage(resp, user_id, action, db_write_callback)
    return text


async def llm_complete_async(
    prompt: str,
    model: str | None = None,
    user_id: int | None = None,
    action: str = "llm.unknown",
    check_budget_callback=None,
    db_write_callback=None,
) -> str:
    """Async LLM completion with automatic provider fallback."""
    import time

    from litellm import acompletion
    from litellm import exceptions as litellm_exc

    check_user_token_budget(check_budget_callback, user_id)

    global _circuit_open, _circuit_open_until, _failure_count

    now = time.time()
    if _circuit_open and now < _circuit_open_until:
        raise RuntimeError("LLM circuit breaker open — service temporarily unavailable")

    if _circuit_open and now >= _circuit_open_until:
        _circuit_open = False
        _failure_count = 0

    messages = [{"role": "user", "content": prompt}]
    target = model or PRIMARY_MODEL

    # See llm_complete's matching comment: retry the primary on a rate limit
    # before allowing litellm's cross-provider fallback, so a transient 429
    # doesn't silently swap models mid-pipeline.
    max_rate_limit_retries = RATE_LIMIT_RETRIES
    rate_limit_backoff_seconds = RATE_LIMIT_BACKOFF_SECONDS

    resp = None
    for attempt in range(max_rate_limit_retries + 1):
        use_fallback = attempt == max_rate_limit_retries
        try:
            resp = await acompletion(
                model=target,
                messages=messages,
                temperature=0,
                seed=LLM_SEED,
                **_reasoning_kwargs(),
                fallbacks=_fallback_list(use_fallback, target),
            )
            text = resp.choices[0].message.content or ""
            _failure_count = 0
            break
        except litellm_exc.RateLimitError as e:
            if not use_fallback:
                delay = _backoff_seconds(rate_limit_backoff_seconds, attempt)
                logger.warning(
                    "Rate limited on %s (attempt %d/%d) -- retrying same model in %ds",
                    target,
                    attempt + 1,
                    max_rate_limit_retries,
                    delay,
                )
                import asyncio

                await asyncio.sleep(delay)
                continue
            # A 429 is "slow down", not "the provider is down". It used to
            # count toward the breaker, so a burst of rate limits on one paper
            # opened the circuit and every following paper then failed
            # instantly without a single request -- backoff turned into a
            # cascade. It also said "circuit breaker tripped" whether or not
            # the breaker had actually opened. Rate limits now exhaust their
            # retries and raise; only genuine API failures feed the breaker.
            raise RuntimeError(
                f"rate limited on {target} after {max_rate_limit_retries + 1} attempts: {e}"
            ) from e
        except (litellm_exc.APIConnectionError, litellm_exc.APIError) as e:
            _failure_count += 1
            if _failure_count >= FAILURE_THRESHOLD:
                _circuit_open = True
                _circuit_open_until = now + CIRCUIT_OPEN_DURATION
            raise RuntimeError(f"LLM circuit breaker tripped for {target}: {e}") from e
        except litellm_exc.AuthenticationError as e:
            raise RuntimeError(f"LLM auth failed for {target}: {e}") from e
        except Exception as e:
            logger.error("llm_complete_async failed: %s", e)
            raise
    _log_usage(resp, user_id, action, db_write_callback)
    return text


def classify_section(text_chunk: str, max_retries: int = 3) -> str:
    """Classify a paper text chunk into a section label. Returns raw LLM text."""
    prompt = f"""You are classifying parts of a research paper.
Possible sections:
abstract, introduction, related_work, method, experiments,
results, discussion, conclusion, other
Return valid JSON ONLY:
{{"section": "<section>", "content": "<original_text>"}}
Text:
\"\"\"{text_chunk[:2000]}\"\"\"
"""
    for attempt in range(1, max_retries + 1):
        try:
            return llm_complete(prompt)
        except Exception as e:
            if attempt == max_retries:
                raise
            logger.warning("classify_section attempt %d failed: %s", attempt, e)
    raise RuntimeError("classify_section failed after retries")


def _log_usage(resp, user_id, action, db_write_callback):
    """Call the db_write_callback to write a UsageLog row if provided."""
    if db_write_callback is None:
        return
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return

        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        cost_usd = getattr(resp, "_hidden_params", {}).get("response_cost", 0.0)
        model = getattr(resp, "model", PRIMARY_MODEL)

        db_write_callback(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            model=model,
            user_id=user_id,
            action=action,
        )
    except Exception as e:
        logger.warning("usage log failed: %s", e)
