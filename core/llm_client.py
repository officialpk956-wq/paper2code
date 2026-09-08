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
PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini/gemini-2.0-flash")
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
) -> str:
    """Synchronous LLM completion with automatic provider fallback."""
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
    max_rate_limit_retries = 2
    rate_limit_backoff_seconds = 8

    resp = None
    for attempt in range(max_rate_limit_retries + 1):
        use_fallback = attempt == max_rate_limit_retries
        try:
            resp = completion(
                model=target,
                messages=messages,
                temperature=0,
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
                    logger.warning(
                        "Empty completion from %s (attempt %d/%d) -- retrying in %ds",
                        target, attempt + 1, max_rate_limit_retries, rate_limit_backoff_seconds,
                    )
                    time.sleep(rate_limit_backoff_seconds)
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
                logger.warning(
                    "Rate limited on %s (attempt %d/%d) -- retrying same model in %ds",
                    target, attempt + 1, max_rate_limit_retries, rate_limit_backoff_seconds,
                )
                time.sleep(rate_limit_backoff_seconds)
                continue
            _failure_count += 1
            if _failure_count >= FAILURE_THRESHOLD:
                _circuit_open = True
                _circuit_open_until = now + CIRCUIT_OPEN_DURATION
            raise RuntimeError(f"LLM circuit breaker tripped for {target}: {e}") from e
        except (litellm_exc.APIConnectionError, litellm_exc.APIError) as e:
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
    max_rate_limit_retries = 2
    rate_limit_backoff_seconds = 8

    resp = None
    for attempt in range(max_rate_limit_retries + 1):
        use_fallback = attempt == max_rate_limit_retries
        try:
            resp = await acompletion(
                model=target,
                messages=messages,
                temperature=0,
                fallbacks=_fallback_list(use_fallback, target),
            )
            text = resp.choices[0].message.content or ""
            _failure_count = 0
            break
        except litellm_exc.RateLimitError as e:
            if not use_fallback:
                logger.warning(
                    "Rate limited on %s (attempt %d/%d) -- retrying same model in %ds",
                    target, attempt + 1, max_rate_limit_retries, rate_limit_backoff_seconds,
                )
                import asyncio

                await asyncio.sleep(rate_limit_backoff_seconds)
                continue
            _failure_count += 1
            if _failure_count >= FAILURE_THRESHOLD:
                _circuit_open = True
                _circuit_open_until = now + CIRCUIT_OPEN_DURATION
            raise RuntimeError(f"LLM circuit breaker tripped for {target}: {e}") from e
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
