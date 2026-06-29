import os
import logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Model config — override via env vars
# ---------------------------------------------------------------------------
PRIMARY_MODEL  = os.getenv("LLM_PRIMARY_MODEL",  "groq/llama-3.3-70b-versatile")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini/gemini-2.0-flash")
# Backward-compat alias (some callers do `if GROQ_API_KEY:`)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Simple circuit breaker state
_circuit_open = False
_circuit_open_until = 0
_failure_count = 0
FAILURE_THRESHOLD = 5
CIRCUIT_OPEN_DURATION = 60  # seconds

def llm_complete(
    prompt: str,
    user_id: int | None = None,
    action: str = "llm.unknown",
    db_session=None,
    model: str | None = None,
) -> str:
    """Synchronous LLM completion with automatic provider fallback."""
    from litellm import completion, exceptions as litellm_exc
    import time
    global _circuit_open, _circuit_open_until, _failure_count
    
    now = time.time()
    if _circuit_open and now < _circuit_open_until:
        raise RuntimeError("LLM circuit breaker open — service temporarily unavailable")
    
    if _circuit_open and now >= _circuit_open_until:
        _circuit_open = False  # half-open: try again
        _failure_count = 0

    messages = [{"role": "user", "content": prompt}]
    target   = model or PRIMARY_MODEL
    try:
        resp = completion(
            model=target,
            messages=messages,
            temperature=0,
            fallbacks=[FALLBACK_MODEL] if target != FALLBACK_MODEL else [],
        )
        text = resp.choices[0].message.content or ""
        _failure_count = 0  # success resets counter
    except (litellm_exc.APIConnectionError, litellm_exc.RateLimitError, litellm_exc.APIError) as e:
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
    _log_usage(resp, user_id, action, db_session)
    return text
async def llm_complete_async(
    prompt: str,
    model: str | None = None,
    user_id: int | None = None,
    action: str = "llm.unknown",
    db_session=None,
) -> str:
    """Async LLM completion with automatic provider fallback."""
    from litellm import acompletion, exceptions as litellm_exc
    import time
    global _circuit_open, _circuit_open_until, _failure_count
    
    now = time.time()
    if _circuit_open and now < _circuit_open_until:
        raise RuntimeError("LLM circuit breaker open — service temporarily unavailable")
    
    if _circuit_open and now >= _circuit_open_until:
        _circuit_open = False
        _failure_count = 0

    messages = [{"role": "user", "content": prompt}]
    target   = model or PRIMARY_MODEL
    try:
        resp = await acompletion(
            model=target,
            messages=messages,
            temperature=0,
            fallbacks=[FALLBACK_MODEL] if target != FALLBACK_MODEL else [],
        )
        text = resp.choices[0].message.content or ""
        _failure_count = 0
    except (litellm_exc.APIConnectionError, litellm_exc.RateLimitError, litellm_exc.APIError) as e:
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
    _log_usage(resp, user_id, action, db_session)
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
def _log_usage(resp, user_id, action, db_session):
    """Write a UsageLog row if db_session is provided."""
    if db_session is None:
        return
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        from backend.models import UsageLog
        db_session.add(UsageLog(
            user_id=user_id,
            action=action,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cost_usd=getattr(resp, "_hidden_params", {}).get("response_cost", 0),
            model=getattr(resp, "model", PRIMARY_MODEL),
        ))
        db_session.commit()
    except Exception as e:
        logger.warning("usage log failed: %s", e)
        try:
            db_session.rollback()
        except Exception:
            pass
