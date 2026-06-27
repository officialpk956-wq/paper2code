import time
import os
import httpx
import logging
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

logger = logging.getLogger(__name__)

def classify_section(text_chunk: str, max_retries=5):
    prompt = f"""
You are classifying parts of a research paper.

Possible sections:
abstract, introduction, related_work, method, experiments,
results, discussion, conclusion, other

Return valid JSON ONLY:
{{"section": "<section>", "content": "<original_text>"}}

Text:
\"\"\"{text_chunk}\"\"\"
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 512,
    }

    for attempt in range(1, max_retries + 1):
        response = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=45.0)

        # ✅ Success
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        # ⏳ Rate limited → wait & retry
        if response.status_code == 429:
            wait_time = 2 * attempt  # exponential backoff
            print(f"  ⏳ Rate limited. Sleeping {wait_time}s (retry {attempt}/{max_retries})")
            time.sleep(wait_time)
            continue

        # ❌ Other error → fail fast
        print("Groq error:", response.status_code, response.text)
        response.raise_for_status()

    raise RuntimeError("Groq API failed after retries")

# src/llm_client.py

GROQ_PRICING = {
    "llama3-70b-8192":       {"input": 0.00059, "output": 0.00079},   # per 1K tokens
    "llama3-8b-8192":        {"input": 0.00005, "output": 0.00008},
    "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant":  {"input": 0.00005, "output": 0.00008},
    "mixtral-8x7b-32768":    {"input": 0.00024, "output": 0.00024},
    "gemma2-9b-it":          {"input": 0.00020, "output": 0.00020},
}
# Fallback for unknown models: $0.001 / 1K tokens both directions
DEFAULT_COST_PER_1K = 0.001

def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = GROQ_PRICING.get(model, {"input": DEFAULT_COST_PER_1K, "output": DEFAULT_COST_PER_1K})
    return (prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]) / 1000

def llm_complete(prompt: str, user_id: int | None = None, action: str = "llm.unknown", db_session = None) -> str:
    """
    Generic LLM completion.
    Returns raw text output.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    response = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=45.0)
    response.raise_for_status()
    resp_json = response.json()

    try:
        usage = resp_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        model_used = resp_json.get("model", MODEL)
        cost = _calculate_cost(model_used, prompt_tokens, completion_tokens)
        
        if db_session is not None:
            from backend.models import UsageLog
            db_session.add(UsageLog(
                user_id=user_id,
                action=action,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                model=model_used,
            ))
            db_session.commit()
    except Exception as e:
        logger.warning(f"[usage_log] failed to log: {e}")
        if db_session is not None:
            try:
                db_session.rollback()
            except Exception:
                pass

    return resp_json["choices"][0]["message"]["content"]

async def llm_complete_async(prompt: str, model: str = MODEL, user_id: int | None = None, action: str = "llm.unknown", db_session = None) -> str:
    """Async version of llm_complete for use in async FastAPI routes."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        )
        resp.raise_for_status()
        resp_json = resp.json()

    try:
        usage = resp_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        model_used = resp_json.get("model", model)
        cost = _calculate_cost(model_used, prompt_tokens, completion_tokens)
        
        if db_session is not None:
            from backend.models import UsageLog
            import inspect
            is_async = hasattr(db_session, "commit") and inspect.iscoroutinefunction(db_session.commit)
            
            db_session.add(UsageLog(
                user_id=user_id,
                action=action,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                model=model_used,
            ))
            if is_async:
                await db_session.commit()
            else:
                db_session.commit()
    except Exception as e:
        logger.warning(f"[usage_log] failed to log async: {e}")
        if db_session is not None:
            try:
                import inspect
                is_async = hasattr(db_session, "rollback") and inspect.iscoroutinefunction(db_session.rollback)
                if is_async:
                    await db_session.rollback()
                else:
                    db_session.rollback()
            except Exception:
                pass

    return resp_json["choices"][0]["message"]["content"]
