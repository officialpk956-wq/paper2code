import os
import logging
from typing import Optional
from pydantic import BaseModel
import instructor
from litellm import completion as litellm_completion

logger = logging.getLogger(__name__)

class PaperAnswer(BaseModel):
    answer:            str
    referenced_papers: list[int] = []

_client = instructor.from_litellm(litellm_completion)
_MODEL  = os.getenv("LLM_PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")

def ask_about_paper(
    target_paper: dict,
    all_papers:   list[dict],
    question:     str,
    check_budget_callback=None,
    user_id: Optional[int] = None,
) -> dict:
    from core.llm_client import check_user_token_budget
    check_user_token_budget(check_budget_callback, user_id)
    
    """Returns {"answer": str, "referenced_papers": list[int]}"""
    ag   = target_paper.get("architecture_graph") or {}
    nodes = ag.get("nodes", []) if isinstance(ag, dict) else []
    node_summary = ", ".join(n.get("type", "") for n in nodes[:10]) or "unknown"

    words   = set(question.lower().split())
    relevant = [
        p for p in all_papers
        if p["id"] != target_paper["id"]
        and any(w in (p.get("title") or "").lower() for w in words if len(w) > 4)
    ][:3]

    related_context = "\n".join(
        f"- {p['title']} (id={p['id']}): {(p.get('abstract') or '')[:200]}"
        for p in relevant
    ) or "None found."

    prompt = f"""You are a research assistant for deep learning papers.
Target paper: {target_paper['title']}
Abstract: {(target_paper.get('abstract') or '')[:600]}
Architecture nodes: {node_summary}

Related papers:
{related_context}

Question: {question}

Answer concisely (150-300 words). If you reference related papers include their IDs in referenced_papers."""

    try:
        result = _client.chat.completions.create(
            model=_MODEL,
            response_model=PaperAnswer,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_retries=2,
        )
        return result.model_dump()
    except Exception as e:
        logger.error("ask_about_paper failed: %s", e)
        return {"answer": "Unable to answer at this time.", "referenced_papers": []}
