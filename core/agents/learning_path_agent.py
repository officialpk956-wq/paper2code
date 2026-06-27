import json
from core.llm_client import llm_complete

def generate_learning_path(
    completed_architectures: list[str],
    weak_topics: list[str],
    solved_problem_ids: list[str],
    available_papers: list[dict],
    available_problems: list[dict],
) -> dict:
    """
    One-shot LLM call with structured output.
    Returns: {"steps": [...], "reasoning": str}
    """
    prompt = f"""You are a machine learning curriculum designer.
Generate a personalized 5-step learning path for a student.

Student profile:

Completed architectures: {', '.join(completed_architectures) or 'None yet'}
Weak topics (< 50% assessment accuracy): {', '.join(weak_topics) or 'None identified'}
Solved problem IDs: {', '.join(solved_problem_ids[:10]) or 'None yet'}
Available papers (ID: title):
{chr(10).join(f" {p['id']}: {p['title']}" for p in available_papers[:20])}

Available problems (ID: title, difficulty):
{chr(10).join(f" {p['id']}: {p['title']} ({p['difficulty']})" for p in available_problems[:20])}

Return ONLY valid JSON in this exact format:
{{
"steps": [
{{
"step": 1,
"type": "paper|problem|concept",
"id": "paper_id_or_problem_id_or_null",
"title": "short title",
"reason": "one sentence why this step"
}}
],
"reasoning": "one paragraph explaining overall strategy"
}}"""
    raw = llm_complete(prompt)
    raw = raw.strip()
    if raw.startswith("```"): raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"steps": [], "reasoning": "Could not generate path."}
