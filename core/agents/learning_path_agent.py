import logging
import os

import instructor
from litellm import completion as litellm_completion
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema — LLM output is always validated against this
# ---------------------------------------------------------------------------
class LearningStep(BaseModel):
    step: int
    type: str  # "paper" | "problem" | "concept"
    id: str | None = None
    title: str
    reason: str


class LearningPath(BaseModel):
    steps: list[LearningStep]
    reasoning: str


# Instructor client wrapping LiteLLM
_client = instructor.from_litellm(litellm_completion)
_MODEL = os.getenv("LLM_PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")


def generate_learning_path(
    completed_architectures: list[str],
    weak_topics: list[str],
    solved_problem_ids: list[str],
    available_papers: list[dict],
    available_problems: list[dict],
) -> dict:
    """
    Returns {"steps": [...], "reasoning": str}
    Always returns a valid dict — never raises on bad LLM output.
    """
    prompt = f"""You are a machine learning curriculum designer.
Generate a personalized 5-step learning path for a student.

Student profile:
Completed architectures: {", ".join(completed_architectures) or "None yet"}
Weak topics (< 50% accuracy): {", ".join(weak_topics) or "None identified"}
Solved problem IDs: {", ".join(solved_problem_ids[:10]) or "None yet"}

Available papers (id: title):
{chr(10).join(f"  {p['id']}: {p['title']}" for p in available_papers[:20])}

Available problems (id: title, difficulty):
{chr(10).join(f"  {p['id']}: {p['title']} ({p['difficulty']})" for p in available_problems[:20])}

Generate exactly 5 steps. Each step must have a clear reason."""

    try:
        path = _client.chat.completions.create(
            model=_MODEL,
            response_model=LearningPath,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_retries=2,
        )
        return path.model_dump()
    except Exception as e:
        logger.error("generate_learning_path failed: %s", e)
        return {"steps": [], "reasoning": "Could not generate a learning path at this time."}
