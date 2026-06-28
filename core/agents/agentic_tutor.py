import os
import json
import logging

logger = logging.getLogger(__name__)

_MODEL = os.getenv("LLM_PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")
_FALLBACK = os.getenv("LLM_FALLBACK_MODEL", "gemini/gemini-2.0-flash")

# OpenAI function-calling format (works with all LiteLLM providers)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_paper_section",
            "description": "Get raw text of a section from a specific paper by paper_id. Use when the user asks about a specific paper's content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "integer", "description": "The paper's database ID"},
                    "section": {"type": "string", "description": "Section name: abstract, method, experiments, introduction"}
                },
                "required": ["paper_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_weak_topics",
            "description": "Get architectures where this user has low assessment accuracy. Use to personalize explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_related_problem",
            "description": "Find a dojo problem related to a concept. Return problem id, title, difficulty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "e.g. 'attention mechanism', 'residual connection'"}
                },
                "required": ["concept"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_papers_by_concept",
            "description": "Search papers in the database that mention a concept in their title or abstract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"}
                },
                "required": ["concept"]
            }
        }
    }
]

_ALLOWED_CONTEXT_KEYS = {"architecture", "title", "domain", "topic"}
_MAX_CONTEXT_VALUE_LEN = 200
_MAX_QUERY_LEN = 2000


def _safe_context_value(value: str) -> str:
    clean = str(value).replace("\n", " ").replace("\r", " ").strip()
    return clean[:_MAX_CONTEXT_VALUE_LEN]


class AgenticTutor:
    def __init__(self, db_session):
        self.db = db_session

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        from backend.models import Paper, Problem, AssessmentAttempt

        if tool_name == "lookup_paper_section":
            paper = self.db.query(Paper).filter_by(id=tool_input["paper_id"]).first()
            if not paper:
                return "Paper not found."
            section = tool_input.get("section", "abstract")
            content = getattr(paper, section, None) or paper.abstract or ""
            return content[:2000] if content else "No content available."

        if tool_name == "get_user_weak_topics":
            attempts = (
                self.db.query(AssessmentAttempt)
                .filter_by(learner_id=str(tool_input["user_id"]))
                .all()
            )
            arch_results: dict[str, list[bool]] = {}
            for a in attempts:
                if a.architecture:
                    arch_results.setdefault(a.architecture, []).append(a.is_correct)
            weak = [
                arch for arch, results in arch_results.items()
                if results and sum(results) / len(results) < 0.5
            ]
            return f"Weak topics: {', '.join(weak)}" if weak else "No weak topics found."

        if tool_name == "find_related_problem":
            concept = tool_input["concept"].lower()
            prob = (
                self.db.query(Problem)
                .filter(
                    Problem.is_retired == False,
                    Problem.description.ilike(f"%{concept}%"),
                )
                .first()
            )
            if prob:
                return f"Problem: {prob.title} (ID: {prob.id}, Difficulty: {prob.difficulty})"
            return "No related problem found."

        if tool_name == "search_papers_by_concept":
            from backend.models import Paper
            concept = tool_input["concept"]
            papers = (
                self.db.query(Paper)
                .filter(
                    Paper.visibility != "private",
                    Paper.title.ilike(f"%{concept}%"),
                )
                .limit(3)
                .all()
            )
            if papers:
                return "; ".join(f"{p.title} (ID: {p.id})" for p in papers)
            return "No papers found."

        return f"Unknown tool: {tool_name}"

    def ask(
        self,
        query: str,
        history: list[dict],
        context_type: str,
        context_data: dict,
        user_id: int | None,
        knowledge_profile: dict | None = None,
    ) -> tuple[dict, list[dict]]:
        """
        Run the agentic tutor via LiteLLM with tool calling.
        Returns (response_dict, updated_history).
        response_dict: {"answer": str, "reasoning_type": str}
        """
        from litellm import completion

        if len(query) > _MAX_QUERY_LEN:
            raise ValueError(f"Query exceeds maximum length of {_MAX_QUERY_LEN} characters")

        safe_arch = _safe_context_value(
            context_data.get("architecture", context_data.get("title", ""))
        )
        safe_ctx = _safe_context_value(context_type)

        system = (
            f"You are an expert AI tutor for deep learning and machine learning architectures.\n"
            f"<context_type>{safe_ctx}</context_type>\n"
            f"<context_subject>{safe_arch}</context_subject>\n"
            "Use tools when the question requires specific data. Otherwise answer directly.\n"
            "Always be educational and specific. Never give away full solutions to coding problems."
        )

        messages: list[dict] = (
            [{"role": "system", "content": system}]
            + list(history)
            + [{"role": "user", "content": query}]
        )

        # Agentic loop: max 3 tool-use rounds
        for _ in range(3):
            resp = completion(
                model=_MODEL,
                messages=messages,
                tools=TOOLS,
                temperature=0,
                fallbacks=[_FALLBACK] if _MODEL != _FALLBACK else [],
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason

            # Append the assistant turn (with or without tool_calls)
            messages.append(choice.message.model_dump(exclude_none=True) if hasattr(choice.message, "model_dump") else {"role": "assistant", "content": choice.message.content or "", "tool_calls": getattr(choice.message, "tool_calls", None)})

            if finish_reason != "tool_calls":
                break

            # Execute each tool call and append results
            tool_calls = choice.message.tool_calls or []
            for tc in tool_calls:
                tool_result = self._execute_tool(
                    tc.function.name,
                    json.loads(tc.function.arguments),
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        # Extract the final text answer
        final_text = choice.message.content or ""
        if not final_text:
            final_text = "I was unable to generate a complete response. Please rephrase your question."

        updated_history = messages[-6:]
        return {"answer": final_text, "reasoning_type": "Agentic"}, updated_history
