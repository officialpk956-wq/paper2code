import os
from anthropic import Anthropic

TOOLS = [
    {
        "name": "lookup_paper_section",
        "description": "Get raw text of a section from a specific paper by paper_id. Useful when the user asks about a specific paper's content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "integer", "description": "The paper's database ID"},
                "section": {"type": "string", "description": "Section name: abstract, method, experiments, introduction"}
            },
            "required": ["paper_id"]
        }
    },
    {
        "name": "get_user_weak_topics",
        "description": "Get the architectures where this user has low assessment accuracy. Use to personalize the explanation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {"type": "integer"}
            },
            "required": ["user_id"]
        }
    },
    {
        "name": "find_related_problem",
        "description": "Find a dojo problem related to a concept. Return problem id, title, difficulty.",
        "input_schema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string", "description": "e.g. 'attention mechanism', 'residual connection', 'batch normalization'"}
            },
            "required": ["concept"]
        }
    },
    {
        "name": "search_papers_by_concept",
        "description": "Search papers in the database that mention a concept in their title or abstract.",
        "input_schema": {
            "type": "object",
            "properties": {
                "concept": {"type": "string"}
            },
            "required": ["concept"]
        }
    }
]

class AgenticTutor:
    def __init__(self, db_session):
        self.db = db_session
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "dummy_key"))

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as a string."""
        from backend.models import Paper, Problem, AssessmentAttempt
        from sqlalchemy import func
        
        if tool_name == "lookup_paper_section":
            paper = self.db.query(Paper).filter_by(
                id=tool_input["paper_id"]
            ).first()
            if not paper:
                return "Paper not found."
            section = tool_input.get("section", "abstract")
            content = getattr(paper, section, None) or paper.abstract or ""
            return content[:2000] if content else "No content available."
            
        if tool_name == "get_user_weak_topics":
            from backend.models import AssessmentAttempt
            attempts = (
                self.db.query(AssessmentAttempt)
                .filter_by(learner_id=str(tool_input["user_id"]))
                .all()
            )
            arch_results = {}
            for a in attempts:
                if a.architecture:
                    arch_results.setdefault(a.architecture, []).append(a.is_correct)
            weak = [
                arch for arch, results in arch_results.items()
                if results and sum(results)/len(results) < 0.5
            ]
            return f"Weak topics: {', '.join(weak)}" if weak else "No weak topics found."
            
        if tool_name == "find_related_problem":
            from backend.models import Problem
            concept = tool_input["concept"].lower()
            prob = (
                self.db.query(Problem)
                .filter(
                    Problem.is_retired == False,
                    Problem.description.ilike(f"%{concept}%")
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
                    Paper.title.ilike(f"%{concept}%")
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
        Run the agentic tutor. Returns (response_dict, updated_history).
        response_dict matches the shape returned by the old ask_with_history:
          {"answer": str, "reasoning_type": str, ...}
        """
        system = f"""You are an expert AI tutor for deep learning and machine learning architectures.
Context: {context_type} - {context_data.get('architecture', context_data.get('title', ''))}
You have access to tools to look up paper content, user progress, and related problems.
Use tools when the question requires specific data. Otherwise answer directly.
Always be educational and specific. Never give away full solutions to coding problems."""

        messages = list(history) + [{"role": "user", "content": query}]
        
        # Agentic loop: max 3 tool-use rounds
        for _ in range(3):
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})
            
            if response.stop_reason == "end_turn":
                break
                
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                messages.append({"role": "user", "content": tool_results})
                
        # Extract final text answer
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text
                break
                
        updated_history = messages[-6:]  # keep last 6 messages as context
        return {"answer": final_text, "reasoning_type": "Agentic"}, updated_history
