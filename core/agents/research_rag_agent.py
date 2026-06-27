from core.llm_client import llm_complete

def ask_about_paper(
    target_paper: dict,
    all_papers: list[dict],
    question: str,
) -> dict:
    """
    Answer a research question about a paper, optionally referencing others.
    Returns {"answer": str, "referenced_papers": list[int]}
    """
    # Build context from target paper
    ag = target_paper.get("architecture_graph") or {}
    nodes = ag.get("nodes", []) if isinstance(ag, dict) else []
    node_summary = ", ".join(n.get("type", "") for n in nodes[:10]) if nodes else "unknown"

    # Find potentially relevant papers (simple title keyword match)
    words = set(question.lower().split())
    relevant = [
        p for p in all_papers
        if p["id"] != target_paper["id"] and
        any(w in (p.get("title") or "").lower() for w in words if len(w) > 4)
    ][:3]
    
    related_context = "\n".join(
        f"- {p['title']}: {(p.get('abstract') or '')[:200]}"
        for p in relevant
    ) if relevant else "None found."
    
    prompt = f"""You are a research assistant for deep learning papers.
Target paper: {target_paper['title']}
Abstract: {(target_paper.get('abstract') or '')[:600]}
Architecture nodes: {node_summary}

Related papers in the database:
{related_context}

User question: {question}

Answer concisely (150-300 words). Reference specific parts of the paper.
If other papers are relevant, mention them by name.
"""
    answer = llm_complete(prompt)
    referenced = [p["id"] for p in relevant]
    return {"answer": answer, "referenced_papers": referenced}
