from typing import TypedDict, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from core.section_splitter import process_text
from core.architecture_extractor import extract_architecture
from core.classification import infer_family_from_name
from core.llm_client import llm_complete
import logging

logger = logging.getLogger(__name__)

class IngestionState(TypedDict):
    paper_id: int
    paper_name: str
    raw_text: str
    sections: dict
    model_family: str
    schema: dict
    code: str
    lint_errors: list[str]
    revision_count: int
    final_code: str
    error: Optional[str]

def split_sections(state: IngestionState) -> dict:
    """Split raw text into labelled sections."""
    try:
        sections = process_text(state["raw_text"])
        return {"sections": sections}
    except Exception as e:
        return {"error": f"Section split failed: {e}", "sections": {}}

def classify_paper(state: IngestionState) -> dict:
    """Determine model family from paper name + section keywords."""
    family = infer_family_from_name(state["paper_name"])
    if not family:
        # fallback: ask LLM
        text_sample = state["sections"].get("method", "")[:500]
        prompt = f"""Classify this ML paper into exactly one family:
resnet, unet, transformer, vit, diffusion, yolo, efficientnet, swin,
gan, densenet, bert_gpt, mobilenet, mae, ldm, cnn, unknown

Paper name: {state["paper_name"]}
Method excerpt: {text_sample}

Reply with ONLY the family name, lowercase, no punctuation."""
        family = llm_complete(prompt).strip().split()[0].lower()
    return {"model_family": family}

def extract_schema(state: IngestionState) -> dict:
    """Extract architecture schema using existing architecture_extractor."""
    try:
        schema = extract_architecture(state["sections"], state["paper_name"])
        schema["model_family"] = state["model_family"]
        return {"schema": schema}
    except Exception as e:
        return {"error": f"Schema extraction failed: {e}", "schema": {}}

def generate_code(state: IngestionState) -> dict:
    """Generate PyTorch code from schema using existing codegen."""
    try:
        from core.paper_to_code_generator import generate_pytorch_code
        code = generate_pytorch_code(state["schema"])
        return {"code": code, "lint_errors": [], "revision_count": 0}
    except Exception as e:
        return {"error": f"Codegen failed: {e}", "code": ""}

def lint_code(state: IngestionState) -> dict:
    """Check code for syntax errors using py_compile."""
    import py_compile, tempfile, os
    code = state.get("code", "")
    if not code:
        return {"lint_errors": ["No code generated"]}
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
            f.write(code)
            fname = f.name
        py_compile.compile(fname, doraise=True)
        os.unlink(fname)
        return {"lint_errors": []}
    except py_compile.PyCompileError as e:
        try: os.unlink(fname)
        except: pass
        return {"lint_errors": [str(e)]}

def revise_code(state: IngestionState) -> dict:
    """Ask LLM to fix lint errors. Max 3 revisions."""
    errors = state.get("lint_errors", [])
    code = state.get("code", "")
    prompt = f"""Fix these Python syntax errors in the following PyTorch code.
Return ONLY the corrected Python code, no explanation.

Errors:
{chr(10).join(errors)}

Code:
{code[:3000]}
"""
    fixed = llm_complete(prompt)
    return {
        "code": fixed,
        "revision_count": state.get("revision_count", 0) + 1,
    }

def commit_result(state: IngestionState) -> dict:
    """Store final code (lint passed or max revisions reached)."""
    return {"final_code": state.get("code", "")}

def should_revise(state: IngestionState) -> str:
    if state.get("error"):
        return "commit"
    errors = state.get("lint_errors", [])
    if not errors:
        return "commit"
    if state.get("revision_count", 0) >= 3:
        logger.warning("Max revisions reached for paper %s", state.get("paper_id"))
        return "commit"
    return "revise"

def build_ingestion_graph() -> StateGraph:
    graph = StateGraph(IngestionState)
    graph.add_node("split_sections", split_sections)
    graph.add_node("classify_paper", classify_paper)
    graph.add_node("extract_schema", extract_schema)
    graph.add_node("generate_code", generate_code)
    graph.add_node("lint_code", lint_code)
    graph.add_node("revise_code", revise_code)
    graph.add_node("commit_result", commit_result)
    
    graph.set_entry_point("split_sections")
    graph.add_edge("split_sections", "classify_paper")
    graph.add_edge("classify_paper", "extract_schema")
    graph.add_edge("extract_schema", "generate_code")
    graph.add_edge("generate_code", "lint_code")
    graph.add_conditional_edges("lint_code", should_revise,
        {"commit": "commit_result", "revise": "revise_code"})
    graph.add_edge("revise_code", "lint_code")
    graph.add_edge("commit_result", END)
    return graph.compile()

ingestion_graph = build_ingestion_graph()

def run_ingestion(paper_id: int, paper_name: str, raw_text: str) -> dict:
    """Entry point called by Celery task."""
    initial_state = IngestionState(
        paper_id=paper_id, paper_name=paper_name, raw_text=raw_text,
        sections={}, model_family="", schema={}, code="",
        lint_errors=[], revision_count=0, final_code="", error=None,
    )
    result = ingestion_graph.invoke(initial_state)
    return {"final_code": result.get("final_code", ""), "error": result.get("error")}
