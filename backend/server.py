from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import io
import json

from core.orchestrator.pipeline import Paper2CodePipeline
from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.config_extractor import ConfigExtractor
from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
from core.explainers.graph_explainer import explain_node
from core.codegen import _node_to_layer
from backend.database import ping_db, get_db
from sqlalchemy.orm import Session
from fastapi import Depends

app = FastAPI(title="Paper2Code API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline = Paper2CodePipeline()
extractor = ConfigExtractor()
generator = PaperToCodeGenerator()

class TextRequest(BaseModel):
    text: str
    
class CompareRequest(BaseModel):
    text_a: str
    text_b: str

class GraphRequest(BaseModel):
    name: str = "Architect Session"
    layers: list[Dict[str, Any]]

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/health/db")
def health_db():
    status = ping_db()
    if not status["ok"]:
        raise HTTPException(status_code=503, detail=status)
    return status

def _build_response(spec, result, code, code_source):
    graph = result["graph"]
    metrics = estimate_metrics_from_graph(graph)
    mem_data = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
    total_mem_mb = sum(row['mem_mb'] for row in mem_data) if mem_data else 0
    
    # Generate layer breakdown for explanation
    layer_breakdown = []
    for node in graph.nodes:
        node_code = _node_to_layer(node) or f"# Custom block implementation needed for {node.type}"
        layer_breakdown.append({
            "id": node.id,
            "label": node.label,
            "type": node.type,
            "params": node.params,
            "semantic": node.semantic_params,
            "description": node.description,
            "explanation": explain_node(node),
            "code_snippet": node_code
        })
        
    return {
        "name": graph.name,
        "svg": result["visual"]["graphviz_dot"],
        "explanation": result["explanation"],
        "metadata": result["metadata"],
        "code": code,
        "code_source": code_source,
        "layer_breakdown": layer_breakdown,
        "metrics": {
            "flops_score": metrics["total_flops_score"],
            "params": metrics["total_params_estimate"],
            "depth": metrics["depth"],
            "memory_mb": round(total_mem_mb, 1)
        },
        "tensor_trace": graph.metadata.get("tensor_trace", []),
        "cross_attention_events": graph.metadata.get("cross_attention_events", []),
        "flops_events": graph.metadata.get("flops_events", []),
        "kag_motifs": result.get("kag_motifs", []),
        "kag_anomalies": result.get("kag_anomalies", []),
        "kag_semantic_roles": result.get("kag_semantic_roles", {}),
    }

@app.post("/api/parse_pdf")
async def parse_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        paper_name = file.filename.replace(".pdf", "")
        # Use generator's built-in PDF extraction which handles pdfplumber safely
        result_dict = generator.from_pdf(pdf_bytes, paper_name)
        
        # result_dict already contains graph, explanation, code, code_source.
        # But we want to reuse _build_response to get metrics and layer_breakdown
        return _build_response(
            spec={"model_family": result_dict.get("family", "")}, 
            result=result_dict, 
            code=result_dict["code"], 
            code_source=result_dict["code_source"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/parse_text")
def parse_text(request: TextRequest):
    try:
        spec = extractor.extract_from_text(request.text)
        result = pipeline.run_single(spec)
        code, code_source = generator._generate_code(spec, result["graph"])
        return _build_response(spec, result, code, code_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/compare_text")
def compare_text(request: CompareRequest):
    try:
        result = pipeline.run_comparison_from_text(request.text_a, request.text_b)
        return {
            "name_a": result["graph_a"].name,
            "name_b": result["graph_b"].name,
            "svg_a": result["visual_a"]["graphviz_dot"],
            "svg_b": result["visual_b"]["graphviz_dot"],
            "explanation": result["explanation"],
            "metadata": result["metadata"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze_graph")
def analyze_graph(request: GraphRequest):
    try:
        # Normalize the layers received from the frontend
        # The frontend sends: [{"type": "Conv2D", "params": {...}}, ...]
        # Pipeline expects normalized ConfigDict.
        
        # We'll use the normalizer directly
        from core.rag.normalizer import normalize_config
        config = normalize_config({
            "name": request.name,
            "layers": request.layers
        })
        
        # Run pipeline starting from config (bypassing extraction)
        result = pipeline.run_single(request.name, config)
        
        return _build_response(
            spec={"model_family": "Architect"}, 
            result=result, 
            code=result.get("code", ""), 
            code_source="Architect Session"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Paper2Code Golden Paper Set APIs
# ==========================================

def get_clean_paper_info(paper) -> tuple[str, str]:
    title_lower = paper.title.lower()
    if "resnet" in title_lower:
        return "ResNet", "CNN"
    elif "transformer" in title_lower:
        return "Transformer", "Attention"
    elif "u-net" in title_lower or "unet" in title_lower:
        return "U-Net", "Encoder-Decoder"
    return paper.title, "Unknown"

@app.get("/api/papers")
def list_papers(db: Session = Depends(get_db)):
    from backend.models import Paper
    papers = db.query(Paper).all()
    results = []
    for p in papers:
        title, arch_type = get_clean_paper_info(p)
        flops_analysis = p.flops_analysis or {}
        results.append({
            "id": p.id,
            "title": title,
            "architecture_type": arch_type,
            "module_count": len(p.modules),
            "parameter_count": flops_analysis.get("total_params_estimate", 0),
            "flops": flops_analysis.get("total_flops_score", 0)
        })
    return results

@app.get("/api/papers/{paper_id}")
def get_paper_details(paper_id: int, db: Session = Depends(get_db)):
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type = get_clean_paper_info(p)
    flops_analysis = p.flops_analysis or {}

    modules_summary = []
    for m in p.modules:
        modules_summary.append({
            "id": m.id,
            "order_index": m.order_index,
            "layer_name": m.layer_name,
            "module_type": m.module_type,
            "explanation": m.explanation,
            "tensor_flow": m.tensor_flow,
            "graph_nodes": m.graph_nodes,
            "flops_context": m.flops_context
        })

    return {
        "metadata": {
            "id": p.id,
            "title": title,
            "full_title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "architecture_type": arch_type
        },
        "module_summary": modules_summary,
        "architecture_statistics": {
            "depth": flops_analysis.get("depth", 0),
            "node_count": len(p.architecture_graph.get("nodes", [])) if p.architecture_graph else 0,
            "edge_count": len(p.architecture_graph.get("edges", [])) if p.architecture_graph else 0
        },
        "architecture_graph": p.architecture_graph or {"nodes": [], "edges": []},
        "flops": flops_analysis.get("total_flops_score", 0),
        "parameter_count": flops_analysis.get("total_params_estimate", 0)
    }

# ==========================================
# Sprint 2: Module-level APIs
# ==========================================

# ---------------------------------------------------------------------------
# Defensive serialization helpers
# ---------------------------------------------------------------------------

def safe_dict(val) -> dict:
    """
    Coerce a persisted JSON column value to a plain dict.

    Handles:
      - dict   → returned as-is
      - str    → parsed as JSON; must decode to a dict, else {}
      - list   → not a dict; returns {} (caller handles lists separately)
      - None   → {}
    """
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def safe_list(val) -> list:
    """
    Coerce a persisted JSON column value to a plain list.

    Handles:
      - list   → returned as-is
      - str    → parsed as JSON; must decode to a list, else []
      - dict   → not a list; returns [] (caller handles dicts separately)
      - None   → []
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []
    return []


def _module_to_dict(m, paper_id: int, total: int) -> dict:
    """
    Serialize a PaperModule ORM object to a canonical API dict.

    Defensive rules for JSON columns:
      flops_context  — always a dict or None in practice; safe_dict handles edge cases.
      tensor_flow    — may be a list (trace rows) or a dict; both are valid.
      graph_nodes    — may be a list of node dicts or a dict with 'nodes'/'edges' keys.
    """
    # --- flops_context ---
    flops = safe_dict(m.flops_context)

    # --- tensor_flow ---
    # Persisted as a list of trace-row dicts  →  treat the list as the trace.
    # Persisted as a dict with structured keys →  use dict lookup.
    raw_tensor = m.tensor_flow
    if isinstance(raw_tensor, list):
        tensor: dict = {"trace": raw_tensor}
    else:
        tensor = safe_dict(raw_tensor)

    # --- graph_nodes ---
    # Persisted as a list of node dicts         →  nodes = the list, edges = [].
    # Persisted as a dict with nodes/edges keys →  extract from dict.
    raw_graph = m.graph_nodes
    if isinstance(raw_graph, list):
        graph_nodes_list: list = raw_graph
        graph_edges_list: list = []
    elif isinstance(raw_graph, dict):
        graph_nodes_list = raw_graph.get("nodes", [])
        graph_edges_list = raw_graph.get("edges", [])
    elif isinstance(raw_graph, str):
        # Rare: stored as a JSON string
        parsed = safe_list(raw_graph) or safe_dict(raw_graph)
        if isinstance(parsed, list):
            graph_nodes_list = parsed
            graph_edges_list = []
        elif isinstance(parsed, dict):
            graph_nodes_list = parsed.get("nodes", [])
            graph_edges_list = parsed.get("edges", [])
        else:
            graph_nodes_list = []
            graph_edges_list = []
    else:
        graph_nodes_list = []
        graph_edges_list = []

    return {
        "id": m.id,
        "confidence": flops.get("confidence", 0.0),
        "paper_id": paper_id,
        "order_index": m.order_index,
        "total_modules": total,
        "layer_name": m.layer_name,
        "module_type": m.module_type,
        "description": flops.get("description") or m.explanation or "",
        "explanation": m.explanation or "",
        # FLOPs context
        "flops_context": {
            "total_flops_score": flops.get("total_flops_score", 0),
            "real_flops_mflops": flops.get("real_flops_mflops", 0.0),
            "total_params_estimate": flops.get("total_params_estimate", 0),
            "depth": flops.get("depth", 0),
            "breakdown": safe_list(flops.get("breakdown", [])),
        },
        # Tensor summary
        "tensor_summary": {
            "input_shape": tensor.get("input_shape") or tensor.get("input"),
            "output_shape": tensor.get("output_shape") or tensor.get("output"),
            "operations": safe_list(tensor.get("operations", [])),
            "trace": safe_list(tensor.get("trace", [])),
        },
        # Graph references
        "graph_nodes": graph_nodes_list,
        "graph_edges": graph_edges_list,
    }


@app.get("/api/papers/{paper_id}/modules")
def list_paper_modules(paper_id: int, db: Session = Depends(get_db)):
    """Return all modules for a paper, ordered by order_index."""
    from backend.models import Paper, PaperModule
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type = get_clean_paper_info(p)
    total = len(p.modules)

    modules = []
    for m in p.modules:
        modules.append({
            "id": m.id,
            "paper_id": paper_id,
            "order_index": m.order_index,
            "total_modules": total,
            "layer_name": m.layer_name,
            "module_type": m.module_type,
        })

    return {
        "paper_id": paper_id,
        "paper_title": title,
        "architecture_type": arch_type,
        "total_modules": total,
        "modules": modules
    }


@app.get("/api/modules/{module_id}")
def get_module(module_id: int, db: Session = Depends(get_db)):
    """Return full detail for a single module, including navigation neighbours."""
    from backend.models import PaperModule, Paper
    m = db.query(PaperModule).filter(PaperModule.id == module_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Module not found")

    # Fetch all sibling modules for this paper (ordered) to build prev/next nav
    siblings = (
        db.query(PaperModule)
        .filter(PaperModule.paper_id == m.paper_id)
        .order_by(PaperModule.order_index)
        .all()
    )
    total = len(siblings)
    ids = [s.id for s in siblings]
    pos = ids.index(m.id)  # 0-based position

    prev_id = ids[pos - 1] if pos > 0 else None
    next_id = ids[pos + 1] if pos < total - 1 else None

    paper_title, arch_type = get_clean_paper_info(m.paper)

    result = _module_to_dict(m, m.paper_id, total)
    result.update({
        "position": pos + 1,          # 1-based for display
        "prev_module_id": prev_id,
        "next_module_id": next_id,
        "paper_title": paper_title,
        "architecture_type": arch_type,
    })
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
