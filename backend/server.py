from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import io
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.orchestrator.pipeline import Paper2CodePipeline
from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.config_extractor import ConfigExtractor
from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
from core.explainers.graph_explainer import explain_node
from core.codegen import _node_to_layer
import dataclasses
from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_vit import build_vit_graph
from core.visualizer_unet import build_unet_graph
from core.rag.tensor_tracker import TensorTracker
from backend.database import ping_db, get_db, engine
from sqlalchemy.orm import Session
from fastapi import Depends, Header, Request
from backend.models import Base
Base.metadata.create_all(bind=engine)

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

class PlaygroundRequest(BaseModel):
    architecture: str
    config: Dict[str, Any]

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
    logger.info("Upload Started for file: %s", file.filename)
    if not file.filename.lower().endswith(".pdf"):
        logger.warning("Upload Failed: non-PDF file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Check size (Max 20MB)
    MAX_SIZE = 20 * 1024 * 1024
    if file.size is not None and file.size > MAX_SIZE:
        logger.warning("Upload Failed: File exceeds 20MB limit.")
        raise HTTPException(status_code=400, detail="File exceeds 20MB limit.")

    try:
        paper_name = file.filename.replace(".pdf", "")
        # Use generator's built-in PDF extraction which handles pdfplumber safely via spooled file
        result_dict = generator.from_pdf(file.file, paper_name)
        
        logger.info("Upload Completed for file: %s", file.filename)
        
        # result_dict already contains graph, explanation, code, code_source.
        # But we want to reuse _build_response to get metrics and layer_breakdown
        return _build_response(
            spec={"model_family": result_dict.get("family", "")}, 
            result=result_dict, 
            code=result_dict["code"], 
            code_source=result_dict["code_source"]
        )
    except Exception as e:
        logger.error("Upload Failed: %s", str(e))
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

@app.post("/api/playground/generate")
def generate_playground(request: PlaygroundRequest):
    logger.info("Playground Generated for architecture: %s", request.architecture)
    try:
        # 1. Build the baseline and modified graphs
        graph = None
        baseline_graph = None
        
        if request.architecture == "ResNet":
            # Using defaults for baseline
            baseline_graph = build_resnet18_graph(base_channels=64, stages=4, blocks_per_stage=2)
            graph = build_resnet18_graph(
                base_channels=request.config.get("base_channels", 64),
                stages=request.config.get("stages", 4),
                blocks_per_stage=request.config.get("blocks_per_stage", 2)
            )
        elif request.architecture == "Transformer":
            baseline_graph = build_vit_graph(hidden_size=768, num_heads=12, depth=4)
            graph = build_vit_graph(
                hidden_size=request.config.get("hidden_size", 768),
                num_heads=request.config.get("num_heads", 12),
                depth=request.config.get("depth", 4)
            )
        elif request.architecture == "U-Net":
            baseline_graph = build_unet_graph(base_channels=64, stages=3)
            graph = build_unet_graph(
                base_channels=request.config.get("base_channels", 64),
                stages=request.config.get("stages", 3)
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported architecture type")
            
        # 2. Track tensors and compute baseline metrics
        tracker = TensorTracker()
        try: tracker.propagate_shapes(baseline_graph)
        except: pass
        baseline_m = estimate_metrics_from_graph(baseline_graph)
        b_mem_data = estimate_activation_memory(baseline_graph, batch_size=1, input_spatial=224)
        baseline_mem = sum(r['mem_mb'] for r in b_mem_data) if b_mem_data else 0
        
        baseline_metrics = {
            "flops_score": baseline_m["total_flops_score"],
            "params": baseline_m["total_params_estimate"],
            "depth": baseline_m["depth"],
            "memory_mb": round(baseline_mem, 1)
        }
        
        # 3. Track tensors and compute current metrics
        tracker = TensorTracker()
        is_valid = True
        validation_error = None
        try:
            tracker.propagate_shapes(graph)
        except Exception as e:
            is_valid = False
            validation_error = str(e)
            
        current_m = estimate_metrics_from_graph(graph)
        c_mem_data = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
        current_mem = sum(r['mem_mb'] for r in c_mem_data) if c_mem_data else 0
        
        metrics = {
            "flops_score": current_m["total_flops_score"],
            "params": current_m["total_params_estimate"],
            "depth": current_m["depth"],
            "memory_mb": round(current_mem, 1)
        }
        
        # Validation output
        validation = {
            "is_valid": is_valid,
            "error": validation_error,
            "anomalies": graph.metadata.get("kag_anomalies", []),
            "motifs": graph.metadata.get("kag_motifs", [])
        }
        
        return {
            "graph": dataclasses.asdict(graph),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "validation": validation
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Paper2Code Golden Paper Set APIs
# ==========================================

def get_paper_info(paper) -> tuple[str, str, str, str]:
    arch_graph = paper.architecture_graph or {}
    classification = arch_graph.get("classification")
    status = arch_graph.get("status", "Published")
    support = arch_graph.get("support_level", "experimental")
    
    if classification:
        return paper.title, classification, status, support

    model_family = arch_graph.get("model_family")
    if model_family in ("cnn", "resnet"):
        return paper.title, "CNN", status, support
    elif model_family == "unet":
        return paper.title, "Encoder-Decoder", status, support
    elif model_family == "transformer":
        return paper.title, "Transformer", status, support

    title_lower = paper.title.lower()
    if "resnet" in title_lower:
        return paper.title, "CNN", status, support
    elif "transformer" in title_lower:
        return paper.title, "Transformer", status, support
    elif "u-net" in title_lower or "unet" in title_lower:
        return paper.title, "Encoder-Decoder", status, support
    return paper.title, "Unknown", status, support

@app.get("/api/papers")
def list_papers(db: Session = Depends(get_db)):
    from backend.models import Paper
    papers = db.query(Paper).all()
    results = []
    
    categories = {}
    largest_model = {"name": None, "params": 0}
    most_complex_model = {"name": None, "flops": 0}
    total_modules = 0
    
    for p in papers:
        title, arch_type, status, support_level = get_paper_info(p)
        flops_analysis = p.flops_analysis or {}
        
        params = flops_analysis.get("total_params_estimate", 0)
        flops = flops_analysis.get("total_flops_score", 0)
        modules_count = len(p.modules)
        
        # Update stats
        if status != "Draft":
            total_modules += modules_count
            categories[arch_type] = categories.get(arch_type, 0) + 1
            if params > largest_model["params"]:
                largest_model = {"name": title, "params": params}
            if flops > most_complex_model["flops"]:
                most_complex_model = {"name": title, "flops": flops}
        
        results.append({
            "id": p.id,
            "title": title,
            "architecture_type": arch_type,
            "module_count": modules_count,
            "parameter_count": params,
            "flops": flops,
            "status": status,
            "support_level": support_level
        })
        
    return {
        "statistics": {
            "total_papers": sum(1 for p in results if p["status"] != "Draft"),
            "total_modules": total_modules,
            "architecture_categories": categories,
            "largest_model": largest_model["name"],
            "most_complex_model": most_complex_model["name"]
        },
        "papers": results
    }

@app.get("/api/papers/{paper_id}")
def get_paper_details(paper_id: int, db: Session = Depends(get_db)):
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, support_level = get_paper_info(p)
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
            "architecture_type": arch_type,
            "status": status
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
    """Return all modules for a paper with full detail (tensor_summary, flops_context)."""
    from backend.models import Paper, PaperModule
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    total = len(p.modules)

    modules = []
    for m in p.modules:
        modules.append(_module_to_dict(m, paper_id, total))

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

    paper_title, arch_type, status, _ = get_paper_info(m.paper)

    result = _module_to_dict(m, m.paper_id, total)
    result.update({
        "position": pos + 1,          # 1-based for display
        "prev_module_id": prev_id,
        "next_module_id": next_id,
        "paper_title": paper_title,
        "architecture_type": arch_type,
    })
    return result

@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info("Upload Started for file: %s", file.filename)
    if not file.filename.lower().endswith(".pdf"):
        logger.warning("Upload Failed: non-PDF file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Check size (Max 20MB)
    MAX_SIZE = 20 * 1024 * 1024
    if file.size is not None and file.size > MAX_SIZE:
        logger.warning("Upload Failed: File exceeds 20MB limit.")
        raise HTTPException(status_code=400, detail="File exceeds 20MB limit.")
        
    try:
        paper_name = file.filename.replace(".pdf", "")
        
        # 1. Run pipeline using file object directly to prevent OOM
        result_dict = generator.from_pdf(file.file, paper_name)
        
        # 2. Extract components
        spec = result_dict.get("spec", {})
        graph = result_dict["graph"]
        
        # 3. Classify architecture
        from core.classification import classify_architecture
        classification = classify_architecture(graph)
        
        # 4. Generate learning modules
        from core.module_generator import generate_modules
        paper_meta, learning_modules = generate_modules(
            paper_name=paper_name,
            schema=spec,
            pipeline_result=result_dict
        )
        
        # Add metadata fields
        paper_meta["architecture_graph"]["classification"] = classification
        paper_meta["architecture_graph"]["status"] = "Draft"
        
        # 5. Save to database
        from backend.models import Paper, PaperModule
        
        paper = Paper(
            title=paper_name,
            authors=paper_meta.get("authors"),
            abstract=paper_meta.get("abstract"),
            architecture_graph=paper_meta.get("architecture_graph"),
            flops_analysis=paper_meta.get("flops_analysis")
        )
        
        db.add(paper)
        db.commit()
        db.refresh(paper)
        
        for m in learning_modules:
            pm = PaperModule(
                paper_id=paper.id,
                layer_name=m.layer_name,
                module_type=m.module_type,
                explanation=m.explanation,
                tensor_flow=m.tensor_flow,
                graph_nodes=m.graph_nodes,
                flops_context=m.flops_context,
                order_index=m.order_index
            )
            db.add(pm)
        
        db.commit()
        
        logger.info("Upload Completed for file: %s", file.filename)
        return {
            "paper_id": paper.id,
            "title": paper_name,
            "classification": classification,
            "status": "Draft",
            "report": {
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "parameters": paper_meta["flops_analysis"]["total_params_estimate"],
                "flops": paper_meta["flops_analysis"]["total_flops_score"],
                "detected_components": list(set(n.type for n in graph.nodes))
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Upload Failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/papers/{paper_id}/publish")
def publish_paper(paper_id: int, db: Session = Depends(get_db)):
    from backend.models import Paper
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
        
    arch_graph = dict(paper.architecture_graph) if paper.architecture_graph else {}
    arch_graph["status"] = "Published"
    
    from sqlalchemy.orm.attributes import flag_modified
    paper.architecture_graph = arch_graph
    flag_modified(paper, "architecture_graph")
    
    db.commit()
    logger.info("Paper Published: %s", paper.title)
    return {"status": "success", "paper_id": paper.id}

# Tutor Endpoints
from core.agents.tutor_agent import tutor_manager

class TutorAskRequest(BaseModel):
    session_id: str
    context_type: str
    context_data: Dict[str, Any]
    query: str

class TutorQuizRequest(BaseModel):
    module_data: Dict[str, Any]

@app.post("/api/tutor/ask")
def tutor_ask(
    request: TutorAskRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.adaptive_engine import adaptive_engine
        profile = adaptive_engine.compute_knowledge_profile(db, x_learner_id)
        response = tutor_manager.ask(request.session_id, request.context_type, request.context_data, request.query, knowledge_profile=profile)
        
        # Log tutor analytics automatically
        from backend.models import TutorAnalytics
        import datetime
        arch = request.context_data.get("paper_title") or request.context_data.get("title") or "General"
        mod = request.context_data.get("layer_name") or "general"
        reasoning_type = response.get("reasoning_type", "General")
        
        existing = db.query(TutorAnalytics).filter(
            TutorAnalytics.learner_id == x_learner_id,
            TutorAnalytics.architecture == arch,
            TutorAnalytics.module == mod
        ).first()
        if existing:
            existing.question_count += 1
            existing.created_at = datetime.datetime.utcnow()
        else:
            t_anal = TutorAnalytics(
                learner_id=x_learner_id,
                architecture=arch,
                module=mod,
                reasoning_type=reasoning_type,
                question_count=1
            )
            db.add(t_anal)
        db.commit()
        
        return response
    except Exception as e:
        logger.error(f"Tutor ask error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tutor/quiz")
def tutor_quiz(
    request: TutorQuizRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.adaptive_engine import adaptive_engine
        weaknesses = adaptive_engine.detect_weaknesses(db, x_learner_id)
        weak_list = [w["topic"] for w in weaknesses["weak_topics"] if w["weakness_score"] > 0.2]
        
        response = tutor_manager.generate_quiz(request.module_data, weak_topics=weak_list)
        return {"questions": response}
    except Exception as e:
        logger.error(f"Tutor quiz error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tutor/learning-path")
def tutor_learning_path(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.adaptive_engine import adaptive_engine
        path = adaptive_engine.get_adaptive_learning_path(db, x_learner_id)
        return {"learning_path": path}
    except Exception as e:
        logger.error(f"Tutor learning path error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Phase 9: Adaptive Learning & Personalization APIs
# ==========================================

@app.get("/api/adaptive/recommendations")
def get_adaptive_recommendations(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.adaptive_engine import adaptive_engine
        recs = adaptive_engine.get_personalized_recommendations(db, x_learner_id)
        return recs
    except Exception as e:
        logger.error(f"Adaptive recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adaptive/review-plan")
def get_adaptive_review_plan(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.adaptive_engine import adaptive_engine
        plan = adaptive_engine.get_daily_review_plan(db, x_learner_id)
        return plan
    except Exception as e:
        logger.error(f"Adaptive review plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/adaptive/concept-graph")
def get_adaptive_concept_graph(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.adaptive_engine import adaptive_engine
        graph_nodes = adaptive_engine.get_concept_graph(db, x_learner_id)
        return {"nodes": graph_nodes}
    except Exception as e:
        logger.error(f"Concept graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Phase 8: Assessment and Analytics APIs
# ==========================================

class ValidateRequest(BaseModel):
    challenge: Dict[str, Any]
    user_answer: str

class ProgressUpdateRequest(BaseModel):
    paper_id: int
    module_id: int
    status: str

@app.get("/api/assessment/challenge")
def get_assessment_challenge(
    type: str = "tensor",
    arch: str = "ResNet",
    difficulty: str = "beginner",
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.assessment.engine import assessment_engine
        
        db_metrics = []
        if type == "comparison":
            from backend.models import Paper
            papers = db.query(Paper).all()
            for p in papers:
                flops_analysis = p.flops_analysis or {}
                arch_graph = p.architecture_graph or {}
                classification = arch_graph.get("classification", "Unknown")
                db_metrics.append({
                    "title": p.title,
                    "architecture_type": classification,
                    "parameter_count": flops_analysis.get("total_params_estimate", 0),
                    "flops": flops_analysis.get("total_flops_score", 0),
                    "module_count": len(p.modules) if p.modules else 0
                })
                
        challenge = assessment_engine.generate(
            challenge_type=type,
            architecture=arch,
            difficulty=difficulty,
            db_metrics=db_metrics
        )
        return challenge
    except Exception as e:
        logger.error(f"Assessment challenge generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/assessment/validate")
def validate_assessment_challenge(
    request: ValidateRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        import datetime
        from core.assessment.engine import assessment_engine
        from backend.models import AssessmentAttempt
        
        challenge = request.challenge
        user_ans = request.user_answer
        val_res = assessment_engine.validate(challenge, user_ans)
        
        question_text = challenge.get("question", "")
        existing = (
            db.query(AssessmentAttempt)
            .filter(
                AssessmentAttempt.learner_id == x_learner_id,
                AssessmentAttempt.question_text == question_text
            )
            .first()
        )
        
        if existing:
            existing.attempt_count += 1
            existing.user_answer = user_ans
            existing.is_correct = val_res["is_correct"]
            existing.score = val_res["score"]
            existing.created_at = datetime.datetime.utcnow()
            db.commit()
            db.refresh(existing)
        else:
            attempt = AssessmentAttempt(
                learner_id=x_learner_id,
                assessment_type=challenge.get("assessment_type", "unknown"),
                architecture=challenge.get("architecture"),
                difficulty=challenge.get("difficulty"),
                question_text=question_text,
                user_answer=user_ans,
                correct_answer=val_res["correct_answer"],
                explanation=val_res["explanation"],
                score=val_res["score"],
                attempt_count=1,
                is_correct=val_res["is_correct"]
            )
            db.add(attempt)
            db.commit()
            
        return val_res
    except Exception as e:
        logger.error(f"Assessment validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/progress/update")
def update_learner_progress(
    request: ProgressUpdateRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        import datetime
        from backend.models import LearnerProgress
        
        progress = (
            db.query(LearnerProgress)
            .filter(
                LearnerProgress.learner_id == x_learner_id,
                LearnerProgress.paper_id == request.paper_id,
                LearnerProgress.module_id == request.module_id
            )
            .first()
        )
        
        now = datetime.datetime.utcnow()
        if progress:
            progress.status = request.status
            if request.status == "completed" and not progress.completed_at:
                progress.completed_at = now
            elif request.status == "in_progress" and not progress.started_at:
                progress.started_at = now
            db.commit()
            db.refresh(progress)
        else:
            progress = LearnerProgress(
                learner_id=x_learner_id,
                paper_id=request.paper_id,
                module_id=request.module_id,
                status=request.status,
                started_at=now if request.status in ("in_progress", "completed") else None,
                completed_at=now if request.status == "completed" else None
            )
            db.add(progress)
            db.commit()
            
        return {"status": "success", "progress_id": progress.id}
    except Exception as e:
        logger.error(f"Progress update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/dashboard")
def get_analytics_dashboard(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from backend.models import Paper, LearnerProgress, AssessmentAttempt, TutorAnalytics, PaperModule
        from collections import defaultdict
        
        # Overview
        papers = db.query(Paper).all()
        total_papers = len(papers)
        
        progress_records = db.query(LearnerProgress).filter(LearnerProgress.learner_id == x_learner_id).all()
        progress_by_module = {p.module_id: p.status for p in progress_records}
        
        papers_started = 0
        papers_completed = 0
        modules_completed = 0
        total_modules_count = 0
        
        for p in papers:
            p_mods = p.modules
            if not p_mods:
                continue
            total_modules_count += len(p_mods)
            mod_statuses = [progress_by_module.get(m.id, "not_started") for m in p_mods]
            
            if any(s in ("in_progress", "completed") for s in mod_statuses):
                papers_started += 1
            if all(s == "completed" for s in mod_statuses):
                papers_completed += 1
            modules_completed += sum(1 for s in mod_statuses if s == "completed")
            
        completion_pct = round(modules_completed / total_modules_count * 100, 1) if total_modules_count > 0 else 0.0
        
        overview = {
            "total_papers": total_papers,
            "papers_started": papers_started,
            "papers_completed": papers_completed,
            "modules_completed": modules_completed,
            "total_modules": total_modules_count,
            "completion_percentage": completion_pct
        }
        
        # Assessment
        attempts = db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == x_learner_id).all()
        total_attempts = len(attempts)
        correct_attempts = sum(1 for a in attempts if a.is_correct)
        accuracy_pct = round(correct_attempts / total_attempts * 100, 1) if total_attempts > 0 else 0.0
        
        arch_attempts = defaultdict(list)
        for a in attempts:
            if a.architecture:
                arch_attempts[a.architecture].append(a.is_correct)
        
        strongest_arch = "None"
        weakest_arch = "None"
        if arch_attempts:
            arch_accuracies = {}
            for arch, results in arch_attempts.items():
                arch_accuracies[arch] = sum(1 for r in results if r) / len(results)
            sorted_archs = sorted(arch_accuracies.items(), key=lambda x: x[1])
            weakest_arch = f"{sorted_archs[0][0]} ({int(sorted_archs[0][1]*100)}% accuracy)"
            strongest_arch = f"{sorted_archs[-1][0]} ({int(sorted_archs[-1][1]*100)}% accuracy)"
            
        assessment = {
            "total_attempts": total_attempts,
            "accuracy_percentage": accuracy_pct,
            "strongest_architecture": strongest_arch,
            "weakest_architecture": weakest_arch
        }
        
        # Tutor
        tutor_records = db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == x_learner_id).all()
        questions_asked = sum(r.question_count for r in tutor_records)
        
        context_counts = defaultdict(int)
        for r in tutor_records:
            if r.architecture:
                context_counts[r.architecture] += r.question_count
        most_asked_context = "None"
        if context_counts:
            most_asked_context = max(context_counts.items(), key=lambda x: x[1])[0]
            
        tutor = {
            "questions_asked": questions_asked,
            "most_asked_context": most_asked_context
        }
        
        # Learning Path
        active_sorted = sorted(
            [p for p in progress_records if p.started_at],
            key=lambda x: x.started_at,
            reverse=True
        )
        current_position = "Get started by reading a paper!"
        next_recommended = "None"
        if active_sorted:
            latest = active_sorted[0]
            m = db.query(PaperModule).filter(PaperModule.id == latest.module_id).first()
            p = db.query(Paper).filter(Paper.id == latest.paper_id).first()
            if m and p:
                current_position = f"{p.title} - {m.layer_name}"
                next_mod = (
                    db.query(PaperModule)
                    .filter(PaperModule.paper_id == p.id, PaperModule.order_index > m.order_index)
                    .order_by(PaperModule.order_index.asc())
                    .first()
                )
                if next_mod:
                    next_recommended = f"{p.title} - {next_mod.layer_name}"
                else:
                    next_paper = (
                        db.query(Paper)
                        .filter(Paper.id > p.id)
                        .order_by(Paper.id.asc())
                        .first()
                    )
                    if next_paper and next_paper.modules:
                        next_recommended = f"{next_paper.title} - {next_paper.modules[0].layer_name}"
                    else:
                        next_recommended = "You have completed all available papers!"
        elif papers:
            first_paper = papers[0]
            if first_paper.modules:
                next_recommended = f"{first_paper.title} - {first_paper.modules[0].layer_name}"
                
        learning_path = {
            "current_position": current_position,
            "next_recommended": next_recommended
        }

        # Build learning_path_items as an array for frontend iteration
        learning_path_items = []
        if current_position != "Get started by reading a paper!":
            learning_path_items.append({
                "title": current_position,
                "url": "",
                "type": "current",
            })
        if next_recommended not in ("None", "You have completed all available papers!"):
            learning_path_items.append({
                "title": next_recommended,
                "url": "",
                "type": "next",
            })

        # Reviews
        from core.analytics.recommendation_engine import recommendation_engine
        recs = recommendation_engine.compute(db, x_learner_id)

        return {
            "learning_overview": overview,
            "assessment_performance": assessment,
            "tutor_usage": tutor,
            "learning_path": learning_path,
            "learning_path_items": learning_path_items,
            "reviews_and_recommendations": recs
        }
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/recommendations")
def get_analytics_recommendations(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        from core.analytics.recommendation_engine import recommendation_engine
        recs = recommendation_engine.compute(db, x_learner_id)
        return recs
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Phase 10: Research Engineer Mode APIs
# ==========================================

from core.implementation.code_mapper import get_module_implementation, get_architecture_implementation
from core.implementation.training_config import get_training_config, get_hyperparameter_explanations
from core.implementation.cost_estimator import estimate_training_cost
from core.implementation.reproduction_cards import get_reproduction_card


@app.get("/api/implementation/{paper_id}")
def get_architecture_implementation_view(paper_id: int, db: Session = Depends(get_db)):
    """Full architecture-to-code view: per-module PyTorch mappings for a paper."""
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    modules_data = []
    for m in p.modules:
        modules_data.append({
            "id": m.id,
            "layer_name": m.layer_name,
            "module_type": m.module_type,
            "explanation": m.explanation or "",
            "graph_nodes": m.graph_nodes if isinstance(m.graph_nodes, list) else [],
        })

    impl_view = get_architecture_implementation(
        paper_title=title,
        classification=arch_type,
        modules=modules_data,
    )
    impl_view["paper_id"] = paper_id
    impl_view["architecture_graph"] = p.architecture_graph or {}
    return impl_view


@app.get("/api/modules/{module_id}/implementation")
def get_module_implementation_view(module_id: int, db: Session = Depends(get_db)):
    """Single module implementation: explanation + pseudocode + PyTorch snippet."""
    from backend.models import PaperModule
    m = db.query(PaperModule).filter(PaperModule.id == module_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Module not found")

    # Extract params from graph nodes if available
    raw_graph = m.graph_nodes
    params = {}
    if isinstance(raw_graph, list) and raw_graph:
        params = raw_graph[0].get("params", {})
    elif isinstance(raw_graph, dict):
        nodes = raw_graph.get("nodes", [])
        if nodes:
            params = nodes[0].get("params", {})

    impl = get_module_implementation(m.module_type or "", params)
    paper_title = m.paper.title if m.paper else "Unknown"

    return {
        "module_id": module_id,
        "layer_name": m.layer_name,
        "module_type": m.module_type,
        "paper_title": paper_title,
        "explanation": m.explanation or "",
        "implementation": impl,
    }


@app.get("/api/training/{paper_id}")
def get_training_pipeline(paper_id: int, db: Session = Depends(get_db)):
    """Training pipeline config for a paper's architecture classification."""
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    config = get_training_config(arch_type)
    return {
        "paper_id": paper_id,
        "paper_title": title,
        "architecture_type": arch_type,
        "training_config": config,
    }


@app.get("/api/hyperparameters")
def get_hyperparameters():
    """All hyperparameter explanation cards with increase/decrease effects."""
    return {"hyperparameters": get_hyperparameter_explanations()}


# ==========================================
# Phase 12 (M1): Code Dojo — implement-it-yourself coding exercises
# ==========================================

from core.dojo import get_exercise_list, get_public_exercise, get_solution


class DojoSubmitRequest(BaseModel):
    exercise_id: str
    passed: bool
    attempts: int = 1


@app.get("/api/dojo/exercises")
def dojo_list_exercises():
    """Lightweight catalog for the Dojo sidebar (no solutions / expected outputs)."""
    return {"exercises": get_exercise_list()}


@app.get("/api/dojo/exercises/{exercise_id}")
def dojo_get_exercise(exercise_id: str):
    """
    Full exercise for the workspace: includes test_inputs + expected_outputs
    (precomputed from the hidden reference solution) but NOT the solution itself.
    """
    ex = get_public_exercise(exercise_id)
    if not ex:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return ex


@app.get("/api/dojo/exercises/{exercise_id}/solution")
def dojo_get_solution(exercise_id: str):
    """Reveal the reference solution (separate endpoint so it stays hidden by default)."""
    sol = get_solution(exercise_id)
    if not sol:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return sol


@app.post("/api/dojo/submit")
def dojo_submit(
    request: DojoSubmitRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    """
    Record a Dojo completion as an AssessmentAttempt with assessment_type='code'
    so it flows into the existing analytics / adaptive engine.
    """
    if not get_solution(request.exercise_id):
        raise HTTPException(status_code=404, detail=f"Exercise '{request.exercise_id}' not found")
    try:
        from backend.models import AssessmentAttempt
        attempt = AssessmentAttempt(
            learner_id=x_learner_id or "default",
            assessment_type="code",
            architecture=None,
            difficulty=None,
            question_text=f"dojo:{request.exercise_id}",
            user_answer="passed" if request.passed else "failed",
            correct_answer="passed",
            explanation="",
            score=1 if request.passed else 0,
            attempt_count=request.attempts,
            is_correct=bool(request.passed),
        )
        db.add(attempt)
        db.commit()
        return {"status": "ok", "recorded": True, "exercise_id": request.exercise_id, "passed": request.passed}
    except Exception as e:
        db.rollback()
        logger.error(f"Dojo submit error: {str(e)}")
        # Non-fatal: client already validated; return ok so UX is not blocked.
        return {"status": "ok", "recorded": False, "detail": str(e)}


class CostEstimatorRequest(BaseModel):
    architecture: str
    dataset_size: int = 1000000
    batch_size: int = 32
    gpu_type: str = "A100"
    epochs: Optional[int] = None
    mixed_precision: bool = False
    gradient_checkpointing: bool = False


@app.post("/api/training-estimator")
def training_cost_estimator(request: CostEstimatorRequest):
    """Training cost estimation from architecture + hardware inputs."""
    try:
        result = estimate_training_cost(
            architecture=request.architecture,
            dataset_size=request.dataset_size,
            batch_size=request.batch_size,
            gpu_type=request.gpu_type,
            epochs=request.epochs,
            mixed_precision=request.mixed_precision,
            gradient_checkpointing=request.gradient_checkpointing,
        )
        return result
    except Exception as e:
        logger.error(f"Cost estimator error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reproduction/{paper_id}")
def get_reproduction_card_view(paper_id: int, db: Session = Depends(get_db)):
    """Research reproduction card for a paper's architecture."""
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    title, arch_type, status, _ = get_paper_info(p)
    card = get_reproduction_card(arch_type)
    return {
        "paper_id": paper_id,
        "paper_title": title,
        "architecture_type": arch_type,
        "reproduction_card": card,
    }


# ==========================================
# Phase 11: Research Lab & Experiment Studio APIs
# ==========================================

from core.lab.diff_engine import compute_diff
from core.lab.mutator import apply_mutations, MUTATION_REGISTRY
from core.lab.hypothesis_engine import hypothesis_engine
from core.lab.tradeoff_analyzer import tradeoff_scatter, get_efficiency_frontiers, get_tradeoff_summary


class LabMutateRequest(BaseModel):
    architecture: str                    # "ResNet" | "Transformer" | "U-Net"
    mutations: list[Dict[str, Any]]      # [{type, params}, ...]
    config: Dict[str, Any] = {}          # optional base config overrides


class LabPredictRequest(BaseModel):
    architecture: str
    mutations: list[Dict[str, Any]]
    hypothesis: Dict[str, Any]           # {predicted_param_direction, predicted_flops_direction, prediction, mutation_type}


class LabExperimentRequest(BaseModel):
    architecture: str
    mutations: list[Dict[str, Any]]
    hypothesis: Optional[Dict[str, Any]] = None


def _build_base_graph(architecture: str, config: Dict[str, Any] = {}):
    """Build a baseline graph for the given architecture with optional config overrides."""
    if architecture == "ResNet":
        return build_resnet18_graph(
            base_channels=config.get("base_channels", 64),
            stages=config.get("stages", 4),
            blocks_per_stage=config.get("blocks_per_stage", 2),
        )
    elif architecture in ("Transformer", "ViT"):
        return build_vit_graph(
            hidden_size=config.get("hidden_size", 768),
            num_heads=config.get("num_heads", 12),
            depth=config.get("depth", 4),
        )
    elif architecture == "U-Net":
        return build_unet_graph(
            base_channels=config.get("base_channels", 64),
            stages=config.get("stages", 3),
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported architecture: '{architecture}'. Valid: ResNet, Transformer, U-Net")


def _graph_to_api_dict(graph) -> Dict[str, Any]:
    """Serialize a graph to a JSON-safe API dict."""
    return {
        "name": graph.name,
        "nodes": [
            {
                "id": n.id,
                "type": n.type,
                "label": n.label,
                "params": n.params or {},
                "block": n.block,
                "semantic_params": n.semantic_params or {},
            }
            for n in graph.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target, "edge_type": e.edge_type}
            for e in graph.edges
        ],
    }


def _graph_metrics(graph) -> Dict[str, Any]:
    from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
    m = estimate_metrics_from_graph(graph)
    mem = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
    mem_mb = round(sum(r["mem_mb"] for r in mem), 2)
    return {
        "flops_score": m["total_flops_score"],
        "params": m["total_params_estimate"],
        "depth": m["depth"],
        "memory_mb": mem_mb,
    }


@app.post("/api/lab/mutate")
def lab_mutate(request: LabMutateRequest):
    """
    Apply a list of mutations to a baseline architecture.
    Returns before graph, after graph, diff, and metrics for both.
    """
    try:
        before_graph = _build_base_graph(request.architecture, request.config)
        after_graph  = apply_mutations(before_graph, request.mutations)

        diff = compute_diff(before_graph, after_graph)

        return {
            "architecture": request.architecture,
            "mutations_applied": request.mutations,
            "before": {
                "graph": _graph_to_api_dict(before_graph),
                "metrics": _graph_metrics(before_graph),
            },
            "after": {
                "graph": _graph_to_api_dict(after_graph),
                "metrics": _graph_metrics(after_graph),
            },
            "diff": diff,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab mutate error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lab/predict")
def lab_predict(request: LabPredictRequest):
    """
    Score a user's prediction against the actual computed diff.
    Implements the 'Predict Before Reveal' challenge workflow.
    """
    try:
        before_graph = _build_base_graph(request.architecture)
        after_graph  = apply_mutations(before_graph, request.mutations)
        diff         = compute_diff(before_graph, after_graph)

        # Enrich hypothesis with mutation_type from first mutation if missing
        hyp = dict(request.hypothesis)
        if "mutation_type" not in hyp and request.mutations:
            hyp["mutation_type"] = request.mutations[0].get("type", "")

        score_result = hypothesis_engine.score_prediction(hyp, diff)

        return {
            "architecture": request.architecture,
            "mutations_applied": request.mutations,
            "diff": diff,
            "before_metrics": _graph_metrics(before_graph),
            "after_metrics":  _graph_metrics(after_graph),
            "scoring": score_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab predict error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/lab/experiment")
def lab_experiment(request: LabExperimentRequest):
    """
    Run a full experiment: apply mutations, compute diff, optionally score hypothesis.
    Returns an ExperimentResult ready for the Research Notebook.
    """
    try:
        before_graph = _build_base_graph(request.architecture)
        after_graph  = apply_mutations(before_graph, request.mutations)
        diff         = compute_diff(before_graph, after_graph)

        result = hypothesis_engine.build_experiment_result(
            hypothesis=request.hypothesis,
            mutations_applied=request.mutations,
            architecture=request.architecture,
            actual_diff=diff,
        )

        return {
            "experiment": result,
            "diff": diff,
            "before_metrics": _graph_metrics(before_graph),
            "after_metrics":  _graph_metrics(after_graph),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab experiment error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lab/tradeoffs")
def lab_tradeoffs(architecture: str = "ResNet"):
    """
    Return scatter-plot data across a grid of mutations for the Tradeoff Explorer.
    Also returns efficiency frontier and tradeoff summaries.
    """
    try:
        base_graph = _build_base_graph(architecture)
        points     = tradeoff_scatter(base_graph, architecture)
        frontier   = get_efficiency_frontiers(points)

        # Include per-mutation tradeoff summaries
        summaries = {mut: get_tradeoff_summary(mut) for mut in MUTATION_REGISTRY.keys()}

        return {
            "architecture": architecture,
            "scatter_points": frontier,
            "tradeoff_summaries": summaries,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab tradeoffs error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lab/prediction-prompt")
def lab_prediction_prompt(mutation_type: str = "increase_depth", architecture: str = "ResNet"):
    """
    Return a structured prediction challenge prompt for the 'Predict Before Reveal' UI.
    """
    try:
        prompt = hypothesis_engine.generate_prediction_prompt(mutation_type, architecture)
        return prompt
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lab/mutations")
def lab_mutation_list():
    """Return the list of all supported mutation types with descriptions."""
    descriptions = {
        "increase_depth":    {"label": "Increase Depth",       "icon": "fa-layer-group",     "description": "Add extra blocks to deepen the network", "category": "depth"},
        "decrease_depth":    {"label": "Decrease Depth",       "icon": "fa-compress-alt",    "description": "Remove blocks to create a shallower network", "category": "depth"},
        "increase_width":    {"label": "Increase Width",       "icon": "fa-expand",          "description": "Scale up channel sizes (×1.5)", "category": "width"},
        "decrease_width":    {"label": "Decrease Width",       "icon": "fa-compress",        "description": "Scale down channel sizes (×0.5)", "category": "width"},
        "add_residual":      {"label": "Add Skip Connections", "icon": "fa-code-branch",     "description": "Inject residual skip connections between blocks", "category": "structure"},
        "remove_residual":   {"label": "Remove Skip Connections","icon": "fa-ban",            "description": "Remove all skip/residual edges", "category": "structure"},
        "add_attention":     {"label": "Add Attention Block",  "icon": "fa-brain",           "description": "Insert a Multi-Head Attention node", "category": "attention"},
        "change_patch_size": {"label": "Change Patch Size",    "icon": "fa-th",              "description": "Change ViT patch size (affects token count)", "category": "embedding"},
        "change_hidden_dim": {"label": "Change Hidden Dim",    "icon": "fa-sliders",         "description": "Adjust transformer hidden dimension (d_model)", "category": "embedding"},
    }
    return {"mutations": descriptions}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)

