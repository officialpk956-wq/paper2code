import logging
import os
import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
from sqlalchemy.orm import Session
from sqlalchemy import func
import dataclasses
import base64

from backend.database import get_db
from backend.services.paper_ingestion_service import ingest_pdf_paper
from core.orchestrator.pipeline import Paper2CodePipeline
from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.config_extractor import ConfigExtractor
from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
from core.explainers.graph_explainer import explain_node
from core.codegen import _node_to_layer
from backend.server import limiter
from backend.dependencies import get_current_user, get_optional_user
from backend.repositories.task_repository import TaskRepository
from backend.tasks.paper_tasks import generate_code_from_pdf_task
from backend.models import Task

logger = logging.getLogger(__name__)

_PAPER_MONTHLY_LIMIT = int(os.getenv("PAPER_MONTHLY_LIMIT", "0"))  # 0 = unlimited
_STORAGE_QUOTA_BYTES = int(os.getenv("STORAGE_QUOTA_MB", "500")) * 1024 * 1024  # default 500 MB


def _check_paper_quota(db: Session, user_id: int) -> None:
    """Raise 429 if the user has exceeded their monthly paper upload quota."""
    if _PAPER_MONTHLY_LIMIT <= 0:
        return
    month_start = datetime.datetime.utcnow().replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    count = (
        db.query(func.count(Task.id))
        .filter(
            Task.user_id == user_id,
            Task.type == "paper.codegen",
            Task.created_at >= month_start,
        )
        .scalar()
        or 0
    )
    if count >= _PAPER_MONTHLY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly paper limit reached ({_PAPER_MONTHLY_LIMIT} papers/month). "
                "Upgrade your plan for unlimited uploads."
            ),
        )

def _check_storage_quota(db: Session, user_id: int, additional_bytes: int = 0) -> None:
    """Raise 429 if user would exceed their storage quota."""
    if _STORAGE_QUOTA_BYTES <= 0:
        return
    from backend.models import User
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        return
    used = (user.storage_bytes_used or 0) + additional_bytes
    if used > _STORAGE_QUOTA_BYTES:
        limit_mb = _STORAGE_QUOTA_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=429,
            detail=f"Storage quota exceeded ({limit_mb} MB limit). Delete papers to free space.",
        )


router = APIRouter(prefix="/api", tags=["Papers"])

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


def _build_response(spec, result, code, code_source):
    graph = result["graph"]
    metrics = estimate_metrics_from_graph(graph)
    mem_data = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
    total_mem_mb = sum(row['mem_mb'] for row in mem_data) if mem_data else 0
    
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


def safe_dict(val) -> dict:
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
    flops = safe_dict(m.flops_context)
    raw_tensor = m.tensor_flow
    if isinstance(raw_tensor, list):
        tensor: dict = {"trace": raw_tensor}
    else:
        tensor = safe_dict(raw_tensor)

    raw_graph = m.graph_nodes
    if isinstance(raw_graph, list):
        graph_nodes_list: list = raw_graph
        graph_edges_list: list = []
    elif isinstance(raw_graph, dict):
        graph_nodes_list = raw_graph.get("nodes", [])
        graph_edges_list = raw_graph.get("edges", [])
    elif isinstance(raw_graph, str):
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
        "flops_context": {
            "total_flops_score": flops.get("total_flops_score", 0),
            "real_flops_mflops": flops.get("real_flops_mflops", 0.0),
            "total_params_estimate": flops.get("total_params_estimate", 0),
            "depth": flops.get("depth", 0),
            "breakdown": safe_list(flops.get("breakdown", [])),
        },
        "tensor_summary": {
            "input_shape": tensor.get("input_shape") or tensor.get("input"),
            "output_shape": tensor.get("output_shape") or tensor.get("output"),
            "operations": safe_list(tensor.get("operations", [])),
            "trace": safe_list(tensor.get("trace", [])),
        },
        "graph_nodes": graph_nodes_list,
        "graph_edges": graph_edges_list,
    }


@router.post("/parse_text")
def parse_text(request: TextRequest):
    try:
        spec = extractor.extract_from_text(request.text)
        result = pipeline.run_single(spec)
        code, code_source = generator._generate_code(spec, result["graph"])
        return _build_response(spec, result, code, code_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare_text")
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

@router.post("/analyze_graph")
def analyze_graph(request: GraphRequest):
    try:
        from core.rag.normalizer import normalize_config
        config = normalize_config({
            "name": request.name,
            "layers": request.layers
        })
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

@router.get("/papers")
def list_papers(
    q:      Optional[str] = Query(None, min_length=2, max_length=200, description="Search title/abstract"),
    domain: Optional[str] = Query(None, description="Filter by architecture type / domain tag"),
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    from backend.models import Paper
    from sqlalchemy import or_
    query = db.query(Paper)
    if current_user:
        # Authenticated: public + unlisted + own private papers
        query = query.filter(
            or_(
                Paper.visibility != "private",
                Paper.uploaded_by == current_user.id,
            )
        )
    else:
        # Unauthenticated: only public + unlisted
        query = query.filter(Paper.visibility != "private")
    if q:
        pat = f"%{q}%"
        query = query.filter(or_(Paper.title.ilike(pat), Paper.abstract.ilike(pat), Paper.authors.ilike(pat)))
    papers = query.all()
    if domain:
        domain_lower = domain.lower()
        papers = [p for p in papers if domain_lower in get_paper_info(p)[1].lower()]
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


# ---------------------------------------------------------------------------
# GET /api/papers/upload-url  — presigned R2 PUT URL for direct client upload
# IMPORTANT: must be registered BEFORE GET /papers/{paper_id} to avoid 422
# ---------------------------------------------------------------------------

@router.get("/papers/upload-url")
@limiter.limit("20/hour")
async def get_upload_url(
    request: Request,
    filename: str,
    content_type: str = "application/pdf",
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    P0 Fix: Client uploads directly to R2 via presigned PUT URL.
    API server never touches the PDF bytes — only a 50-byte key enters Redis.
    """
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    _check_paper_quota(db, current_user.id)
    _check_storage_quota(db, current_user.id)

    from backend.services.storage_service import generate_presigned_upload_url, R2_AVAILABLE
    if not R2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Direct upload not available — R2 not configured. Use /api/papers/upload instead.",
        )

    return generate_presigned_upload_url(filename, content_type)


@router.get("/papers/{paper_id}")
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

    ingestion_data = (p.architecture_graph or {}).get("ingestion", {})
    return {
        "metadata": {
            "id": p.id,
            "title": title,
            "full_title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "architecture_type": arch_type,
            "status": status,
            "source_filename": ingestion_data.get("source_filename"),
            "figure_count": ingestion_data.get("figure_count", 0),
            "equation_count": ingestion_data.get("equation_count", 0),
        },
        "module_summary": modules_summary,
        "architecture_statistics": {
            "depth": flops_analysis.get("depth", 0),
            "node_count": len(p.architecture_graph.get("nodes", [])) if p.architecture_graph else 0,
            "edge_count": len(p.architecture_graph.get("edges", [])) if p.architecture_graph else 0
        },
        "architecture_graph": p.architecture_graph or {"nodes": [], "edges": []},
        "flops": flops_analysis.get("total_flops_score", 0),
        "parameter_count": flops_analysis.get("total_params_estimate", 0),
        "ingestion": ingestion_data,
    }

@router.get("/papers/{paper_id}/modules")
def list_paper_modules(paper_id: int, db: Session = Depends(get_db)):
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

@router.get("/modules/{module_id}")
def get_module(module_id: int, db: Session = Depends(get_db)):
    from backend.models import PaperModule, Paper
    m = db.query(PaperModule).filter(PaperModule.id == module_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Module not found")

    siblings = (
        db.query(PaperModule)
        .filter(PaperModule.paper_id == m.paper_id)
        .order_by(PaperModule.order_index)
        .all()
    )
    total = len(siblings)
    ids = [s.id for s in siblings]
    pos = ids.index(m.id)

    prev_id = ids[pos - 1] if pos > 0 else None
    next_id = ids[pos + 1] if pos < total - 1 else None

    paper_title, arch_type, status, _ = get_paper_info(m.paper)

    result = _module_to_dict(m, m.paper_id, total)
    result.update({
        "position": pos + 1,
        "prev_module_id": prev_id,
        "next_module_id": next_id,
        "paper_title": paper_title,
        "architecture_type": arch_type,
    })
    return result

MAX_SIZE = 20 * 1024 * 1024  # 20 MB hard cap

_VALID_VISIBILITY = {"public", "unlisted", "private"}

async def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks = []
    received = 0
    while True:
        chunk = await file.read(65536)  # 64 KB chunks
        if not chunk:
            break
        received += len(chunk)
        if received > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {max_bytes // (1024*1024)} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)

@router.post("/papers/upload")
@limiter.limit("10/hour")
async def upload_paper(
    request: Request,
    file: UploadFile = File(...),
    terms_accepted: bool = Form(False),
    visibility: str = Form("public"),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info("Upload Started for file: %s", file.filename)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning("Upload Failed: non-PDF file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if not terms_accepted:
        raise HTTPException(status_code=400, detail="You must accept the Terms of Service to upload a paper.")

    if visibility not in _VALID_VISIBILITY:
        raise HTTPException(status_code=400, detail=f"visibility must be one of: {', '.join(_VALID_VISIBILITY)}")

    # ── Monthly paper quota ───────────────────────────────────────────────────
    _check_paper_quota(db, current_user.id)

    try:
        pdf_bytes = await _read_limited(file, MAX_SIZE)

        from backend.services.storage_service import store_pdf
        paper_name = file.filename.removesuffix(".pdf") if file.filename else "uploaded_paper"
        storage_ref = store_pdf(pdf_bytes, file.filename)

        task = TaskRepository(db).create("paper.codegen", current_user.id, paper_name)
        generate_code_from_pdf_task.delay(
            task.id, storage_ref, paper_name,
            current_user.id, visibility, terms_accepted,
        )

        from backend.services.progress_service import update_user_activity, award_xp
        update_user_activity(db, current_user.id)
        award_xp(db, current_user.id, "paper.uploaded")
        try:
            from backend.services.analytics_service import track
            from backend.services.achievement_service import check_and_award
            track(current_user.id, "paper_uploaded", {"visibility": visibility})
            check_and_award(db, current_user.id, "paper.uploaded")
        except Exception:
            pass

        return {
            "task_id":  task.id,
            "status":   "pending",
            "poll_url": f"/api/tasks/{task.id}",
            "message":  "PDF uploaded. Poll poll_url every 3s for the generated code.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload Failed: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))

# ---------------------------------------------------------------------------
# POST /api/papers/confirm-upload  — confirm direct upload, queue processing
# ---------------------------------------------------------------------------

class ConfirmUploadRequest(BaseModel):
    key: str
    paper_name: str
    visibility: str = "public"
    terms_accepted: bool = False
    file_size_bytes: int = 0


@router.post("/papers/confirm-upload")
@limiter.limit("10/hour")
async def confirm_upload(
    request: Request,
    body: ConfirmUploadRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Confirm that the client has finished uploading to R2 and queue processing."""
    if not body.terms_accepted:
        raise HTTPException(status_code=400, detail="You must accept the Terms of Service.")
    if body.visibility not in _VALID_VISIBILITY:
        raise HTTPException(status_code=400, detail=f"visibility must be one of: {', '.join(_VALID_VISIBILITY)}")
    if not body.key.startswith("papers/"):
        raise HTTPException(status_code=400, detail="Invalid R2 key.")

    _check_paper_quota(db, current_user.id)
    _check_storage_quota(db, current_user.id, additional_bytes=body.file_size_bytes)

    storage_ref = f"r2://{body.key}"
    task = TaskRepository(db).create("paper.codegen", current_user.id, body.paper_name)
    generate_code_from_pdf_task.delay(
        task.id,
        storage_ref,
        body.paper_name,
        current_user.id,
        body.visibility,
        body.terms_accepted,
    )

    # Increment user storage quota counter
    if body.file_size_bytes > 0:
        from backend.models import User
        user = db.query(User).filter_by(id=current_user.id).first()
        if user:
            user.storage_bytes_used = (user.storage_bytes_used or 0) + body.file_size_bytes
            db.commit()

    from backend.services.progress_service import update_user_activity, award_xp
    update_user_activity(db, current_user.id)
    award_xp(db, current_user.id, "paper.uploaded")
    try:
        from backend.services.analytics_service import track
        from backend.services.achievement_service import check_and_award
        track(current_user.id, "paper_uploaded", {"visibility": body.visibility, "via": "presigned"})
        check_and_award(db, current_user.id, "paper.uploaded")
    except Exception:
        pass

    return {
        "task_id": task.id,
        "status": "pending",
        "poll_url": f"/api/tasks/{task.id}",
        "message": "Upload confirmed. Poll poll_url every 3s for generated code.",
    }


# ---------------------------------------------------------------------------
# GET /api/papers/{paper_id}/download  — presigned R2 download URL (1h expiry)
# ---------------------------------------------------------------------------

@router.get("/papers/{paper_id}/download")
async def download_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    """
    Returns a 302 redirect to a time-limited R2 presigned download URL.
    Private papers require ownership. Zero API server bandwidth.
    """
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    if p.visibility == "private":
        if not current_user or current_user.id != p.uploaded_by:
            raise HTTPException(status_code=403, detail="Access denied")

    if not p.r2_key:
        raise HTTPException(status_code=404, detail="PDF not available for download")

    from backend.services.storage_service import presigned_download_url
    url = presigned_download_url(f"r2://{p.r2_key}", expires_in=3600)
    if not url:
        raise HTTPException(status_code=503, detail="Download not available — R2 not configured")

    return RedirectResponse(url=url, status_code=302)


@router.get("/papers/{paper_id}/blueprint")
def get_architecture_blueprint(paper_id: int, db: Session = Depends(get_db)):
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    bp = (p.architecture_graph or {}).get("ingestion", {}).get("architecture_blueprint")
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not generated yet")
    return bp


@router.get("/papers/{paper_id}/executable-graph")
def get_executable_graph(paper_id: int, db: Session = Depends(get_db)):
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    eg = (p.architecture_graph or {}).get("ingestion", {}).get("executable_graph")
    if not eg:
        raise HTTPException(status_code=404, detail="Executable graph not generated yet")
    return eg


@router.get("/papers/{paper_id}/graph-export")
def export_paper_graph(
    paper_id: int,
    format: str = "json",
    db: Session = Depends(get_db),
):
    from backend.models import Paper
    from backend.services.architecture_graph_compiler import (
        ExecutableGraph, export_graph_json, export_graph_mermaid, export_graph_dot
    )
    from fastapi.responses import PlainTextResponse

    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    eg_dict = (p.architecture_graph or {}).get("ingestion", {}).get("executable_graph")
    if not eg_dict:
        raise HTTPException(status_code=404, detail="Executable graph not generated yet")

    if format not in ("json", "mermaid", "dot"):
        raise HTTPException(status_code=400, detail="format must be json, mermaid, or dot")

    graph = ExecutableGraph.from_dict(eg_dict)
    if format == "json":
        return PlainTextResponse(export_graph_json(graph), media_type="application/json")
    elif format == "mermaid":
        return PlainTextResponse(export_graph_mermaid(graph), media_type="text/plain")
    else:
        return PlainTextResponse(export_graph_dot(graph), media_type="text/plain")


@router.get("/papers/{paper_id}/knowledge-graph")
def get_knowledge_graph(paper_id: int, db: Session = Depends(get_db)):
    from backend.models import Paper
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    kg = (p.architecture_graph or {}).get("ingestion", {}).get("knowledge_graph", {})
    return {"nodes": kg.get("nodes", []), "edges": kg.get("edges", [])}


@router.post("/papers/{paper_id}/publish")
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
