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
from backend.models import Paper
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


router = APIRouter(prefix="/api", tags=["Papers Analysis"])

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


def _assert_paper_readable(p, current_user) -> None:
    """Raise 403 if the paper is private and the caller is not the owner."""
    if p.visibility == "private":
        if not current_user or current_user.id != p.uploaded_by:
            raise HTTPException(status_code=403, detail="Access denied")


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


@router.post("/papers/text-parse")
# deprecated alias
@router.post("/parse_text", deprecated=True)
def parse_text(request: TextRequest):
    try:
        spec = extractor.extract_from_text(request.text)
        result = pipeline.run_single(spec)
        code, code_source = generator._generate_code(spec, result["graph"])
        return _build_response(spec, result, code, code_source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/papers/text-compare")
# deprecated alias
@router.post("/compare_text", deprecated=True)
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

@router.post("/papers/graph-analysis")
# deprecated alias
@router.post("/analyze_graph", deprecated=True)
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



# ---------------------------------------------------------------------------
# GET /api/papers/upload-url  — presigned R2 PUT URL for direct client upload
# IMPORTANT: must be registered BEFORE GET /papers/{paper_id} to avoid 422
# ---------------------------------------------------------------------------






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


# ---------------------------------------------------------------------------
# POST /api/papers/confirm-upload  — confirm direct upload, queue processing
# ---------------------------------------------------------------------------

class ConfirmUploadRequest(BaseModel):
    key: str
    paper_name: str
    visibility: str = "public"
    terms_accepted: bool = False
    file_size_bytes: int = 0




# ---------------------------------------------------------------------------
# GET /api/papers/{paper_id}/download  — presigned R2 download URL (1h expiry)
# ---------------------------------------------------------------------------













# ---------------------------------------------------------------------------
# PATCH /api/papers/{id}/visibility  — change visibility (P0, ownership req.)
# ---------------------------------------------------------------------------

class VisibilityUpdate(BaseModel):
    visibility: str




# ---------------------------------------------------------------------------
# DELETE /api/papers/{id}  — delete own paper, R2 object, free quota (P1)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# POST /api/papers/{id}/flag  — user reports a paper for review (P1)
# ---------------------------------------------------------------------------

class PaperFlagRequest(BaseModel):
    reason: str = "inappropriate"




# ---------------------------------------------------------------------------
# GET /api/papers/{id}/similar  — architecture-type similarity (P2)
# ---------------------------------------------------------------------------


class PaperAskRequest(BaseModel):
    question: str



