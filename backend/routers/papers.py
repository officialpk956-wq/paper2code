import datetime
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user, get_optional_user
from backend.models import Paper, Task
from core.codegen import _node_to_layer
from core.explainers.graph_explainer import explain_node
from core.metrics_estimator import estimate_activation_memory, estimate_metrics_from_graph
from core.orchestrator.pipeline import Paper2CodePipeline
from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.config_extractor import ConfigExtractor

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
    text: str = Field(..., max_length=50000)


class CompareRequest(BaseModel):
    text_a: str = Field(..., max_length=50000)
    text_b: str = Field(..., max_length=50000)


class GraphRequest(BaseModel):
    name: str = Field(default="Architect Session", max_length=500)
    layers: list[dict[str, Any]]


def _build_response(spec, result, code, code_source):
    graph = result["graph"]
    metrics = estimate_metrics_from_graph(graph)
    mem_data = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
    total_mem_mb = sum(row["mem_mb"] for row in mem_data) if mem_data else 0

    layer_breakdown = []
    for node in graph.nodes:
        node_code = _node_to_layer(node) or f"# Custom block implementation needed for {node.type}"
        layer_breakdown.append(
            {
                "id": node.id,
                "label": node.label,
                "type": node.type,
                "params": node.params,
                "semantic": node.semantic_params,
                "description": node.description,
                "explanation": explain_node(node),
                "code_snippet": node_code,
            }
        )

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
            "memory_mb": round(total_mem_mb, 1),
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


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


@router.get("/papers")
def list_papers(
    q: str | None = Query(None, min_length=2, max_length=200, description="Search title/abstract"),
    domain: str | None = Query(None, description="Filter by architecture type / domain tag"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    import json

    from backend.redis_config import cache_redis

    user_id = current_user.id if current_user else None
    cache_key = f"papers:list:{q}:{domain}:{user_id}"

    if cache_redis:
        cached = cache_redis.get(cache_key)
        if cached:
            return json.loads(cached)

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
        from backend.services.vector_service import semantic_search

        paper_ids = semantic_search(q, limit=20)
        if paper_ids:
            # Reorder SQL results to match Qdrant ranking
            from sqlalchemy.sql.expression import case

            ordering = case({_id: index for index, _id in enumerate(paper_ids)}, value=Paper.id)
            query = query.filter(Paper.id.in_(paper_ids)).order_by(ordering)
        else:
            # Fallback to ILIKE if vector search returns nothing (or fails)
            pat = f"%{q}%"
            query = query.filter(
                or_(Paper.title.ilike(pat), Paper.abstract.ilike(pat), Paper.authors.ilike(pat))
            )
    from sqlalchemy.orm import selectinload

    papers = query.options(selectinload(Paper.modules)).offset(skip).limit(limit).all()
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

        results.append(
            {
                "id": p.id,
                "title": title,
                "authors": p.authors,
                "abstract": p.abstract,
                "visibility": p.visibility,
                "uploaded_by": p.uploaded_by,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "architecture_type": arch_type,
                "module_count": modules_count,
                "parameter_count": params,
                "flops": flops,
                "status": status,
                "support_level": support_level,
            }
        )

    res = {
        "summary": {
            "total_papers": sum(1 for p in results if p["status"] != "Draft"),
            "total_modules": total_modules,
            "architecture_categories": categories,
            "largest_model": largest_model["name"],
            "most_complex_model": most_complex_model["name"],
        },
        "papers": results,
    }

    if cache_redis:
        cache_redis.setex(cache_key, 30, json.dumps(res))

    return res


# ---------------------------------------------------------------------------
# GET /api/papers/upload-url  — presigned R2 PUT URL for direct client upload
# IMPORTANT: must be registered BEFORE GET /papers/{paper_id} to avoid 422
# ---------------------------------------------------------------------------


@router.get("/papers/{paper_id}")
def get_paper_details(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)

    title, arch_type, status, support_level = get_paper_info(p)
    flops_analysis = p.flops_analysis or {}

    modules_summary = []
    for m in p.modules:
        modules_summary.append(
            {
                "id": m.id,
                "order_index": m.order_index,
                "layer_name": m.layer_name,
                "module_type": m.module_type,
                "explanation": m.explanation,
                "tensor_flow": m.tensor_flow,
                "graph_nodes": m.graph_nodes,
                "flops_context": m.flops_context,
            }
        )

    ingestion_data = (p.architecture_graph or {}).get("ingestion", {})
    return {
        "metadata": {
            "id": p.id,
            "title": title,
            "full_title": p.title,
            "authors": p.authors,
            "abstract": p.abstract,
            "visibility": p.visibility,
            "uploaded_by": p.uploaded_by,
            "created_at": p.created_at.isoformat() if p.created_at else None,
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
            "edge_count": len(p.architecture_graph.get("edges", [])) if p.architecture_graph else 0,
        },
        "architecture_graph": p.architecture_graph or {"nodes": [], "edges": []},
        "flops": flops_analysis.get("total_flops_score", 0),
        "parameter_count": flops_analysis.get("total_params_estimate", 0),
        "ingestion": ingestion_data,
    }


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
                detail=f"File exceeds {max_bytes // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# POST /api/papers/confirm-upload  — confirm direct upload, queue processing
# ---------------------------------------------------------------------------


class ConfirmUploadRequest(BaseModel):
    key: str = Field(..., max_length=2048)
    paper_name: str = Field(..., max_length=500)
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


@router.delete("/papers/{paper_id}", status_code=200)
def delete_paper(
    paper_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.models import User as _User

    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if paper.uploaded_by != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your paper")

    # Free storage quota
    if paper.file_size_bytes and paper.uploaded_by:
        owner = db.query(_User).filter_by(id=paper.uploaded_by).first()
        if owner:
            owner.storage_bytes_used = max(
                0, (owner.storage_bytes_used or 0) - paper.file_size_bytes
            )

    # Delete R2 object (best-effort)
    try:
        from backend.services import storage_service

        if paper.r2_key:
            storage_service.cleanup(f"r2://{paper.r2_key}")
    except Exception:
        pass

    db.delete(paper)
    db.commit()
    return {"deleted": True, "paper_id": paper_id}


# ---------------------------------------------------------------------------
# POST /api/papers/{id}/flag  — user reports a paper for review (P1)
# ---------------------------------------------------------------------------


class PaperFlagRequest(BaseModel):
    reason: str = Field(default="inappropriate", max_length=500)


# ---------------------------------------------------------------------------
# GET /api/papers/{id}/similar  — architecture-type similarity (P2)
# ---------------------------------------------------------------------------


class PaperAskRequest(BaseModel):
    question: str = Field(..., max_length=10000)
