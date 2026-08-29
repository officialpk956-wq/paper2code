import datetime
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user, get_optional_user
from backend.models import Paper, Task
from backend.repositories.task_repository import TaskRepository

router = APIRouter(prefix="/api", tags=["Papers Pipeline"])

from backend.server import limiter
from backend.tasks.paper_tasks import generate_code_from_pdf_task
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

    from backend.services.storage_service import R2_AVAILABLE, generate_presigned_upload_url

    if not R2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Direct upload not available — R2 not configured. Use /api/papers/upload instead.",
        )

    return generate_presigned_upload_url(
        filename, content_type, expires_in=900, user_id=current_user.id
    )


@router.get("/papers/{paper_id}/modules")
def list_paper_modules(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    from backend.models import Paper

    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)

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
        "modules": modules,
    }


@router.get("/modules/{module_id}")
def get_module(
    module_id: int, db: Session = Depends(get_db), current_user=Depends(get_optional_user)
):
    from backend.models import Paper, PaperModule

    m = db.query(PaperModule).filter(PaperModule.id == module_id).first()
    if not m:
        raise HTTPException(status_code=404, detail="Module not found")

    paper = db.query(Paper).filter_by(id=m.paper_id).first()
    if paper:
        _assert_paper_readable(paper, current_user)

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
    result.update(
        {
            "position": pos + 1,
            "prev_module_id": prev_id,
            "next_module_id": next_id,
            "paper_title": paper_title,
            "architecture_type": arch_type,
        }
    )
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
                detail=f"File exceeds {max_bytes // (1024 * 1024)} MB limit.",
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
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    logger.info("Upload Started for file: %s", file.filename)
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        logger.warning("Upload Failed: non-PDF file: %s", file.filename)
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    if not terms_accepted:
        raise HTTPException(
            status_code=400, detail="You must accept the Terms of Service to upload a paper."
        )

    if visibility not in _VALID_VISIBILITY:
        raise HTTPException(
            status_code=400, detail=f"visibility must be one of: {', '.join(_VALID_VISIBILITY)}"
        )

    # ── Monthly paper quota ───────────────────────────────────────────────────
    _check_paper_quota(db, current_user.id)

    try:
        pdf_bytes = await _read_limited(file, MAX_SIZE)

        if not pdf_bytes.startswith(b"%PDF-"):
            logger.warning("Upload Failed: File missing PDF magic bytes: %s", file.filename)
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Only legitimate PDF files are allowed.",
            )

        from backend.services.storage_service import store_pdf

        paper_name = file.filename.removesuffix(".pdf") if file.filename else "uploaded_paper"
        storage_ref = store_pdf(pdf_bytes, file.filename)

        task = TaskRepository(db).create("paper.codegen", current_user.id, paper_name)
        generate_code_from_pdf_task.delay(
            task.id,
            storage_ref,
            paper_name,
            current_user.id,
            visibility,
            terms_accepted,
        )

        from backend.services.progress_service import award_xp, update_user_activity

        update_user_activity(db, current_user.id)
        award_xp(db, current_user.id, "paper.uploaded")
        try:
            from backend.services.achievement_service import check_and_award
            from backend.services.analytics_service import track

            track(current_user.id, "paper_uploaded", {"visibility": visibility})
            check_and_award(db, current_user.id, "paper.uploaded")
        except Exception:
            pass

        from backend import metrics

        metrics.increment("papers_uploaded_total")

        return {
            "task_id": task.id,
            "status": "pending",
            "poll_url": f"/api/tasks/{task.id}",
            "paper_id": None,
            "message": "PDF uploaded. Poll poll_url every 1s for the generated code.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload Failed: %s", str(e))
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
        raise HTTPException(
            status_code=400, detail=f"visibility must be one of: {', '.join(_VALID_VISIBILITY)}"
        )

    if not body.key.startswith("papers/"):
        raise HTTPException(status_code=400, detail="Invalid R2 key.")

    # 1. Verify that the object actually exists in R2 and get its real byte size
    from backend.services.storage_service import get_object_size

    verified_file_size = get_object_size(body.key)
    if verified_file_size <= 0:
        raise HTTPException(
            status_code=400,
            detail="Upload not found in storage — ensure the PUT to the presigned URL completed before confirming.",
        )

    # 2. Validate upload intent in Redis
    from backend.redis_config import cache_redis

    if cache_redis:
        try:
            intent_key = f"upload_intent:{body.key}"
            stored_user_id = cache_redis.get(intent_key)
            if stored_user_id is None:
                raise HTTPException(
                    status_code=403,
                    detail="Upload URL expired or invalid — request a new one.",
                )
            if isinstance(stored_user_id, bytes):
                stored_user_id = stored_user_id.decode("utf-8")

            if str(stored_user_id) != str(current_user.id):
                raise HTTPException(
                    status_code=403,
                    detail="This upload was not issued to your account.",
                )

            # Consume intent key immediately for one-time use (replay protection)
            cache_redis.delete(intent_key)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning(
                "DEGRADED SECURITY: Redis error during confirm_upload intent check for user %s, key %s: %s",
                current_user.id,
                body.key,
                exc,
            )
    else:
        logger.warning(
            "DEGRADED SECURITY: Redis unavailable during confirm_upload for user %s, key %s. "
            "Upload intent check skipped; relying on Paper row ownership check.",
            current_user.id,
            body.key,
        )

    _check_paper_quota(db, current_user.id)

    _check_storage_quota(db, current_user.id, additional_bytes=verified_file_size)

    from backend.models import Paper

    paper = db.query(Paper).filter(Paper.r2_key == body.key).first()

    if paper and paper.uploaded_by != current_user.id:
        raise HTTPException(status_code=403, detail="Forbidden")

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

    # Increment user storage quota counter using verified R2 size
    if verified_file_size > 0:
        from backend.models import User

        user = db.query(User).filter_by(id=current_user.id).first()

        if user:
            user.storage_bytes_used = (user.storage_bytes_used or 0) + verified_file_size

            db.commit()

    from backend.services.progress_service import award_xp, update_user_activity

    update_user_activity(db, current_user.id)

    award_xp(db, current_user.id, "paper.uploaded")

    try:
        from backend.services.achievement_service import check_and_award
        from backend.services.analytics_service import track

        track(
            current_user.id, "paper_uploaded", {"visibility": body.visibility, "via": "presigned"}
        )

        check_and_award(db, current_user.id, "paper.uploaded")

    except Exception:
        pass

    return {
        "task_id": task.id,
        "status": "pending",
        "poll_url": f"/api/tasks/{task.id}",
        "paper_id": None,
        "message": "Upload confirmed. Poll poll_url every 1s for generated code.",
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
def get_architecture_blueprint(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    from backend.models import Paper

    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)
    bp = (p.architecture_graph or {}).get("ingestion", {}).get("architecture_blueprint")
    if not bp:
        raise HTTPException(status_code=404, detail="Blueprint not generated yet")
    return bp


@router.get("/papers/{paper_id}/executable-graph")
def get_executable_graph(
    paper_id: int, db: Session = Depends(get_db), current_user=Depends(get_optional_user)
):
    from backend.models import Paper

    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)
    if p.generated_code_source:
        return {
            "status": p.generation_status or "needs_review",
            "code": p.generated_code_source,
            "language": "python",
            "compiled": p.generated_code_compiled,
            "verification_report": p.verification_report,
            "last_generation_error": p.last_generation_error,
        }

    eg = (p.architecture_graph or {}).get("ingestion", {}).get("executable_graph")
    if not eg:
        raise HTTPException(status_code=404, detail="Executable graph not generated yet")
    return eg


@router.get("/papers/{paper_id}/graph-export")
def export_paper_graph(
    paper_id: int,
    format: str = "json",
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    from fastapi.responses import PlainTextResponse

    from backend.models import Paper
    from backend.services.architecture_graph_compiler import (
        ExecutableGraph,
        export_graph_dot,
        export_graph_json,
        export_graph_mermaid,
    )

    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)
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
def get_knowledge_graph(
    paper_id: int, db: Session = Depends(get_db), current_user=Depends(get_optional_user)
):
    from backend.models import Paper

    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)
    kg = (p.architecture_graph or {}).get("ingestion", {}).get("knowledge_graph", {})
    return {"nodes": kg.get("nodes", []), "edges": kg.get("edges", [])}


@router.post("/papers/{paper_id}/publish")
def publish_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    from backend.models import Paper

    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if paper.uploaded_by != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your paper")

    arch_graph = dict(paper.architecture_graph) if paper.architecture_graph else {}
    arch_graph["status"] = "Published"

    from sqlalchemy.orm.attributes import flag_modified

    paper.architecture_graph = arch_graph
    flag_modified(paper, "architecture_graph")

    db.commit()
    logger.info("Paper Published: %s", paper.title)
    return {"status": "success", "paper_id": paper.id}


# ---------------------------------------------------------------------------
# PATCH /api/papers/{id}/visibility  — change visibility (P0, ownership req.)
# ---------------------------------------------------------------------------


class VisibilityUpdate(BaseModel):
    visibility: str


@router.patch("/papers/{paper_id}/visibility")
def update_paper_visibility(
    paper_id: int,
    body: VisibilityUpdate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.visibility not in _VALID_VISIBILITY:
        raise HTTPException(
            status_code=400,
            detail=f"visibility must be one of: {', '.join(_VALID_VISIBILITY)}",
        )
    from backend.models import Paper

    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if paper.uploaded_by != current_user.id and not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not your paper")
    paper.visibility = body.visibility
    db.commit()
    return {"paper_id": paper_id, "visibility": body.visibility}


# ---------------------------------------------------------------------------
# DELETE /api/papers/{id}  — delete own paper, R2 object, free quota (P1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# POST /api/papers/{id}/flag  — user reports a paper for review (P1)
# ---------------------------------------------------------------------------


class PaperFlagRequest(BaseModel):
    reason: str = "inappropriate"


@router.post("/papers/{paper_id}/flag", status_code=200)
def flag_paper(
    paper_id: int,
    body: PaperFlagRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.models import Paper

    paper = db.query(Paper).filter_by(id=paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    paper.is_flagged = True
    paper.flag_reason = body.reason
    db.commit()
    return {"paper_id": paper_id, "flagged": True, "reason": body.reason}


# ---------------------------------------------------------------------------
# GET /api/papers/{id}/similar  — architecture-type similarity (P2)
# ---------------------------------------------------------------------------


@router.get("/papers/{paper_id}/similar")
def similar_papers(
    paper_id: int,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    from sqlalchemy import or_

    from backend.models import Paper

    p = db.query(Paper).filter_by(id=paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)

    _, arch_type, _, _ = get_paper_info(p)

    vis_filter = Paper.visibility != "private"
    if current_user:
        vis_filter = or_(Paper.visibility != "private", Paper.uploaded_by == current_user.id)

    candidates = db.query(Paper).filter(Paper.id != paper_id, vis_filter).limit(200).all()

    def _similarity(arch_a: str | None, arch_b: str | None) -> float:
        if not arch_a or not arch_b:
            return 0.1
        if arch_a == arch_b:
            return 1.0
        if arch_a.split("-")[0] == arch_b.split("-")[0]:
            return 0.5
        return 0.1

    scored = []
    for c in candidates:
        _, c_arch, _, _ = get_paper_info(c)
        score = _similarity(c_arch, arch_type)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    return {
        "paper_id": paper_id,
        "similar": [
            {
                "id": c.id,
                "title": c.title,
                "architecture_type": get_paper_info(c)[1],
                "similarity_score": round(score, 4),
            }
            for score, c in scored[:limit]
        ],
    }


class PaperAskRequest(BaseModel):
    question: str


@router.post("/papers/{paper_id}/ask")
def ask_paper(
    paper_id: int,
    body: PaperAskRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    from core.agents.research_rag_agent import ask_about_paper

    p = db.query(Paper).filter_by(id=paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")
    _assert_paper_readable(p, current_user)

    all_papers = db.query(Paper).filter(Paper.visibility == "public").limit(100).all()
    paper_dicts = [
        {
            "id": pp.id,
            "title": pp.title,
            "abstract": pp.abstract,
            "architecture_graph": pp.architecture_graph,
        }
        for pp in all_papers
    ]
    from core.llm_client import BudgetExceededError

    target = {
        "id": p.id,
        "title": p.title,
        "abstract": p.abstract,
        "architecture_graph": p.architecture_graph,
    }
    try:
        from backend.routers.learning import _get_budget_callback

        user_id = current_user.id if current_user else None
        result = ask_about_paper(
            target,
            paper_dicts,
            body.question,
            check_budget_callback=_get_budget_callback(db),
            user_id=user_id,
        )
        return result
    except BudgetExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))


@router.get("/papers/{paper_id}/implement")
def implement_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
):
    import re

    from backend.routers.architectures import dict_to_arch_graph
    from core.rag.semantic_explainer import SemanticExplainer
    from core.rag.tensor_tracker import TensorTracker

    p = db.query(Paper).filter_by(id=paper_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paper not found")

    _assert_paper_readable(p, current_user)

    title_slug = re.sub(r"[^a-zA-Z0-9]", "", p.title.title()) or "Model"

    default_starter = f"# Implement {p.title}\nimport torch\nimport torch.nn as nn\n\nclass {title_slug}(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: add layers\n    \n    def forward(self, x):\n        # TODO: implement forward pass\n        return x"

    if not p.architecture_graph:
        return {
            "status": "no_graph",
            "starter_code": default_starter,
            "shapes": {},
            "layer_docs": {},
        }

    graph = dict_to_arch_graph(p.title, p.architecture_graph)

    shapes = {}
    tracker = TensorTracker()
    try:
        tracker.propagate_shapes(graph)
        for node in graph.nodes:
            if node.input_shape or node.output_shape:
                shapes[node.id] = {
                    "input": list(node.input_shape)
                    if isinstance(node.input_shape, tuple)
                    else node.input_shape,
                    "output": list(node.output_shape)
                    if isinstance(node.output_shape, tuple)
                    else node.output_shape,
                }
    except Exception as e:
        logger.warning(f"TensorTracker failed: {e}")

    explainer = SemanticExplainer()
    layer_docs = {}

    nodes_data = p.architecture_graph.get("nodes", [])
    total_nodes = len(nodes_data)
    for i, n in enumerate(nodes_data):
        ntype = n.get("type")
        if ntype:
            layer_docs[n["id"]] = {
                "summary": explainer.explain(
                    node_type=ntype,
                    semantic_role=n.get("semantic_role") or n.get("compute_role", ""),
                    params=n.get("params", {}),
                    context={"index": i, "total": total_nodes},
                )
            }

    # Generate PyTorch Scaffold
    imports = {"import torch", "import torch.nn as nn"}
    init_lines = []
    forward_lines = []

    for i, node in enumerate(graph.nodes):
        ntype = node.type.lower()
        if not ntype:
            continue

        layer_name = f"self.layer_{i}"

        # Build layer init string mapping
        layer_init = None
        if ntype in ["conv2d", "conv"]:
            c_in = node.input_shape[1] if node.input_shape and len(node.input_shape) == 4 else 3
            c_out = node.params.get(
                "channels", node.params.get("out_channels", node.params.get("filters", 64))
            )
            k = node.params.get("kernel_size", 3)
            s = node.params.get("stride", 1)
            p_pad = node.params.get("padding", 1)
            layer_init = f"nn.Conv2d({c_in}, {c_out}, kernel_size={k}, stride={s}, padding={p_pad})"
        elif ntype in ["linear", "dense"]:
            d_in = node.input_shape[-1] if node.input_shape else 512
            d_out = node.params.get(
                "hidden_size", node.params.get("channels", node.params.get("out_features", 1000))
            )
            layer_init = f"nn.Linear({d_in}, {d_out})"
        elif ntype == "maxpool2d":
            k = node.params.get("kernel_size", 2)
            s = node.params.get("stride", 2)
            layer_init = f"nn.MaxPool2d(kernel_size={k}, stride={s})"
        elif ntype == "avgpool2d":
            layer_init = "nn.AdaptiveAvgPool2d((1, 1))"
        elif ntype in ["relu", "activation"]:
            layer_init = "nn.ReLU()"
        elif ntype == "layernorm":
            d_in = node.input_shape[-1] if node.input_shape else 768
            layer_init = f"nn.LayerNorm({d_in})"
        elif ntype == "batchnorm2d":
            c_in = node.input_shape[1] if node.input_shape and len(node.input_shape) == 4 else 64
            layer_init = f"nn.BatchNorm2d({c_in})"
        elif ntype in ["dropout"]:
            p_drop = node.params.get("p", 0.1)
            layer_init = f"nn.Dropout(p={p_drop})"
        elif ntype == "flatten":
            layer_init = "nn.Flatten()"
        else:
            layer_init = f"nn.Module() # TODO: implement {ntype}"

        if layer_init:
            shape_info = shapes.get(node.id, {})
            if shape_info:
                in_s = shape_info.get("input", "?")
                out_s = shape_info.get("output", "?")
                init_lines.append(f"        # {ntype}: input {in_s} -> output {out_s}")
            else:
                init_lines.append(f"        # {ntype}")
            init_lines.append(f"        {layer_name} = {layer_init}")
            forward_lines.append(f"        x = {layer_name}(x)")

    if not init_lines:
        init_lines.append("        # TODO: add layers")
        forward_lines.append("        # TODO: implement forward pass")

    init_str = "\n".join(init_lines)
    forward_str = "\n".join(forward_lines)

    starter_code = f"# Implement {p.title}\n"
    starter_code += "\n".join(sorted(imports)) + "\n\n"
    starter_code += f"class {title_slug}(nn.Module):\n"
    starter_code += "    def __init__(self):\n"
    starter_code += "        super().__init__()\n"
    starter_code += f"{init_str}\n\n"
    starter_code += "    def forward(self, x):\n"
    starter_code += f"{forward_str}\n"
    starter_code += "        return x"

    return {
        "status": "ok",
        "starter_code": starter_code,
        "shapes": shapes,
        "layer_docs": layer_docs,
    }


@router.get("/tasks/{task_id}/stream")
def stream_task_status(
    task_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    "Server-Sent Events endpoint for real-time task status."
    import asyncio
    import json

    from fastapi.responses import StreamingResponse

    from backend.repositories.task_repository import TaskRepository

    repo = TaskRepository(db)

    async def event_generator():
        last_status = None
        while True:
            task = repo.get(task_id)
            if not task:
                yield f"data: {json.dumps({'status': 'error', 'error': 'Task not found'})}\n\n"
                break

            current_status = task.status
            if current_status != last_status:
                payload = {
                    "id": task.id,
                    "status": task.status,
                    "result": task.result,
                    "error": task.error,
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last_status = current_status

            if current_status in ("complete", "failed"):
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
