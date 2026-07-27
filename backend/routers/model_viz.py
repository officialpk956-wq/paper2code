import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import ModelGraph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/model", tags=["Model Viz"])

MAX_BYTES = 50 * 1024 * 1024  # 50 MB
CHUNK = 1024 * 1024  # stream in 1 MB chunks

_PYTORCH_EXTS = {".pt", ".pth"}


# ── helpers ───────────────────────────────────────────────────────────────────


async def _read_upload(file: UploadFile) -> bytes:
    """Stream-read UploadFile enforcing MAX_BYTES limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50 MB.")
        chunks.append(chunk)
    return b"".join(chunks)


# ── GET /api/model/health  (dependency availability) ──────────────────────────


@router.get("/health")
def model_viz_health():
    """Report whether the heavy parsers' dependencies are importable on THIS
    server — so you can confirm ONNX/PyTorch support in prod without uploading a
    file (audit #1: onnx is a big package and a build may silently skip it)."""

    def _check(mod: str) -> dict:
        try:
            m = __import__(mod)
            return {"available": True, "version": getattr(m, "__version__", None)}
        except Exception as e:  # ImportError or a broken native build
            return {"available": False, "error": str(e)[:160]}

    onnx = _check("onnx")
    e2b = _check("e2b_code_interpreter")
    return {
        "onnx": onnx,  # required by POST /parse
        "e2b": e2b,  # required by POST /parse-pytorch
        "onnx_parse_ready": onnx["available"],
        "pytorch_parse_ready": e2b["available"],
    }


# ── POST /api/model/parse  (ONNX) ─────────────────────────────────────────────


@router.post("/parse")
async def parse_model(
    file: UploadFile = File(...),
    _user=Depends(get_current_user),
):
    """Accept an .onnx file, parse its computation graph, return nodes/edges/meta."""
    if not file.filename or not file.filename.lower().endswith(".onnx"):
        raise HTTPException(status_code=400, detail="Only .onnx files are supported.")

    content = await _read_upload(file)

    try:
        from backend.services.onnx_parser import parse_onnx

        graph = parse_onnx(content)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="ONNX support unavailable. The 'onnx' package is not installed on the server.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("ONNX parse error for file %s", file.filename)
        raise HTTPException(status_code=422, detail=f"Could not parse ONNX file: {e}")

    return graph


# ── POST /api/model/parse-pytorch  (PyTorch via E2B) ─────────────────────────


@router.post("/parse-pytorch")
async def parse_pytorch_model(
    file: UploadFile = File(...),
    input_shape: str = Form(...),  # JSON array string, e.g. "[3,224,224]"
    _user=Depends(get_current_user),
):
    """
    Accept a .pt / .pth file plus an input_shape JSON array, run torch.fx inside
    an E2B sandbox, and return the same graph format as /parse.

    input_shape must be the spatial dims WITHOUT the batch dimension, e.g. [3,224,224].
    Batch dim (1) is prepended automatically inside the sandbox.
    """
    fname = (file.filename or "").lower()
    if not any(fname.endswith(ext) for ext in _PYTORCH_EXTS):
        raise HTTPException(status_code=400, detail="Only .pt / .pth files are supported here.")

    # Parse and validate input_shape
    import json as _json

    try:
        shape: list[int] = _json.loads(input_shape)
        if not isinstance(shape, list) or not shape:
            raise ValueError
        shape = [int(d) for d in shape]
        if any(d <= 0 for d in shape):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="input_shape must be a JSON array of positive integers, e.g. [3,224,224].",
        )

    content = await _read_upload(file)

    try:
        from backend.services.pytorch_parser import parse_pytorch

        graph = parse_pytorch(content, shape)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="E2B not installed. The 'e2b-code-interpreter' package is required for PyTorch parsing.",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("PyTorch parse error for file %s", file.filename)
        raise HTTPException(status_code=422, detail=f"Could not parse PyTorch model: {e}")

    return graph


# ── POST /api/model/save ──────────────────────────────────────────────────────


@router.post("/save")
def save_graph(
    payload: Annotated[dict, Body()],
    user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Persist a parsed graph to the DB so it can be retrieved later via a
    shareable link.

    Body: { name: string, format: "onnx"|"pytorch", graph_data: {...} }
    Returns: { id: number }
    """
    name: str = payload.get("name", "")[:255] or "model"
    fmt: str = payload.get("format", "onnx")
    if fmt not in ("onnx", "pytorch"):
        raise HTTPException(status_code=400, detail="format must be 'onnx' or 'pytorch'.")

    graph_data = payload.get("graph_data")
    if not isinstance(graph_data, dict):
        raise HTTPException(status_code=400, detail="graph_data must be an object.")

    record = ModelGraph(
        user_id=user.id,
        name=name,
        format=fmt,
        graph_data=graph_data,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {"id": record.id}


# ── GET /api/model/{graph_id} ────────────────────────────────────────────────


@router.get("/{graph_id}")
def get_graph(
    graph_id: int,
    db: Session = Depends(get_db),
):
    """
    Return a previously saved graph by ID.
    No auth required — anyone with the link can view the architecture.
    The graph_data itself contains no user-identifying info.
    """
    record = db.get(ModelGraph, graph_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Graph not found.")

    return {
        "id": record.id,
        "name": record.name,
        "format": record.format,
        "graph_data": record.graph_data,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }
