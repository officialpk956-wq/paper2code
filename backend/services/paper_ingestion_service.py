"""
Paper ingestion service for Phase 13.

This module centralizes the deterministic PDF ingestion flow so both the
FastAPI backend and the Next.js upload proxy can reuse the same behavior.

Responsibilities:
  - Extract raw text from PDF bytes with pdfplumber + PyMuPDF fallback
  - Extract figure metadata and equation candidates
  - Run the existing Paper2Code pipeline
  - Generate learning modules
  - Persist the result into the existing Paper / PaperModule tables

The extracted figures and equations are persisted inside the Paper JSON payload
to avoid a schema migration while keeping the data durable.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from backend.models import Paper, PaperChunk, PaperModule
from backend.services.architecture_graph_compiler import compile_blueprint
from backend.services.architecture_reconstruction_service import reconstruct_architecture
from backend.services.knowledge_extraction_service import build_knowledge_graph
from core.classification import classify_architecture
from core.evidence_tracking import build_evidence_map
from core.llm_client import GROQ_API_KEY, llm_complete
from core.module_generator import generate_modules
from core.paper_to_code_generator import PaperToCodeGenerator
from core.utils import (
    chunk_pages_with_provenance,
    extract_caption_chunks,
    extract_table_chunks,
)


def _chunk_retriever(query: str, texts: list[str], top_k: int) -> list[str]:
    from backend.services.vector_service import hybrid_rank_texts

    return hybrid_rank_texts(query, texts, top_k=top_k)


_GENERATOR = PaperToCodeGenerator(chunk_retriever=_chunk_retriever)
log = logging.getLogger(__name__)

OCR_MIN_CHARS_PER_PAGE = 50
_OCR_UNAVAILABLE_LOGGED = False


def _normalize_title(paper_name: str) -> str:
    paper_name = Path(paper_name).stem.strip()
    return paper_name or "paper"


def _resolve_unique_title(db: Session, base_title: str) -> str:
    candidate = base_title
    suffix = 2

    while db.query(Paper.id).filter(Paper.title == candidate).first() is not None:
        candidate = f"{base_title} ({suffix})"
        suffix += 1

    return candidate


def _needs_ocr(page_texts: list[str]) -> bool:
    """Return True when the PDF text layer is too sparse to be usable."""
    if not page_texts:
        return True
    total_chars = sum(len((page_text or "").strip()) for page_text in page_texts)
    return total_chars / len(page_texts) < OCR_MIN_CHARS_PER_PAGE


def _get_ocr_engine():
    """Reserved adapter hook for a future optional OCR engine."""
    raise ImportError("No OCR engine is installed or configured")


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 30) -> list[tuple[int, str]]:
    """Rasterize + OCR PDF pages when an optional engine is configured.

    PyMuPDF can provide rasterization because it is already installed. No OCR
    engine is enabled in this deployment, so this returns [] without raising.
    """
    global _OCR_UNAVAILABLE_LOGGED
    try:
        _get_ocr_engine()
    except ImportError:
        if not _OCR_UNAVAILABLE_LOGGED:
            log.warning("OCR requested for a sparse PDF, but no OCR engine is enabled")
            _OCR_UNAVAILABLE_LOGGED = True
        return []
    except Exception as exc:
        log.warning("OCR initialization failed: %s", exc)
        return []

    log.warning("OCR engine adapter is not configured; skipping OCR")
    return []


def extract_pdf_pages(pdf_bytes: bytes) -> tuple[list[str], str]:
    """Extract page-level text from a PDF byte stream."""
    pdfplumber_pages: list[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages[:30]:
                # x_tolerance=1: pdfplumber's default (3) merges adjacent words on these
                # PDFs -- measured across all 10 benchmark papers, spaces ran 3.3-11.1%
                # of characters against ~16% for normal prose, and transformer_base came
                # out at an average letter-run length of 12.4 chars ('Weuseself-
                # attentionat'). That breaks regex word boundaries, BM25 tokenisation
                # and embeddings alike. At x_tolerance=1 the same papers land at
                # 13.4-15.3% spaces and 4.7-5.4 char runs, which is normal English.
                text = page.extract_text(x_tolerance=1) or ""
                pages.append(text)

        pdfplumber_pages = pages
        if not _needs_ocr(pages):
            return pages, "pdfplumber"
    except Exception:
        pass

    try:
        import fitz  # PyMuPDF

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_count = min(len(document), 30)
            pages = [document[index].get_text("text") or "" for index in range(page_count)]
        if not _needs_ocr(pages):
            return pages, "pymupdf"

        ocr_pages = ocr_pdf_pages(pdf_bytes, max_pages=30)
        if ocr_pages:
            return [text for _, text in ocr_pages], "ocr"
        return pages or pdfplumber_pages, "pymupdf"
    except Exception as exc:
        if _needs_ocr(pdfplumber_pages):
            ocr_pages = ocr_pdf_pages(pdf_bytes, max_pages=30)
            if ocr_pages:
                return [text for _, text in ocr_pages], "ocr"
        raise ValueError(f"Failed to extract text from PDF: {exc}") from exc


def extract_raw_text(pdf_bytes: bytes) -> tuple[str, list[str], str]:
    pages, method = extract_pdf_pages(pdf_bytes)
    raw_text = "\n\n".join(page for page in pages if page is not None)
    if not raw_text.strip():
        if _needs_ocr(pages):
            raise ValueError(
                "This PDF appears to be scanned or image-only. OCR is not enabled, "
                "so please upload a PDF with a selectable text layer."
            )
        raise ValueError("Could not extract any text from the PDF. It might be scanned or empty.")
    return raw_text, pages, method


def extract_figures(pdf_bytes: bytes, page_texts: list[str]) -> list[dict[str, Any]]:
    """Extract figure metadata from PDF pages using PyMuPDF image xrefs."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            figures: list[dict[str, Any]] = []
            seen_xrefs: set[int] = set()

            page_count = min(len(document), 30)

            for page_offset in range(page_count):
                page_index = page_offset + 1
                page = document[page_offset]
                page_text = page_texts[page_index - 1] if page_index - 1 < len(page_texts) else ""
                caption_match = None
                for match in re.finditer(
                    r"(?:Figure|Fig\.)\s*\d+[\s:\-–.]*([^\n]{0,200})", page_text, re.IGNORECASE
                ):
                    caption_match = match
                    break

                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = image_info[0]
                    if xref in seen_xrefs:
                        continue
                    seen_xrefs.add(xref)

                    try:
                        extracted = document.extract_image(xref)
                    except Exception:
                        extracted = {}

                    figures.append(
                        {
                            "id": f"p{page_index}-img{image_index}",
                            "page": page_index,
                            "xref": xref,
                            "width": extracted.get("width"),
                            "height": extracted.get("height"),
                            "ext": extracted.get("ext"),
                            "caption": caption_match.group(1).strip() if caption_match else None,
                            "has_binary": bool(extracted.get("image")),
                        }
                    )

            return figures
    except Exception:
        return []


def extract_equations(page_texts: list[str]) -> list[dict[str, Any]]:
    """Extract equation-like text spans from OCR/text extraction output."""
    equations: list[dict[str, Any]] = []
    equation_re = re.compile(
        r"(?:\$\$[^$]{4,240}\$\$|\$[^$\n]{4,180}\$|.*[=±∑∫∂∇→≤≥≈][^\n]{0,180})"
    )

    for page_index, page_text in enumerate(page_texts, start=1):
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if len(line) > 240:
                continue
            if equation_re.fullmatch(line) or (
                any(symbol in line for symbol in ("=", "±", "∑", "∫", "∂", "∇", "→", "≤", "≥", "≈"))
                and sum(ch.isalpha() for ch in line) < len(line)
            ):
                equations.append(
                    {
                        "id": f"p{page_index}-eq{len(equations) + 1}",
                        "page": page_index,
                        "text": line,
                    }
                )

    return equations[:80]


_SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?("
    r"abstract|introduction|related\s+work|background"
    r"|method(?:ology|s)?|approach|model(?:\s+architecture)?"
    r"|experiment(?:al(?:\s+(?:setup|results))?|s)?"
    r"|results?(?:\s+and\s+discussion)?|discussion"
    r"|conclusion(?:s)?|references?|appendix"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_CANONICAL_NAMES: dict[str, str] = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "methodology": "Methods",
    "methods": "Methods",
    "method": "Methods",
    "approach": "Approach",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "experimental results": "Experiments",
    "results": "Results",
    "results and discussion": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
    "reference": "References",
    "appendix": "Appendix",
}


def extract_sections(page_texts: list[str]) -> list[dict[str, Any]]:
    """Extract structured sections from page texts using heading detection."""
    full_text = "\n".join(page_texts)
    matches = list(_SECTION_HEADING_RE.finditer(full_text))

    sections: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        if not content:
            continue

        keyword = match.group(1).lower()
        canonical = _CANONICAL_NAMES.get(keyword, match.group(1).title())
        level = 2 if canonical == "Appendix" else 1

        sections.append(
            {
                "id": f"sec-{i + 1}",
                "title": canonical,
                "content": content[:4000],
                "level": level,
            }
        )

    return sections


def build_ingestion_payload(pdf_bytes: bytes, source_filename: str, title: str) -> dict[str, Any]:
    raw_text, page_texts, text_method = extract_raw_text(pdf_bytes)
    figures = extract_figures(pdf_bytes, page_texts)
    equations = extract_equations(page_texts)
    sections = extract_sections(page_texts)

    page_chunks = [(index, text) for index, text in enumerate(page_texts, start=1) if text]
    source_chunks = chunk_pages_with_provenance(page_chunks)
    source_chunks.extend(extract_table_chunks(page_chunks))
    source_chunks.extend(extract_caption_chunks(page_chunks))

    page_offsets: dict[int, int] = {}
    global_offset = 0
    for page_index, page_text in enumerate(page_texts, start=1):
        page_offsets[page_index] = global_offset
        global_offset += len(page_text or "") + 2
    for equation in equations:
        page = equation.get("page")
        text = str(equation.get("text") or "").strip()
        page_text = ""
        if isinstance(page, int) and page is not None and 1 <= page <= len(page_texts):
            page_text = page_texts[page - 1] or ""
        start = page_text.find(text) if text and page_text else -1
        source_chunks.append(
            {
                "section": "other",
                "page": page if isinstance(page, int) else None,
                "chunk_type": "equation",
                "text": text,
                "source_offset_start": page_offsets.get(page) + start
                if start >= 0 and page is not None and page in page_offsets
                else None,
                "source_offset_end": page_offsets.get(page) + start + len(text)
                if start >= 0 and page is not None and page in page_offsets
                else None,
            }
        )

    seen_chunks: set[tuple[str, int | None, str]] = set()
    source_chunks = [
        chunk
        for chunk in source_chunks
        if not (
            (key := (chunk["chunk_type"], chunk["page"], chunk["text"])) in seen_chunks
            or seen_chunks.add(key)
        )
    ]

    return {
        "source_filename": source_filename,
        "title": title,
        "page_count": len(page_texts),
        "text_extraction_method": text_method,
        "text_source": text_method,
        "sections": sections,
        "section_count": len(sections),
        "figures": figures,
        "figure_count": len(figures),
        "equations": equations,
        "equation_count": len(equations),
        "raw_text_excerpt": raw_text[:4000],
        "source_chunks": source_chunks,
    }


def _positive_int(value: Any) -> int | None:
    """Return a positive integer accepted by ArchitectureSpec, otherwise None."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _architecture_spec_payload(
    extracted_spec: dict[str, Any], result_dict: dict[str, Any]
) -> dict[str, Any]:
    """Adapt ConfigExtractor/graph output to the learning-module schema."""
    extracted_spec = extracted_spec or {}
    family = str(
        result_dict.get("family")
        or extracted_spec.get("model_family")
        or extracted_spec.get("family")
        or "unknown"
    )
    input_config = extracted_spec.get("input") or {}
    spatial = input_config.get("spatial_dims") or []
    if isinstance(spatial, (list, tuple)) and spatial:
        input_shape = [
            _positive_int(input_config.get("channels")) or 3,
            *[_positive_int(value) or 224 for value in spatial[:2]],
        ]
    elif _positive_int(input_config.get("seq_len")) or family in (
        "transformer",
        "bert_gpt",
    ):
        input_shape = [_positive_int(input_config.get("seq_len")) or 64]
    else:
        input_shape = [3, 224, 224]

    graph = result_dict.get("graph")
    graph_nodes = list(getattr(graph, "nodes", None) or [])
    layers: list[dict[str, Any]] = []
    for index, node in enumerate(graph_nodes):
        params = getattr(node, "params", None) or {}
        layers.append(
            {
                "type": str(getattr(node, "type", None) or "identity"),
                "name": str(
                    getattr(node, "label", None) or getattr(node, "id", None) or f"layer_{index}"
                ),
                "channels": _positive_int(params.get("channels") or params.get("out_channels")),
                "kernel_size": _positive_int(params.get("kernel_size") or params.get("kernel")),
                "stride": _positive_int(params.get("stride")),
                "heads": _positive_int(params.get("num_heads") or params.get("heads")),
                "hidden_size": _positive_int(
                    params.get("hidden_size") or params.get("embed_dim") or params.get("d_model")
                ),
            }
        )

    if not layers:
        for index, layer in enumerate(extracted_spec.get("layers") or []):
            if not isinstance(layer, dict):
                continue
            params = layer.get("params") or {}
            layers.append(
                {
                    "type": str(layer.get("type") or "identity"),
                    "name": str(layer.get("name") or layer.get("id") or f"layer_{index}"),
                    "channels": _positive_int(params.get("channels") or params.get("out_channels")),
                    "kernel_size": _positive_int(params.get("kernel_size") or params.get("kernel")),
                    "stride": _positive_int(params.get("stride")),
                    "heads": _positive_int(params.get("num_heads") or params.get("heads")),
                    "hidden_size": _positive_int(
                        params.get("hidden_size")
                        or params.get("embed_dim")
                        or params.get("d_model")
                    ),
                }
            )

    return {
        "family": family,
        "input_shape": input_shape,
        "layers": layers,
    }


def ingest_pdf_paper(
    db: Session,
    pdf_bytes: bytes,
    source_filename: str,
    paper_name: str | None = None,
) -> dict[str, Any]:
    """Run the full ingestion pipeline and persist the resulting paper."""
    base_title = _normalize_title(paper_name or source_filename)
    title = _resolve_unique_title(db, base_title)

    ingestion = build_ingestion_payload(pdf_bytes, source_filename, title)
    result_dict = _GENERATOR.from_pdf(io.BytesIO(pdf_bytes), title)

    spec = result_dict.get("spec", {})
    import logging

    from pydantic import ValidationError

    from backend.schemas.architecture_spec import ArchitectureSpec

    logger = logging.getLogger(__name__)
    try:
        validated_spec = ArchitectureSpec(**_architecture_spec_payload(spec, result_dict))
        spec = validated_spec.model_dump()
    except ValidationError as e:
        logger.warning("Architecture spec validation failed: %s — using partial spec", e)
        spec = {"family": "unknown", "layers": [], "input_shape": [3, 224, 224]}

    paper_meta, learning_modules = generate_modules(
        paper_name=title,
        schema=spec,
        pipeline_result=result_dict,
    )

    graph = result_dict["graph"]
    classification = classify_architecture(graph)

    knowledge_graph = build_knowledge_graph(
        paper_title=title,
        abstract=paper_meta.get("abstract") or "",
        sections=ingestion.get("sections", []),
        equations=ingestion.get("equations", []),
    )

    architecture_blueprint = reconstruct_architecture(
        paper_title=title,
        abstract=paper_meta.get("abstract") or "",
        sections=ingestion.get("sections", []),
        equations=ingestion.get("equations", []),
        knowledge_graph=knowledge_graph,
        paper_id=None,
    )

    executable_graph = compile_blueprint(architecture_blueprint).to_dict()

    paper_meta["architecture_graph"]["classification"] = classification
    paper_meta["architecture_graph"]["status"] = "Draft"
    ingestion_for_json = {key: value for key, value in ingestion.items() if key != "source_chunks"}
    paper_meta["architecture_graph"]["ingestion"] = {
        **ingestion_for_json,
        "detected_components": sorted({node.type for node in graph.nodes}),
        "module_count": len(learning_modules),
        "knowledge_graph": knowledge_graph,
        "architecture_blueprint": architecture_blueprint,
        "executable_graph": executable_graph,
    }

    paper = Paper(
        title=title,
        authors=paper_meta.get("authors"),
        abstract=paper_meta.get("abstract"),
        architecture_graph=paper_meta.get("architecture_graph"),
        flops_analysis=paper_meta.get("flops_analysis"),
        generated_code_source=result_dict.get("code") or None,
        generated_code_compiled={
            "language": "python",
            "framework": "pytorch",
            "code_source": result_dict.get("code_source", "skeleton"),
            "entrypoint_class": (result_dict.get("verification_report") or {}).get(
                "entrypoint_class"
            ),
        },
        generation_status=result_dict.get("generation_status", "needs_review"),
        verification_report=result_dict.get("verification_report"),
        last_generation_error=(result_dict.get("verification_report") or {}).get("error"),
    )

    db.add(paper)
    db.commit()
    db.refresh(paper)

    source_chunks = list(result_dict.get("source_chunks") or ingestion.get("source_chunks") or [])
    structured_chunks = [
        chunk for chunk in ingestion.get("source_chunks") or [] if chunk.get("chunk_type") != "text"
    ]
    source_chunks.extend(structured_chunks)
    seen_chunk_keys: set[tuple[str, int | None, str]] = set()
    source_chunks = [
        chunk
        for chunk in source_chunks
        if not (
            (
                key := (
                    str(chunk.get("chunk_type") or "text"),
                    chunk.get("page") if isinstance(chunk.get("page"), int) else None,
                    str(chunk.get("text") or "").strip(),
                )
            )
            in seen_chunk_keys
            or seen_chunk_keys.add(key)
        )
    ]
    persisted_chunks: list[PaperChunk] = []
    for chunk in source_chunks:
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        persisted_chunks.append(
            PaperChunk(
                paper_id=paper.id,
                section=str(chunk.get("section") or "other"),
                page=chunk.get("page") if isinstance(chunk.get("page"), int) else None,
                chunk_type=str(chunk.get("chunk_type") or "text"),
                text=text,
                source_offset_start=chunk.get("source_offset_start")
                if isinstance(chunk.get("source_offset_start"), int)
                else None,
                source_offset_end=chunk.get("source_offset_end")
                if isinstance(chunk.get("source_offset_end"), int)
                else None,
            )
        )
    if persisted_chunks:
        db.add_all(persisted_chunks)
        db.flush()

        # Only durable row IDs can appear in a citation. Candidate LLM quotes
        # are checked against the stored source text before becoming "cited".
        report = dict(paper.verification_report or {})
        report["evidence"] = build_evidence_map(
            result_dict.get("spec") or {},
            [
                {"id": chunk.id, "text": chunk.text, "page": chunk.page}
                for chunk in persisted_chunks
            ],
            complete=llm_complete if GROQ_API_KEY else None,
        )
        paper.verification_report = report
        paper_meta["architecture_graph"]["ingestion"]["chunk_count"] = len(persisted_chunks)
        paper.architecture_graph = paper_meta["architecture_graph"]

    for module in learning_modules:
        db.add(
            PaperModule(
                paper_id=paper.id,
                layer_name=module.layer_name,
                module_type=module.module_type,
                explanation=module.explanation,
                tensor_flow=module.tensor_flow,
                graph_nodes=module.graph_nodes,
                flops_context=module.flops_context,
                order_index=module.order_index,
            )
        )

    db.commit()

    ingestion_summary = paper_meta["architecture_graph"]["ingestion"]
    return {
        "paper_id": paper.id,
        "title": paper.title,
        "classification": classification,
        "status": "Draft",
        "paper_meta": paper_meta,
        "modules": [module.__dict__ for module in learning_modules],
        "ingestion": ingestion_summary,
        "code": result_dict.get("code", ""),
        "code_source": result_dict.get("code_source", "skeleton"),
        "family": result_dict.get("family", "unknown"),
        "generation_status": paper.generation_status,
        "verification_report": paper.verification_report,
        "report": {
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "modules": len(learning_modules),
            "parameters": paper_meta["flops_analysis"]["total_params_estimate"],
            "flops": paper_meta["flops_analysis"]["total_flops_score"],
            "detected_components": ingestion_summary["detected_components"],
            "figure_count": ingestion_summary["figure_count"],
            "equation_count": ingestion_summary["equation_count"],
        },
    }
