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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.models import Paper, PaperModule
from backend.services.knowledge_extraction_service import build_knowledge_graph
from backend.services.architecture_reconstruction_service import reconstruct_architecture
from backend.services.architecture_graph_compiler import compile_blueprint
from core.classification import classify_architecture
from core.module_generator import generate_modules
from core.paper_to_code_generator import PaperToCodeGenerator


_GENERATOR = PaperToCodeGenerator()


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


def extract_pdf_pages(pdf_bytes: bytes) -> Tuple[List[str], str]:
    """Extract page-level text from a PDF byte stream."""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages[:30]:
                text = page.extract_text() or ""
                pages.append(text)

        if any(page.strip() for page in pages):
            return pages, "pdfplumber"
    except Exception:
        pass

    try:
        import fitz  # PyMuPDF

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = min(len(document), 30)
        pages = [document[index].get_text("text") or "" for index in range(page_count)]
        return pages, "pymupdf"
    except Exception as exc:
        raise ValueError(f"Failed to extract text from PDF: {exc}") from exc


def extract_raw_text(pdf_bytes: bytes) -> Tuple[str, List[str], str]:
    pages, method = extract_pdf_pages(pdf_bytes)
    raw_text = "\n\n".join(page for page in pages if page is not None)
    if not raw_text.strip():
        raise ValueError("Could not extract any text from the PDF. It might be scanned or empty.")
    return raw_text, pages, method


def extract_figures(pdf_bytes: bytes, page_texts: List[str]) -> List[Dict[str, Any]]:
    """Extract figure metadata from PDF pages using PyMuPDF image xrefs."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    figures: List[Dict[str, Any]] = []
    seen_xrefs: set[int] = set()

    page_count = min(len(document), 30)

    for page_offset in range(page_count):
        page_index = page_offset + 1
        page = document[page_offset]
        page_text = page_texts[page_index - 1] if page_index - 1 < len(page_texts) else ""
        caption_match = None
        for match in re.finditer(r"(?:Figure|Fig\.)\s*\d+[\s:\-–.]*([^\n]{0,200})", page_text, re.IGNORECASE):
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

            figures.append({
                "id": f"p{page_index}-img{image_index}",
                "page": page_index,
                "xref": xref,
                "width": extracted.get("width"),
                "height": extracted.get("height"),
                "ext": extracted.get("ext"),
                "caption": caption_match.group(1).strip() if caption_match else None,
                "has_binary": bool(extracted.get("image")),
            })

    return figures


def extract_equations(page_texts: List[str]) -> List[Dict[str, Any]]:
    """Extract equation-like text spans from OCR/text extraction output."""
    equations: List[Dict[str, Any]] = []
    equation_re = re.compile(r"(?:\$\$[^$]{4,240}\$\$|\$[^$\n]{4,180}\$|.*[=±∑∫∂∇→≤≥≈][^\n]{0,180})")

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
                equations.append({
                    "id": f"p{page_index}-eq{len(equations) + 1}",
                    "page": page_index,
                    "text": line,
                })

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

_CANONICAL_NAMES: Dict[str, str] = {
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


def extract_sections(page_texts: List[str]) -> List[Dict[str, Any]]:
    """Extract structured sections from page texts using heading detection."""
    full_text = "\n".join(page_texts)
    matches = list(_SECTION_HEADING_RE.finditer(full_text))

    sections: List[Dict[str, Any]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()
        if not content:
            continue

        keyword = match.group(1).lower()
        canonical = _CANONICAL_NAMES.get(keyword, match.group(1).title())
        level = 2 if canonical == "Appendix" else 1

        sections.append({
            "id": f"sec-{i + 1}",
            "title": canonical,
            "content": content[:4000],
            "level": level,
        })

    return sections


def build_ingestion_payload(pdf_bytes: bytes, source_filename: str, title: str) -> Dict[str, Any]:
    raw_text, page_texts, text_method = extract_raw_text(pdf_bytes)
    figures = extract_figures(pdf_bytes, page_texts)
    equations = extract_equations(page_texts)
    sections = extract_sections(page_texts)

    return {
        "source_filename": source_filename,
        "title": title,
        "page_count": len(page_texts),
        "text_extraction_method": text_method,
        "sections": sections,
        "section_count": len(sections),
        "figures": figures,
        "figure_count": len(figures),
        "equations": equations,
        "equation_count": len(equations),
        "raw_text_excerpt": raw_text[:4000],
    }


def ingest_pdf_paper(
    db: Session,
    pdf_bytes: bytes,
    source_filename: str,
    paper_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full ingestion pipeline and persist the resulting paper."""
    base_title = _normalize_title(paper_name or source_filename)
    title = _resolve_unique_title(db, base_title)

    ingestion = build_ingestion_payload(pdf_bytes, source_filename, title)
    result_dict = _GENERATOR.from_pdf(io.BytesIO(pdf_bytes), title)

    spec = result_dict.get("spec", {})
    graph = result_dict["graph"]
    classification = classify_architecture(graph)

    paper_meta, learning_modules = generate_modules(
        paper_name=title,
        schema=spec,
        pipeline_result=result_dict,
    )

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
    paper_meta["architecture_graph"]["ingestion"] = {
        **ingestion,
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
    )

    db.add(paper)
    db.commit()
    db.refresh(paper)

    for module in learning_modules:
        db.add(PaperModule(
            paper_id=paper.id,
            layer_name=module.layer_name,
            module_type=module.module_type,
            explanation=module.explanation,
            tensor_flow=module.tensor_flow,
            graph_nodes=module.graph_nodes,
            flops_context=module.flops_context,
            order_index=module.order_index,
        ))

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
