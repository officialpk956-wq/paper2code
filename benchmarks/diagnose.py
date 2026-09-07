"""Inspect a live benchmark extraction without changing its scoring cache.

Usage:
    python -m benchmarks.diagnose LABEL_ID [--retrieval production|legacy] [--live]

Without ``--live``, this reads an existing companion diagnostic artifact and
never calls an LLM. ``--live`` performs one extraction and refreshes only the
``<id>.<mode>.diag.json`` companion file.
"""

import argparse
import io
import json
from pathlib import Path

import httpx

from benchmarks.harness import LABELS_DIR, _chunk_retriever, _live_extractor, diagnostic_path, load_label
from core.rag.config_extractor import ConfigExtractor
from core.utils import chunk_pages_with_provenance, extract_caption_chunks, extract_table_chunks


_REQUIRED_STAGES = {
    "raw_text_length",
    "chunk_count_by_type",
    "focused_text",
    "raw_llm_response",
    "parsed_spec_pre_normalization",
    "spec_post_normalization",
}


def _label_path(label_id: str) -> Path:
    path = LABELS_DIR / f"{label_id}.json"
    if not path.exists():
        raise ValueError(f"Unknown benchmark label {label_id!r}: expected {path}")
    return path


def focus_only(label: dict) -> dict:
    """Rebuild production chunks and focus context without calling an LLM."""
    source = str(label["source"])
    if not source.startswith("arxiv:"):
        raise ValueError(f"focus-only diagnostics support arXiv labels, got {source}")
    response = httpx.get(
        f"https://arxiv.org/pdf/{source.removeprefix('arxiv:')}.pdf",
        follow_redirects=True,
        timeout=60.0,
    )
    response.raise_for_status()
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("focus-only diagnostics require pdfplumber") from exc
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        page_texts = [
            (page_number, text)
            for page_number, page in enumerate(pdf.pages[:30], start=1)
            if (text := page.extract_text())
        ]
    text = "\n\n".join(page_text for _, page_text in page_texts)
    source_chunks = chunk_pages_with_provenance(page_texts)
    source_chunks.extend(extract_table_chunks(page_texts))
    source_chunks.extend(extract_caption_chunks(page_texts))
    extractor = ConfigExtractor(
        use_llm=False,
        verify=False,
        chunk_retriever=_chunk_retriever,
    )
    ranked = extractor._select_focus_chunks(source_chunks)
    selected = extractor._select_focus_chunks(
        source_chunks,
        expand_neighbors=True,
        max_context_chars=extractor.max_context_chars,
    )
    focused = extractor._focus_text(text, source_chunks=source_chunks)
    ranked_indices = [
        index for index, chunk in enumerate(source_chunks) if chunk.get("text") in ranked
    ]
    selected_indices = [
        index for index, chunk in enumerate(source_chunks) if chunk.get("text") in selected
    ]
    selected_structured = [
        {
            "chunk_type": chunk.get("chunk_type"),
            "text": selected_text,
        }
        for selected_text in selected
        for chunk in source_chunks
        if (
            chunk.get("text") == selected_text
            and chunk.get("chunk_type") in ("table", "caption")
        )
    ]
    return {
        "paper_id": label["paper_id"],
        "raw_text_length": len(text),
        "ranked_indices": ranked_indices,
        "expanded_indices": selected_indices,
        "added_indices": [index for index in selected_indices if index not in ranked_indices],
        "selected_count": len(selected),
        "focused_text_length": len(focused),
        "selected_structured": selected_structured,
        "focused_text": focused,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Paper2Code benchmark extraction stages")
    parser.add_argument("label_id", help="benchmark label id, for example ddpm")
    parser.add_argument("--retrieval", choices=("production", "legacy"), default="production")
    parser.add_argument("--live", action="store_true", help="perform one live extraction before reading diagnostics")
    parser.add_argument("--focus-only", action="store_true", help="rebuild production focus context without an LLM")
    args = parser.parse_args(argv)

    label = load_label(_label_path(args.label_id))
    if args.focus_only:
        print(json.dumps(focus_only(label), indent=2))
        return 0
    path = diagnostic_path(label["paper_id"], args.retrieval)
    if args.live:
        _live_extractor(label, retrieval=args.retrieval)
    if not path.exists():
        raise FileNotFoundError(
            f"No live diagnostic at {path}. Run `python -m benchmarks.diagnose "
            f"{label['paper_id']} --retrieval {args.retrieval} --live`."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = _REQUIRED_STAGES - payload.keys()
    if missing:
        raise ValueError(f"Diagnostic {path} is missing stages: {', '.join(sorted(missing))}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
