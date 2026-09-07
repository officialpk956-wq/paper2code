"""Honest citation tracking for extracted architecture fields.

The extractor is allowed to infer values, but it is not allowed to present an
invented citation as paper evidence.  This module therefore treats LLM output
only as a *candidate quote*: a field is cited solely after the quote is found
in a real, persisted chunk.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Callable


FUZZY_QUOTE_THRESHOLD = 0.92
MIN_QUOTE_LENGTH = 12


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def _leaf_fields(value: Any, path: str = "") -> dict[str, Any]:
    """Flatten a JSON-like extraction result into stable, displayable paths."""
    fields: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            fields.update(_leaf_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.update(_leaf_fields(child, f"{path}[{index}]"))
    else:
        fields[path] = value
    return fields


def _quote_prompt(spec: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    fields = _leaf_fields(spec)
    # A bounded source sample keeps the additive evidence pass predictable.
    source = "\n\n".join(str(chunk.get("text") or "") for chunk in chunks)[:18000]
    requested = {path: value for path, value in fields.items() if value not in (None, "")}
    return f"""You are verifying an already-extracted neural-network architecture.
Do not extract new values. For each field below, return a short exact phrase
from the supplied paper text that supports that field, only when one exists.
If there is no explicit support, omit the field. Never invent a quote.

Return JSON only in this format:
{{"field.path": "short exact supporting phrase"}}

Extracted fields:
{json.dumps(requested, ensure_ascii=False)}

Paper text:
{source}
"""


def _parse_quote_map(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(path): quote.strip()
        for path, quote in parsed.items()
        if isinstance(quote, str) and len(_normalise(quote)) >= MIN_QUOTE_LENGTH
    }


def _matching_chunks(quote: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised_quote = _normalise(quote)
    matches: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        text = _normalise(str(chunk.get("text") or ""))
        if not isinstance(chunk_id, int) or not text:
            continue
        if normalised_quote in text:
            matches.append(chunk)
            continue
        # Fuzzy matching is deliberately conservative and only compares the
        # candidate phrase against same-length windows in the real chunk.
        width = len(normalised_quote)
        if width > len(text):
            continue
        step = max(1, width // 4)
        if any(
            SequenceMatcher(None, normalised_quote, text[start : start + width]).ratio()
            >= FUZZY_QUOTE_THRESHOLD
            for start in range(0, len(text) - width + 1, step)
        ):
            matches.append(chunk)
    return matches


def build_evidence_map(
    spec: dict[str, Any],
    chunks: list[dict[str, Any]],
    complete: Callable[[str], str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return an evidence entry for every extracted leaf field.

    ``complete`` is injected so failures (or unavailable credentials) degrade
    cleanly to inferred/default statuses without affecting code extraction.
    """
    fields = _leaf_fields(spec)
    evidence: dict[str, dict[str, Any]] = {
        path: {
            "status": "default" if value in (None, "") else "inferred",
            "chunk_ids": [],
            "value": value,
        }
        for path, value in fields.items()
    }
    if not fields or not chunks or complete is None:
        return evidence

    try:
        quotes = _parse_quote_map(complete(_quote_prompt(spec, chunks)))
    except Exception:
        return evidence

    for path, quote in quotes.items():
        if path not in evidence:
            continue
        matched_chunks = _matching_chunks(quote, chunks)
        if matched_chunks:
            evidence[path].update(
                {
                    "status": "cited",
                    "chunk_ids": [int(chunk["id"]) for chunk in matched_chunks],
                    "pages": sorted(
                        {
                            int(chunk["page"])
                            for chunk in matched_chunks
                            if isinstance(chunk.get("page"), int)
                        }
                    ),
                    "quote": quote,
                }
            )
    return evidence
