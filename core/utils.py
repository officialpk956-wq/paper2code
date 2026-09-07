import re


SECTION_VOCABULARY = {
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "other",
}

_SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?("
    r"abstract|introduction|related\s+work|background"
    r"|method(?:ology|s)?|approach|model(?:\s+architecture)?"
    r"|experiment(?:al(?:\s+(?:setup|results))?|s)?"
    r"|results?(?:\s+and\s+discussion)?|discussion|conclusion(?:s)?"
    r")\s*$",
    re.IGNORECASE,
)

_SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "related_work",
    "methodology": "method",
    "methods": "method",
    "method": "method",
    "approach": "method",
    "model": "method",
    "model architecture": "method",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "experimental results": "experiments",
    "results": "results",
    "results and discussion": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}


def chunk_text(text, max_chars=1200):
    """
    Conservative chunking to stay under Groq TPM limits.
    1200 chars ≈ 300–400 tokens (safe).
    """
    chunks = []
    current = ""

    for line in text.splitlines():
        if len(current) + len(line) < max_chars:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_pages_with_provenance(
    pages: list[tuple[int, str]], max_chars: int = 1200
) -> list[dict]:
    """Create conservative text chunks without losing PDF page provenance.

    The existing :func:`chunk_text` algorithm is intentionally line-oriented
    to keep LLM requests small.  This companion keeps that granularity, but
    records the original page and offsets in the flattened text passed to the
    extractor.  Section assignment is deterministic heading detection: it is
    useful provenance metadata and must never require a second LLM call.
    """
    chunks: list[dict] = []
    section = "other"
    global_offset = 0

    for page_index, page_text in pages:
        text = page_text or ""
        current_start: int | None = None
        current_end: int | None = None

        def flush() -> None:
            nonlocal current_start, current_end, section
            if current_start is None or current_end is None:
                return
            raw_chunk = text[current_start:current_end]
            leading = len(raw_chunk) - len(raw_chunk.lstrip())
            trimmed = raw_chunk.strip()
            if trimmed:
                heading = _SECTION_HEADING_RE.search(trimmed.splitlines()[0])
                if heading:
                    section = _SECTION_ALIASES.get(heading.group(1).lower(), "other")
                chunks.append(
                    {
                        "section": section if section in SECTION_VOCABULARY else "other",
                        "page": page_index,
                        "chunk_type": "text",
                        "text": trimmed,
                        "source_offset_start": global_offset + current_start + leading,
                        "source_offset_end": global_offset + current_end
                        - (len(raw_chunk) - len(raw_chunk.rstrip())),
                    }
                )
            current_start = None
            current_end = None

        for match in re.finditer(r"[^\n]*(?:\n|$)", text):
            line_start, line_end = match.span()
            if line_start == line_end:
                continue
            line = match.group(0)
            if current_start is not None and line_end - current_start > max_chars:
                flush()

            # A single unusually long line is split deterministically.  It is
            # rare in normal PDFs, but prevents an unbounded prompt/chunk row.
            if current_start is None and len(line) > max_chars:
                for offset in range(0, len(line), max_chars):
                    current_start = line_start + offset
                    current_end = min(line_start + offset + max_chars, line_end)
                    flush()
                continue

            if current_start is None:
                current_start = line_start
            current_end = line_end

        flush()
        # This exactly mirrors the two newlines used by the flattened
        # extraction input, including an empty extracted page.
        global_offset += len(text) + 2

    return chunks


_TABLE_HEADER_RE = re.compile(r"^\s*Table\s+\d+", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^\s*(?:Figure|Fig\.|Table)\s+\d+[.:)]?\s+\S", re.IGNORECASE)
_NUMERICISH_TOKEN_RE = re.compile(
    r"(?<!\S)[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?(?!\S)", re.IGNORECASE
)


def _structured_chunk(
    text: str,
    section: str,
    page: int | None,
    chunk_type: str,
    start: int,
    end: int,
    page_offset: int,
) -> dict | None:
    raw = text[start:end]
    trimmed = raw.strip()
    if not trimmed:
        return None
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    return {
        "section": section,
        "page": page,
        "chunk_type": chunk_type,
        "text": trimmed,
        "source_offset_start": page_offset + start + leading,
        "source_offset_end": page_offset + end - trailing,
    }


def _append_capped_region(
    chunks: list[dict],
    text: str,
    section: str,
    page: int | None,
    chunk_type: str,
    start: int,
    end: int,
    page_offset: int,
    max_chars: int = 1200,
) -> None:
    """Append one or more line-preserving structured chunks below max_chars."""
    region = text[start:end]
    if len(region) <= max_chars:
        chunk = _structured_chunk(text, section, page, chunk_type, start, end, page_offset)
        if chunk is not None:
            chunks.append(chunk)
        return

    current_start = start
    current_end = start
    for match in re.finditer(r"[^\n]*(?:\n|$)", region):
        line_start = start + match.start()
        line_end = start + match.end()
        if line_start == line_end:
            continue
        if current_end > current_start and line_end - current_start > max_chars:
            _append_capped_region(
                chunks, text, section, page, chunk_type, current_start, current_end, page_offset, max_chars
            )
            current_start = line_start
        if line_end - current_start > max_chars:
            for offset in range(current_start, line_end, max_chars):
                _append_capped_region(
                    chunks,
                    text,
                    section,
                    page,
                    chunk_type,
                    offset,
                    min(offset + max_chars, line_end),
                    page_offset,
                    max_chars,
                )
            current_start = line_end
        current_end = line_end
    if current_end > current_start:
        _append_capped_region(
            chunks, text, section, page, chunk_type, current_start, current_end, page_offset, max_chars
        )


def _deduplicate_structured_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[tuple[str, int | None, str]] = set()
    unique: list[dict] = []
    for chunk in chunks:
        key = (chunk["chunk_type"], chunk["page"], chunk["text"])
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def extract_table_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """Detect table-like regions in page text and emit deterministic table chunks."""
    chunks: list[dict] = []
    global_offset = 0
    for page, page_text in pages:
        text = page_text or ""
        lines = [
            (match.start(), match.end(), match.group(0).rstrip("\n"))
            for match in re.finditer(r"[^\n]*(?:\n|$)", text)
            if match.start() != match.end()
        ]

        def table_like(line: str) -> bool:
            return len(_NUMERICISH_TOKEN_RE.findall(line)) >= 2 or (
                line.count("|") + line.count("\t") >= 2
            )

        run_start: int | None = None
        for index, (_, _, line) in enumerate(lines + [(0, 0, "")]):
            if index < len(lines) and table_like(line):
                if run_start is None:
                    run_start = index
                continue
            if run_start is not None and index - run_start >= 3:
                start_index = run_start
                if start_index > 0 and _TABLE_HEADER_RE.match(lines[start_index - 1][2]):
                    start_index -= 1
                _append_capped_region(
                    chunks,
                    text,
                    "other",
                    page,
                    "table",
                    lines[start_index][0],
                    lines[index - 1][1],
                    global_offset,
                )
            run_start = None
        global_offset += len(text) + 2
    return _deduplicate_structured_chunks(chunks)


def extract_caption_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """Emit caption chunks for figure and table caption lines."""
    chunks: list[dict] = []
    global_offset = 0
    for page, page_text in pages:
        text = page_text or ""
        lines = [
            (match.start(), match.end(), match.group(0).rstrip("\n"))
            for match in re.finditer(r"[^\n]*(?:\n|$)", text)
            if match.start() != match.end()
        ]
        for index, (start, end, line) in enumerate(lines):
            if not _CAPTION_RE.match(line):
                continue
            caption_end = end
            for following in lines[index + 1 : index + 3]:
                if not following[2].strip():
                    break
                caption_end = following[1]
            _append_capped_region(
                chunks, text, "other", page, "caption", start, caption_end, global_offset
            )
        global_offset += len(text) + 2
    return _deduplicate_structured_chunks(chunks)


def sanitize_llm_json(text: str) -> str:
    """
    Remove common LLM JSON violations:
    - Python-style comments (# ...)
    - C++-style comments (// ...)
    """

    # Remove # comments
    text = re.sub(r"#.*", "", text)

    # Remove // comments
    text = re.sub(r"//.*", "", text)

    return text.strip()
