"""
Section-Aware Smart Chunker for RAG pipeline.

Improvement R2: Isolates "Method" and "Architecture" sections from raw
PDF text using keyword-based heuristics, regex patterns, and scoring.
No LLM required — fully deterministic.

Pipeline:
    raw_pdf_text
        -> split_into_sections()       # Detect section boundaries
        -> score_sections()            # Score each section for relevance
        -> get_architecture_text()     # Return only the best section(s)
        -> chunk_for_retrieval()       # Split into retrieval-sized chunks
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Section boundary detection patterns
# Ordered: highest priority first (most explicit headings)
# ---------------------------------------------------------------------------

# Headings that indicate architecture/method content
_ARCH_HEADING_PATTERNS = [
    r"^\s*\d+\.?\s*(?:our\s+)?(?:proposed\s+)?(?:architecture|network|model)\b",
    r"^\s*\d+\.?\s*method(?:ology)?\b",
    r"^\s*\d+\.?\s*(?:approach|framework|design)\b",
    r"^\s*\d+\.?\s*(?:network|model)\s+(?:design|structure|overview)\b",
    r"^\s*\b(?:architecture|method|proposed\s+method)\b\s*$",
]

# Headings that signal the END of architecture content (stop-words)
_STOP_HEADING_PATTERNS = [
    r"^\s*\d+\.?\s*(?:experiments?|experimental|evaluation|ablation)\b",
    r"^\s*\d+\.?\s*(?:results?|findings?|discussion)\b",
    r"^\s*\d+\.?\s*(?:conclusion|future\s+work|acknowledgement)\b",
    r"^\s*\d+\.?\s*related\s+work\b",
    r"^\s*\d+\.?\s*introduction\b",
    r"^\s*\d+\.?\s*abstract\b",
]

# Architecture-related keyword density indicators
_ARCH_KEYWORDS = [
    r"\bconv(?:olution(?:al)?)?\b",
    r"\bkernel(?:_size)?\b",
    r"\bstride\b",
    r"\bchannels?\b",
    r"\bfilters?\b",
    r"\bpooling\b",
    r"\bbatch\s*norm\b",
    r"\blayer\s*norm\b",
    r"\bdropout\b",
    r"\blinear\b",
    r"\battention\b",
    r"\btransformer\b",
    r"\bresidual\b",
    r"\bskip\s*connection\b",
    r"\bencoder\b",
    r"\bdecoder\b",
    r"\bembedding\b",
    r"\bactivation\b",
    r"\brelu\b",
    r"\bgelu\b",
    r"\bsoftmax\b",
    r"\bupsample\b",
    r"\bdownsample\b",
    r"\bpatch\b",
    r"\bhead\b",
    r"\bd_model\b",
    r"\bhidden(?:_size|_dim)?\b",
    r"\bfc\b",
    r"\bmlp\b",
    r"\bfpn\b",
    r"\bneck\b",
]

# Compile all patterns once
_arch_heading_re = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in _ARCH_HEADING_PATTERNS]
_stop_heading_re = [re.compile(p, re.IGNORECASE | re.MULTILINE)
                    for p in _STOP_HEADING_PATTERNS]
_arch_keyword_re = [re.compile(p, re.IGNORECASE) for p in _ARCH_KEYWORDS]

# Generic section heading: "3. Method" or "3 Method" or "Method" on its own line
_SECTION_BOUNDARY_RE = re.compile(
    r"(?:^|\n)(\s*(?:\d+\.?\s+)?[A-Z][a-zA-Z ]{2,40})\s*\n",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextSection:
    """A segment of paper text with its heading and relevance score."""
    heading: str
    content: str
    start_char: int
    end_char: int
    arch_score: float = 0.0
    is_arch_section: bool = False
    is_stop_section: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> List[TextSection]:
    """
    Split raw PDF text into sections using heading detection.

    Args:
        text: Full raw text from a PDF.

    Returns:
        List of TextSection objects in order of appearance.
    """
    # Find all candidate section boundaries
    boundaries: List[Tuple[int, str]] = []

    for match in _SECTION_BOUNDARY_RE.finditer(text):
        heading = match.group(1).strip()
        if heading:
            boundaries.append((match.start(), heading))

    if not boundaries:
        # No boundaries found — treat entire text as one section
        return [TextSection(
            heading="Full Text",
            content=text,
            start_char=0,
            end_char=len(text),
        )]

    # Build sections from boundaries
    sections: List[TextSection] = []
    for i, (start, heading) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        content = text[start:end].strip()

        section = TextSection(
            heading=heading,
            content=content,
            start_char=start,
            end_char=end,
        )

        # Tag section types
        section.is_arch_section = _is_arch_heading(heading)
        section.is_stop_section = _is_stop_heading(heading)
        section.arch_score = _score_section(content)

        sections.append(section)

    return sections


def get_architecture_text(text: str, max_chars: int = 12_000) -> str:
    """
    Extract the most architecturally relevant text from a paper.

    Strategy:
    1. Split into sections.
    2. If explicit arch/method section found → use it.
    3. Otherwise → use top-scoring sections by keyword density.
    4. Truncate to max_chars to fit LLM context.

    Args:
        text: Full raw PDF text.
        max_chars: Maximum character budget for returned text.

    Returns:
        A focused string containing only architecturally relevant text.
    """
    sections = split_into_sections(text)

    if not sections:
        return text[:max_chars]

    # First preference: explicit arch/method headings
    arch_sections = [s for s in sections if s.is_arch_section]

    # Second preference: top scoring sections (exclude stop sections)
    if not arch_sections:
        scored = sorted(
            [s for s in sections if not s.is_stop_section],
            key=lambda s: s.arch_score,
            reverse=True,
        )
        # Take top 2 sections
        arch_sections = scored[:2]

    # Merge and truncate
    merged = "\n\n".join(s.content for s in arch_sections)
    return merged[:max_chars]


def chunk_for_retrieval(text: str, chunk_size: int = 1_000, overlap: int = 150) -> List[str]:
    """
    Split text into overlapping chunks for BM25 / semantic retrieval.

    Args:
        text: Source text to chunk.
        chunk_size: Target character size per chunk.
        overlap: Overlap between consecutive chunks (for context continuity).

    Returns:
        List of string chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Snap to nearest sentence boundary to avoid mid-sentence cuts
        if end < len(text):
            snap = text.rfind(".", start, end + 50)
            if snap != -1 and snap > start:
                end = snap + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


def score_chunks_by_density(chunks: List[str], top_k: int = 5) -> List[str]:
    """
    Score and rank chunks by architectural keyword density.

    This is a lightweight alternative to BM25 — computes keyword hit
    frequency per character to avoid bias towards longer chunks.

    Args:
        chunks: List of text chunks to score.
        top_k: Number of top chunks to return.

    Returns:
        Top-k chunks ordered by descending keyword density.
    """
    scored: List[Tuple[float, str]] = []

    for chunk in chunks:
        score = _score_section(chunk)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_arch_heading(heading: str) -> bool:
    """Return True if this heading signals an architecture/method section."""
    return any(p.search(heading) for p in _arch_heading_re)


def _is_stop_heading(heading: str) -> bool:
    """Return True if this heading signals a non-architecture section."""
    return any(p.search(heading) for p in _stop_heading_re)


def _score_section(text: str) -> float:
    """
    Compute keyword density score for a section.

    Score = (keyword_hits / text_length) * 10_000
    Normalized so length doesn't bias the result.
    """
    if not text:
        return 0.0

    hits = sum(len(p.findall(text)) for p in _arch_keyword_re)
    return (hits / max(len(text), 1)) * 10_000
