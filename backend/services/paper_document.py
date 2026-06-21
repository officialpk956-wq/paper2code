"""
Structured data models for the paper ingestion pipeline output.

These dataclasses represent the parsed view of a research PDF and are
distinct from the SQLAlchemy ORM models in backend/models.py.  They are
used as intermediate representations inside the ingestion service and can
be serialized to dicts for persistence inside the Paper.architecture_graph
JSON column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Section:
    id: str
    title: str
    content: str
    level: int = 1


@dataclass
class Figure:
    id: str
    page: int
    caption: Optional[str] = None
    xref: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    ext: Optional[str] = None
    has_binary: bool = False


@dataclass
class Equation:
    id: str
    page: int
    text: str


@dataclass
class PaperDocumentMetadata:
    page_count: int
    file_size: int
    text_extraction_method: str
    source_filename: str


@dataclass
class PaperDocument:
    id: str
    title: str
    authors: List[str]
    abstract: str
    sections: List[Section]
    figures: List[Figure]
    equations: List[Equation]
    metadata: PaperDocumentMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "sections": [
                {"id": s.id, "title": s.title, "content": s.content, "level": s.level}
                for s in self.sections
            ],
            "figures": [
                {
                    "id": f.id,
                    "page": f.page,
                    "caption": f.caption,
                    "xref": f.xref,
                    "width": f.width,
                    "height": f.height,
                    "ext": f.ext,
                    "has_binary": f.has_binary,
                }
                for f in self.figures
            ],
            "equations": [
                {"id": e.id, "page": e.page, "text": e.text}
                for e in self.equations
            ],
            "metadata": {
                "page_count": self.metadata.page_count,
                "file_size": self.metadata.file_size,
                "text_extraction_method": self.metadata.text_extraction_method,
                "source_filename": self.metadata.source_filename,
            },
        }
