"""Regression coverage for table, caption, and equation retrieval chunks."""

from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, PaperChunk
from backend.services.paper_ingestion_service import ingest_pdf_paper
from core.utils import (
    chunk_pages_with_provenance,
    extract_caption_chunks,
    extract_table_chunks,
)


REQUIRED_KEYS = {
    "text",
    "section",
    "page",
    "chunk_type",
    "source_offset_start",
    "source_offset_end",
}


def test_table_chunks_capture_one_complete_numeric_table():
    page = (
        "Table 1: Residual stages\n"
        "1    64    3\n"
        "2    128   4\n"
        "3    256   6\n"
        "4    512   3\n"
    )

    chunks = extract_table_chunks([(1, page)])

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "table"
    assert all(row in chunks[0]["text"] for row in ("1    64    3", "4    512   3"))


def test_caption_chunks_capture_figure_caption():
    caption = "Figure 3: The residual block adds the input to the output."

    chunks = extract_caption_chunks([(1, caption)])

    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "caption"
    assert caption in chunks[0]["text"]


def test_prose_does_not_create_structured_chunks():
    prose = "This paragraph explains the model in ordinary prose without a caption or numeric table."

    assert extract_table_chunks([(1, prose)]) == []
    assert extract_caption_chunks([(1, prose)]) == []


def test_text_chunk_snapshot_and_structured_chunk_keys_are_stable():
    pages = [(1, "Abstract\nA short paper.\n"), (2, "Methods\nA 64 channel block.\n")]
    expected = [
        {
            "section": "abstract",
            "page": 1,
            "chunk_type": "text",
            "text": "Abstract\nA short paper.",
            "source_offset_start": 0,
            "source_offset_end": 23,
        },
        {
            "section": "method",
            "page": 2,
            "chunk_type": "text",
            "text": "Methods\nA 64 channel block.",
            "source_offset_start": 26,
            "source_offset_end": 53,
        },
    ]

    assert chunk_pages_with_provenance(pages) == expected
    structured = extract_table_chunks([(3, "1  64  3\n2  128  4\n3  256  6\n")])
    structured += extract_caption_chunks([(4, "Fig. 2: Encoder overview\n")])
    assert all(set(chunk) == REQUIRED_KEYS for chunk in structured)
    assert all(isinstance(chunk["page"], int) or chunk["page"] is None for chunk in structured)


def test_ingestion_persists_non_text_chunks_from_a_real_pdf():
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Figure 3: The residual block adds the input to the output.\n"
        "y = x + residual\n",
    )
    pdf_bytes = document.tobytes()
    document.close()

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    result = {
        "spec": {"model_family": "resnet", "input": {"channels": 3, "spatial_dims": [224, 224]}},
        "graph": MagicMock(nodes=[], edges=[]),
        "code": "import torch\n",
        "code_source": "builder",
        "family": "resnet",
        "generation_status": "success",
        "verification_report": {"passed": True},
        "source_chunks": [{"section": "other", "page": 1, "chunk_type": "text", "text": "generator text", "source_offset_start": 0, "source_offset_end": 14}],
    }
    paper_meta = {"abstract": "fixture", "architecture_graph": {}, "flops_analysis": {"total_params_estimate": 0, "total_flops_score": 0}}

    try:
        with (
            patch("backend.services.paper_ingestion_service._GENERATOR.from_pdf", return_value=result),
            patch("backend.services.paper_ingestion_service.generate_modules", return_value=(paper_meta, [])),
            patch("backend.services.paper_ingestion_service.classify_architecture", return_value="resnet"),
            patch("backend.services.paper_ingestion_service.build_knowledge_graph", return_value={}),
            patch("backend.services.paper_ingestion_service.reconstruct_architecture", return_value={}),
            patch("backend.services.paper_ingestion_service.compile_blueprint") as compile_blueprint,
            patch("backend.services.paper_ingestion_service.GROQ_API_KEY", None),
        ):
            compile_blueprint.return_value.to_dict.return_value = {}
            outcome = ingest_pdf_paper(session, pdf_bytes, "structured-fixture.pdf")

        chunk_types = {
            row.chunk_type
            for row in session.query(PaperChunk).filter_by(paper_id=outcome["paper_id"])
        }
        assert "caption" in chunk_types
        assert "equation" in chunk_types
    finally:
        session.close()
