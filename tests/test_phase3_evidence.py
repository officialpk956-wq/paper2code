"""Phase 3A regression tests: page provenance, durable chunks, honest citations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, PaperChunk
from backend.services.paper_ingestion_service import ingest_pdf_paper
from core.evidence_tracking import build_evidence_map
from core.utils import chunk_pages_with_provenance


def test_page_chunks_keep_page_section_and_flattened_offsets():
    chunks = chunk_pages_with_provenance(
        [
            (1, "Abstract\nWe introduce a residual network.\n"),
            (2, "1. Methods\nStage 1 has 3 bottleneck blocks with 64 channels.\n"),
        ]
    )

    assert [chunk["page"] for chunk in chunks] == [1, 2]
    assert chunks[0]["section"] == "abstract"
    assert chunks[1]["section"] == "method"
    assert chunks[1]["source_offset_start"] > chunks[0]["source_offset_end"]
    assert all(chunk["chunk_type"] == "text" for chunk in chunks)


def test_citation_tracking_only_accepts_quotes_found_in_real_chunks():
    spec = {
        "model_family": "resnet",
        "stages": [{"num_blocks": 3, "out_channels": 64}],
        "optional_value": None,
    }
    chunks = [
        {
            "id": 41,
            "page": 2,
            "text": "Stage 1 has 3 bottleneck blocks with 64 channels.",
        }
    ]

    evidence = build_evidence_map(
        spec,
        chunks,
        complete=lambda _: (
            '{"stages[0].num_blocks": "Stage 1 has 3 bottleneck blocks", '
            '"stages[0].out_channels": "made up statement that is not in the paper"}'
        ),
    )

    assert evidence["stages[0].num_blocks"]["status"] == "cited"
    assert evidence["stages[0].num_blocks"]["chunk_ids"] == [41]
    assert evidence["stages[0].num_blocks"]["pages"] == [2]
    assert evidence["stages[0].out_channels"]["status"] == "inferred"
    assert evidence["optional_value"]["status"] == "default"


def test_ingestion_persists_source_chunks_with_real_page_numbers():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    source_chunks = [
        {
            "section": "abstract",
            "page": 1,
            "chunk_type": "text",
            "text": "We introduce a compact residual network.",
            "source_offset_start": 0,
            "source_offset_end": 39,
        },
        {
            "section": "method",
            "page": 2,
            "chunk_type": "text",
            "text": "Stage 1 has 3 bottleneck blocks with 64 channels.",
            "source_offset_start": 41,
            "source_offset_end": 92,
        },
    ]
    result = {
        "spec": {"model_family": "resnet", "input": {"channels": 3, "spatial_dims": [224, 224]}},
        "graph": MagicMock(nodes=[], edges=[]),
        "code": "import torch\n",
        "code_source": "builder",
        "family": "resnet",
        "generation_status": "success",
        "verification_report": {"passed": True},
        "source_chunks": source_chunks,
    }
    ingestion = {
        "source_filename": "fixture.pdf",
        "title": "fixture",
        "page_count": 2,
        "text_extraction_method": "fixture",
        "sections": [],
        "section_count": 0,
        "figures": [],
        "figure_count": 0,
        "equations": [],
        "equation_count": 0,
        "raw_text_excerpt": "fixture",
        "source_chunks": source_chunks,
    }
    paper_meta = {"abstract": "fixture", "architecture_graph": {}, "flops_analysis": {"total_params_estimate": 0, "total_flops_score": 0}}

    try:
        with (
            patch("backend.services.paper_ingestion_service.build_ingestion_payload", return_value=ingestion),
            patch("backend.services.paper_ingestion_service._GENERATOR.from_pdf", return_value=result),
            patch("backend.services.paper_ingestion_service.generate_modules", return_value=(paper_meta, [])),
            patch("backend.services.paper_ingestion_service.classify_architecture", return_value="resnet"),
            patch("backend.services.paper_ingestion_service.build_knowledge_graph", return_value={}),
            patch("backend.services.paper_ingestion_service.reconstruct_architecture", return_value={}),
            patch("backend.services.paper_ingestion_service.compile_blueprint") as compile_blueprint,
            patch("backend.services.paper_ingestion_service.GROQ_API_KEY", None),
        ):
            compile_blueprint.return_value.to_dict.return_value = {}
            outcome = ingest_pdf_paper(session, b"%PDF-fixture", "fixture.pdf")

        rows = session.query(PaperChunk).filter_by(paper_id=outcome["paper_id"]).all()
        assert len(rows) == 2
        assert [row.page for row in rows] == [1, 2]
        assert all(row.text for row in rows)
        assert outcome["verification_report"]["evidence"]["model_family"]["status"] == "inferred"
    finally:
        session.close()
