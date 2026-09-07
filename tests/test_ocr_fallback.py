"""Regression coverage for the optional, disabled-by-default OCR fallback."""

import base64
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.services import paper_ingestion_service as ingestion


def test_needs_ocr_thresholds():
    assert ingestion._needs_ocr(["", "", ""]) is True
    assert ingestion._needs_ocr(["x" * (ingestion.OCR_MIN_CHARS_PER_PAGE - 1)]) is True
    assert ingestion._needs_ocr(["x" * ingestion.OCR_MIN_CHARS_PER_PAGE]) is False
    assert ingestion._needs_ocr(["x" * 800]) is False


def test_ocr_returns_empty_when_optional_engine_is_unavailable():
    with patch.object(ingestion, "_get_ocr_engine", side_effect=ImportError):
        assert ingestion.ocr_pdf_pages(b"%PDF-empty") == []


def test_text_layer_fixture_does_not_trigger_ocr_and_reports_source():
    pdf_bytes = base64.b64decode(
        Path("tests/fixtures/phase1_architecture.pdf.b64").read_text().strip()
    )
    with patch.object(ingestion, "ocr_pdf_pages", wraps=ingestion.ocr_pdf_pages) as ocr:
        payload = ingestion.build_ingestion_payload(pdf_bytes, "fixture.pdf", "Fixture")

    ocr.assert_not_called()
    assert payload["text_source"] in {"pdfplumber", "pymupdf"}


def test_sparse_pdf_raises_actionable_scanned_pdf_error():
    import fitz

    document = fitz.open()
    document.new_page()
    pdf_bytes = document.tobytes()
    document.close()

    with pytest.raises(ValueError, match="scanned or image-only.*OCR is not enabled"):
        ingestion.extract_raw_text(pdf_bytes)
