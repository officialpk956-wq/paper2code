"""Hermetic Phase 1 upload -> task -> paper -> workspace API integration."""

from unittest.mock import MagicMock, patch

from backend.models import Paper
from backend.tasks.paper_tasks import generate_code_from_pdf_task


def test_e2e_paper_upload_to_workspace(
    client, db_session, regular_user_headers, tmp_path
):
    stored_pdf = tmp_path / "phase1-e2e.pdf"
    stored_pdf.write_bytes(b"%PDF-1.4 deterministic phase 1 fixture")
    source = "import torch\nprint('phase1-e2e-ready')"
    report = {
        "passed": True,
        "status": "success",
        "entrypoint_class": "Phase1Model",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
    }

    def deterministic_ingest(*, db, pdf_bytes, source_filename, paper_name):
        assert pdf_bytes.startswith(b"%PDF-")
        paper = Paper(
            title=paper_name,
            visibility="private",
            architecture_graph={"nodes": [], "edges": [], "ingestion": {}},
            generated_code_source=source,
            generated_code_compiled={"language": "python", "framework": "pytorch"},
            generation_status="success",
            verification_report=report,
        )
        db.add(paper)
        db.commit()
        db.refresh(paper)
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "code": source,
            "code_source": "builder",
            "family": "resnet",
            "generation_status": "success",
            "verification_report": report,
        }

    original_close = db_session.close
    db_session.close = MagicMock()
    try:
        with (
            patch("backend.services.storage_service.store_pdf", return_value=str(stored_pdf)),
            patch("backend.tasks.paper_tasks.SessionLocal", return_value=db_session),
            patch(
                "backend.services.paper_ingestion_service.ingest_pdf_paper",
                side_effect=deterministic_ingest,
            ),
            patch("backend.services.vector_service.index_paper"),
            patch("backend.tasks.paper_tasks._notify_paper_done"),
            patch.object(
                generate_code_from_pdf_task,
                "delay",
                side_effect=lambda *args: generate_code_from_pdf_task(*args),
            ),
        ):
            upload = client.post(
                "/api/papers/upload",
                headers=regular_user_headers,
                data={"terms_accepted": "true", "visibility": "private"},
                files={
                    "file": (
                        "phase1-e2e.pdf",
                        stored_pdf.read_bytes(),
                        "application/pdf",
                    )
                },
            )

            assert upload.status_code == 200, upload.text
            envelope = upload.json()
            assert envelope["paper_id"] is None

            task = client.get(envelope["poll_url"], headers=regular_user_headers)
            assert task.status_code == 200, task.text
            assert task.json()["status"] == "completed"
            paper_id = task.json()["result"]["paper_id"]

            detail = client.get(f"/api/papers/{paper_id}", headers=regular_user_headers)
            assert detail.status_code == 200, detail.text
            assert detail.json()["generated_code_source"] == source
            assert detail.json()["verification_report"]["passed"] is True

            executable = client.get(
                f"/api/papers/{paper_id}/executable-graph",
                headers=regular_user_headers,
            )
            assert executable.status_code == 200, executable.text
            assert executable.json()["code"] == source
    finally:
        db_session.close = original_close
