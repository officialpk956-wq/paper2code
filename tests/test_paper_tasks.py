import pytest
from unittest.mock import patch, MagicMock
from backend.tasks.paper_tasks import generate_code_from_pdf_task
from backend.models import Paper, Task

def test_generate_code_from_pdf_task_rewired(db_session):
    # Setup mock data & mocks
    task_id = "test-task-123"
    storage_ref = "r2://temp/paper.pdf"
    paper_name = "Attention Is All You Need"
    user_id = 999
    visibility = "private"
    terms_accepted = True

    # Seed Task row
    from backend.repositories.task_repository import TaskRepository
    repo = TaskRepository(db_session)
    task_row = repo.create("paper.codegen", user_id, paper_name)
    task_row.id = task_id
    db_session.commit()

    # Seed a dummy paper that ingest_pdf_paper would return
    dummy_paper = Paper(
        title=paper_name,
        architecture_graph={"nodes": [], "edges": [], "ingestion": {}},
    )
    db_session.add(dummy_paper)
    db_session.commit()
    db_session.refresh(dummy_paper)

    # Prevent db_session.close() from detaching objects during task execution
    original_close = db_session.close
    db_session.close = MagicMock()

    # Mock SessionLocal to return db_session (forces test to use in-memory SQLite)
    mock_db = patch("backend.tasks.paper_tasks.SessionLocal", return_value=db_session)
    # Mock fetch_pdf
    mock_fetch = patch("backend.tasks.paper_tasks.fetch_pdf", return_value=b"%PDF-1.4 dummy contents")
    # Mock ingest_pdf_paper
    mock_ingest = patch(
        "backend.services.paper_ingestion_service.ingest_pdf_paper",
        return_value={
            "paper_id": dummy_paper.id,
            "title": paper_name,
            "code": "import torch\nclass Model(torch.nn.Module):\n    pass",
            "code_source": "builder",
            "family": "transformer",
        }
    )
    # Mock vector indexing
    mock_index = patch("backend.services.vector_service.index_paper")
    # Mock notification
    mock_notify = patch("backend.tasks.paper_tasks._notify_paper_done")

    try:
        with mock_db, mock_fetch as mf, mock_ingest as mi, mock_index as mx, mock_notify as mn:
            # Run Celery task synchronously
            generate_code_from_pdf_task(
                task_id=task_id,
                storage_ref=storage_ref,
                paper_name=paper_name,
                user_id=user_id,
                visibility=visibility,
                terms_accepted=terms_accepted,
            )

            # Assertions
            mf.assert_called_once_with(storage_ref)
            mi.assert_called_once()
            mx.assert_called_once_with(
                paper_id=dummy_paper.id,
                title=paper_name,
                abstract="",
                authors="",
            )
            mn.assert_called_once()

            # Check DB updates on the Paper row
            db_session.refresh(dummy_paper)
            assert dummy_paper.uploaded_by == user_id
            assert dummy_paper.visibility == visibility
            assert dummy_paper.r2_key == "temp/paper.pdf"
            assert dummy_paper.terms_accepted_at is not None

            # Check task completion status
            task = repo.get(task_id)
            assert task.status == "completed"
            assert task.result["paper_id"] == dummy_paper.id
            assert task.result["code_source"] == "builder"
            assert task.result["family"] == "transformer"
    finally:
        db_session.close = original_close
