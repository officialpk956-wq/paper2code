import datetime
import io
import logging

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.models import Notification, Paper, User
from backend.repositories.task_repository import TaskRepository
from backend.services.storage_service import cleanup, fetch_pdf, r2_key_from_ref

log = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=5)
def generate_code_from_pdf_task(
    self,
    task_id: str,
    storage_ref: str,
    paper_name: str,
    user_id: int = None,
    visibility: str = "public",
    terms_accepted: bool = False,
):
    """
    PDF → architecture spec → PyTorch code → DB.

    storage_ref is either "r2://papers/{key}" (R2 mode) or a local
    tempfile path (dev/test). Resolved by storage_service.fetch_pdf().

    Stage progression emitted to task.result.stage for frontend polling:
      pending → extracting → analyzing → generating → saving → complete
    """
    db = SessionLocal()
    repo = TaskRepository(db)
    try:
        repo.set_running(task_id)

        # ── Stage 1: fetch the PDF ────────────────────────────────────────────
        repo.set_stage(task_id, "extracting")
        pdf_bytes = fetch_pdf(storage_ref)

        # ── Stage 2: run the generator ────────────────────────────────────────
        repo.set_stage(task_id, "generating")
        from backend.services.paper_ingestion_service import ingest_pdf_paper

        ingest_result = ingest_pdf_paper(
            db=db,
            pdf_bytes=pdf_bytes,
            source_filename=f"{paper_name}.pdf",
            paper_name=paper_name,
        )

        # ── Stage 3: persist Paper row ────────────────────────────────────────
        repo.set_stage(task_id, "saving")
        paper_id = ingest_result["paper_id"]
        paper = db.query(Paper).filter_by(id=paper_id).first()
        if not paper:
            raise ValueError(f"Paper not found in database after ingestion: {paper_id}")

        paper.uploaded_by = user_id
        paper.visibility = visibility
        paper.r2_key = r2_key_from_ref(storage_ref)
        if terms_accepted:
            paper.terms_accepted_at = datetime.datetime.utcnow()
        db.commit()

        repo.set_complete(
            task_id,
            {
                "paper_id": paper.id,
                "code": ingest_result.get("code", ""),
                "code_source": ingest_result.get("code_source", "skeleton"),
                "family": ingest_result.get("family", "unknown"),
                "stage": "complete",
            },
        )

        # ── Stage 5: index in vector store ───────────────────────────────────
        try:
            from backend.services.vector_service import index_paper

            index_paper(
                paper_id=paper.id,
                title=paper.title,
                abstract=paper.abstract or "",
                authors=paper.authors or "",
            )
        except Exception as _ve:
            log.warning("vector index failed (non-fatal): %s", _ve)

        # ── Stage 4: notify user ──────────────────────────────────────────────
        if user_id:
            _notify_paper_done(db, user_id, paper.title, paper.id)

    except Exception as exc:
        repo.set_failed(task_id, str(exc))
        if self.request.retries >= self.max_retries:
            cleanup(storage_ref)
        raise self.retry(exc=exc)
    finally:
        db.close()


def _notify_paper_done(db, user_id: int, paper_name: str, paper_id: int) -> None:
    """Create an in-app notification and attempt a transactional email."""
    try:
        notif = Notification(
            user_id=user_id,
            type="paper.done",
            title=f"Paper ready: {paper_name}",
            body="Your paper has been processed and PyTorch code has been generated.",
            payload={"paper_id": paper_id},
        )
        db.add(notif)
        db.commit()
    except Exception as exc:
        log.warning("Failed to create paper.done notification: %s", exc)

    try:
        user = db.query(User).filter_by(id=user_id).first()
        if user and user.email:
            from backend.services.email_service import send_paper_done_email_sync

            send_paper_done_email_sync(user.email, paper_name, paper_id)
    except Exception as exc:
        log.warning("Failed to send paper-done email for user %s: %s", user_id, exc)
