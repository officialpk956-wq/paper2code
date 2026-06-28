import io
import logging
import datetime
from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.repositories.task_repository import TaskRepository
from backend.models import Paper, Notification, User
from backend.services.storage_service import fetch_pdf, cleanup, r2_key_from_ref
from core.paper_to_code_generator import PaperToCodeGenerator

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
        repo.set_stage(task_id, "analyzing")
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_pages = [page.extract_text() for page in pdf.pages[:30] if page.extract_text()]
        extracted_text = "\n\n".join(text_pages)

        repo.set_stage(task_id, "generating")
        from core.agents.ingestion_agent import run_ingestion
        agent_result = run_ingestion(
            paper_id=0,
            paper_name=paper_name,
            raw_text=extracted_text,
        )
        generated_code = agent_result["final_code"]
        if agent_result.get("error"):
            log.warning("Ingestion agent error for paper %s: %s",
                           0, agent_result["error"])
                           
        result = {
            "graph": {},
            "code": generated_code,
            "code_source": "agent",
            "family": "unknown"
        }

        # ── Stage 3: persist Paper row ────────────────────────────────────────
        repo.set_stage(task_id, "saving")
        existing = db.query(Paper).filter_by(title=paper_name).first()
        if existing:
            paper = existing
        else:
            import dataclasses
            graph_json = (
                dataclasses.asdict(result["graph"])
                if dataclasses.is_dataclass(result["graph"])
                else {}
            )
            paper = Paper(
                title=paper_name,
                architecture_graph=graph_json,
                uploaded_by=user_id,
                visibility=visibility,
                r2_key=r2_key_from_ref(storage_ref),
                terms_accepted_at=(
                    datetime.datetime.utcnow() if terms_accepted else None
                ),
            )
            db.add(paper)
            db.commit()
            db.refresh(paper)

        repo.set_complete(task_id, {
            "paper_id":    paper.id,
            "code":        result.get("code", ""),
            "code_source": result.get("code_source", "skeleton"),
            "family":      result.get("family", "unknown"),
            "stage":       "complete",
        })

        # ── Stage 5: index in vector store ───────────────────────────────────
        try:
            from backend.services.vector_service import index_paper
            index_paper(
                paper_id=paper.id,
                title=paper_name,
                abstract=getattr(paper, "abstract", "") or "",
                authors=getattr(paper, "authors", "") or "",
            )
        except Exception as _ve:
            log.warning("vector index failed (non-fatal): %s", _ve)

        # ── Stage 4: notify user ──────────────────────────────────────────────
        if user_id:
            _notify_paper_done(db, user_id, paper_name, paper.id)

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
