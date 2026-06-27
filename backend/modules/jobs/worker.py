import uuid
import datetime
import threading
import logging
from typing import Dict, Any, Callable, Optional
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.modules.jobs.models import BackgroundJob
from backend.modules.jobs.queue import QueueBackend
from backend.modules.jobs.retry import handle_job_failure

logger = logging.getLogger(__name__)

class WorkerPool:
    def __init__(self, queue_backend: QueueBackend, queue_name: str, concurrency: int = 2):
        self.queue_backend = queue_backend
        self.queue_name = queue_name
        self.concurrency = concurrency
        self.handlers: Dict[str, Callable[[Dict[str, Any], BackgroundJob, Session], None]] = {}
        
        self._threads = []
        self._stop_event = threading.Event()

    def register_handler(self, job_name: str, handler: Callable[[Dict[str, Any], BackgroundJob, Session], None]) -> None:
        self.handlers[job_name] = handler

    def start(self) -> None:
        self._stop_event.clear()
        for i in range(self.concurrency):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()

    def create_and_enqueue_job(
        self,
        db: Session,
        name: str,
        payload: Dict[str, Any],
        user_id: Optional[int] = None,
        org_id: Optional[int] = None,
        api_key_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        max_retries: int = 3
    ) -> BackgroundJob:
        """Create database entry and enqueue the job."""
        job_id = str(uuid.uuid4())
        job = BackgroundJob(
            id=job_id,
            name=name,
            correlation_id=correlation_id or str(uuid.uuid4()),
            user_id=user_id,
            organization_id=org_id,
            api_key_id=api_key_id,
            status="Queued",
            payload=payload,
            max_retries=max_retries,
            created_at=datetime.datetime.utcnow()
        )
        db.add(job)
        db.commit()

        self.queue_backend.enqueue(self.queue_name, job_id, payload)
        return job

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self.queue_backend.dequeue(self.queue_name)
                if not task:
                    self._stop_event.wait(0.2)
                    continue

                job_id, payload = task
                self._execute_job(job_id, payload)
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)

    def _execute_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        db = SessionLocal()
        try:
            job = db.get(BackgroundJob, job_id)
            if not job:
                return

            if job.status == "Cancelled":
                return

            # Update job state to Running
            job.status = "Running"
            job.started_at = datetime.datetime.utcnow()
            db.commit()

            handler = self.handlers.get(job.name)
            if not handler:
                raise ValueError(f"No handler registered for job name: {job.name}")

            # Run handler
            handler(payload, job, db)

            # Success path
            job.status = "Succeeded"
            job.progress_pct = 100
            job.completed_at = datetime.datetime.utcnow()
            db.commit()

        except Exception as e:
            logger.error(f"Job execution failed (ID: {job_id}): {e}")
            db.rollback()
            # Refetch job to avoid stale state in handles
            job = db.get(BackgroundJob, job_id)
            if job:
                handle_job_failure(db, job, str(e), self.queue_backend, self.queue_name)
        finally:
            db.close()
