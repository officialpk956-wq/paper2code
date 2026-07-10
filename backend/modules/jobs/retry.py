import datetime
import random

from sqlalchemy.orm import Session

from backend.modules.jobs.models import BackgroundJob


def calculate_backoff_delay(
    retry_count: int, base_delay: float = 2.0, max_delay: float = 60.0
) -> float:
    """Calculate exponential backoff delay with random jitter (NIST standard)."""
    # Exponential part
    delay = base_delay * (2**retry_count)
    # Cap delay
    delay = min(delay, max_delay)
    # Add jitter: +/- 10%
    jitter = random.uniform(-0.1, 0.1) * delay
    return max(0.0, delay + jitter)


def handle_job_failure(
    db: Session, job: BackgroundJob, error_message: str, queue_backend, queue_name: str
) -> None:
    """Increment retry counter, schedule retry, or move permanently to Dead Letter Queue."""
    job.retry_count += 1
    job.failure_reason = error_message[:1024]

    if job.retry_count <= job.max_retries:
        job.status = "Retrying"
        delay = calculate_backoff_delay(job.retry_count)
        # In a real environment, we'd schedule a delayed redis job. For our mock/test,
        # we can simulate re-enqueuing immediately or tracking wait time
        job.metadata_json = {**(job.metadata_json or {}), "next_retry_delay": delay}
        db.commit()

        # Re-enqueue
        queue_backend.enqueue(queue_name, job.id, job.payload or {})
    else:
        job.status = "Dead Letter"
        job.completed_at = datetime.datetime.utcnow()
        db.commit()
