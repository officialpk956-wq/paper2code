import logging

from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.models import DojoSubmission, Problem
from core.agents.code_review_agent import generate_code_review

logger = logging.getLogger(__name__)


@celery_app.task(name="review_submission")
def review_submission_task(submission_id: int):
    db = SessionLocal()
    try:
        sub = db.query(DojoSubmission).filter_by(id=submission_id).first()
        if not sub:
            logger.warning("Submission %s not found for review", submission_id)
            return
        prob = db.query(Problem).filter_by(id=sub.problem_id).first()
        if not prob:
            return
        review = generate_code_review(
            problem_title=prob.title or "",
            problem_description=prob.description or "",
            test_cases=prob.test_cases or [],
            user_code=sub.code or "",
            stdout=sub.stdout or "",
            stderr=sub.stderr or "",
            passed=bool(sub.passed),
            time_ms=sub.time_ms,
        )
        sub.review_text = review
        db.commit()
        logger.info("Review generated for submission %s", submission_id)
    except Exception as exc:
        logger.exception("Review task failed for %s: %s", submission_id, exc)
    finally:
        db.close()
