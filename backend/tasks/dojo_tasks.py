from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.repositories.task_repository import TaskRepository
from backend.services.dojo_execution_service import RUN_TIMEOUT_MS


def _mark_best_submission(db, user_id: int, problem_id: str, submission_id: int, time_ms) -> None:
    """Mark `submission_id` as is_best=True when it improves the user's best time."""
    from backend.models import DojoSubmission

    prev_best = (
        db.query(DojoSubmission)
        .filter_by(user_id=user_id, problem_id=problem_id, is_best=True)
        .first()
    )
    is_better = (
        prev_best is None
        or prev_best.time_ms is None
        or time_ms is None
        or time_ms < prev_best.time_ms
    )
    if is_better:
        if prev_best:
            prev_best.is_best = False
        sub = db.query(DojoSubmission).filter_by(id=submission_id).first()
        if sub:
            sub.is_best = True
    db.commit()


def _update_acceptance_rate(db, problem_id: str) -> None:
    """Recalculate and persist Problem.acceptance_rate after each submission."""
    from sqlalchemy import func

    from backend.models import DojoSubmission, Problem

    total = (
        db.query(func.count(func.distinct(DojoSubmission.user_id)))
        .filter_by(problem_id=problem_id)
        .scalar()
    ) or 0
    passed = (
        db.query(func.count(func.distinct(DojoSubmission.user_id)))
        .filter_by(problem_id=problem_id, passed=True)
        .scalar()
    ) or 0
    rate = round(passed / total, 4) if total > 0 else 0.0
    problem = db.query(Problem).filter_by(id=problem_id).first()
    if problem:
        problem.acceptance_rate = rate
        db.commit()


def grade_and_record(db, user_id: int, problem, code: str) -> dict:
    """
    Grade `code` against `problem.test_cases` (v2 structured cases or legacy
    harness), persist a DojoSubmission with the per-case breakdown, update
    activity / XP / streak / is_best / acceptance rate / achievements, and
    return the grading result dict.

    Shared by the synchronous route and the Celery task so grading logic lives
    in exactly one place.
    """
    from backend.models import DojoSubmission
    from backend.services.dojo_grading import grade
    from backend.services.progress_service import award_xp, update_user_activity

    cpu_ms = problem.time_limit_ms or RUN_TIMEOUT_MS
    mem_mb = problem.memory_limit_mb or 256
    result = grade(problem.test_cases, code, cpu_time_ms=cpu_ms, memory_mb=mem_mb)

    submission = DojoSubmission(
        user_id=user_id,
        problem_id=problem.id,
        code=code,
        passed=result.get("passed", False),
        stdout=result.get("stdout"),
        stderr=result.get("stderr"),
        time_ms=result.get("time_ms"),
        problem_version=problem.version or 1,
        cases_json=result.get("cases"),
        num_passed=result.get("num_passed"),
        num_total=result.get("total"),
    )
    db.add(submission)
    db.commit()
    db.refresh(submission)

    update_user_activity(db, user_id)
    if submission.passed:
        _mark_best_submission(db, user_id, problem.id, submission.id, submission.time_ms)
        difficulty = problem.difficulty.lower() if problem.difficulty else "easy"
        event = f"dojo.solved.{difficulty}"
        award_xp(db, user_id, event, entity_id=problem.id)
        try:
            from backend.services.achievement_service import check_and_award

            check_and_award(db, user_id, event, {"problem_id": problem.id})
        except Exception:
            pass
    else:
        award_xp(db, user_id, "dojo.attempt", entity_id=problem.id)

    _update_acceptance_rate(db, problem.id)
    result["submission_id"] = submission.id
    return result


@celery_app.task(bind=True, max_retries=1, time_limit=120)
def run_dojo_submission_task(self, task_id: str, code: str, stdin: str = ""):
    """Async grading path (used when DOJO_SYNC_GRADING is off). Delegates to
    grade_and_record so behaviour matches the synchronous route exactly."""
    db = SessionLocal()
    try:
        repo = TaskRepository(db)
        repo.set_running(task_id)
        task = repo.get(task_id)
        if not task or not task.input_ref or not task.user_id:
            repo.set_failed(task_id, "Task missing problem/user reference")
            return
        from backend.models import Problem

        problem = db.query(Problem).filter_by(id=task.input_ref).first()
        if not problem:
            repo.set_failed(task_id, "Problem not found")
            return
        result = grade_and_record(db, task.user_id, problem, code)
        repo.set_complete(task_id, result)
    except Exception as e:
        try:
            TaskRepository(db).set_failed(task_id, str(e))
        except Exception:
            db.rollback()
        raise self.retry(exc=e, countdown=3)
    finally:
        db.close()
