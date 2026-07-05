from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.repositories.task_repository import TaskRepository
from backend.services.dojo_execution_service import execute_python, RUN_TIMEOUT_MS
import asyncio


def _mark_best_submission(db, user_id: int, problem_id: str, submission_id: int, time_ms) -> None:
    """
    Mark `submission_id` as is_best=True when it improves the user's best time.
    Also unmarks the previous best. Only applies to passing submissions.
    """
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
    from backend.models import DojoSubmission, Problem
    from sqlalchemy import func
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


@celery_app.task(bind=True, max_retries=1, time_limit=60)
def run_dojo_submission_task(self, task_id: str, code: str, stdin: str = ""):
    db = SessionLocal()
    try:
        TaskRepository(db).set_running(task_id)

        # Resolve per-problem timeout + judge harness
        task = TaskRepository(db).get(task_id)
        run_timeout_ms = RUN_TIMEOUT_MS
        exec_code = code
        if task and task.input_ref:
            from backend.models import Problem
            prob = db.query(Problem).filter_by(id=task.input_ref).first()
            if prob and prob.time_limit_ms:
                run_timeout_ms = prob.time_limit_ms
            # Graded submissions run the user's code with the problem's assert
            # harness appended — passed == harness asserts all hold (exit 0).
            if prob and isinstance(prob.test_cases, list):
                for tc in prob.test_cases:
                    if isinstance(tc, dict) and tc.get("type") == "harness" and tc.get("code"):
                        exec_code = code + "\n\n# ── judge harness ──\n" + tc["code"]
                        break

        # Run execute_python in a dedicated thread (avoids running event loop conflicts)
        import threading
        res = []
        err = []

        def target():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res.append(loop.run_until_complete(
                    execute_python(exec_code, stdin, run_timeout_ms=run_timeout_ms)
                ))
            except Exception as ex:
                err.append(ex)
            finally:
                loop.close()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        join_timeout = run_timeout_ms / 1000 + 30  # 30s grace beyond code run timeout
        t.join(timeout=join_timeout)
        if t.is_alive():
            raise TimeoutError(
                f"Execution thread did not finish within {join_timeout:.0f}s — "
                "Piston may be unresponsive"
            )
        if err:
            raise err[0]
        result = res[0]
        TaskRepository(db).set_complete(task_id, result)

        # Save DojoSubmission and update XP / streak / acceptance rate / is_best
        task = TaskRepository(db).get(task_id)
        if task and task.user_id:
            from backend.models import DojoSubmission, Problem
            from backend.services.progress_service import update_user_activity, award_xp

            problem = db.query(Problem).filter_by(id=task.input_ref).first()
            prob_version = problem.version if problem else 1

            submission = DojoSubmission(
                user_id=task.user_id,
                problem_id=task.input_ref,
                code=code,
                passed=result.get("passed", False),
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                time_ms=result.get("time_ms"),
                problem_version=prob_version,
            )
            db.add(submission)
            db.commit()
            db.refresh(submission)

            update_user_activity(db, task.user_id)
            if submission.passed:
                # Track best submission for this user+problem
                _mark_best_submission(db, task.user_id, task.input_ref, submission.id, submission.time_ms)

                difficulty = problem.difficulty.lower() if (problem and problem.difficulty) else "easy"
                event = f"dojo.solved.{difficulty}"
                award_xp(db, task.user_id, event, entity_id=task.input_ref)
                try:
                    from backend.services.achievement_service import check_and_award
                    check_and_award(db, task.user_id, event, {"problem_id": task.input_ref})
                except Exception:
                    pass
            else:
                award_xp(db, task.user_id, "dojo.attempt", entity_id=task.input_ref)

            # Update acceptance rate on every submission
            _update_acceptance_rate(db, task.input_ref)

    except Exception as e:
        if self.request.retries >= self.max_retries:
            try:
                TaskRepository(db).set_failed(task_id, str(e))
                db.commit()
            except Exception:
                db.rollback()
        raise self.retry(exc=e, countdown=3)
    finally:
        db.close()
