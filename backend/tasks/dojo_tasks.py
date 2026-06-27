from backend.celery_app import celery_app
from backend.database import SessionLocal
from backend.repositories.task_repository import TaskRepository
from backend.services.dojo_execution_service import execute_python
import asyncio

@celery_app.task(bind=True, max_retries=1, time_limit=30)
def run_dojo_submission_task(self, task_id: str, code: str, stdin: str = ""):
    db = SessionLocal()
    try:
        TaskRepository(db).set_running(task_id)
        # Run execute_python in a new loop in a separate thread to avoid "cannot be called from a running event loop"
        import threading
        res = []
        err = []
        def target():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res.append(loop.run_until_complete(execute_python(code, stdin)))
            except Exception as ex:
                err.append(ex)
            finally:
                loop.close()
        t = threading.Thread(target=target)
        t.start()
        t.join()
        if err:
            raise err[0]
        result = res[0]
        TaskRepository(db).set_complete(task_id, result)
        
        # Save DojoSubmission and update XP/streak
        task = TaskRepository(db).get(task_id)
        if task and task.user_id:
            from backend.models import DojoSubmission, Problem
            from backend.services.progress_service import update_user_activity, award_xp
            
            submission = DojoSubmission(
                user_id=task.user_id,
                problem_id=task.input_ref,
                code=code,
                passed=result.get("passed", False),
                stdout=result.get("stdout"),
                stderr=result.get("stderr"),
                time_ms=result.get("time_ms"),
            )
            db.add(submission)
            db.commit()
            
            problem = db.query(Problem).filter_by(id=task.input_ref).first()
            update_user_activity(db, task.user_id)
            if submission.passed:
                difficulty = problem.difficulty.lower() if (problem and problem.difficulty) else "easy"
                event = f"dojo.solved.{difficulty}"
                award_xp(db, task.user_id, event)
            else:
                award_xp(db, task.user_id, "dojo.attempt")
    except Exception as e:
        TaskRepository(db).set_failed(task_id, str(e))
        raise self.retry(exc=e, countdown=3)
    finally:
        db.close()
