import logging
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import datetime as _dt
from backend.server import limiter

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import Problem, DojoSubmission, User

from core.dojo import get_exercise_list, get_public_exercise, get_solution

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Dojo"])

@router.get("/problems")
def get_problems(db: Session = Depends(get_db)):
    return db.query(Problem).filter(Problem.is_retired == False).all()

@router.get("/problems/{problem_id}")
def get_problem(problem_id: str, db: Session = Depends(get_db)):
    prob = db.query(Problem).filter_by(id=problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")
    return prob


class DojoExerciseSubmitRequest(BaseModel):
    exercise_id: str
    passed: bool
    attempts: int = 1

@router.get("/dojo/exercises")
def dojo_list_exercises():
    return {"exercises": get_exercise_list()}

@router.get("/dojo/exercises/{exercise_id}")
def dojo_get_exercise(exercise_id: str):
    ex = get_public_exercise(exercise_id)
    if not ex:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return ex

@router.get("/dojo/exercises/{exercise_id}/solution")
def dojo_get_solution(exercise_id: str):
    sol = get_solution(exercise_id)
    if not sol:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return sol

@router.post("/dojo/submit_exercise") # Adjusted path slightly to avoid collision with the piston submit
def dojo_submit(
    request: DojoExerciseSubmitRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    if not get_solution(request.exercise_id):
        raise HTTPException(status_code=404, detail=f"Exercise '{request.exercise_id}' not found")
    try:
        from backend.models import AssessmentAttempt
        attempt = AssessmentAttempt(
            learner_id=x_learner_id or "default",
            assessment_type="code",
            architecture=None,
            difficulty=None,
            question_text=f"dojo:{request.exercise_id}",
            user_answer="passed" if request.passed else "failed",
            correct_answer="passed",
            explanation="",
            score=1 if request.passed else 0,
            attempt_count=request.attempts,
            is_correct=bool(request.passed),
        )
        db.add(attempt)
        db.commit()
        
        # Award XP and update user activity
        from backend.services.progress_service import update_user_activity, award_xp
        user = None
        if x_learner_id:
            try:
                user = db.query(User).filter(User.id == int(x_learner_id)).first()
            except ValueError:
                user = db.query(User).filter(User.name == x_learner_id).first()
        if user:
            update_user_activity(db, user.id)
            if request.passed:
                award_xp(db, user.id, "dojo.solved.easy")
                try:
                    from backend.services.analytics_service import track
                    from backend.services.achievement_service import check_and_award
                    track(user.id, "problem_solved", {"exercise_id": request.exercise_id})
                    check_and_award(db, user.id, "dojo.solved.easy", {"exercise_id": request.exercise_id})
                except Exception:
                    pass
            else:
                award_xp(db, user.id, "dojo.attempt")

        return {"status": "ok", "recorded": True, "exercise_id": request.exercise_id, "passed": request.passed}
    except Exception as e:
        db.rollback()
        logger.error(f"Dojo submit error: {str(e)}")
        return {"status": "ok", "recorded": False, "detail": str(e)}

@router.get("/problems/{problem_id}/submissions")
def problem_submission_history(
    problem_id: str,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's submission history for a problem, newest first."""
    prob = db.query(Problem).filter_by(id=problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")

    rows = (
        db.query(DojoSubmission)
        .filter_by(user_id=current_user.id, problem_id=problem_id)
        .order_by(DojoSubmission.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "problem_id": problem_id,
        "total": len(rows),
        "submissions": [
            {
                "id":         s.id,
                "passed":     s.passed,
                "time_ms":    s.time_ms,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "stdout":     (s.stdout or "")[:500],
                "stderr":     (s.stderr or "")[:300],
            }
            for s in rows
        ],
    }


@router.get("/problems/{problem_id}/stats")
def problem_stats(problem_id: str, db: Session = Depends(get_db)):
    """
    Return acceptance rate and submission totals for a problem.
    Acceptance rate = distinct users who passed / distinct users who attempted.
    """
    prob = db.query(Problem).filter_by(id=problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")

    total_users = (
        db.query(func.count(func.distinct(DojoSubmission.user_id)))
        .filter_by(problem_id=problem_id)
        .scalar()
    ) or 0

    passed_users = (
        db.query(func.count(func.distinct(DojoSubmission.user_id)))
        .filter_by(problem_id=problem_id, passed=True)
        .scalar()
    ) or 0

    total_submissions = (
        db.query(func.count(DojoSubmission.id))
        .filter_by(problem_id=problem_id)
        .scalar()
    ) or 0

    acceptance_rate = round(passed_users / total_users, 4) if total_users > 0 else 0.0

    return {
        "problem_id":       problem_id,
        "total_users":      total_users,
        "passed_users":     passed_users,
        "total_submissions": total_submissions,
        "acceptance_rate":  acceptance_rate,
    }


class DojoCodeSubmitRequest(BaseModel):
    problem_id: str
    code: str
    stdin: Optional[str] = None

@router.post("/dojo/submit")
@limiter.limit("60/hour")
async def submit_dojo_code(
    req: DojoCodeSubmitRequest,
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.repositories.task_repository import TaskRepository
    from backend.tasks.dojo_tasks import run_dojo_submission_task

    prob = db.query(Problem).filter_by(id=req.problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")

    task = TaskRepository(db).create(
        "dojo.execute",
        current_user.id,
        req.problem_id,
    )
    run_dojo_submission_task.delay(task.id, req.code, req.stdin or "")

    return {
        "task_id": task.id,
        "status": "pending",
        "poll_url": f"/api/tasks/{task.id}",
    }

