import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from slowapi.util import get_remote_address
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import DojoSubmission, Problem, User
from backend.server import limiter
from core.dojo import get_exercise_list, get_public_exercise, get_solution

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Dojo"])


@router.get("/problems")
def get_problems(
    difficulty: str | None = Query(None, description="Filter by difficulty: Easy, Medium, Hard"),
    category: str | None = Query(None, description="Filter by category (partial match)"),
    q: str | None = Query(None, min_length=2, max_length=200, description="Full-text search"),
    db: Session = Depends(get_db),
):
    """List active problems with optional filtering."""
    query = db.query(Problem).filter(Problem.is_retired == False)
    if difficulty:
        query = query.filter(Problem.difficulty.ilike(difficulty))
    if category:
        query = query.filter(Problem.category.ilike(f"%{category}%"))
    if q:
        pat = f"%{q}%"
        query = query.filter(
            or_(
                Problem.title.ilike(pat),
                Problem.description.ilike(pat),
                Problem.category.ilike(pat),
            )
        )
    problems = query.all()
    return [
        {
            "id": p.id,
            "slug": p.slug,
            "title": p.title,
            "difficulty": p.difficulty,
            "category": p.category,
            "estimated_time": p.estimated_time,
            "tags": p.tags,
            "is_retired": p.is_retired,
            "version": p.version,
            "time_limit_ms": p.time_limit_ms,
            "acceptance_rate": float(p.acceptance_rate) if p.acceptance_rate is not None else None,
        }
        for p in problems
    ]


@router.get("/problems/{problem_id}")
def get_problem(problem_id: str, db: Session = Depends(get_db)):
    prob = db.query(Problem).filter_by(id=problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")
    return {
        "id": prob.id,
        "slug": prob.slug,
        "title": prob.title,
        "difficulty": prob.difficulty,
        "category": prob.category,
        "description": prob.description,
        "estimated_time": prob.estimated_time,
        "tags": prob.tags,
        "related_architectures": prob.related_architectures,
        "related_papers": prob.related_papers,
        "related_math": prob.related_math,
        "learning_points": prob.learning_points,
        "visualization_url": prob.visualization_url,
        "python_template": prob.python_template,
        "hints": prob.hints,
        "explanation": prob.explanation,
        "is_retired": prob.is_retired,
        "version": prob.version,
        "time_limit_ms": prob.time_limit_ms,
        "acceptance_rate": float(prob.acceptance_rate)
        if prob.acceptance_rate is not None
        else None,
    }


class DojoExerciseSubmitRequest(BaseModel):
    exercise_id: str = Field(..., max_length=256)
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
def dojo_get_solution(
    exercise_id: str, current_user=Depends(get_current_user), db: Session = Depends(get_db)
):
    from backend.models import DojoSubmission

    passed = (
        db.query(DojoSubmission)
        .filter(
            DojoSubmission.user_id == current_user.id,
            DojoSubmission.problem_id == exercise_id,
            DojoSubmission.passed == True,
        )
        .first()
    )

    if not passed and not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Solve the problem before viewing the solution")

    sol = get_solution(exercise_id)
    if not sol:
        raise HTTPException(status_code=404, detail=f"Exercise '{exercise_id}' not found")
    return sol


@router.post("/dojo/submissions")
# deprecated alias
@router.post("/dojo/submit_exercise", deprecated=True)
def dojo_submit(
    request: DojoExerciseSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not get_solution(request.exercise_id):
        raise HTTPException(status_code=404, detail=f"Exercise '{request.exercise_id}' not found")

    from backend import metrics

    metrics.increment("submissions_total")

    try:
        from backend.models import AssessmentAttempt

        attempt = AssessmentAttempt(
            learner_id=current_user.id,
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
        from backend.services.progress_service import award_xp, update_user_activity

        user = current_user
        if user:
            update_user_activity(db, user.id)
            if request.passed:
                award_xp(db, user.id, "dojo.solved.easy")
                try:
                    from backend.services.achievement_service import check_and_award
                    from backend.services.analytics_service import track

                    track(user.id, "problem_solved", {"exercise_id": request.exercise_id})
                    check_and_award(
                        db, user.id, "dojo.solved.easy", {"exercise_id": request.exercise_id}
                    )
                except Exception:
                    pass
            else:
                award_xp(db, user.id, "dojo.attempt")

        return {
            "status": "ok",
            "recorded": True,
            "exercise_id": request.exercise_id,
            "passed": request.passed,
        }
    except Exception as e:
        db.rollback()
        logger.exception(f"Dojo submit error: {str(e)}")
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

    total = (
        db.query(func.count(DojoSubmission.id))
        .filter_by(user_id=current_user.id, problem_id=problem_id)
        .scalar()
        or 0
    )
    rows = (
        db.query(DojoSubmission)
        .filter_by(user_id=current_user.id, problem_id=problem_id)
        .order_by(DojoSubmission.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "problem_id": problem_id,
        "total": total,
        "submissions": [
            {
                "id": s.id,
                "passed": s.passed,
                "time_ms": s.time_ms,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "stdout": (s.stdout or "")[:500],
                "stderr": (s.stderr or "")[:300],
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
        db.query(func.count(DojoSubmission.id)).filter_by(problem_id=problem_id).scalar()
    ) or 0

    acceptance_rate = round(passed_users / total_users, 4) if total_users > 0 else 0.0

    return {
        "problem_id": problem_id,
        "total_users": total_users,
        "passed_users": passed_users,
        "total_submissions": total_submissions,
        "acceptance_rate": acceptance_rate,
    }


@router.get("/dojo/submissions")
def user_submission_history(
    problem_id: str | None = Query(None, description="Filter to a specific problem"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the current user's submission history (all problems unless problem_id given), newest first."""
    query = db.query(DojoSubmission).filter_by(user_id=current_user.id)
    if problem_id:
        query = query.filter(DojoSubmission.problem_id == problem_id)
    total = query.count()
    rows = query.order_by(DojoSubmission.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "submissions": [
            {
                "id": s.id,
                "problem_id": s.problem_id,
                "passed": s.passed,
                "is_best": s.is_best,
                "time_ms": s.time_ms,
                "problem_version": s.problem_version,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "stdout": (s.stdout or "")[:500],
                "stderr": (s.stderr or "")[:300],
            }
            for s in rows
        ],
    }


def _dojo_user_key(request: Request) -> str:
    """Per-user rate limit key for dojo code submission — falls back to IP."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        try:
            import jwt as _jwt

            from backend.modules.security.jwt_rotation import resolve_signing_secret

            token = auth[7:]
            payload = _jwt.decode(
                token,
                resolve_signing_secret(token),
                algorithms=["HS256"],
                options={"verify_aud": False, "verify_iss": False, "verify_exp": False},
            )
            uid = payload.get("user_id")
            if uid is not None:
                return f"dojo:{uid}"
        except Exception:
            pass
    return get_remote_address(request)


class DojoCodeSubmitRequest(BaseModel):
    problem_id: str = Field(..., max_length=256)
    code: str = Field(..., max_length=65536)
    stdin: str | None = Field(default=None, max_length=65536)


@router.post("/dojo/code-submissions")
# deprecated alias
@router.post("/dojo/submit", deprecated=True)
@limiter.limit("30/hour", key_func=_dojo_user_key)
async def submit_dojo_code(
    req: DojoCodeSubmitRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.repositories.task_repository import TaskRepository
    from backend.tasks.dojo_tasks import run_dojo_submission_task

    prob = db.query(Problem).filter_by(id=req.problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")

    if len(req.code.encode("utf-8")) > 10000:
        raise HTTPException(status_code=400, detail="Code exceeds maximum size of 10KB")

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


# ---------------------------------------------------------------------------
# POST /api/dojo/runs — synchronous, UNGRADED execution ("Run" button)
# Executes the code as-is in the Piston sandbox and returns the result
# directly. No DojoSubmission row, no XP, no streak — unlike code-submissions
# this is a scratchpad run, and being synchronous it works without a Celery
# worker deployed.
# ---------------------------------------------------------------------------


class DojoRunRequest(BaseModel):
    problem_id: str = Field(..., max_length=256)
    code: str = Field(..., max_length=65536)
    stdin: str | None = Field(default=None, max_length=65536)


@router.post("/dojo/runs")
@limiter.limit("60/hour", key_func=_dojo_user_key)
async def run_dojo_code(
    req: DojoRunRequest,
    request: Request,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.services.dojo_execution_service import RUN_TIMEOUT_MS, execute_python

    prob = db.query(Problem).filter_by(id=req.problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")

    if len(req.code.encode("utf-8")) > 10000:
        raise HTTPException(status_code=400, detail="Code exceeds maximum size of 10KB")

    timeout_ms = prob.time_limit_ms or RUN_TIMEOUT_MS
    result = await execute_python(req.code, req.stdin or "", run_timeout_ms=timeout_ms)
    return result


# ---------------------------------------------------------------------------
# POST /api/submissions/{id}/share  — mark best submission as public (P1)
# ---------------------------------------------------------------------------


@router.post("/submissions/{submission_id}/share")
def share_submission(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.query(DojoSubmission).filter_by(id=submission_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if sub.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your submission")
    if not sub.is_best:
        raise HTTPException(status_code=400, detail="Only best submissions can be shared")
    sub.is_public = True
    db.commit()
    return {"submission_id": submission_id, "is_public": True}


# ---------------------------------------------------------------------------
# GET /api/problems/{id}/solutions  — top-N public best submissions (P1)
# ---------------------------------------------------------------------------


@router.get("/problems/{problem_id}/solutions")
def problem_solutions(
    problem_id: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    prob = db.query(Problem).filter_by(id=problem_id).first()
    if not prob:
        raise HTTPException(status_code=404, detail="Problem not found")
    rows = (
        db.query(DojoSubmission)
        .filter_by(problem_id=problem_id, is_best=True, is_public=True)
        .order_by(DojoSubmission.time_ms.asc())
        .limit(limit)
        .all()
    )
    return {
        "problem_id": problem_id,
        "total": len(rows),
        "solutions": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "time_ms": s.time_ms,
                "code": s.code,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in rows
        ],
    }


# ---------------------------------------------------------------------------
# GET /api/dojo/global-stats  — global aggregate stats (P2)
# ---------------------------------------------------------------------------


@router.get("/dojo/global-stats")
@router.get("/dojo/stats")
def dojo_global_stats(db: Session = Depends(get_db)):
    import json

    from backend.redis_config import cache_redis

    cache_key = "dojo:global_stats"
    if cache_redis:
        cached = cache_redis.get(cache_key)
        if cached:
            return json.loads(cached)

    total_users = db.query(func.count(User.id)).scalar() or 0

    total_submissions = db.query(func.count(DojoSubmission.id)).scalar() or 0

    total_problems = (
        db.query(func.count(Problem.id)).filter(Problem.is_retired == False).scalar() or 0
    )

    rates = [
        float(r[0])
        for r in db.query(Problem.acceptance_rate)
        .filter(
            Problem.is_retired == False,
            Problem.acceptance_rate.isnot(None),
        )
        .all()
    ]
    avg_acceptance = round(sum(rates) / len(rates), 4) if rates else 0.0

    res = {
        "total_problems": total_problems,
        "total_submissions": total_submissions,
        "total_users": total_users,
        "avg_acceptance_rate": avg_acceptance,
    }

    if cache_redis:
        cache_redis.setex(cache_key, 300, json.dumps(res))

    return res


# ---------------------------------------------------------------------------
# GET /api/me/dojo-stats  — per-user summary (P2)
# ---------------------------------------------------------------------------


@router.get("/me/dojo-stats")
def my_dojo_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Best passing submissions (one per problem)
    best_subs = (
        db.query(DojoSubmission).filter_by(user_id=current_user.id, is_best=True, passed=True).all()
    )
    solved_problem_ids = {s.problem_id for s in best_subs}

    # Count solved problems by difficulty
    solved_by_difficulty: dict = {}

    if solved_problem_ids:
        problems = (
            db.query(Problem.id, Problem.difficulty)
            .filter(Problem.id.in_(solved_problem_ids))
            .all()
        )
        for pid, diff_val in problems:
            diff = diff_val or "Unknown"
            solved_by_difficulty[diff] = solved_by_difficulty.get(diff, 0) + 1

    total_submissions = (
        db.query(func.count(DojoSubmission.id)).filter_by(user_id=current_user.id).scalar() or 0
    )

    problems_attempted = (
        db.query(func.count(func.distinct(DojoSubmission.problem_id)))
        .filter_by(user_id=current_user.id)
        .scalar()
        or 0
    )

    # Leaderboard rank by all-time points (1-indexed)
    higher_count = (
        db.query(func.count(User.id)).filter(User.points > (current_user.points or 0)).scalar() or 0
    )
    leaderboard_rank = higher_count + 1

    return {
        "user_id": current_user.id,
        "total_solved": len(solved_problem_ids),
        "solved_by_difficulty": solved_by_difficulty,
        "total_submissions": total_submissions,
        "problems_attempted": problems_attempted,
        "streak": current_user.streak,
        "points": current_user.points,
        "leaderboard_rank": leaderboard_rank,
    }


# ---------------------------------------------------------------------------
# POST /api/submissions/{id}/review  — trigger an AI code review
# ---------------------------------------------------------------------------


@router.post("/submissions/{submission_id}/review")
def trigger_submission_review(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.query(DojoSubmission).filter_by(id=submission_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if sub.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your submission")

    from backend.tasks.review_tasks import review_submission_task

    review_submission_task.delay(submission_id)

    return {"status": "pending", "submission_id": submission_id}


# ---------------------------------------------------------------------------
# GET /api/submissions/{id}/review  — get the AI code review text
# ---------------------------------------------------------------------------


@router.get("/submissions/{submission_id}/review")
def get_submission_review(
    submission_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sub = db.query(DojoSubmission).filter_by(id=submission_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")
    if sub.user_id != current_user.id and not sub.is_public:
        raise HTTPException(status_code=403, detail="Not your submission")

    if not sub.review_text:
        return {"status": "pending", "review_text": None}

    return {"status": "ready", "review_text": sub.review_text}
