import datetime
import logging
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user, get_optional_user
from backend.models import (
    AssessmentAttempt,
    InterviewQuestion,
    LearnerProgress,
    Paper,
    PaperModule,
    Roadmap,
    TutorAnalytics,
)

# Imported for its side effect of exposing the shared tutor_session_store
# singleton on this module's namespace — tests patch it via
# backend.routers.learning.tutor_session_store even though this file's own
# code doesn't call it directly (the real /tutor/ask endpoint lives in
# backend/routers/tutor.py, which imports the same singleton).
from backend.services.tutor_session_store import tutor_session_store  # noqa: F401
from core.analytics.adaptive_engine import adaptive_engine
from core.analytics.recommendation_engine import recommendation_engine
from core.llm_client import BudgetExceededError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Learning"])


def _fetch_adaptive_data(db: Session, learner_id: str):
    attempts = db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == learner_id).all()
    progress_records = (
        db.query(LearnerProgress).filter(LearnerProgress.learner_id == learner_id).all()
    )
    tutor_records = db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == learner_id).all()
    all_modules = db.query(PaperModule).all()

    attempts_data = [
        {
            "question_text": getattr(a, "question_text", None),
            "assessment_type": getattr(a, "assessment_type", None),
            "architecture": getattr(a, "architecture", None),
            "is_correct": getattr(a, "is_correct", False),
        }
        for a in attempts
    ]
    progress_data = [{"module_id": p.module_id, "status": p.status} for p in progress_records]
    tutor_data = [
        {
            "module": t.module,
            "architecture": t.architecture,
            "question_count": getattr(t, "question_count", 0),
        }
        for t in tutor_records
    ]
    modules_data = [
        {
            "id": m.id,
            "explanation": m.explanation,
            "module_type": m.module_type,
            "layer_name": m.layer_name,
        }
        for m in all_modules
    ]

    return attempts_data, progress_data, tutor_data, modules_data


def _fetch_all_papers_data(db: Session):
    all_papers = db.query(Paper).all()
    papers_data = []
    for p in all_papers:
        mods = [
            {
                "id": m.id,
                "explanation": m.explanation,
                "module_type": m.module_type,
                "layer_name": m.layer_name,
            }
            for m in p.modules
        ]
        papers_data.append(
            {
                "id": p.id,
                "title": p.title,
                "architecture_graph": p.architecture_graph,
                "modules": mods,
            }
        )
    return papers_data


def _fetch_recommendation_data(db: Session, learner_id: str):
    attempts = db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == learner_id).all()
    tutor_rows = db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == learner_id).all()

    attempts_data = [
        {
            "architecture": a.architecture,
            "assessment_type": a.assessment_type,
            "is_correct": a.is_correct,
        }
        for a in attempts
    ]
    tutor_data = [
        {"architecture": t.architecture, "module": t.module, "question_count": t.question_count}
        for t in tutor_rows
    ]
    return attempts_data, tutor_data


from backend.routers.tutor import _get_tutor_callbacks


class ProgressUpdate(BaseModel):
    status: str
    time_spent_seconds: int = 0


class ValidateRequest(BaseModel):
    challenge: dict[str, Any]
    user_answer: str


class ProgressUpdateRequest(BaseModel):
    entity_type: str
    entity_id: str
    status: str


class TutorAskRequest(BaseModel):
    session_id: str | None = Field(
        default=None, max_length=128
    )  # server-generated; null = auto-create
    context_type: str = Field(..., max_length=500)
    context_data: dict[str, Any]
    query: str = Field(..., max_length=10000)


class TutorQuizRequest(BaseModel):
    module_data: dict[str, Any]


@router.get("/progress/{entity_type}")
def get_progress(
    entity_type: str, current_user=Depends(get_current_user), db: Session = Depends(get_db)
):
    progress = (
        db.query(LearnerProgress)
        .filter_by(learner_id=str(current_user.id), entity_type=entity_type)
        .all()
    )
    return progress


@router.post("/progress/{entity_type}/{entity_id}")
def update_progress(
    entity_type: str,
    entity_id: str,
    update: ProgressUpdate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prog = (
        db.query(LearnerProgress)
        .filter_by(
            learner_id=str(current_user.id),
            entity_type=entity_type,
            entity_id=entity_id,
        )
        .first()
    )

    newly_completed = False
    if not prog:
        prog = LearnerProgress(
            learner_id=str(current_user.id),
            entity_type=entity_type,
            entity_id=entity_id,
            status=update.status,
            time_spent_seconds=update.time_spent_seconds,
            started_at=datetime.datetime.utcnow(),
        )
        if update.status == "completed":
            prog.completed_at = datetime.datetime.utcnow()
            newly_completed = True
        db.add(prog)
    else:
        prog.time_spent_seconds += update.time_spent_seconds
        if update.status == "completed" and not prog.completed_at:
            prog.completed_at = datetime.datetime.utcnow()
            newly_completed = True
        prog.status = update.status

    db.commit()
    db.refresh(prog)

    # ── XP + streak on topic / architecture completion ───────────────────────
    if newly_completed and entity_type in ("topic", "architecture"):
        try:
            from backend.services.progress_service import (
                award_xp,
                check_domain_completion,
                update_user_activity,
            )

            update_user_activity(db, current_user.id)
            award_xp(db, current_user.id, "topic.completed", entity_id=entity_id)

            completed_domain = check_domain_completion(db, current_user.id, entity_id)
            if completed_domain:
                award_xp(db, current_user.id, "domain.completed", entity_id=completed_domain)
                try:
                    from backend.models import Notification

                    db.add(
                        Notification(
                            user_id=current_user.id,
                            type="domain.completed",
                            title=f"Domain complete: {completed_domain.replace('-', ' ').title()}",
                            body="You've mastered all architectures in this domain! +500 XP awarded.",
                            payload={"domain": completed_domain},
                        )
                    )
                    db.commit()
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("XP award failed for user %s: %s", current_user.id, exc)

    return prog


@router.post("/progress/update")
def update_learner_progress(
    request: ProgressUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        progress = (
            db.query(LearnerProgress)
            .filter(
                LearnerProgress.learner_id == str(current_user.id),
                LearnerProgress.entity_type == request.entity_type,
                LearnerProgress.entity_id == request.entity_id,
            )
            .first()
        )

        now = datetime.datetime.utcnow()
        if progress:
            progress.status = request.status
            if request.status == "completed" and not progress.completed_at:
                progress.completed_at = now
            elif request.status == "in_progress" and not progress.started_at:
                progress.started_at = now
            db.commit()
            db.refresh(progress)
        else:
            progress = LearnerProgress(
                learner_id=str(current_user.id),
                entity_type=request.entity_type,
                entity_id=request.entity_id,
                status=request.status,
                started_at=now if request.status in ("in_progress", "completed") else None,
                completed_at=now if request.status == "completed" else None,
            )
            db.add(progress)
            db.commit()

        return {"status": "success", "progress_id": progress.id}
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as e:
        logger.exception(f"Progress update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard")
def get_analytics_dashboard(
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    # Authenticated requests always use the real user id — the header is only
    # a fallback for anonymous "guest" progress tracking (see test_05_auth_overrides_header_learner_id).
    if current_user is not None:
        x_learner_id = str(current_user.id)
    try:
        # Overview
        total_papers = db.query(func.count(Paper.id)).scalar() or 0
        total_modules_count = db.query(func.count(PaperModule.id)).scalar() or 0

        modules_completed = (
            db.query(func.count(LearnerProgress.id))
            .filter(
                LearnerProgress.learner_id == x_learner_id, LearnerProgress.status == "completed"
            )
            .scalar()
            or 0
        )

        papers_started = 0
        papers_completed = 0

        if total_modules_count > 0:
            papers_with_modules = (
                db.query(PaperModule.paper_id, func.count(PaperModule.id))
                .group_by(PaperModule.paper_id)
                .all()
            )
            paper_module_counts = {p_id: count for p_id, count in papers_with_modules}

            learner_progress = (
                db.query(PaperModule.paper_id, LearnerProgress.status)
                .join(PaperModule, PaperModule.id == LearnerProgress.entity_id)
                .filter(
                    LearnerProgress.learner_id == x_learner_id,
                    LearnerProgress.entity_type == "paper_module",
                )
                .all()
            )

            progress_by_paper = defaultdict(list)
            for p_id, status in learner_progress:
                progress_by_paper[p_id].append(status)

            for p_id, statuses in progress_by_paper.items():
                if any(s in ("in_progress", "completed") for s in statuses):
                    papers_started += 1
                if all(s == "completed" for s in statuses) and len(
                    statuses
                ) == paper_module_counts.get(p_id, -1):
                    papers_completed += 1

        completion_pct = (
            round(modules_completed / total_modules_count * 100, 1)
            if total_modules_count > 0
            else 0.0
        )

        overview = {
            "total_papers": total_papers,
            "papers_started": papers_started,
            "papers_completed": papers_completed,
            "modules_completed": modules_completed,
            "total_modules": total_modules_count,
            "completion_percentage": completion_pct,
        }

        # Assessment
        attempts = (
            db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == x_learner_id).all()
        )
        total_attempts = len(attempts)
        correct_attempts = sum(1 for a in attempts if a.is_correct)
        accuracy_pct = (
            round(correct_attempts / total_attempts * 100, 1) if total_attempts > 0 else 0.0
        )

        arch_attempts = defaultdict(list)
        for a in attempts:
            if a.architecture:
                arch_attempts[a.architecture].append(a.is_correct)

        strongest_arch = "None"
        weakest_arch = "None"
        if arch_attempts:
            arch_accuracies = {}
            for arch, results in arch_attempts.items():
                arch_accuracies[arch] = sum(1 for r in results if r) / len(results)
            sorted_archs = sorted(arch_accuracies.items(), key=lambda x: x[1])
            weakest_arch = f"{sorted_archs[0][0]} ({int(sorted_archs[0][1] * 100)}% accuracy)"
            strongest_arch = f"{sorted_archs[-1][0]} ({int(sorted_archs[-1][1] * 100)}% accuracy)"

        assessment = {
            "total_attempts": total_attempts,
            "accuracy_percentage": accuracy_pct,
            "strongest_architecture": strongest_arch,
            "weakest_architecture": weakest_arch,
        }

        # Tutor
        tutor_records = (
            db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == x_learner_id).all()
        )
        questions_asked = sum(r.question_count for r in tutor_records)

        context_counts = defaultdict(int)
        for r in tutor_records:
            if r.architecture:
                context_counts[r.architecture] += r.question_count
        most_asked_context = "None"
        if context_counts:
            most_asked_context = max(context_counts.items(), key=lambda x: x[1])[0]

        tutor = {"questions_asked": questions_asked, "most_asked_context": most_asked_context}

        # Learning Path
        progress_records = (
            db.query(LearnerProgress).filter(LearnerProgress.learner_id == x_learner_id).all()
        )
        active_sorted = sorted(
            [p for p in progress_records if p.started_at], key=lambda x: x.started_at, reverse=True
        )
        current_position = "Get started by reading a paper!"
        next_recommended = "None"
        first_paper = db.query(Paper).order_by(Paper.id.asc()).first()
        if active_sorted:
            latest = active_sorted[0]
            m = db.query(PaperModule).filter(PaperModule.id == latest.module_id).first()
            p = db.query(Paper).filter(Paper.id == latest.paper_id).first()
            if m and p:
                current_position = f"{p.title} - {m.layer_name}"
                next_mod = (
                    db.query(PaperModule)
                    .filter(PaperModule.paper_id == p.id, PaperModule.order_index > m.order_index)
                    .order_by(PaperModule.order_index.asc())
                    .first()
                )
                if next_mod:
                    next_recommended = f"{p.title} - {next_mod.layer_name}"
                else:
                    next_paper = (
                        db.query(Paper).filter(Paper.id > p.id).order_by(Paper.id.asc()).first()
                    )
                    if next_paper and next_paper.modules:
                        next_recommended = (
                            f"{next_paper.title} - {next_paper.modules[0].layer_name}"
                        )
                    else:
                        next_recommended = "You have completed all available papers!"
        elif first_paper and first_paper.modules:
            next_recommended = f"{first_paper.title} - {first_paper.modules[0].layer_name}"

        learning_path = {"current_position": current_position, "next_recommended": next_recommended}

        learning_path_items = []
        if current_position != "Get started by reading a paper!":
            learning_path_items.append(
                {
                    "title": current_position,
                    "url": "",
                    "type": "current",
                }
            )
        if next_recommended not in ("None", "You have completed all available papers!"):
            learning_path_items.append(
                {
                    "title": next_recommended,
                    "url": "",
                    "type": "next",
                }
            )

        recs = recommendation_engine.compute(*_fetch_recommendation_data(db, x_learner_id))

        return {
            "learning_overview": overview,
            "assessment_performance": assessment,
            "tutor_usage": tutor,
            "learning_path": learning_path,
            "learning_path_items": learning_path_items,
            "reviews_and_recommendations": recs,
        }
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as e:
        logger.exception(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/recommendations")
def get_analytics_recommendations(
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    if current_user is not None:
        x_learner_id = str(current_user.id)
    try:
        recs = recommendation_engine.compute(*_fetch_recommendation_data(db, x_learner_id))
        return recs
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as e:
        logger.exception(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive/recommendations")
def get_adaptive_recommendations(
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    if current_user is not None:
        x_learner_id = str(current_user.id)
    try:
        recs = adaptive_engine.get_personalized_recommendations(
            *_fetch_adaptive_data(db, x_learner_id), _fetch_all_papers_data(db)
        )
        return {"recommendations": recs}
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as e:
        logger.exception(f"Adaptive recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive/review-plan")
def get_adaptive_review_plan(
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    if current_user is not None:
        x_learner_id = str(current_user.id)
    try:
        plan = adaptive_engine.get_daily_review_plan(
            *_fetch_adaptive_data(db, x_learner_id), _fetch_all_papers_data(db)
        )
        return plan
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as e:
        logger.exception(f"Adaptive review plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptive/concept-graph")
def get_adaptive_concept_graph(
    db: Session = Depends(get_db),
    current_user=Depends(get_optional_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    if current_user is not None:
        x_learner_id = str(current_user.id)
    try:
        graph_nodes = adaptive_engine.get_concept_graph(*_fetch_adaptive_data(db, x_learner_id))
        return {"nodes": graph_nodes}
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as e:
        logger.exception(f"Concept graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


@router.get("/interviews/questions")
def get_interview_questions(db: Session = Depends(get_db)):
    return db.query(InterviewQuestion).all()


@router.get("/roadmaps")
def get_roadmaps(db: Session = Depends(get_db)):
    return db.query(Roadmap).all()


# ---------------------------------------------------------------------------
# GET /api/recommendations  — personalised next actions (Sprint F)
# ---------------------------------------------------------------------------


@router.get("/recommendations")
def get_recommendations(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Personalised next-action recommendations for the authenticated user."""
    learner_key = str(current_user.id)
    try:
        recs = recommendation_engine.compute(*_fetch_recommendation_data(db, learner_key))
        adaptive = adaptive_engine.get_personalized_recommendations(
            *_fetch_adaptive_data(db, learner_key), _fetch_all_papers_data(db)
        )
        return {
            "user_id": current_user.id,
            "recommendations": recs,
            "adaptive": adaptive,
        }
    except BudgetExceededError:
        raise HTTPException(
            status_code=429, detail="Daily LLM token budget exceeded. Try again tomorrow."
        )
    except Exception as exc:
        logger.exception("Recommendations error for user %s: %s", current_user.id, exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# GET /api/learning-paths  — curated learning sequences (Sprint F)
# ---------------------------------------------------------------------------

_LEARNING_PATHS = [
    {
        "id": "ai-engineer",
        "title": "AI Engineer",
        "description": "From ML fundamentals to production-ready transformer systems.",
        "icon": "🤖",
        "steps": [
            {"id": "resnet", "type": "architecture", "title": "ResNet", "domain": "cnns"},
            {
                "id": "transformer",
                "type": "architecture",
                "title": "Transformer",
                "domain": "transformers",
            },
            {"id": "bert", "type": "architecture", "title": "BERT", "domain": "transformers"},
            {"id": "gpt", "type": "architecture", "title": "GPT", "domain": "transformers"},
            {
                "id": "vit",
                "type": "architecture",
                "title": "Vision Transformer",
                "domain": "transformers",
            },
            {
                "id": "stable-diffusion",
                "type": "architecture",
                "title": "Stable Diffusion",
                "domain": "diffusion",
            },
        ],
    },
    {
        "id": "llm-specialist",
        "title": "LLM Specialist",
        "description": "Deep dive into large language models and alignment.",
        "icon": "🧠",
        "steps": [
            {
                "id": "transformer",
                "type": "architecture",
                "title": "Transformer",
                "domain": "transformers",
            },
            {"id": "bert", "type": "architecture", "title": "BERT", "domain": "transformers"},
            {"id": "gpt", "type": "architecture", "title": "GPT", "domain": "transformers"},
            {"id": "llama", "type": "architecture", "title": "LLaMA", "domain": "transformers"},
            {"id": "clip", "type": "architecture", "title": "CLIP", "domain": "transformers"},
        ],
    },
    {
        "id": "vision-engineer",
        "title": "Vision Engineer",
        "description": "Computer vision from classic CNNs to diffusion models.",
        "icon": "👁️",
        "steps": [
            {"id": "alexnet", "type": "architecture", "title": "AlexNet", "domain": "cnns"},
            {"id": "resnet", "type": "architecture", "title": "ResNet", "domain": "cnns"},
            {
                "id": "vit",
                "type": "architecture",
                "title": "Vision Transformer",
                "domain": "transformers",
            },
            {"id": "clip", "type": "architecture", "title": "CLIP", "domain": "transformers"},
            {"id": "ddpm", "type": "architecture", "title": "DDPM", "domain": "diffusion"},
            {
                "id": "stable-diffusion",
                "type": "architecture",
                "title": "Stable Diffusion",
                "domain": "diffusion",
            },
        ],
    },
    {
        "id": "research-track",
        "title": "Research Track",
        "description": "Broad survey of modern architectures for aspiring ML researchers.",
        "icon": "🔬",
        "steps": [
            {
                "id": "transformer",
                "type": "architecture",
                "title": "Transformer",
                "domain": "transformers",
            },
            {"id": "gan", "type": "architecture", "title": "GAN", "domain": "generative"},
            {"id": "vae", "type": "architecture", "title": "VAE", "domain": "generative"},
            {"id": "gcn", "type": "architecture", "title": "GCN", "domain": "graph"},
            {"id": "ddpm", "type": "architecture", "title": "DDPM", "domain": "diffusion"},
            {"id": "ppo", "type": "architecture", "title": "PPO", "domain": "rl"},
        ],
    },
    {
        "id": "interview-prep",
        "title": "Interview Prep",
        "description": "Targeted practice problems for ML engineering interviews.",
        "icon": "💼",
        "steps": [
            {
                "id": "transformer",
                "type": "architecture",
                "title": "Implement Attention",
                "domain": "transformers",
            },
            {
                "id": "resnet",
                "type": "architecture",
                "title": "Implement ResNet Block",
                "domain": "cnns",
            },
            {
                "id": "vae",
                "type": "architecture",
                "title": "Implement VAE Loss",
                "domain": "generative",
            },
            {"id": "ppo", "type": "architecture", "title": "Policy Gradient", "domain": "rl"},
        ],
    },
]


@router.get("/learning-paths")
def get_learning_paths(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Curated learning sequences with per-step completion status overlaid."""
    completed_ids = {
        r.entity_id
        for r in db.query(LearnerProgress)
        .filter_by(
            learner_id=str(current_user.id),
            entity_type="architecture",
            status="completed",
        )
        .all()
    }

    result = []
    for path in _LEARNING_PATHS:
        steps_with_status = []
        done = 0
        for step in path["steps"]:
            is_done = step["id"] in completed_ids
            if is_done:
                done += 1
            steps_with_status.append({**step, "completed": is_done})
        result.append(
            {
                **path,
                "steps": steps_with_status,
                "progress_pct": round(done / len(path["steps"]) * 100) if path["steps"] else 0,
            }
        )
    return {"paths": result}


# ---------------------------------------------------------------------------
# POST /api/tutor/feedback  — thumbs up/down on a tutor message (Sprint F)
# ---------------------------------------------------------------------------


class TutorFeedbackRequest(BaseModel):
    session_id: str = Field(..., max_length=128)
    message_index: int
    rating: int  # 1 = thumbs up, -1 = thumbs down


# ---------------------------------------------------------------------------
# GET /api/tutor/sessions  — conversation history (Sprint F)
# ---------------------------------------------------------------------------


def _get_budget_callback(db):
    def callback(user_id: int):
        import datetime

        from sqlalchemy import func

        from backend.models import UsageLog, User

        user = db.query(User).filter_by(id=user_id).first()
        if user and user.is_admin:
            return -1
        today = datetime.date.today()
        total = (
            db.query(func.sum(UsageLog.prompt_tokens + UsageLog.completion_tokens))
            .filter(UsageLog.user_id == user_id, func.date(UsageLog.created_at) == today)
            .scalar()
        )
        return total or 0

    return callback
