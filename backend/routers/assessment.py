import datetime
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_optional_user
from backend.models import (
    AssessmentAttempt,
    LearnerProgress,
    Paper,
    PaperModule,
    TutorAnalytics,
)
from backend.services.progress_service import award_xp
from core.assessment.engine import assessment_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Assessment"])


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
    progress_data = []
    for p in progress_records:
        if getattr(p, "entity_type", "paper_module") in ("paper_module", "module"):
            try:
                mod_id = int(p.entity_id) if hasattr(p, "entity_id") else getattr(p, "module_id", None)
                if mod_id is not None:
                    progress_data.append({"module_id": mod_id, "status": p.status})
            except (ValueError, TypeError):
                continue
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
    session_id: str | None = None  # server-generated; null = auto-create
    context_type: str
    context_data: dict[str, Any]
    query: str


class TutorQuizRequest(BaseModel):
    module_data: dict[str, Any]


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


# deprecated alias, remove after frontend migration


@router.get("/assessment/challenge")
def get_assessment_challenge(
    type: str = "tensor",
    arch: str = "ResNet",
    difficulty: str = "beginner",
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    try:
        db_metrics = []
        if type == "comparison":
            papers = db.query(Paper).all()
            for p in papers:
                flops_analysis = p.flops_analysis or {}
                arch_graph = p.architecture_graph or {}
                classification = arch_graph.get("classification", "Unknown")
                db_metrics.append(
                    {
                        "title": p.title,
                        "architecture_type": classification,
                        "parameter_count": flops_analysis.get("total_params_estimate", 0),
                        "flops": flops_analysis.get("total_flops_score", 0),
                        "module_count": len(p.modules) if p.modules else 0,
                    }
                )

        challenge = assessment_engine.generate(
            challenge_type=type, architecture=arch, difficulty=difficulty, db_metrics=db_metrics
        )
        return challenge
    except Exception as e:
        logger.exception(f"Assessment challenge generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assessments/validations")
# deprecated alias
@router.post("/assessment/validate", deprecated=True)
def validate_assessment_challenge(
    request: ValidateRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
    current_user=Depends(get_optional_user),
):
    try:
        challenge = request.challenge
        user_ans = request.user_answer
        val_res = assessment_engine.validate(challenge, user_ans)

        question_text = challenge.get("question", "")
        existing = (
            db.query(AssessmentAttempt)
            .filter(
                AssessmentAttempt.learner_id == x_learner_id,
                AssessmentAttempt.question_text == question_text,
            )
            .first()
        )

        if existing:
            existing.attempt_count += 1
            existing.user_answer = user_ans
            existing.is_correct = val_res["is_correct"]
            existing.score = val_res["score"]
            existing.created_at = datetime.datetime.utcnow()
            db.commit()
            db.refresh(existing)
        else:
            attempt = AssessmentAttempt(
                learner_id=x_learner_id,
                assessment_type=challenge.get("assessment_type", "unknown"),
                architecture=challenge.get("architecture"),
                difficulty=challenge.get("difficulty"),
                question_text=question_text,
                user_answer=user_ans,
                correct_answer=val_res["correct_answer"],
                explanation=val_res["explanation"],
                score=val_res["score"],
                attempt_count=1,
                is_correct=val_res["is_correct"],
            )
            db.add(attempt)
            db.commit()

        if current_user and val_res["is_correct"]:
            award_xp(db, current_user.id, "assessment.completed", question_text)

        return val_res
    except Exception as e:
        logger.exception(f"Assessment validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/recommendations  — personalised next actions (Sprint F)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# POST /api/tutor/feedback  — thumbs up/down on a tutor message (Sprint F)
# ---------------------------------------------------------------------------


class TutorFeedbackRequest(BaseModel):
    session_id: str
    message_index: int
    rating: int  # 1 = thumbs up, -1 = thumbs down


# ---------------------------------------------------------------------------
# GET /api/tutor/sessions  — conversation history (Sprint F)
# ---------------------------------------------------------------------------
