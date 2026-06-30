import logging
import datetime
from typing import Dict, Any, Optional
from core.llm_client import BudgetExceededError
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func
from collections import defaultdict

from backend.database import get_db
from backend.dependencies import get_current_user, get_optional_user
from backend.models import (
    LearnerProgress, Paper, PaperModule, AssessmentAttempt, TutorAnalytics,
    InterviewQuestion, Roadmap, TutorFeedback, TutorSessionRecord,
)
from backend.services.tutor_session_store import tutor_session_store
from backend.services.progress_service import award_xp

from core.analytics.adaptive_engine import adaptive_engine
from core.analytics.recommendation_engine import recommendation_engine
from core.assessment.engine import assessment_engine
from core.agents.tutor_agent import tutor_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Tutor"])


def _fetch_adaptive_data(db: Session, learner_id: str):
    attempts = db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == learner_id).all()
    progress_records = db.query(LearnerProgress).filter(LearnerProgress.learner_id == learner_id).all()
    tutor_records = db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == learner_id).all()
    all_modules = db.query(PaperModule).all()
    
    attempts_data = [{"question_text": getattr(a, "question_text", None), "assessment_type": getattr(a, "assessment_type", None), "architecture": getattr(a, "architecture", None), "is_correct": getattr(a, "is_correct", False)} for a in attempts]
    progress_data = [{"module_id": p.module_id, "status": p.status} for p in progress_records]
    tutor_data = [{"module": t.module, "architecture": t.architecture, "question_count": getattr(t, "question_count", 0)} for t in tutor_records]
    modules_data = [{"id": m.id, "explanation": m.explanation, "module_type": m.module_type, "layer_name": m.layer_name} for m in all_modules]

    return attempts_data, progress_data, tutor_data, modules_data

def _fetch_all_papers_data(db: Session):
    all_papers = db.query(Paper).all()
    papers_data = []
    for p in all_papers:
        mods = [{"id": m.id, "explanation": m.explanation, "module_type": m.module_type, "layer_name": m.layer_name} for m in p.modules]
        papers_data.append({"id": p.id, "title": p.title, "architecture_graph": p.architecture_graph, "modules": mods})
    return papers_data

def _fetch_recommendation_data(db: Session, learner_id: str):
    attempts = db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == learner_id).all()
    tutor_rows = db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == learner_id).all()
    
    attempts_data = [{"architecture": a.architecture, "assessment_type": a.assessment_type, "is_correct": a.is_correct} for a in attempts]
    tutor_data = [{"architecture": t.architecture, "module": t.module, "question_count": t.question_count} for t in tutor_rows]
    return attempts_data, tutor_data

def _get_tutor_callbacks(db: Session):
    def lookup_paper_section(paper_id: int, section: str = "abstract"):
        paper = db.query(Paper).filter_by(id=paper_id).first()
        if not paper: return "Paper not found."
        content = getattr(paper, section, None) or paper.abstract or ""
        return content[:2000] if content else "No content available."

    def get_user_weak_topics(user_id: int):
        attempts = db.query(AssessmentAttempt).filter_by(learner_id=str(user_id)).all()
        arch_results = {}
        for a in attempts:
            if a.architecture:
                arch_results.setdefault(a.architecture, []).append(a.is_correct)
        weak = [arch for arch, results in arch_results.items() if results and sum(results) / len(results) < 0.5]
        return f"Weak topics: {', '.join(weak)}" if weak else "No weak topics found."

    def find_related_problem(concept: str):
        concept = concept.lower()
        from backend.models import Problem
        prob = db.query(Problem).filter(Problem.is_retired == False, Problem.description.ilike(f"%{concept}%")).first()
        if prob: return f"Problem: {prob.title} (ID: {prob.id}, Difficulty: {prob.difficulty})"
        return "No related problem found."

    def search_papers_by_concept(concept: str):
        papers = db.query(Paper).filter(Paper.visibility != "private", Paper.title.ilike(f"%{concept}%")).limit(3).all()
        if papers: return "; ".join(f"{p.title} (ID: {p.id})" for p in papers)
        return "No papers found."

    return {
        "lookup_paper_section": lookup_paper_section,
        "get_user_weak_topics": get_user_weak_topics,
        "find_related_problem": find_related_problem,
        "search_papers_by_concept": search_papers_by_concept
    }


class ProgressUpdate(BaseModel):
    status: str
    time_spent_seconds: int = 0

class ValidateRequest(BaseModel):
    challenge: Dict[str, Any]
    user_answer: str = Field(..., max_length=5000)

class ProgressUpdateRequest(BaseModel):
    entity_type: str
    entity_id: str
    status: str

class TutorAskRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, max_length=128)  # server-generated; null = auto-create
    context_type: str = Field(..., max_length=500)
    context_data: Dict[str, Any]
    query: str = Field(..., max_length=10000)

class TutorQuizRequest(BaseModel):
    module_data: Dict[str, Any]









@router.post("/tutor/sessions")
# deprecated alias
@router.post("/tutor/start-session", deprecated=True)
def tutor_start_session(current_user=Depends(get_current_user)):
    """Create a server-generated session ID owned by the authenticated user."""
    session_id = tutor_session_store.create_session(current_user.id)
    
    from backend import metrics
    metrics.increment("tutor_sessions_total")
    
    return {"session_id": session_id}


@router.post("/tutor/messages")
# deprecated alias
@router.post("/tutor/ask", deprecated=True)
def tutor_ask(
    request: TutorAskRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    try:
        # ── Session security ─────────────────────────────────────────────────
        session_id = request.session_id
        if session_id and not tutor_session_store.validate_ownership(session_id, current_user.id):
            # Foreign or expired session — issue a fresh one
            session_id = tutor_session_store.create_session(current_user.id)
        elif not session_id:
            session_id = tutor_session_store.create_session(current_user.id)

        # ── Rate limiting ────────────────────────────────────────────────────
        allowed, count, limit = tutor_session_store.check_and_incr_rate(current_user.id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Daily tutor query limit reached ({limit} queries/day). Try again tomorrow.",
            )

        # ── LLM call with session history ────────────────────────────────────
        history = tutor_session_store.get_history(session_id)
        learner_key = x_learner_id or str(current_user.id)
        profile = adaptive_engine.compute_knowledge_profile(*_fetch_adaptive_data(db, learner_key))

        from core.agents.agentic_tutor import AgenticTutor
        agentic_tutor = AgenticTutor(_get_tutor_callbacks(db))
        response, updated_history = agentic_tutor.ask(
            query=request.query,
            history=history,
            context_type=request.context_type,
            context_data=request.context_data,
            user_id=current_user.id if current_user else None,
            knowledge_profile=profile,
        )
        tutor_session_store.update_history(session_id, updated_history)

        # ── Analytics ────────────────────────────────────────────────────────
        arch = request.context_data.get("architecture") or request.context_type
        mod = request.context_data.get("layer_name") or "general"
        reasoning_type = response.get("reasoning_type", "General")

        existing = db.query(TutorAnalytics).filter(
            TutorAnalytics.learner_id == learner_key,
            TutorAnalytics.architecture == arch,
            TutorAnalytics.module == mod,
        ).first()
        if existing:
            existing.created_at = datetime.datetime.utcnow()
        else:
            db.add(TutorAnalytics(
                learner_id=learner_key,
                architecture=arch,
                module=mod,
                reasoning_type=reasoning_type,
                question_count=1,
            ))

        # ── Persist session record (for /api/tutor/sessions history) ─────────
        history_snapshot = tutor_session_store.get_history(session_id)
        now = datetime.datetime.utcnow()
        rec = db.query(TutorSessionRecord).filter_by(session_id=session_id).first()
        if rec:
            rec.messages = history_snapshot
            rec.last_active_at = now
        else:
            db.add(TutorSessionRecord(
                user_id=current_user.id,
                session_id=session_id,
                context_type=request.context_type,
                messages=history_snapshot,
                last_active_at=now,
            ))

        db.commit()

        return {
            **response,
            "session_id": session_id,
            "queries_today": count,
            "queries_limit": limit,
        }
    except HTTPException:
        raise
    except BudgetExceededError as e:
        logger.exception("Tutor ask error")
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as exc:
        logger.exception("Tutor ask error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/tutor/stream-messages")
# deprecated alias
@router.post("/tutor/stream", deprecated=True)
def tutor_stream(
    request: TutorAskRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
):
    """
    SSE endpoint for streaming tutor responses via real LiteLLM token streaming.
    Uses a stateless system prompt with conversation history — no tool-use agentic
    loop (that remains in /tutor/ask). First token arrives in ~300ms.

    Event stream format:
      data: {"type": "meta",  "session_id": "<id>"}
      data: {"type": "chunk", "text": "<token>"}
      data: {"type": "done"}
      data: {"type": "error", "detail": "<msg>"}
    """
    from fastapi.responses import StreamingResponse
    import json as _json
    import os

    # ── Session security ──────────────────────────────────────────────────────
    session_id = request.session_id
    if session_id and not tutor_session_store.validate_ownership(session_id, current_user.id):
        session_id = tutor_session_store.create_session(current_user.id)
    elif not session_id:
        session_id = tutor_session_store.create_session(current_user.id)

    allowed, _count, limit = tutor_session_store.check_and_incr_rate(current_user.id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"Daily tutor limit of {limit} reached")

    history  = tutor_session_store.get_history(session_id)
    ctx_data = request.context_data or {}
    arch     = str(ctx_data.get("architecture", ctx_data.get("title", "")))[:200]
    ctx_type = str(request.context_type or "general")[:100]

    system_msg = (
        f"You are an expert AI tutor for deep learning architectures.\n"
        f"<context_type>{ctx_type}</context_type>\n"
        f"<context_subject>{arch}</context_subject>\n"
        "Be educational and specific. Never give away full solutions to coding problems."
    )
    messages = (
        [{"role": "system", "content": system_msg}]
        + list(history)
        + [{"role": "user", "content": request.query}]
    )

    async def event_generator():
        yield f"data: {_json.dumps({'type': 'meta', 'session_id': session_id})}\n\n"
        collected: list[str] = []
        try:
            from litellm import acompletion
            model = os.getenv("LLM_PRIMARY_MODEL", "groq/llama-3.3-70b-versatile")
            fallback = os.getenv("LLM_FALLBACK_MODEL", "gemini/gemini-2.0-flash")
            resp = await acompletion(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.3,
                fallbacks=[fallback] if model != fallback else [],
            )
            async for chunk in resp:
                token = chunk.choices[0].delta.content or ""
                if token:
                    collected.append(token)
                    yield f"data: {_json.dumps({'type': 'chunk', 'text': token})}\n\n"
        except BudgetExceededError as e:
            logger.exception("Tutor stream error")
            yield f"data: {_json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
            return
        except Exception as exc:
            logger.exception("tutor_stream error: %s", exc)
            yield f"data: {_json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return

        # Persist conversation history (last 6 turns)
        full_answer = "".join(collected)
        new_history = list(history) + [
            {"role": "user",      "content": request.query},
            {"role": "assistant", "content": full_answer},
        ]
        tutor_session_store.update_history(session_id, new_history[-6:])
        yield f"data: {_json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/tutor/quizzes")
# deprecated alias
@router.post("/tutor/quiz", deprecated=True)
def tutor_quiz(
    request: TutorQuizRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        weaknesses = adaptive_engine.detect_weaknesses(*_fetch_adaptive_data(db, x_learner_id))
        weak_list = [w["topic"] for w in weaknesses["weak_topics"] if w["weakness_score"] > 0.2]
        
        response = tutor_manager.generate_quiz(request.module_data, weak_topics=weak_list)
        return {"questions": response}
    except Exception as e:
        logger.exception(f"Tutor quiz error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tutor/learning-path")
def tutor_learning_path(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        path = adaptive_engine.get_adaptive_learning_path(*_fetch_adaptive_data(db, x_learner_id))
        return {"learning_path": path}
    except Exception as e:
        logger.exception(f"Tutor learning path error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# deprecated alias, remove after frontend migration





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
            {"id": "resnet",      "type": "architecture", "title": "ResNet",       "domain": "cnns"},
            {"id": "transformer", "type": "architecture", "title": "Transformer",  "domain": "transformers"},
            {"id": "bert",        "type": "architecture", "title": "BERT",         "domain": "transformers"},
            {"id": "gpt",         "type": "architecture", "title": "GPT",          "domain": "transformers"},
            {"id": "vit",         "type": "architecture", "title": "Vision Transformer", "domain": "transformers"},
            {"id": "stable-diffusion", "type": "architecture", "title": "Stable Diffusion", "domain": "diffusion"},
        ],
    },
    {
        "id": "llm-specialist",
        "title": "LLM Specialist",
        "description": "Deep dive into large language models and alignment.",
        "icon": "🧠",
        "steps": [
            {"id": "transformer", "type": "architecture", "title": "Transformer",  "domain": "transformers"},
            {"id": "bert",        "type": "architecture", "title": "BERT",         "domain": "transformers"},
            {"id": "gpt",         "type": "architecture", "title": "GPT",          "domain": "transformers"},
            {"id": "llama",       "type": "architecture", "title": "LLaMA",        "domain": "transformers"},
            {"id": "clip",        "type": "architecture", "title": "CLIP",         "domain": "transformers"},
        ],
    },
    {
        "id": "vision-engineer",
        "title": "Vision Engineer",
        "description": "Computer vision from classic CNNs to diffusion models.",
        "icon": "👁️",
        "steps": [
            {"id": "alexnet",     "type": "architecture", "title": "AlexNet",      "domain": "cnns"},
            {"id": "resnet",      "type": "architecture", "title": "ResNet",       "domain": "cnns"},
            {"id": "vit",         "type": "architecture", "title": "Vision Transformer", "domain": "transformers"},
            {"id": "clip",        "type": "architecture", "title": "CLIP",         "domain": "transformers"},
            {"id": "ddpm",        "type": "architecture", "title": "DDPM",         "domain": "diffusion"},
            {"id": "stable-diffusion", "type": "architecture", "title": "Stable Diffusion", "domain": "diffusion"},
        ],
    },
    {
        "id": "research-track",
        "title": "Research Track",
        "description": "Broad survey of modern architectures for aspiring ML researchers.",
        "icon": "🔬",
        "steps": [
            {"id": "transformer", "type": "architecture", "title": "Transformer",  "domain": "transformers"},
            {"id": "gan",         "type": "architecture", "title": "GAN",          "domain": "generative"},
            {"id": "vae",         "type": "architecture", "title": "VAE",          "domain": "generative"},
            {"id": "gcn",         "type": "architecture", "title": "GCN",          "domain": "graph"},
            {"id": "ddpm",        "type": "architecture", "title": "DDPM",         "domain": "diffusion"},
            {"id": "ppo",         "type": "architecture", "title": "PPO",          "domain": "rl"},
        ],
    },
    {
        "id": "interview-prep",
        "title": "Interview Prep",
        "description": "Targeted practice problems for ML engineering interviews.",
        "icon": "💼",
        "steps": [
            {"id": "transformer", "type": "architecture", "title": "Implement Attention", "domain": "transformers"},
            {"id": "resnet",      "type": "architecture", "title": "Implement ResNet Block", "domain": "cnns"},
            {"id": "vae",         "type": "architecture", "title": "Implement VAE Loss",  "domain": "generative"},
            {"id": "ppo",         "type": "architecture", "title": "Policy Gradient",     "domain": "rl"},
        ],
    },
]




# ---------------------------------------------------------------------------
# POST /api/tutor/feedback  — thumbs up/down on a tutor message (Sprint F)
# ---------------------------------------------------------------------------

class TutorFeedbackRequest(BaseModel):
    session_id:    str
    message_index: int
    rating:        int   # 1 = thumbs up, -1 = thumbs down


@router.post("/tutor/feedback", status_code=201)
def tutor_feedback(
    body: TutorFeedbackRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="rating must be 1 (up) or -1 (down)")
    if body.message_index < 0:
        raise HTTPException(status_code=400, detail="message_index must be non-negative")
    if not tutor_session_store.validate_ownership(body.session_id, current_user.id):
        raise HTTPException(status_code=403, detail="Session not found or not owned by you")

    existing = db.query(TutorFeedback).filter_by(
        session_id=body.session_id,
        message_index=body.message_index,
        user_id=current_user.id,
    ).first()
    if existing:
        existing.rating = body.rating
    else:
        db.add(TutorFeedback(
            user_id=current_user.id,
            session_id=body.session_id,
            message_index=body.message_index,
            rating=body.rating,
        ))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    return {"recorded": True, "session_id": body.session_id, "message_index": body.message_index, "rating": body.rating}


# ---------------------------------------------------------------------------
# GET /api/tutor/sessions  — conversation history (Sprint F)
# ---------------------------------------------------------------------------

@router.get("/tutor/sessions")
def tutor_sessions(
    limit: int = 20,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the authenticated user's persisted tutor sessions, newest first."""
    rows = (
        db.query(TutorSessionRecord)
        .filter_by(user_id=current_user.id)
        .order_by(TutorSessionRecord.created_at.desc())
        .limit(min(limit, 100))
        .all()
    )
    return {
        "total": len(rows),
        "sessions": [
            {
                "session_id":     r.session_id,
                "context_type":   r.context_type,
                "message_count":  len(r.messages or []),
                "created_at":     r.created_at.isoformat() if r.created_at else None,
                "last_active_at": r.last_active_at.isoformat() if r.last_active_at else None,
            }
            for r in rows
        ],
    }

