import logging
import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
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

router = APIRouter(prefix="/api", tags=["Learning"])

class ProgressUpdate(BaseModel):
    status: str
    time_spent_seconds: int = 0

class ValidateRequest(BaseModel):
    challenge: Dict[str, Any]
    user_answer: str

class ProgressUpdateRequest(BaseModel):
    entity_type: str
    entity_id: str
    status: str

class TutorAskRequest(BaseModel):
    session_id: Optional[str] = None  # server-generated; null = auto-create
    context_type: str
    context_data: Dict[str, Any]
    query: str

class TutorQuizRequest(BaseModel):
    module_data: Dict[str, Any]

@router.get("/progress/{entity_type}")
def get_progress(entity_type: str, current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    progress = db.query(LearnerProgress).filter_by(
        learner_id=str(current_user.id), 
        entity_type=entity_type
    ).all()
    return progress

@router.post("/progress/{entity_type}/{entity_id}")
def update_progress(
    entity_type: str,
    entity_id: str,
    update: ProgressUpdate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prog = db.query(LearnerProgress).filter_by(
        learner_id=str(current_user.id),
        entity_type=entity_type,
        entity_id=entity_id,
    ).first()

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
            from backend.services.progress_service import update_user_activity, award_xp, check_domain_completion
            update_user_activity(db, current_user.id)
            award_xp(db, current_user.id, "topic.completed", entity_id=entity_id)

            completed_domain = check_domain_completion(db, current_user.id, entity_id)
            if completed_domain:
                award_xp(db, current_user.id, "domain.completed", entity_id=completed_domain)
                try:
                    from backend.models import Notification
                    db.add(Notification(
                        user_id=current_user.id,
                        type="domain.completed",
                        title=f"Domain complete: {completed_domain.replace('-', ' ').title()}",
                        body="You've mastered all architectures in this domain! +500 XP awarded.",
                        payload={"domain": completed_domain},
                    ))
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
    current_user=Depends(get_current_user)
):
    try:
        progress = (
            db.query(LearnerProgress)
            .filter(
                LearnerProgress.learner_id == str(current_user.id),
                LearnerProgress.entity_type == request.entity_type,
                LearnerProgress.entity_id == request.entity_id
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
                completed_at=now if request.status == "completed" else None
            )
            db.add(progress)
            db.commit()
            
        return {"status": "success", "progress_id": progress.id}
    except Exception as e:
        logger.error(f"Progress update error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/dashboard")
def get_analytics_dashboard(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        # Overview
        total_papers = db.query(func.count(Paper.id)).scalar() or 0
        total_modules_count = db.query(func.count(PaperModule.id)).scalar() or 0
        
        modules_completed = db.query(func.count(LearnerProgress.id)).filter(
            LearnerProgress.learner_id == x_learner_id,
            LearnerProgress.status == "completed"
        ).scalar() or 0
        
        papers_started = 0
        papers_completed = 0
        
        if total_modules_count > 0:
            papers_with_modules = db.query(
                PaperModule.paper_id, func.count(PaperModule.id)
            ).group_by(PaperModule.paper_id).all()
            paper_module_counts = {p_id: count for p_id, count in papers_with_modules}

            learner_progress = db.query(
                PaperModule.paper_id, LearnerProgress.status
            ).join(
                PaperModule, PaperModule.id == LearnerProgress.entity_id
            ).filter(
                LearnerProgress.learner_id == x_learner_id,
                LearnerProgress.entity_type == "paper_module",
            ).all()
            
            progress_by_paper = defaultdict(list)
            for p_id, status in learner_progress:
                progress_by_paper[p_id].append(status)
                
            for p_id, statuses in progress_by_paper.items():
                if any(s in ("in_progress", "completed") for s in statuses):
                    papers_started += 1
                if all(s == "completed" for s in statuses) and len(statuses) == paper_module_counts.get(p_id, -1):
                    papers_completed += 1
            
        completion_pct = round(modules_completed / total_modules_count * 100, 1) if total_modules_count > 0 else 0.0
        
        overview = {
            "total_papers": total_papers,
            "papers_started": papers_started,
            "papers_completed": papers_completed,
            "modules_completed": modules_completed,
            "total_modules": total_modules_count,
            "completion_percentage": completion_pct
        }
        
        # Assessment
        attempts = db.query(AssessmentAttempt).filter(AssessmentAttempt.learner_id == x_learner_id).all()
        total_attempts = len(attempts)
        correct_attempts = sum(1 for a in attempts if a.is_correct)
        accuracy_pct = round(correct_attempts / total_attempts * 100, 1) if total_attempts > 0 else 0.0
        
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
            weakest_arch = f"{sorted_archs[0][0]} ({int(sorted_archs[0][1]*100)}% accuracy)"
            strongest_arch = f"{sorted_archs[-1][0]} ({int(sorted_archs[-1][1]*100)}% accuracy)"
            
        assessment = {
            "total_attempts": total_attempts,
            "accuracy_percentage": accuracy_pct,
            "strongest_architecture": strongest_arch,
            "weakest_architecture": weakest_arch
        }
        
        # Tutor
        tutor_records = db.query(TutorAnalytics).filter(TutorAnalytics.learner_id == x_learner_id).all()
        questions_asked = sum(r.question_count for r in tutor_records)
        
        context_counts = defaultdict(int)
        for r in tutor_records:
            if r.architecture:
                context_counts[r.architecture] += r.question_count
        most_asked_context = "None"
        if context_counts:
            most_asked_context = max(context_counts.items(), key=lambda x: x[1])[0]
            
        tutor = {
            "questions_asked": questions_asked,
            "most_asked_context": most_asked_context
        }
        
        # Learning Path
        progress_records = db.query(LearnerProgress).filter(
            LearnerProgress.learner_id == x_learner_id
        ).all()
        active_sorted = sorted(
            [p for p in progress_records if p.started_at],
            key=lambda x: x.started_at,
            reverse=True
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
                        db.query(Paper)
                        .filter(Paper.id > p.id)
                        .order_by(Paper.id.asc())
                        .first()
                    )
                    if next_paper and next_paper.modules:
                        next_recommended = f"{next_paper.title} - {next_paper.modules[0].layer_name}"
                    else:
                        next_recommended = "You have completed all available papers!"
        elif first_paper and first_paper.modules:
            next_recommended = f"{first_paper.title} - {first_paper.modules[0].layer_name}"
                
        learning_path = {
            "current_position": current_position,
            "next_recommended": next_recommended
        }

        learning_path_items = []
        if current_position != "Get started by reading a paper!":
            learning_path_items.append({
                "title": current_position,
                "url": "",
                "type": "current",
            })
        if next_recommended not in ("None", "You have completed all available papers!"):
            learning_path_items.append({
                "title": next_recommended,
                "url": "",
                "type": "next",
            })

        recs = recommendation_engine.compute(db, x_learner_id)

        return {
            "learning_overview": overview,
            "assessment_performance": assessment,
            "tutor_usage": tutor,
            "learning_path": learning_path,
            "learning_path_items": learning_path_items,
            "reviews_and_recommendations": recs
        }
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/recommendations")
def get_analytics_recommendations(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        recs = recommendation_engine.compute(db, x_learner_id)
        return recs
    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/adaptive/recommendations")
def get_adaptive_recommendations(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        recs = adaptive_engine.get_personalized_recommendations(db, x_learner_id)
        return {"recommendations": recs}
    except Exception as e:
        logger.error(f"Adaptive recommendations error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/adaptive/review-plan")
def get_adaptive_review_plan(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        plan = adaptive_engine.get_daily_review_plan(db, x_learner_id)
        return plan
    except Exception as e:
        logger.error(f"Adaptive review plan error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/adaptive/concept-graph")
def get_adaptive_concept_graph(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        graph_nodes = adaptive_engine.get_concept_graph(db, x_learner_id)
        return {"nodes": graph_nodes}
    except Exception as e:
        logger.error(f"Concept graph error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tutor/start-session")
def tutor_start_session(current_user=Depends(get_current_user)):
    """Create a server-generated session ID owned by the authenticated user."""
    session_id = tutor_session_store.create_session(current_user.id)
    return {"session_id": session_id}


@router.post("/tutor/ask")
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
        profile = adaptive_engine.compute_knowledge_profile(db, learner_key)

        from core.agents.agentic_tutor import AgenticTutor
        agentic_tutor = AgenticTutor(db)
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
    except Exception as exc:
        logger.error("Tutor ask error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/tutor/stream")
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
        except Exception as exc:
            logger.error("tutor_stream error: %s", exc)
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

@router.post("/tutor/quiz")
def tutor_quiz(
    request: TutorQuizRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        weaknesses = adaptive_engine.detect_weaknesses(db, x_learner_id)
        weak_list = [w["topic"] for w in weaknesses["weak_topics"] if w["weakness_score"] > 0.2]
        
        response = tutor_manager.generate_quiz(request.module_data, weak_topics=weak_list)
        return {"questions": response}
    except Exception as e:
        logger.error(f"Tutor quiz error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tutor/learning-path")
def tutor_learning_path(
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        path = adaptive_engine.get_adaptive_learning_path(db, x_learner_id)
        return {"learning_path": path}
    except Exception as e:
        logger.error(f"Tutor learning path error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/assessment/challenge")
def get_assessment_challenge(
    type: str = "tensor",
    arch: str = "ResNet",
    difficulty: str = "beginner",
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default="")
):
    try:
        db_metrics = []
        if type == "comparison":
            papers = db.query(Paper).all()
            for p in papers:
                flops_analysis = p.flops_analysis or {}
                arch_graph = p.architecture_graph or {}
                classification = arch_graph.get("classification", "Unknown")
                db_metrics.append({
                    "title": p.title,
                    "architecture_type": classification,
                    "parameter_count": flops_analysis.get("total_params_estimate", 0),
                    "flops": flops_analysis.get("total_flops_score", 0),
                    "module_count": len(p.modules) if p.modules else 0
                })
                
        challenge = assessment_engine.generate(
            challenge_type=type,
            architecture=arch,
            difficulty=difficulty,
            db_metrics=db_metrics
        )
        return challenge
    except Exception as e:
        logger.error(f"Assessment challenge generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assessment/validate")
def validate_assessment_challenge(
    request: ValidateRequest,
    db: Session = Depends(get_db),
    x_learner_id: str = Header(alias="X-Learner-ID", default=""),
    current_user=Depends(get_optional_user)
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
                AssessmentAttempt.question_text == question_text
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
                is_correct=val_res["is_correct"]
            )
            db.add(attempt)
            db.commit()
            
        if current_user and val_res["is_correct"]:
            award_xp(
                db, 
                current_user.id, 
                "assessment.completed",
                question_text
            )
            
        return val_res
    except Exception as e:
        logger.error(f"Assessment validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        recs = recommendation_engine.compute(db, learner_key)
        adaptive = adaptive_engine.get_personalized_recommendations(db, learner_key)
        return {
            "user_id": current_user.id,
            "recommendations": recs,
            "adaptive": adaptive,
        }
    except Exception as exc:
        logger.error("Recommendations error for user %s: %s", current_user.id, exc)
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


@router.get("/learning-paths")
def get_learning_paths(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Curated learning sequences with per-step completion status overlaid."""
    completed_ids = {
        r.entity_id
        for r in db.query(LearnerProgress).filter_by(
            learner_id=str(current_user.id),
            entity_type="architecture",
            status="completed",
        ).all()
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
        result.append({
            **path,
            "steps": steps_with_status,
            "progress_pct": round(done / len(path["steps"]) * 100) if path["steps"] else 0,
        })
    return {"paths": result}


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

