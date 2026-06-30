from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel

from backend.database import get_db
from backend.models import Paper, PaperChallenge, PaperChallengePart, PaperPartSubmission, User
from backend.modules.auth.dependencies import get_current_user
from backend.services.e2b_service import run_code_in_sandbox
from backend.services.progress_service import award_xp


router = APIRouter(prefix="/api/papers", tags=["Paper Challenges"])

class RunCodeRequest(BaseModel):
    code: str

@router.get("/{paper_id}/challenges")
def get_paper_challenges(paper_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    challenges = db.query(PaperChallenge).filter(
        PaperChallenge.paper_id == paper_id,
        PaperChallenge.is_published == True
    ).order_by(PaperChallenge.order_idx).all()

    # Prefetch all passed submissions for this user & paper's challenges
    part_ids = [p.id for c in challenges for p in c.parts]
    passed_subs = db.query(PaperPartSubmission.part_id).filter(
        PaperPartSubmission.user_id == current_user.id,
        PaperPartSubmission.part_id.in_(part_ids),
        PaperPartSubmission.passed == True
    ).all()
    passed_part_ids = {s[0] for s in passed_subs}

    result = []
    for challenge in challenges:
        parts_res = []
        for part in challenge.parts:
            user_passed = part.id in passed_part_ids
            is_locked = False
            if part.unlock_requires_part_id and part.unlock_requires_part_id not in passed_part_ids:
                is_locked = True
                
            parts_res.append({
                "id": part.id,
                "challenge_id": part.challenge_id,
                "order_idx": part.order_idx,
                "title": part.title,
                "description_md": part.description_md,
                "paper_section_md": part.paper_section_md,
                "setup_code": part.setup_code,
                "starter_code": part.starter_code,
                "unlock_requires_part_id": part.unlock_requires_part_id,
                "xp_reward": part.xp_reward,
                "created_at": part.created_at,
                "user_passed": user_passed,
                "is_locked": is_locked
            })
            
        result.append({
            "id": challenge.id,
            "title": challenge.title,
            "description": challenge.description,
            "order_idx": challenge.order_idx,
            "parts": parts_res
        })
        
    return result

@router.get("/{paper_id}/challenges/{challenge_id}/parts/{part_id}")
def get_challenge_part(
    paper_id: int, challenge_id: int, part_id: int,
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    part = db.query(PaperChallengePart).filter(
        PaperChallengePart.id == part_id,
        PaperChallengePart.challenge_id == challenge_id
    ).first()
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")

    if part.unlock_requires_part_id:
        has_passed = db.query(PaperPartSubmission).filter(
            PaperPartSubmission.user_id == current_user.id,
            PaperPartSubmission.part_id == part.unlock_requires_part_id,
            PaperPartSubmission.passed == True
        ).first()
        if not has_passed:
            raise HTTPException(status_code=403, detail="Complete the previous part first")

    return {
        "id": part.id,
        "challenge_id": part.challenge_id,
        "order_idx": part.order_idx,
        "title": part.title,
        "description_md": part.description_md,
        "paper_section_md": part.paper_section_md,
        "setup_code": part.setup_code,
        "starter_code": part.starter_code,
        "unlock_requires_part_id": part.unlock_requires_part_id,
        "xp_reward": part.xp_reward,
    }

@router.post("/{paper_id}/challenges/{challenge_id}/parts/{part_id}/executions")
# deprecated alias
@router.post("/{paper_id}/challenges/{challenge_id}/parts/{part_id}/run", deprecated=True)
def run_challenge_part(
    paper_id: int, challenge_id: int, part_id: int,
    request: RunCodeRequest,
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    part = db.query(PaperChallengePart).filter(
        PaperChallengePart.id == part_id,
        PaperChallengePart.challenge_id == challenge_id
    ).first()
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")

    if part.unlock_requires_part_id:
        has_passed = db.query(PaperPartSubmission).filter(
            PaperPartSubmission.user_id == current_user.id,
            PaperPartSubmission.part_id == part.unlock_requires_part_id,
            PaperPartSubmission.passed == True
        ).first()
        if not has_passed:
            raise HTTPException(status_code=403, detail="Complete the previous part first")

    # E2B run
    res = run_code_in_sandbox(part.setup_code or "", request.code, part.test_code)
    passed = res.get("passed", False)
    
    # Check if they already solved it BEFORE adding new submission
    prior_pass = None
    if passed:
        prior_pass = db.query(PaperPartSubmission).filter(
            PaperPartSubmission.user_id == current_user.id,
            PaperPartSubmission.part_id == part.id,
            PaperPartSubmission.passed == True
        ).first()

    submission = PaperPartSubmission(
        user_id=current_user.id,
        part_id=part.id,
        code=request.code,
        passed=passed,
        stdout=res.get("stdout"),
        stderr=res.get("stderr"),
        time_ms=res.get("time_ms")
    )
    db.add(submission)
    
    xp_earned = 0
    if passed and not prior_pass:
        submission.is_best = True
        db.flush()
        xp_earned = part.xp_reward
        award_xp(db, current_user.id, "paper.part.solved", f"paper_part-{part.id}", amount_override=xp_earned)

            
    db.commit()

    return {
        "passed": passed,
        "stdout": res.get("stdout"),
        "stderr": res.get("stderr"),
        "time_ms": res.get("time_ms"),
        "xp_earned": xp_earned
    }

@router.get("/{paper_id}/challenges/{challenge_id}/parts/{part_id}/progress")
def get_part_progress(
    paper_id: int, challenge_id: int, part_id: int,
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    subs = db.query(PaperPartSubmission).filter(
        PaperPartSubmission.user_id == current_user.id,
        PaperPartSubmission.part_id == part_id
    ).order_by(PaperPartSubmission.created_at.desc()).all()
    
    passed = any(s.passed for s in subs)
    best_sub = next((s.id for s in subs if s.is_best), None)
    
    attempts = [
        {"id": s.id, "passed": s.passed, "created_at": s.created_at} for s in subs
    ]

    return {
        "attempts": len(subs),
        "passed": passed,
        "best_submission_id": best_sub,
        "history": attempts
    }
