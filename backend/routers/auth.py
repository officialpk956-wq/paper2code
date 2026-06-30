import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.server import limiter
from backend.modules.auth.middleware.rate_limit import rate_limiter
from backend.models import User
from backend.repositories.token_repository import TokenRepository
from backend.services.email_service import (
    send_verification_email_sync,
    send_password_reset_email_sync
)
import sqlalchemy

router = APIRouter(prefix="/api/auth", tags=["Auth"])

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str

@router.post("/register")
@limiter.limit("5/hour")
async def register_user(
    request: Request,
    req: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    from backend.repositories.user_repository import UserRepository
    from backend.modules.auth.security.hashing import hash_password
    from backend.modules.auth.services.session_service import SessionService
    repo = UserRepository(db)
    session_svc = SessionService(db)
    try:
        hashed = hash_password(req.password)
        user = repo.create(email=req.email, name=req.name, hashed_password=hashed)
        db.commit()

        # Create email verification token
        token_repo = TokenRepository(db)
        token = token_repo.create_email_verification(user.id)
        
        # Fire-and-forget email sending
        background_tasks.add_task(send_verification_email_sync, user.email, token)

        access = session_svc.create_access_token(user)
        return {
            "access_token": access,
            "token_type": "bearer",
            "email_verification_sent": True,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name
            }
        }
    except sqlalchemy.exc.IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")

@router.post("/login")
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
    _rl=Depends(rate_limiter(limit=10, window_seconds=60)),
):
    from backend.repositories.user_repository import UserRepository
    from backend.modules.auth.security.hashing import verify_password_and_needs_rehash
    from backend.modules.auth.services.session_service import SessionService
    repo = UserRepository(db)
    user = repo.get_by_email(form_data.username)
    
    if not user or not user.hashed_password:
        from backend import metrics
        metrics.increment("login_failures_total")
        raise HTTPException(status_code=401, detail="Incorrect email or password")
        
    is_verified, needs_rehash = verify_password_and_needs_rehash(form_data.password, user.hashed_password)
    if not is_verified:
        from backend import metrics
        metrics.increment("login_failures_total")
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    from backend import metrics
    metrics.increment("logins_total")
    
    session_svc = SessionService(db)
    access_token = session_svc.create_access_token(user)
    refresh_token = session_svc.create_refresh_token(user)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    session_svc.register_session(user, refresh_token, ip_address=ip_address, user_agent=user_agent)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }

class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/refresh")
def refresh_access_token(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    from backend.modules.auth.services.session_service import SessionService
    session_svc = SessionService(db)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    new_access, new_refresh = session_svc.rotate_session(body.refresh_token, ip_address, user_agent)
    return {"access_token": new_access, "refresh_token": new_refresh, "token_type": "bearer"}

@router.post("/logout")
def logout(
    request: Request,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from backend.modules.auth.services.session_service import SessionService
    session_svc = SessionService(db)
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    session_svc.revoke_all_sessions(current_user.id, ip_address, user_agent)
    return {"status": "ok", "message": "Logged out successfully"}

@router.get("/me")
def get_me(current_user = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "points": current_user.points,
        "streak": current_user.streak,
        "is_email_verified": getattr(current_user, "is_email_verified", False)
    }

@router.get("/verify-email")
async def verify_email(token: str, db: Session = Depends(get_db)):
    repo = TokenRepository(db)
    user_id = repo.verify_email_token(token)
    if not user_id:
        raise HTTPException(400, "Invalid or expired verification link")
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.is_email_verified = True
    user.email_verified_at = datetime.now(timezone.utc)
    db.commit()
    return RedirectResponse(
        url=f"{os.getenv('FRONTEND_URL','http://localhost:3000')}/auth/verified",
        status_code=302
    )

@router.post("/resend-verification")
@limiter.limit("3/hour")
async def resend_verification(
    request: Request,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    if current_user.is_email_verified:
        raise HTTPException(400, "Email already verified")
    token = TokenRepository(db).create_email_verification(current_user.id)
    background_tasks.add_task(send_verification_email_sync, current_user.email, token)
    return {"detail": "Verification email sent"}

class ForgotPasswordRequest(BaseModel):
    email: str

@router.post("/forgot-password")
@limiter.limit("3/hour")
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(email=body.email.lower()).first()
    if user:
        token = TokenRepository(db).create_password_reset(user.id)
        background_tasks.add_task(send_password_reset_email_sync, user.email, token)
    return {"detail": "If that email exists, a reset link has been sent"}

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    user_id = TokenRepository(db).verify_reset_token(body.token)
    if not user_id:
        raise HTTPException(400, "Invalid or expired reset link")
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404)
    from backend.modules.auth.security.hashing import hash_password
    user.hashed_password = hash_password(body.new_password)
    user.token_version = (user.token_version or 0) + 1
    db.commit()
    return {"detail": "Password updated. Please log in again."}
