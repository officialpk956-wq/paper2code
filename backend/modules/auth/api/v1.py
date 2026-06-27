import datetime
import jwt
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Request, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.modules.auth.dependencies import get_current_user, get_current_verified_user
from backend.modules.auth.middleware.rate_limit import rate_limiter
from backend.modules.auth.schemas import (
    RegisterRequest,
    UserResponse,
    LoginResponse,
    RefreshRequest,
    RefreshResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
    ResendVerificationRequest,
    ChangePasswordRequest,
    ChangeEmailRequest,
    UpdateProfileRequest,
    SessionResponse,
    EnableMFAResponse,
    VerifyMFARequest,
    DisableMFARequest,
)
from backend.modules.auth.services import (
    AuthService,
    SessionService,
    VerificationService,
    ResetService,
    MFAService,
    OAuthService,
)
from backend.services.auth_service import SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/api/auth", tags=["Auth"])

# Temporary token generator for MFA flow
def generate_mfa_temp_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5),
        "type": "mfa_temp"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_mfa_temp_token(token: str) -> int:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "mfa_temp":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return int(payload["sub"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired MFA token")

# Helper to extract IP and UA
def get_client_details(request: Request) -> tuple[Optional[str], Optional[str]]:
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    return ip, ua

# ---------------------------------------------------------------------------
# Core Auth Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(rate_limiter(5, 3600))])
async def register(request: Request, body: RegisterRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    ip, ua = get_client_details(request)
    user = auth_service.register(email=body.email, name=body.name, password=body.password, ip_address=ip, user_agent=ua)
    
    # Create email verification token via TokenRepository
    from backend.repositories.token_repository import TokenRepository
    from backend.services.email_service import send_verification_email
    token_repo = TokenRepository(db)
    token = token_repo.create_email_verification(user.id)
    
    # Fire-and-forget send_verification_email
    background_tasks.add_task(send_verification_email, user.email, token)

    # PostHog + early-adopter achievement (non-blocking)
    try:
        from backend.services.analytics_service import track
        track(user.id, "signup", {"method": "email"})
        from backend.services.achievement_service import check_and_award
        check_and_award(db, user.id, "user.registered")
    except Exception:
        pass

    user.email_verification_sent = True
    return user

@router.post("/login", dependencies=[Depends(rate_limiter(10, 60))])
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    session_service = SessionService(db)
    ip, ua = get_client_details(request)
    
    user = auth_service.login(email=form_data.username, password=form_data.password, ip_address=ip, user_agent=ua)
    
    # If MFA is enabled, do not issue tokens. Issue a short-lived temp token instead.
    if user.mfa_enabled:
        temp_token = generate_mfa_temp_token(user.id)
        return {
            "mfa_required": True,
            "mfa_token": temp_token
        }

    # Issue standard tokens
    access_token = session_service.create_access_token(user)
    refresh_token = session_service.create_refresh_token(user)
    session_service.register_session(user, refresh_token, ip, ua)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }

class LoginMFARequest(BaseModel):
    mfa_token: str
    code: str

@router.post("/login/mfa", dependencies=[Depends(rate_limiter(10, 60))])
def login_mfa(request: Request, body: LoginMFARequest, db: Session = Depends(get_db)):
    session_service = SessionService(db)
    mfa_service = MFAService(db)
    ip, ua = get_client_details(request)
    
    user_id = verify_mfa_temp_token(body.mfa_token)
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
        
    # Check code against TOTP or backup codes
    is_valid = mfa_service.verify_totp(user, body.code) or mfa_service.verify_backup_code(user, body.code)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid MFA code")

    access_token = session_service.create_access_token(user)
    refresh_token = session_service.create_refresh_token(user)
    session_service.register_session(user, refresh_token, ip, ua)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }

@router.post("/refresh", response_model=RefreshResponse, dependencies=[Depends(rate_limiter(30, 60))])
def refresh(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    session_service = SessionService(db)
    ip, ua = get_client_details(request)
    
    access_token, refresh_token = session_service.rotate_session(body.refresh_token, ip, ua)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/logout")
def logout(request: Request, body: RefreshRequest, db: Session = Depends(get_db)):
    session_service = SessionService(db)
    ip, ua = get_client_details(request)
    session_service.revoke_session_by_token(body.refresh_token, ip, ua)
    return {"status": "ok", "message": "Logged out successfully"}

@router.post("/logout-all")
def logout_all(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session_service = SessionService(db)
    ip, ua = get_client_details(request)
    session_service.revoke_all_sessions(current_user.id, ip, ua)
    return {"status": "ok", "message": "Logged out of all devices"}

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ---------------------------------------------------------------------------
# Email Verification Endpoints
# ---------------------------------------------------------------------------

@router.post("/verify-email")
def verify_email(request: Request, body: VerifyEmailRequest, db: Session = Depends(get_db)):
    verification_service = VerificationService(db)
    ip, ua = get_client_details(request)
    verification_service.verify_email(body.token, ip, ua)
    return {"status": "ok", "message": "Email verified successfully"}

@router.get("/verify-email")
async def verify_email_get(token: str, db: Session = Depends(get_db)):
    from backend.repositories.token_repository import TokenRepository
    repo = TokenRepository(db)
    user_id = repo.verify_email_token(token)
    if not user_id:
        raise HTTPException(400, "Invalid or expired verification link")
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.is_email_verified = True
    user.email_verified_at = datetime.datetime.now(datetime.timezone.utc)
    db.commit()
    from fastapi.responses import RedirectResponse
    import os
    return RedirectResponse(
        url=f"{os.getenv('FRONTEND_URL','http://localhost:3000')}/auth/verified",
        status_code=302
    )

from backend.server import limiter

@router.post("/resend-verification")
@limiter.limit("3/hour")
async def resend_verification(
    request: Request,
    body: Optional[ResendVerificationRequest] = None,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    # Support both current_user verification and legacy verification request by body.email
    if body and body.email:
        verification_service = VerificationService(db)
        ip, ua = get_client_details(request)
        verification_service.resend_verification(body.email, ip, ua)
        return {"status": "ok", "message": "Verification link sent if account exists"}
        
    if current_user.is_email_verified:
        raise HTTPException(400, "Email already verified")
    from backend.repositories.token_repository import TokenRepository
    from backend.services.email_service import send_verification_email
    token = TokenRepository(db).create_email_verification(current_user.id)
    background_tasks.add_task(send_verification_email, current_user.email, token)
    return {"detail": "Verification email sent"}

# ---------------------------------------------------------------------------
# Password Reset Endpoints
# ---------------------------------------------------------------------------

@router.post("/forgot-password")
@limiter.limit("3/hour")
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # Check both the new flow and trigger the legacy service
    user = db.query(User).filter_by(email=body.email.lower()).first()
    if user:
        from backend.repositories.token_repository import TokenRepository
        from backend.services.email_service import send_password_reset_email
        token = TokenRepository(db).create_password_reset(user.id)
        background_tasks.add_task(send_password_reset_email, user.email, token)
        
    # Also trigger legacy reset service so verification/audit logs are created correctly
    try:
        reset_service = ResetService(db)
        ip, ua = get_client_details(request)
        reset_service.forgot_password(body.email, ip, ua)
    except Exception:
        pass
        
    return {"status": "ok", "message": "Password reset link sent if account exists", "detail": "If that email exists, a reset link has been sent"}

@router.post("/reset-password")
async def reset_password(request: Request, body: ResetPasswordRequest, db: Session = Depends(get_db)):
    # Try the new token first
    from backend.repositories.token_repository import TokenRepository
    user_id = TokenRepository(db).verify_reset_token(body.token)
    if user_id:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(404)
        from backend.services.auth_service import get_password_hash
        user.hashed_password = get_password_hash(body.new_password)
        user.token_version = (user.token_version or 0) + 1
        db.commit()
        return {"status": "ok", "detail": "Password updated. Please log in again.", "message": "Password reset successfully"}
        
    # Fallback to legacy verification token
    reset_service = ResetService(db)
    ip, ua = get_client_details(request)
    reset_service.reset_password(body.token, body.new_password, ip, ua)
    return {"status": "ok", "message": "Password reset successfully"}

# ---------------------------------------------------------------------------
# Session Management Endpoints
# ---------------------------------------------------------------------------

@router.get("/sessions", response_model=List[SessionResponse])
def get_sessions(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session_service = SessionService(db)
    active_sessions = session_service.get_active_sessions(current_user.id)
    
    # Try to find current request session to mark is_current
    ip, ua = get_client_details(request)
    from backend.modules.auth.repositories.session_repository import hash_token
    
    # Decode authorization header token if needed or find session matching ip/ua
    res = []
    for s in active_sessions:
        resp_obj = SessionResponse.from_orm(s)
        if s.ip_address == ip and s.user_agent == ua:
            resp_obj.is_current = True
        res.append(resp_obj)
    return res

@router.delete("/sessions/{session_id}")
def revoke_session(session_id: str, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    session_service = SessionService(db)
    ip, ua = get_client_details(request)
    success = session_service.revoke_session_by_id(session_id, current_user.id, ip, ua)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or not owned by user")
    return {"status": "ok", "message": "Session revoked successfully"}

# ---------------------------------------------------------------------------
# MFA / Two-Factor Authentication Endpoints
# ---------------------------------------------------------------------------

@router.post("/mfa/setup", response_model=EnableMFAResponse)
def setup_mfa(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    mfa_service = MFAService(db)
    secret, qr_uri, backup_codes = mfa_service.generate_mfa_setup(current_user)
    return {
        "secret": secret,
        "qr_code_data_uri": qr_uri,
        "backup_codes": backup_codes
    }

@router.post("/mfa/enable")
def enable_mfa(request: Request, body: VerifyMFARequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    mfa_service = MFAService(db)
    ip, ua = get_client_details(request)
    success = mfa_service.verify_and_enable(current_user, body.code, ip, ua)
    if not success:
        raise HTTPException(status_code=400, detail="Invalid MFA verification code")
    return {"status": "ok", "message": "MFA enabled successfully"}

@router.post("/mfa/disable")
def disable_mfa(request: Request, body: DisableMFARequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    mfa_service = MFAService(db)
    ip, ua = get_client_details(request)
    
    # Verify password before disabling MFA
    auth_service.login(email=current_user.email, password=body.password, ip_address=ip, user_agent=ua)
    
    mfa_service.disable_mfa(current_user, ip, ua)
    return {"status": "ok", "message": "MFA disabled successfully"}

# ---------------------------------------------------------------------------
# OAuth Endpoints
# ---------------------------------------------------------------------------

class OAuthLoginRequest(BaseModel):
    provider: str
    token: str

@router.post("/oauth/login")
async def oauth_login(request: Request, body: OAuthLoginRequest, db: Session = Depends(get_db)):
    oauth_service = OAuthService(db)
    session_service = SessionService(db)
    ip, ua = get_client_details(request)
    
    user = await oauth_service.authenticate_oauth(body.provider, body.token, ip, ua)
    if not user:
        raise HTTPException(status_code=401, detail="OAuth authentication failed")

    access_token = session_service.create_access_token(user)
    refresh_token = session_service.create_refresh_token(user)
    session_service.register_session(user, refresh_token, ip, ua)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": UserResponse.from_orm(user)
    }

# ---------------------------------------------------------------------------
# Account Settings Endpoints
# ---------------------------------------------------------------------------

@router.post("/settings/password")
def change_password(request: Request, body: ChangePasswordRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    ip, ua = get_client_details(request)
    auth_service.change_password(current_user, body.current_password, body.new_password, ip, ua)
    return {"status": "ok", "message": "Password changed successfully"}

@router.post("/settings/email")
def change_email(request: Request, body: ChangeEmailRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    ip, ua = get_client_details(request)
    auth_service.change_email(current_user, body.new_email, body.password, ip, ua)
    return {"status": "ok", "message": "Email updated successfully. Please verify your new email."}

@router.post("/settings/profile", response_model=UserResponse)
def update_profile(body: UpdateProfileRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    return auth_service.update_profile(current_user, body.name, body.avatar_url)

@router.delete("/settings/delete")
def delete_account(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    auth_service = AuthService(db)
    ip, ua = get_client_details(request)
    auth_service.delete_account(current_user, ip, ua)
    return {"status": "ok", "message": "Account deleted successfully"}
