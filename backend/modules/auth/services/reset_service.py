import datetime
import secrets
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from backend.models import User
from backend.modules.auth.repositories.verification_repository import VerificationRepository
from backend.modules.auth.repositories.session_repository import SessionRepository
from backend.modules.auth.security.hashing import hash_password, validate_password_strength
from backend.modules.auth.services.email_service import EmailService
from backend.modules.auth.services.audit_service import AuditService

class ResetService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = VerificationRepository(db)
        self.session_repo = SessionRepository(db)
        self.audit_service = AuditService(db)

    def forgot_password(self, email: str, ip_address: Optional[str], user_agent: Optional[str]) -> None:
        from backend.repositories.user_repository import UserRepository
        user_repo = UserRepository(self.db)
        user = user_repo.get_by_email(email)
        if not user:
            # Silent return to avoid email enumeration
            return

        # Generate 15-minute token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        
        self.repo.create_reset_token(user_id=user.id, token=token, expires_at=expires_at)
        self.db.commit()

        # Send reset email
        EmailService.send_reset_password_email(user.email, token)
        
        self.audit_service.log(
            action="forgot_password_requested",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent
        )

    def reset_password(self, token: str, new_password: str, ip_address: Optional[str], user_agent: Optional[str]) -> bool:
        rt = self.repo.get_reset_token(token)
        if not rt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )

        if rt.expires_at < datetime.datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired"
            )

        # Validate strength of new password
        is_strong, err_msg = validate_password_strength(new_password)
        if not is_strong:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg
            )

        user = rt.user
        
        # Hash new password
        user.hashed_password = hash_password(new_password)
        
        # Invalidate current session tokens & increment token_version
        user.token_version += 1
        self.session_repo.revoke_all_user_sessions(user.id)
        
        # Mark token used
        self.repo.mark_reset_token_used(rt.id)
        self.db.commit()

        # Log event
        self.audit_service.log(
            action="password_reset_success",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent
        )
        return True
