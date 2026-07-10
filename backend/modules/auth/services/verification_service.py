import datetime
import secrets

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.auth.repositories.verification_repository import VerificationRepository
from backend.modules.auth.services.audit_service import AuditService
from backend.modules.auth.services.email_service import EmailService


class VerificationService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = VerificationRepository(db)
        self.audit_service = AuditService(db)

    def generate_and_send_verification(self, user: User) -> str:
        # Generate token
        token = secrets.token_urlsafe(32)
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=24)

        self.repo.create_verification_token(user_id=user.id, token=token, expires_at=expires_at)
        self.db.commit()

        # Send email
        EmailService.send_verification_email(user.email, token)
        return token

    def verify_email(self, token: str, ip_address: str | None, user_agent: str | None) -> bool:
        vt = self.repo.get_verification_token(token)
        if not vt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token",
            )

        if vt.expires_at < datetime.datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Verification token has expired"
            )

        # Mark token used
        self.repo.mark_verification_token_used(vt.id)

        # Verify user
        user = vt.user
        user.is_verified = True
        self.db.commit()

        # Log event
        self.audit_service.log(
            action="email_verified", user_id=user.id, ip_address=ip_address, device=user_agent
        )
        return True

    def resend_verification(
        self, email: str, ip_address: str | None, user_agent: str | None
    ) -> None:
        from backend.repositories.user_repository import UserRepository

        user_repo = UserRepository(self.db)
        user = user_repo.get_by_email(email)
        if not user:
            # Silent return to prevent email enumeration
            return

        if user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email is already verified"
            )

        self.generate_and_send_verification(user)
        self.audit_service.log(
            action="resend_verification_requested",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent,
        )
