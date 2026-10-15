import datetime
import secrets

from fastapi import HTTPException, status
from sqlalchemy import update
from sqlalchemy.orm import Session

from backend.models import EmailVerificationToken, User
from backend.modules.auth.models import VerificationToken
from backend.modules.auth.repositories.verification_repository import (
    VerificationRepository,
    hash_token,
)
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
        vt = self.db.query(VerificationToken).filter_by(token_hash=hash_token(token)).first()
        legacy = vt is None
        if legacy:
            vt = self.db.query(EmailVerificationToken).filter_by(token=token).first()
        if not vt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token",
            )

        used = vt.used_at is not None if legacy else vt.used
        if used:
            raise HTTPException(status_code=400, detail="Verification link already used")
        if vt.expires_at.replace(tzinfo=None) < datetime.datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Verification token has expired"
            )

        # Consume either issuer's token atomically with the account update.
        now = datetime.datetime.utcnow()
        if legacy:
            claim = (
                update(EmailVerificationToken)
                .where(EmailVerificationToken.id == vt.id, EmailVerificationToken.used_at.is_(None))
                .values(used_at=now)
            )
        else:
            claim = (
                update(VerificationToken)
                .where(VerificationToken.id == vt.id, VerificationToken.used.is_(False))
                .values(used=True)
            )
        if self.db.execute(claim).rowcount != 1:
            self.db.rollback()
            raise HTTPException(status_code=400, detail="Verification link already used")
        user = self.db.get(User, vt.user_id)
        if not user:
            self.db.rollback()
            raise HTTPException(status_code=400, detail="Invalid verification token")
        if user.is_email_verified:
            self.db.commit()
            raise HTTPException(status_code=400, detail="Verification link already used")
        user.is_verified = True
        user.is_email_verified = True
        user.email_verified_at = now
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
