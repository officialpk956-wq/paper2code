import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from backend.models import EmailVerificationToken, PasswordResetToken


class TokenRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_email_verification(self, user_id: int) -> str:
        token = secrets.token_urlsafe(48)
        expires = datetime.now(UTC) + timedelta(hours=24)
        self.db.add(EmailVerificationToken(user_id=user_id, token=token, expires_at=expires))
        self.db.commit()
        return token

    def verify_email_token(self, token: str) -> int | None:
        """Returns user_id if valid and unused, else None."""
        row = self.db.query(EmailVerificationToken).filter_by(token=token, used_at=None).first()
        if not row or row.expires_at < datetime.now(UTC):
            return None
        row.used_at = datetime.now(UTC)
        self.db.commit()
        return row.user_id

    def create_password_reset(self, user_id: int) -> str:
        # Invalidate any existing tokens for this user first
        self.db.query(PasswordResetToken).filter_by(user_id=user_id, used_at=None).update(
            {"used_at": datetime.now(UTC)}
        )
        token = secrets.token_urlsafe(48)
        expires = datetime.now(UTC) + timedelta(minutes=15)
        self.db.add(PasswordResetToken(user_id=user_id, token=token, expires_at=expires))
        self.db.commit()
        return token

    def verify_reset_token(self, token: str) -> int | None:
        row = self.db.query(PasswordResetToken).filter_by(token=token, used_at=None).first()
        if not row or row.expires_at < datetime.now(UTC):
            return None
        row.used_at = datetime.now(UTC)
        self.db.commit()
        return row.user_id
