import datetime
import hashlib

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.modules.auth.models import ResetToken, VerificationToken


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class VerificationRepository:
    def __init__(self, db: Session):
        self.db = db

    # Verification Tokens
    def create_verification_token(
        self, user_id: int, token: str, expires_at: datetime.datetime
    ) -> VerificationToken:
        hashed = hash_token(token)
        vt = VerificationToken(
            user_id=user_id, token_hash=hashed, expires_at=expires_at, used=False
        )
        self.db.add(vt)
        self.db.flush()
        return vt

    def get_verification_token(self, token: str) -> VerificationToken | None:
        hashed = hash_token(token)
        stmt = select(VerificationToken).where(
            VerificationToken.token_hash == hashed, VerificationToken.used == False
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def mark_verification_token_used(self, token_id: int) -> None:
        stmt = update(VerificationToken).where(VerificationToken.id == token_id).values(used=True)
        self.db.execute(stmt)
        self.db.flush()

    # Reset Tokens
    def create_reset_token(
        self, user_id: int, token: str, expires_at: datetime.datetime
    ) -> ResetToken:
        hashed = hash_token(token)
        rt = ResetToken(user_id=user_id, token_hash=hashed, expires_at=expires_at, used=False)
        self.db.add(rt)
        self.db.flush()
        return rt

    def get_reset_token(self, token: str) -> ResetToken | None:
        hashed = hash_token(token)
        stmt = select(ResetToken).where(ResetToken.token_hash == hashed, ResetToken.used == False)
        return self.db.execute(stmt).scalar_one_or_none()

    def mark_reset_token_used(self, token_id: int) -> None:
        stmt = update(ResetToken).where(ResetToken.id == token_id).values(used=True)
        self.db.execute(stmt)
        self.db.flush()
