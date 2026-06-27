import uuid
import datetime
import hashlib
from typing import Optional, Sequence
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from backend.modules.auth.models import UserSession
from backend.modules.auth.utils.ua_parser import parse_user_agent

def hash_token(token: str) -> str:
    """Hash refresh tokens before comparison or storage using SHA256."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

class SessionRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_session(
        self,
        user_id: int,
        refresh_token: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        expires_at: datetime.datetime,
    ) -> UserSession:
        token_hash = hash_token(refresh_token)
        browser, os = parse_user_agent(user_agent or "")
        
        session_id = str(uuid.uuid4())
        session = UserSession(
            id=session_id,
            user_id=user_id,
            refresh_token_hash=token_hash,
            ip_address=ip_address,
            user_agent=user_agent,
            browser=browser,
            os=os,
            expires_at=expires_at,
            revoked=False
        )
        self.db.add(session)
        self.db.flush()
        return session

    def get_session_by_token_hash(self, token_hash: str, for_update: bool = False) -> Optional[UserSession]:
        stmt = select(UserSession).where(UserSession.refresh_token_hash == token_hash)
        if for_update:
            stmt = stmt.with_for_update()
        return self.db.execute(stmt).scalar_one_or_none()

    def get_session_by_id(self, session_id: str) -> Optional[UserSession]:
        return self.db.get(UserSession, session_id)

    def update_last_used(self, session_id: str, new_token: str) -> None:
        new_hash = hash_token(new_token)
        stmt = (
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(
                refresh_token_hash=new_hash,
                last_used_at=datetime.datetime.utcnow()
            )
        )
        self.db.execute(stmt)
        self.db.flush()

    def revoke_session(self, session_id: str) -> bool:
        stmt = (
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(revoked=True)
        )
        res = self.db.execute(stmt)
        self.db.flush()
        return res.rowcount > 0

    def revoke_all_user_sessions(self, user_id: int) -> int:
        stmt = (
            update(UserSession)
            .where(UserSession.user_id == user_id, UserSession.revoked == False)
            .values(revoked=True)
        )
        res = self.db.execute(stmt)
        self.db.flush()
        return res.rowcount

    def get_active_sessions_for_user(self, user_id: int) -> Sequence[UserSession]:
        now = datetime.datetime.utcnow()
        stmt = (
            select(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.revoked == False,
                UserSession.expires_at > now
            )
            .order_by(UserSession.last_used_at.desc())
        )
        return self.db.execute(stmt).scalars().all()
