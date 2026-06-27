import datetime
import secrets
from typing import Optional, Sequence
import jwt
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.modules.auth.models import UserSession
from backend.models import User
from backend.modules.auth.repositories.session_repository import SessionRepository, hash_token
from backend.modules.auth.services.audit_service import AuditService
from backend.services.auth_service import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

class SessionService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = SessionRepository(db)
        self.audit_service = AuditService(db)

    def create_access_token(self, user: User) -> str:
        now = datetime.datetime.utcnow()
        expire = now + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        payload = {
            "sub": user.email,
            "user_id": user.id,
            "token_version": user.token_version,
            "exp": expire,
            "iss": "paper2code-auth",
            "aud": "paper2code-app",
            "iat": now,
            "nbf": now,
            "type": "access"
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def create_refresh_token(self, user: User) -> str:
        # Generates a random secure string for refresh token, not a JWT payload to avoid JWT leaks
        # and support absolute revocation check by matching hash.
        return secrets.token_hex(40)

    def register_session(
        self,
        user: User,
        refresh_token: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> UserSession:
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        session = self.repo.create_session(
            user_id=user.id,
            refresh_token=refresh_token,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        self.db.commit()
        return session

    def rotate_session(
        self,
        refresh_token: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> tuple[str, str]:
        """
        Rotates refresh token. Implements replay attack detection and concurrency grace period.
        """
        token_hash = hash_token(refresh_token)
        # Lock session row to prevent race conditions during concurrent requests
        session = self.repo.get_session_by_token_hash(token_hash, for_update=True)

        if not session:
            # Token not found - could be a replay attack of a token that has been rotated/revoked!
            # To be safe, we reject and raise 401
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )

        if session.revoked or session.expires_at < datetime.datetime.utcnow():
            # Grace Period Check:
            # If the session was revoked recently (within last 10 seconds), and the request
            # metadata (IP address and User Agent) matches exactly, we allow the request
            # to succeed by generating a new rotated session. This prevents false positive
            # replay attack triggers due to transient network retries or client concurrency.
            if session.revoked and session.rotated_at:
                grace_limit = session.rotated_at + datetime.timedelta(seconds=10)
                if datetime.datetime.utcnow() <= grace_limit and session.ip_address == ip_address and session.user_agent == user_agent:
                    user = session.user
                    new_access = self.create_access_token(user)
                    new_refresh = self.create_refresh_token(user)
                    
                    expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
                    self.repo.create_session(
                        user_id=user.id,
                        refresh_token=new_refresh,
                        ip_address=ip_address,
                        user_agent=user_agent,
                        expires_at=expires_at
                    )
                    self.db.commit()
                    return new_access, new_refresh

            # Replay attack: Revoke all sessions for this user to protect the account
            self.repo.revoke_all_user_sessions(session.user_id)
            self.db.commit()
            
            self.audit_service.log(
                action="replay_attack_detected",
                user_id=session.user_id,
                ip_address=ip_address,
                device=user_agent,
                metadata_dict={"revoked_token_hash": token_hash}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Replay attack detected. All sessions revoked."
            )

        # Generate new tokens
        user = session.user
        new_access = self.create_access_token(user)
        new_refresh = self.create_refresh_token(user)

        # Revoke the old session and record the rotation time
        session.revoked = True
        session.rotated_at = datetime.datetime.utcnow()

        # Create a new session for the rotated token
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        self.repo.create_session(
            user_id=user.id,
            refresh_token=new_refresh,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at
        )
        self.db.commit()

        return new_access, new_refresh

    def revoke_session_by_token(self, refresh_token: str, ip_address: Optional[str], user_agent: Optional[str]) -> None:
        token_hash = hash_token(refresh_token)
        session = self.repo.get_session_by_token_hash(token_hash)
        if session:
            self.repo.revoke_session(session.id)
            self.db.commit()
            self.audit_service.log(
                action="logout",
                user_id=session.user_id,
                ip_address=ip_address,
                device=user_agent,
                metadata_dict={"session_id": session.id}
            )

    def revoke_session_by_id(self, session_id: str, user_id: int, ip_address: Optional[str], user_agent: Optional[str]) -> bool:
        session = self.repo.get_session_by_id(session_id)
        if session and session.user_id == user_id:
            res = self.repo.revoke_session(session_id)
            self.db.commit()
            self.audit_service.log(
                action="session_revoked",
                user_id=user_id,
                ip_address=ip_address,
                device=user_agent,
                metadata_dict={"revoked_session_id": session_id}
            )
            return res
        return False

    def revoke_all_sessions(self, user_id: int, ip_address: Optional[str], user_agent: Optional[str]) -> None:
        self.repo.revoke_all_user_sessions(user_id)
        
        # Increment token version to invalidate all current access tokens
        user = self.db.get(User, user_id)
        if user:
            user.token_version += 1
            
        self.db.commit()
        
        self.audit_service.log(
            action="logout_all_devices",
            user_id=user_id,
            ip_address=ip_address,
            device=user_agent,
            metadata_dict={"new_token_version": user.token_version if user else None}
        )

    def get_active_sessions(self, user_id: int) -> Sequence[UserSession]:
        return self.repo.get_active_sessions_for_user(user_id)
