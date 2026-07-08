from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from backend.models import User
from backend.repositories.user_repository import UserRepository
from backend.modules.auth.security.hashing import (
    validate_password_strength,
    hash_password,
    verify_password_and_needs_rehash,
)
from backend.modules.auth.services.verification_service import VerificationService
from backend.modules.auth.services.session_service import SessionService
from backend.modules.auth.services.audit_service import AuditService
from backend.modules.auth.services.email_service import EmailService
from backend.modules.security.brute_force import check_login_brute_force, record_failed_attempt, reset_failed_attempts
from backend.modules.auth.allowlist import is_email_allowed

class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = UserRepository(db)
        self.verification_service = VerificationService(db)
        self.session_service = SessionService(db)
        self.audit_service = AuditService(db)

    def register(
        self,
        email: str,
        name: str,
        password: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> User:
        # 0. Private-preview allowlist (no-op unless ALLOWLIST_EMAILS is set)
        if not is_email_allowed(email):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Sign-ups are limited to invited accounts during the private preview.",
            )

        # 1. Validate password strength
        is_strong, err_msg = validate_password_strength(password)
        if not is_strong:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg
            )

        # 2. Check if user already exists
        existing = self.repo.get_by_email(email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # 3. Create user
        hashed = hash_password(password)
        user = self.repo.create(email=email, name=name, hashed_password=hashed)
        self.db.commit()

        # 4. Generate verification token and send email
        self.verification_service.generate_and_send_verification(user)

        self.audit_service.log(
            action="registration",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent
        )
        return user

    def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> User:
        # Enforce account lockout checks
        check_login_brute_force(email)

        # Private-preview allowlist (no-op unless ALLOWLIST_EMAILS is set)
        if not is_email_allowed(email):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This site is in private preview and your account is not on the access list.",
            )

        user = self.repo.get_by_email(email)
        
        if not user or not user.hashed_password:
            # Increment failed attempts and trigger lockout if limit exceeded
            record_failed_attempt(email, self.db)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        # Verify password and check for rehashing requirement
        is_verified, needs_rehash = verify_password_and_needs_rehash(password, user.hashed_password)
        if not is_verified:
            # Increment failed attempts and trigger lockout if limit exceeded
            record_failed_attempt(email, self.db)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )

        # Success path: reset failed attempts
        reset_failed_attempts(email)
        user.failed_login_attempts = 0
        user.lockout_until = None

        # Automatic password rehash when parameters change
        if needs_rehash:
            user.hashed_password = hash_password(password)
            self.audit_service.log(
                action="password_rehashed",
                user_id=user.id,
                ip_address=ip_address,
                device=user_agent
            )

        self.db.commit()
        
        self.audit_service.log(
            action="login_success",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent
        )
        return user

    def change_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> None:
        # Verify current password
        is_verified, _ = verify_password_and_needs_rehash(current_password, user.hashed_password)
        if not is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect current password"
            )

        # Validate strength of new password
        is_strong, err_msg = validate_password_strength(new_password)
        if not is_strong:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg
            )

        user.hashed_password = hash_password(new_password)
        user.token_version += 1
        
        # Revoke all sessions
        self.session_service.revoke_all_sessions(user.id, ip_address, user_agent)
        self.db.commit()

        self.audit_service.log(
            action="password_changed",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent
        )

    def change_email(
        self,
        user: User,
        new_email: str,
        password: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> None:
        # Verify password
        is_verified, _ = verify_password_and_needs_rehash(password, user.hashed_password)
        if not is_verified:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect password"
            )

        # Check if email is already taken
        existing = self.repo.get_by_email(new_email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already taken"
            )

        user.email = new_email
        user.is_verified = False  # Mark unverified and force verification
        self.db.commit()

        # Send new verification
        self.verification_service.generate_and_send_verification(user)

        self.audit_service.log(
            action="email_changed",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent,
            metadata_dict={"new_email": new_email}
        )

    def update_profile(self, user: User, name: Optional[str], avatar_url: Optional[str]) -> User:
        if name:
            user.name = name
        if avatar_url:
            user.avatar_url = avatar_url
        self.db.commit()
        return user

    def delete_account(self, user: User, ip_address: Optional[str], user_agent: Optional[str]) -> None:
        # Log event BEFORE purging user data (to keep a trace, or write to general log)
        self.audit_service.log(
            action="account_deletion",
            user_id=user.id,
            ip_address=ip_address,
            device=user_agent,
            metadata_dict={"email": user.email}
        )
        
        # Revoke all sessions
        self.session_service.revoke_all_sessions(user.id, ip_address, user_agent)
        
        # Purge user row (cascades database deletions)
        email = user.email
        name = user.name
        self.repo.delete(user.id)
        self.db.commit()

        # Send confirmation email
        EmailService.send_account_deleted_email(email, name)
