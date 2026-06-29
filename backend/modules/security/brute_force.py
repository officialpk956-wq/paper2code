import os
import time
import redis
import logging
from typing import Optional
from fastapi import HTTPException, status
from backend.modules.auth.services.audit_service import AuditService

logger = logging.getLogger(__name__)

from backend.redis_config import rate_limit_redis as _redis_client

# Simple in-memory fallback for lockout
_in_memory_failed_attempts = {}
_in_memory_lockouts = {}

MAX_ATTEMPTS = int(os.getenv("BRUTE_FORCE_MAX_ATTEMPTS", "5"))
LOCKOUT_DURATION = int(os.getenv("BRUTE_FORCE_LOCKOUT_DURATION", "900")) # 15 min

def get_failed_key(email: str) -> str:
    return f"auth:failed:{email}"

def is_locked_out(email: str) -> bool:
    key = get_failed_key(email)
    now = int(time.time())
    
    if _redis_client is not None:
        try:
            lockout_time_str = _redis_client.get(f"auth:lockout:{email}")
            if lockout_time_str:
                lockout_until = int(lockout_time_str)
                if now < lockout_until:
                    return True
        except Exception:
            pass
            
    # In-memory check
    lockout_until = _in_memory_lockouts.get(email, 0)
    return now < lockout_until

def record_failed_attempt(email: str, db_session) -> None:
    key = get_failed_key(email)
    now = int(time.time())
    
    attempts = 0
    if _redis_client is not None:
        try:
            attempts = _redis_client.incr(key)
            if attempts == 1:
                _redis_client.expire(key, LOCKOUT_DURATION)
            if attempts >= MAX_ATTEMPTS:
                _redis_client.set(f"auth:lockout:{email}", str(now + LOCKOUT_DURATION), ex=LOCKOUT_DURATION)
                _redis_client.delete(key)
        except Exception:
            pass
    else:
        attempts = _in_memory_failed_attempts.get(email, 0) + 1
        _in_memory_failed_attempts[email] = attempts
        if attempts >= MAX_ATTEMPTS:
            _in_memory_lockouts[email] = now + LOCKOUT_DURATION
            _in_memory_failed_attempts[email] = 0

    # Log security event in audit logs table
    # Resolve user if exists to link ID
    from backend.models import User
    user = db_session.query(User).filter_by(email=email).first()
    audit_service = AuditService(db_session)
    audit_service.log(
        action="login_failed_lockout" if attempts >= MAX_ATTEMPTS else "login_failed_invalid_credentials",
        user_id=user.id if user else None,
        ip_address=None,
        device=None,
        metadata_dict={"email": email}
    )

def reset_failed_attempts(email: str) -> None:
    key = get_failed_key(email)
    if _redis_client is not None:
        try:
            _redis_client.delete(key)
            _redis_client.delete(f"auth:lockout:{email}")
        except Exception:
            pass
    else:
        _in_memory_failed_attempts.pop(email, None)
        _in_memory_lockouts.pop(email, None)

def check_login_brute_force(email: str):
    """Enforce lockout and prevent user enumeration by throwing identical 401 errors."""
    if is_locked_out(email):
        logger.info(f"Failed login attempt for locked account: {email}")
        # Return identical 401 error to prevent enumeration of lockout status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
