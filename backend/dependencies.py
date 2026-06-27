from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend.database import get_db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
_optional_bearer = HTTPBearer(auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from backend.modules.auth.dependencies import get_current_user as _get_current_user
    return _get_current_user(token, db)

def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
    db: Session = Depends(get_db),
):
    """Returns the authenticated User, or None if no valid Bearer token is present."""
    if credentials is None:
        return None
    try:
        from backend.modules.auth.dependencies import get_current_user as _get_current_user
        return _get_current_user(credentials.credentials, db)
    except Exception:
        return None

