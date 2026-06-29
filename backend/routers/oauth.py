"""
backend/routers/oauth.py

OAuth 2.0 authorization-code flow helpers.

Existing token-based flow (frontend handles redirect, sends token):
  POST /api/auth/oauth/login  ← already in backend/modules/auth/api/v1.py

This router adds the server-side code-exchange flow:
  GET  /api/auth/oauth/{provider}/authorize-url  → {url, state}
  POST /api/auth/oauth/{provider}/exchange        → {access_token, ...}

Also exposes:
  GET  /api/auth/oauth/providers                  → list of providers user has linked
"""

import os
import secrets
import logging
import httpx

from backend.modules.security.rate_limit import check_sliding_window_rate_limit
from fastapi import Request, APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import User, OAuthAccount

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth/oauth", tags=["OAuth"])

_PROVIDERS = {
    "google": {
        "auth_url":    "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url":   "https://oauth2.googleapis.com/token",
        "userinfo_url":"https://openidconnect.googleapis.com/v1/userinfo",
        "scope":       "openid email profile",
        "client_id":     os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
    },
    "github": {
        "auth_url":    "https://github.com/login/oauth/authorize",
        "token_url":   "https://github.com/login/oauth/access_token",
        "userinfo_url":"https://api.github.com/user",
        "scope":       "user:email",
        "client_id":     os.getenv("GITHUB_CLIENT_ID", ""),
        "client_secret": os.getenv("GITHUB_CLIENT_SECRET", ""),
    },
}


# ---------------------------------------------------------------------------
# Authorize-URL  (GET)
# ---------------------------------------------------------------------------

def _get_redis():
    from backend.redis_config import session_redis
    return session_redis

@router.get("/{provider}/authorize-url")
def get_authorize_url(provider: str, request: Request, redirect_uri: str = ""):
    """
    Return the OAuth authorization URL the frontend should redirect the user to.
    """
    from backend.modules.auth.middleware.rate_limit import get_client_ip
    client_ip = get_client_ip(request)
    allowed = check_sliding_window_rate_limit(f"oauth:authorize:{client_ip}", limit=10, window_seconds=60)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many authorization requests. Try again in a minute.")
    cfg = _PROVIDERS.get(provider)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Unknown OAuth provider: {provider}")
    if not cfg["client_id"]:
        raise HTTPException(status_code=503, detail=f"{provider} OAuth not configured")

    state = secrets.token_urlsafe(24)
    
    try:
        r = _get_redis()
        r.setex(f"oauth:state:{state}", 600, "1")
    except Exception as e:
        log.warning("Redis oauth state set failed: %s", e)
        
    if provider == "google":
        url = (
            f"{cfg['auth_url']}?response_type=code"
            f"&client_id={cfg['client_id']}"
            f"&scope={cfg['scope'].replace(' ', '%20')}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
            f"&access_type=offline"
        )
    else:  # github
        url = (
            f"{cfg['auth_url']}?client_id={cfg['client_id']}"
            f"&scope={cfg['scope']}"
            f"&redirect_uri={redirect_uri}"
            f"&state={state}"
        )
    return {"url": url, "state": state, "provider": provider}


# ---------------------------------------------------------------------------
# Code Exchange  (POST)
# ---------------------------------------------------------------------------

class ExchangeRequest(BaseModel):
    code: str
    state: str = ""
    redirect_uri: str = ""


@router.post("/{provider}/exchange")
async def exchange_code(provider: str, body: ExchangeRequest, request: Request, db: Session = Depends(get_db)):
    """
    Exchange an authorization code for a JWT. Creates or links the user account.
    """
    from backend.modules.auth.middleware.rate_limit import get_client_ip
    client_ip = get_client_ip(request)
    allowed = check_sliding_window_rate_limit(f"oauth:exchange:{client_ip}", limit=5, window_seconds=60)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many exchange requests. Try again in a minute.")
    
    allowed_state = check_sliding_window_rate_limit(f"oauth:state_attempt:{body.state[:8]}", limit=3, window_seconds=300)
    if not allowed_state:
        raise HTTPException(status_code=429, detail="Too many attempts with this state token.")
    cfg = _PROVIDERS.get(provider)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Unknown OAuth provider: {provider}")
    if not cfg["client_id"] or not cfg["client_secret"]:
        raise HTTPException(status_code=503, detail=f"{provider} OAuth not configured")

    if not body.state:
        raise HTTPException(status_code=400, detail="Missing state parameter")

    try:
        r = _get_redis()
        # strict check: if redis is working but state not found, it's CSRF
        val = r.get(f"oauth:state:{body.state}")
        if val is None:
            raise HTTPException(status_code=400, detail="Invalid or expired state (CSRF)")
        r.delete(f"oauth:state:{body.state}")
    except HTTPException:
        raise
    except Exception as e:
        log.error("Redis oauth state check failed — returning 503 to prevent CSRF bypass: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Auth service temporarily unavailable. Please try again."
        )

    # Exchange code → access token
    async with httpx.AsyncClient(timeout=10.0) as client:
        token_resp = await client.post(
            cfg["token_url"],
            data={
                "code":          body.code,
                "client_id":     cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uri":  body.redirect_uri,
                "grant_type":    "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="OAuth token exchange failed")

    token_data = token_resp.json()
    access_token = token_data.get("access_token") or token_data.get("id_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="No access token returned")

    # Fetch user info
    userinfo = await _fetch_userinfo(provider, cfg, access_token, token_data)
    if not userinfo:
        raise HTTPException(status_code=401, detail="Failed to fetch user info from provider")

    # Upsert user + OAuthAccount
    user = _upsert_user(db, provider, userinfo)

    # Issue platform JWT
    from backend.modules.auth.services import SessionService
    session_svc = SessionService(db)
    jwt_access  = session_svc.create_access_token(user)
    jwt_refresh = session_svc.create_refresh_token(user)
    session_svc.register_session(user, jwt_refresh, None, None)

    return {
        "access_token":  jwt_access,
        "refresh_token": jwt_refresh,
        "token_type":    "bearer",
        "user": {
            "id":    user.id,
            "email": user.email,
            "name":  user.name,
        },
    }


# ---------------------------------------------------------------------------
# List linked providers  (GET, authenticated)
# ---------------------------------------------------------------------------

@router.get("/providers")
def list_linked_providers(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    accounts = db.query(OAuthAccount).filter_by(user_id=current_user.id).all()
    return {
        "providers": [
            {"provider": a.provider, "email": a.provider_email, "linked_at": a.created_at}
            for a in accounts
        ]
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_userinfo(provider: str, cfg: dict, access_token: str, token_data: dict) -> dict | None:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        if provider == "google":
            # Prefer id_token claims; fallback to userinfo endpoint
            r = await client.get(cfg["userinfo_url"], headers=headers)
        else:
            r = await client.get(cfg["userinfo_url"], headers=headers)
    if r.status_code != 200:
        log.error("userinfo fetch failed: %s %s", r.status_code, r.text[:200])
        return None
    data = r.json()
    if provider == "github":
        # GitHub may not expose email in /user; try /user/emails
        if not data.get("email"):
            async with httpx.AsyncClient(timeout=10.0) as client:
                er = await client.get("https://api.github.com/user/emails", headers=headers)
            if er.status_code == 200:
                for e in er.json():
                    if e.get("primary") and e.get("verified"):
                        data["email"] = e["email"]
                        break
    return data if data.get("email") else None


def _upsert_user(db: Session, provider: str, userinfo: dict) -> User:
    email     = userinfo.get("email", "")
    name      = userinfo.get("name") or userinfo.get("login") or email.split("@")[0]
    avatar    = userinfo.get("picture") or userinfo.get("avatar_url")
    uid       = str(userinfo.get("sub") or userinfo.get("id") or email)

    user = db.query(User).filter_by(email=email).first()
    if not user:
        from backend.repositories.user_repository import UserRepository
        user = UserRepository(db).create(email=email, name=name, avatar_url=avatar)
        user.is_verified = True
        user.is_email_verified = True
        db.commit()
        db.refresh(user)

        try:
            from backend.services.achievement_service import check_and_award
            check_and_award(db, user.id, "user.registered")
        except Exception:
            pass

    # Upsert OAuthAccount
    existing = db.query(OAuthAccount).filter_by(
        provider=provider, provider_user_id=uid
    ).first()
    if not existing:
        oa = OAuthAccount(
            user_id=user.id, provider=provider,
            provider_user_id=uid, provider_email=email,
        )
        db.add(oa)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
    return user
