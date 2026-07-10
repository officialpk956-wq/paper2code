"""
backend/middleware/sentry_context.py

Enriches Sentry error reports with the authenticated user's ID and email
so engineers can filter events by user in the Sentry UI.

Decodes the JWT with verify_exp=False — we only want the identity for
diagnostic purposes, not to gate access. No-op when SENTRY_DSN is unset or
the request carries no valid Bearer token.
"""

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_SENTRY_ACTIVE = bool(os.getenv("SENTRY_DSN", ""))


class SentryUserContextMiddleware(BaseHTTPMiddleware):
    """Attach JWT identity to every Sentry event for this request scope."""

    async def dispatch(self, request: Request, call_next):
        if _SENTRY_ACTIVE:
            _attach_user(request)
        return await call_next(request)


def _attach_user(request: Request) -> None:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return
    try:
        import jwt as _jwt
        import sentry_sdk

        from backend.modules.auth.config import ALGORITHM, SECRET_KEY

        payload = _jwt.decode(
            auth[7:],
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"verify_aud": False, "verify_iss": False, "verify_exp": False},
        )
        uid = payload.get("user_id")
        email = payload.get("sub")
        if uid is not None:
            sentry_sdk.set_user({"id": uid, "email": email})
    except Exception:
        pass
