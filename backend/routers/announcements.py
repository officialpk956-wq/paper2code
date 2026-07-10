import json
import logging
import os
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.dependencies import get_current_user
from backend.models import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["announcements"])

# ---------------------------------------------------------------------------
# Redis helpers (soft-import so the app boots without Redis in tests)
# ---------------------------------------------------------------------------

_ANNOUNCE_KEY = "announcement:current"


def _get_redis():
    try:
        import redis as redis_lib

        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return redis_lib.from_url(url, decode_responses=True)
    except Exception:
        return None


def _admin_required(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ---------------------------------------------------------------------------
# POST /api/admin/announcements
# ---------------------------------------------------------------------------


class AnnouncementCreate(BaseModel):
    message: str
    level: str = "info"  # info | warning | error
    expires_in_seconds: int | None = None


@router.post("/api/admin/announcements", include_in_schema=False)
def create_announcement(
    body: AnnouncementCreate,
    admin: User = Depends(_admin_required),
):
    payload = {
        "message": body.message,
        "level": body.level,
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": admin.id,
    }
    r = _get_redis()
    if r:
        try:
            if body.expires_in_seconds:
                r.setex(_ANNOUNCE_KEY, body.expires_in_seconds, json.dumps(payload))
            else:
                r.set(_ANNOUNCE_KEY, json.dumps(payload))
        except Exception as exc:
            logger.warning("Redis set announcement failed: %s", exc)
    return {"ok": True, "announcement": payload}


# ---------------------------------------------------------------------------
# DELETE /api/admin/announcements
# ---------------------------------------------------------------------------


@router.delete("/api/admin/announcements", include_in_schema=False, status_code=200)
def delete_announcement(admin: User = Depends(_admin_required)):
    r = _get_redis()
    if r:
        try:
            r.delete(_ANNOUNCE_KEY)
        except Exception as exc:
            logger.warning("Redis delete announcement failed: %s", exc)
    return {"ok": True, "cleared": True}


# ---------------------------------------------------------------------------
# GET /api/announcements  (public — no auth)
# ---------------------------------------------------------------------------


@router.get("/api/announcements", include_in_schema=False)
def get_announcement():
    r = _get_redis()
    if not r:
        return {"announcement": None}
    try:
        raw = r.get(_ANNOUNCE_KEY)
    except Exception as exc:
        logger.debug("Redis get announcement failed: %s", exc)
        return {"announcement": None}
    if not raw:
        return {"announcement": None}
    try:
        return {"announcement": json.loads(raw)}
    except json.JSONDecodeError:
        return {"announcement": None}
