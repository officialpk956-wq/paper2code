"""
backend/services/tutor_session_store.py

Redis-backed store for AI tutor sessions with in-memory fallback for dev/test.

Security guarantees:
  - Session IDs are server-generated UUIDs (not user-supplied)
  - Each session is bound to a user_id; cross-user access is rejected
  - Per-user daily query limit enforced via Redis INCR + TTL
"""

import datetime
import json
import logging
import os
import time
import uuid

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TUTOR_DAILY_LIMIT = int(os.getenv("TUTOR_DAILY_LIMIT", "30"))
SESSION_TTL = 1800  # 30 minutes of inactivity


class TutorSessionStore:
    def __init__(self):
        self._redis = None
        self._mem_sessions: dict = {}  # session_id -> data dict
        self._mem_rates: dict = {}  # "user_id:date" -> count
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            from backend.redis_config import session_redis

            if session_redis:
                session_redis.ping()
                self._redis = session_redis
                log.info("TutorSessionStore: Redis connected")
            else:
                self._redis = None
                log.warning("TutorSessionStore: session_redis is None; falling back to in-memory")
        except Exception as exc:
            log.warning("TutorSessionStore: Redis unavailable (%s); falling back to in-memory", exc)
            self._redis = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, user_id: int) -> str:
        """Create a server-generated session owned by user_id. Returns session_id."""
        session_id = str(uuid.uuid4())
        data = {
            "user_id": str(user_id),
            "history": [],
            "created_at": time.time(),
            "last_active": time.time(),
        }
        if self._redis:
            self._redis.setex(f"tutor:session:{session_id}", SESSION_TTL, json.dumps(data))
        else:
            self._mem_sessions[session_id] = data
        return session_id

    def validate_ownership(self, session_id: str, user_id: int) -> bool:
        """Return True iff session_id exists and belongs to user_id."""
        data = self._load(session_id)
        if not data:
            return False
        return str(data.get("user_id")) == str(user_id)

    def get_history(self, session_id: str) -> list:
        data = self._load(session_id)
        return data.get("history", []) if data else []

    def update_history(self, session_id: str, history: list) -> None:
        data = self._load(session_id) or {}
        data["history"] = history[-6:]  # cap at 6 messages to avoid token bloat
        data["last_active"] = time.time()
        self._save(session_id, data)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_and_incr_rate(self, user_id: int) -> tuple:
        """
        Atomically increment today's query counter for user_id.
        Returns (allowed: bool, count: int, limit: int).
        In-memory fallback always allows (dev/test).
        """
        if self._redis:
            today = datetime.date.today().isoformat()
            key = f"tutor:rate:{user_id}:{today}"
            count = self._redis.incr(key)
            if count == 1:
                self._redis.expire(key, 86400)  # reset at midnight + up to 1 day
            allowed = count <= TUTOR_DAILY_LIMIT
            return allowed, count, TUTOR_DAILY_LIMIT

        # In-memory: no enforcement
        return True, 0, TUTOR_DAILY_LIMIT

    def get_rate_info(self, user_id: int) -> tuple:
        """Returns (current_count, limit) without incrementing."""
        if self._redis:
            today = datetime.date.today().isoformat()
            raw = self._redis.get(f"tutor:rate:{user_id}:{today}")
            count = int(raw) if raw else 0
            return count, TUTOR_DAILY_LIMIT
        return 0, TUTOR_DAILY_LIMIT

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, session_id: str):
        if self._redis:
            raw = self._redis.get(f"tutor:session:{session_id}")
            if not raw:
                return None
            self._redis.expire(f"tutor:session:{session_id}", SESSION_TTL)
            return json.loads(raw)
        return self._mem_sessions.get(session_id)

    def _save(self, session_id: str, data: dict) -> None:
        if self._redis:
            self._redis.setex(f"tutor:session:{session_id}", SESSION_TTL, json.dumps(data))
        else:
            self._mem_sessions[session_id] = data


# Singleton — imported by learning.py
tutor_session_store = TutorSessionStore()
