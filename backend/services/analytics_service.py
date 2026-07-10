"""
backend/services/analytics_service.py

PostHog event tracking. Silently no-ops when POSTHOG_API_KEY is unset.

Usage:
    from backend.services.analytics_service import track
    track(user_id, "paper_uploaded", {"paper_id": 7, "visibility": "public"})
"""

import logging
import os
import threading

import httpx

log = logging.getLogger(__name__)

POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY", "")
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://app.posthog.com").rstrip("/")
_CAPTURE_URL = f"{POSTHOG_HOST}/capture/"


def track(user_id: int | None, event: str, properties: dict | None = None) -> None:
    """
    Fire-and-forget PostHog event. Never raises; runs in a daemon thread.
    Silently returns when POSTHOG_API_KEY is not configured.
    """
    if not POSTHOG_API_KEY:
        return
    payload = {
        "api_key": POSTHOG_API_KEY,
        "event": event,
        "distinct_id": str(user_id) if user_id is not None else "anonymous",
        "properties": properties or {},
    }
    threading.Thread(target=_send, args=(payload,), daemon=True).start()


def _send(payload: dict) -> None:
    try:
        httpx.post(_CAPTURE_URL, json=payload, timeout=5.0)
    except Exception as exc:
        log.debug("PostHog capture error: %s", exc)
