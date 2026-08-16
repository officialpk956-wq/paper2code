"""
backend/middleware/alerting.py

Slack webhook alerting for 5xx errors.
Fire-and-forget daemon thread — never blocks the response.
No-op when SLACK_WEBHOOK_URL is unset.
"""

import logging
import os
import threading
import time

import httpx
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger(__name__)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
_ALERT_COOLDOWN_SECONDS = int(os.getenv("SLACK_ALERT_COOLDOWN_SECONDS", "60"))

# Deduplicate: track last alert per route to avoid storms
_last_alert: dict[str, float] = {}
_lock = threading.Lock()


def _should_alert(route: str) -> bool:
    now = time.monotonic()
    with _lock:
        last = _last_alert.get(route, 0)
        if now - last < _ALERT_COOLDOWN_SECONDS:
            return False
        _last_alert[route] = now
    return True


def _send_slack(payload: dict) -> None:
    if not SLACK_WEBHOOK_URL:
        return
    try:
        httpx.post(SLACK_WEBHOOK_URL, json=payload, timeout=5.0)
    except Exception as exc:
        log.debug("Slack alert failed: %s", exc)


def alert_5xx(method: str, path: str, status_code: int, request_id: str = "") -> None:
    """Non-blocking 5xx alert. Deduped by route + cooldown window."""
    if not SLACK_WEBHOOK_URL:
        return
    route_key = f"{method}:{path}"
    if not _should_alert(route_key):
        return
    payload = {
        "text": f":rotating_light: *Paper2Code 5xx* — `{method} {path}` → {status_code}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f":rotating_light: *5xx Error*\n"
                        f"*Route:* `{method} {path}`\n"
                        f"*Status:* `{status_code}`\n"
                        f"*Request-ID:* `{request_id or 'n/a'}`"
                    ),
                },
            }
        ],
    }
    threading.Thread(target=_send_slack, args=(payload,), daemon=True).start()


def alert_backup_status(outcome: str, detail: str = "") -> None:
    """Non-blocking backup status alert. Fire-and-forget via daemon thread."""
    if not SLACK_WEBHOOK_URL:
        return
    emoji = ":x:" if "fail" in outcome.lower() else ":warning:"
    payload = {
        "text": f"{emoji} *Paper2Code Backup* — {outcome}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{emoji} *Database Backup Alert*\n"
                        f"*Outcome:* `{outcome}`\n"
                        f"*Detail:* {detail or 'n/a'}"
                    ),
                },
            }
        ],
    }
    threading.Thread(target=_send_slack, args=(payload,), daemon=True).start()


class SlackAlertingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code >= 500:
            request_id = request.state.request_id if hasattr(request.state, "request_id") else ""
            alert_5xx(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                request_id=request_id,
            )
        return response
