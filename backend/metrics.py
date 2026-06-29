from dataclasses import dataclass, field
from threading import Lock
import time

# Simple in-memory counters (replace with Prometheus in production)
_lock = Lock()
_counters = {
    "submissions_total": 0,
    "papers_uploaded_total": 0,
    "tutor_sessions_total": 0,
    "logins_total": 0,
    "login_failures_total": 0,
}
_last_reset = time.time()

def increment(key: str, amount: int = 1):
    with _lock:
        _counters[key] = _counters.get(key, 0) + amount

def get_metrics() -> dict:
    with _lock:
        return {**_counters, "uptime_seconds": int(time.time() - _last_reset)}
