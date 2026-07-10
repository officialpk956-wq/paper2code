"""
backend/middleware/metrics.py

Prometheus metrics middleware.
Tracks: http_requests_total, http_request_duration_seconds per method/route/status.

Exposed at GET /metrics (restrict to internal IPs via Nginx in production).
No-op if prometheus_client is not installed.
"""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    log.info("prometheus_client not installed — metrics disabled")

if _PROMETHEUS_AVAILABLE:
    _REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "route", "status_code"],
    )
    _REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency",
        ["method", "route"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )


def _normalise_path(path: str) -> str:
    """Replace numeric path segments with {id} to avoid cardinality explosion."""
    import re

    path = re.sub(r"/\d+", "/{id}", path)
    return path


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not _PROMETHEUS_AVAILABLE:
            return await call_next(request)

        route = _normalise_path(request.url.path)
        method = request.method
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as exc:
            _REQUEST_COUNT.labels(method=method, route=route, status_code="500").inc()
            raise exc
        finally:
            duration = time.perf_counter() - start
            _REQUEST_LATENCY.labels(method=method, route=route).observe(duration)

        _REQUEST_COUNT.labels(method=method, route=route, status_code=status).inc()
        return response


def metrics_endpoint():
    """FastAPI route handler for GET /metrics."""
    if not _PROMETHEUS_AVAILABLE:
        return Response("prometheus_client not installed", status_code=503)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
