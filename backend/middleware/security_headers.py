"""
backend/middleware/security_headers.py

Consolidated security headers middleware:
- Reads Content-Security-Policy from CONTENT_SECURITY_POLICY env var (validated at production startup)
- Falls back to a restrictive default CSP (scoped connect-src, no unsafe-eval)
- Applies comprehensive security headers (XFO, XCTO, Referrer-Policy, Permissions-Policy, COOP, CORP)
- Enforces HSTS when on HTTPS or in production environment
- Uses setdefault() to allow endpoint-level header customization
"""

import os

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https:; "
    "font-src 'self' data: https://cdn.jsdelivr.net; "
    "connect-src 'self' https://paper2code-1-81y5.onrender.com https://observablehq.com https://us.i.posthog.com; "
    "frame-ancestors 'none';"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # Apply standard security headers with setdefault to allow endpoint-level overrides
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

        # Content Security Policy (CSP)
        csp_header = os.getenv("CONTENT_SECURITY_POLICY") or DEFAULT_CSP
        response.headers.setdefault("Content-Security-Policy", csp_header)

        # HSTS only for HTTPS requests or in production
        is_https = (
            request.headers.get("x-forwarded-proto") == "https"
            or request.url.scheme == "https"
            or os.getenv("ENVIRONMENT") == "production"
        )
        if is_https:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )

        return response
