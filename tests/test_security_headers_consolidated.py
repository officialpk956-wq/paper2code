"""
tests/test_security_headers_consolidated.py

Tests verifying:
1. Content-Security-Policy header respects CONTENT_SECURITY_POLICY environment variable when set.
2. Content-Security-Policy falls back to a restrictive default when unset (no wildcard connect-src, no unsafe-eval).
3. Comprehensive security headers are present (XFO, XCTO, Referrer-Policy, Permissions-Policy, COOP, CORP).
4. Strict-Transport-Security (HSTS) is attached on HTTPS requests and in production mode.
5. Only ONE SecurityHeadersMiddleware exists in the codebase (no duplicate dead code).
"""

import os
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from backend.middleware.security_headers import SecurityHeadersMiddleware, DEFAULT_CSP


def create_test_app():
    async def sample_endpoint(request):
        return JSONResponse({"status": "ok"})

    app = Starlette(routes=[Route("/api/test", sample_endpoint)])
    app.add_middleware(SecurityHeadersMiddleware)
    return app


def test_csp_env_var_applied(monkeypatch):
    """Scenario 1: CONTENT_SECURITY_POLICY env var is read and applied to responses."""
    custom_csp = "default-src 'self'; connect-src 'self' https://api.custom.com; frame-ancestors 'none';"
    monkeypatch.setenv("CONTENT_SECURITY_POLICY", custom_csp)

    app = create_test_app()
    client = TestClient(app)
    resp = client.get("/api/test")

    assert resp.status_code == 200
    assert resp.headers.get("Content-Security-Policy") == custom_csp


def test_csp_default_fallback_is_safe_and_scoped(monkeypatch):
    """Scenario 2: Unset CONTENT_SECURITY_POLICY falls back to safe default (no wildcard connect-src)."""
    monkeypatch.delenv("CONTENT_SECURITY_POLICY", raising=False)

    app = create_test_app()
    client = TestClient(app)
    resp = client.get("/api/test")

    assert resp.status_code == 200
    csp = resp.headers.get("Content-Security-Policy", "")
    assert csp == DEFAULT_CSP
    # Verify connect-src is NOT wide open to all http/https/ws/wss
    assert "connect-src 'self' ws: wss: http: https:" not in csp
    assert "https://paper2code-1-81y5.onrender.com" in csp
    assert "'unsafe-eval'" not in csp


def test_all_standard_security_headers_present(monkeypatch):
    """Scenario 3: Full suite of security headers (XFO, XCTO, Referrer, Permissions, COOP, CORP) are present."""
    monkeypatch.delenv("CONTENT_SECURITY_POLICY", raising=False)

    app = create_test_app()
    client = TestClient(app)
    resp = client.get("/api/test")

    assert resp.status_code == 200
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "camera=()" in resp.headers.get("Permissions-Policy", "")
    assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert resp.headers.get("Cross-Origin-Resource-Policy") == "same-origin"
    # HTTP non-prod should not have HSTS
    assert resp.headers.get("Strict-Transport-Security") is None


def test_hsts_applied_on_https_and_production(monkeypatch):
    """Scenario 3b: HSTS applied on HTTPS request header or in production environment."""
    app = create_test_app()
    client = TestClient(app)

    # 1. Via x-forwarded-proto https
    resp_proto = client.get("/api/test", headers={"x-forwarded-proto": "https"})
    assert resp_proto.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"

    # 2. Via production environment flag
    monkeypatch.setenv("ENVIRONMENT", "production")
    resp_prod = client.get("/api/test")
    assert resp_prod.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"


def test_no_duplicate_security_headers_middleware():
    """Scenario 4: Verify only ONE SecurityHeadersMiddleware file exists across backend."""
    import pathlib

    backend_dir = pathlib.Path("backend")
    found_middleware_files = []
    for py_file in backend_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        if "class SecurityHeadersMiddleware" in text:
            found_middleware_files.append(py_file.as_posix())

    # Must find exactly one canonical file: backend/middleware/security_headers.py
    assert len(found_middleware_files) == 1
    assert found_middleware_files[0] == "backend/middleware/security_headers.py"
