import pytest
from fastapi.testclient import TestClient
from backend.server import app

def test_security_headers_present():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "default-src 'self'" in response.headers.get("Content-Security-Policy", "")
    assert response.headers.get("Strict-Transport-Security") is None

def test_security_headers_hsts_on_https():
    client = TestClient(app)
    response = client.get("/api/health", headers={"x-forwarded-proto": "https"})
    assert response.status_code == 200
    assert response.headers.get("Strict-Transport-Security") == "max-age=31536000; includeSubDomains"

def test_security_headers_on_404():
    client = TestClient(app)
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert response.headers.get("X-Frame-Options") == "DENY"
