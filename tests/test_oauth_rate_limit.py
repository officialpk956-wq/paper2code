import pytest
from fastapi.testclient import TestClient
from backend.server import app
from backend.modules.security.rate_limit import _in_memory_store

def test_oauth_authorize_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    client = TestClient(app)
    _in_memory_store.clear()
    
    for _ in range(10):
        response = client.get("/api/auth/oauth/testprovider/authorize-url", headers={"X-Forwarded-For": "10.0.0.1"})
        assert response.status_code == 404
        
    response = client.get("/api/auth/oauth/testprovider/authorize-url", headers={"X-Forwarded-For": "10.0.0.1"})
    assert response.status_code == 429
    assert "Too many authorization requests" in response.text

def test_oauth_exchange_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    client = TestClient(app)
    _in_memory_store.clear()
    
    # Use different states so we don't hit the state token limit (which is 3, truncated to 12 chars)
    for i in range(5):
        response = client.post("/api/auth/oauth/testprovider/exchange", json={"code": "123", "state": f"{i}_valid_state"}, headers={"X-Forwarded-For": "10.0.0.2"})
        assert response.status_code == 404
        
    response = client.post("/api/auth/oauth/testprovider/exchange", json={"code": "123", "state": "99_valid_state"}, headers={"X-Forwarded-For": "10.0.0.2"})
    assert response.status_code == 429
    assert "Too many exchange requests" in response.text

def test_oauth_exchange_state_rate_limit(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    client = TestClient(app)
    _in_memory_store.clear()
    
    for i in range(3):
        response = client.post("/api/auth/oauth/testprovider/exchange", json={"code": "123", "state": "brute_state"}, headers={"X-Forwarded-For": f"10.0.1.{i}"})
        assert response.status_code == 404
        
    response = client.post("/api/auth/oauth/testprovider/exchange", json={"code": "123", "state": "brute_state"}, headers={"X-Forwarded-For": "10.0.1.99"})
    assert response.status_code == 429
    assert "Too many attempts with this state token" in response.text

def test_oauth_normal_flow_not_blocked_when_disabled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    client = TestClient(app)
    _in_memory_store.clear()
    
    for _ in range(15):
        response = client.get("/api/auth/oauth/testprovider/authorize-url", headers={"X-Forwarded-For": "10.0.0.5"})
        assert response.status_code == 404
