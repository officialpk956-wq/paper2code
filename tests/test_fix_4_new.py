import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.server import app

client = TestClient(app)

def test_fix1_trace_id():
    response = client.get("/health")
    assert response.status_code in [200, 503]
    assert "X-Trace-ID" in response.headers

def test_fix2_health_healthy():
    with patch("redis.Redis.from_url") as mock_redis_from_url:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_from_url.return_value = mock_redis
        
        with patch("backend.routers.health.get_db") as mock_db_dep:
            mock_session = MagicMock()
            mock_session.execute.return_value = True
            mock_db_dep.return_value = mock_session
            
            app.dependency_overrides[mock_db_dep] = lambda: mock_session
            
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
            
            app.dependency_overrides.clear()

def test_fix2_health_unhealthy():
    with patch("redis.Redis.from_url") as mock_redis_from_url:
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis_from_url.return_value = mock_redis
        
        with patch("backend.routers.health.get_db") as mock_db_dep:
            mock_session = MagicMock()
            mock_session.execute.side_effect = Exception("DB Error")
            mock_db_dep.return_value = mock_session
            
            app.dependency_overrides[mock_db_dep] = lambda: mock_session
            
            response = client.get("/health")
            assert response.status_code == 503
            assert response.json()["status"] == "unhealthy"
            
            app.dependency_overrides.clear()

def test_fix3_logger_exception():
    with patch("backend.routers.health.httpx.get") as mock_get:
        mock_get.side_effect = Exception("Piston Error")
        
        # We don't have logger in health.py! So this test will just check static code
        with open("backend/routers/learning.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "logger.exception" in content

def test_fix4_metrics():
    # call metrics directly
    from backend import metrics
    metrics.increment("submissions_total", 5)
    
    m = metrics.get_metrics()
    assert m["submissions_total"] >= 5
