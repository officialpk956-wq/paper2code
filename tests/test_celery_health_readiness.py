from unittest.mock import MagicMock

import pytest

from backend.celery_app import celery_app


@pytest.fixture
def broker(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(celery_app, "connection_for_read", lambda **kwargs: connection)
    monkeypatch.setattr(celery_app.control, "ping", lambda **kwargs: [])
    return connection


def test_anonymous_health_hides_fake_credentials(client, monkeypatch, broker):
    for name in ("REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
        monkeypatch.setenv(name, "redis://:FAKE_SECRET_NEGATIVE_CONTROL@fake-broker:6379/0")
    response = client.get("/api/health/celery")
    assert "://" not in response.text
    assert "FAKE_SECRET_NEGATIVE_CONTROL" not in response.text
    assert "fake-broker" not in response.text
    assert response.json() == {"ready": False, "broker_reachable": True, "worker_responding": False}
    assert response.status_code == 503


def test_unreachable_broker_is_not_ready(client, broker):
    broker.ensure_connection.side_effect = ConnectionError("redis://secret@host")
    response = client.get("/api/health/celery")
    assert response.status_code == 503
    assert response.json() == {
        "ready": False,
        "broker_reachable": False,
        "worker_responding": False,
    }


def test_worker_pong_is_required(client, broker, monkeypatch):
    monkeypatch.setattr(celery_app.control, "ping", lambda **kwargs: [{"worker": {"ok": "pong"}}])
    response = client.get("/api/health/celery")
    assert response.status_code == 200
    assert response.json() == {"ready": True, "broker_reachable": True, "worker_responding": True}


def test_details_require_auth(client):
    assert client.get("/api/health/celery/details").status_code == 401


def test_details_require_admin(auth_client):
    assert auth_client.get("/api/health/celery/details").status_code == 403


def test_admin_can_view_details(admin_client):
    response = admin_client.get("/api/health/celery/details")
    assert response.status_code == 200
    assert "celery_backend_url" in response.json()
