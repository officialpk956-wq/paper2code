"""
tests/test_storage_infra.py

Storage & Infrastructure tests (~30 tests):

  TestPresignedUploadURL     — GET /api/papers/upload-url
  TestConfirmUpload          — POST /api/papers/confirm-upload
  TestPresignedDownload      — GET /api/papers/{id}/download
  TestStorageQuota           — User.storage_bytes_used enforcement
  TestPrometheusMetrics      — /metrics endpoint
  TestSlackAlerting          — 5xx alert logic (unit)
  TestStorageService         — generate_presigned_upload_url, get_object_size helpers
  TestNginxConfig            — nginx.conf syntax and required directives
"""

import datetime
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import Base, User, Paper, Task
from backend.modules.auth.security.hashing import hash_password


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset in-memory rate-limit counters before each test so upload tests don't hit 10/hour cap."""
    try:
        from backend.server import limiter
        limiter.reset()  # Limiter.reset() clears the MemoryStorage backend
    except Exception:
        try:
            from backend.server import limiter
            limiter._storage.reset()
        except Exception:
            pass
    yield

# ---------------------------------------------------------------------------
# Helpers (reuse conftest fixtures)
# ---------------------------------------------------------------------------

_PASS = "StorageTest999!"


def _seed_user(db: Session, email: str, is_admin: bool = False,
               storage_bytes_used: int = 0) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        return existing
    u = User(
        email=email,
        name=email.split("@")[0],
        hashed_password=hash_password(_PASS),
        is_verified=True,
        is_email_verified=True,
        is_admin=is_admin,
        storage_bytes_used=storage_bytes_used,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": _PASS})
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_paper(db: Session, title: str, r2_key: str = None,
                file_size_bytes: int = None, uploaded_by: int = None) -> Paper:
    p = Paper(
        title=title,
        r2_key=r2_key,
        file_size_bytes=file_size_bytes,
        uploaded_by=uploaded_by,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ---------------------------------------------------------------------------
# TestPresignedUploadURL — GET /api/papers/upload-url
# ---------------------------------------------------------------------------

class TestPresignedUploadURL:
    def test_01_r2_not_configured_returns_503(self, client, db_session):
        user = _seed_user(db_session, email="pu01@si.com")
        token = _login(client, user.email)
        with patch("backend.services.storage_service.R2_AVAILABLE", False):
            r = client.get(
                "/api/papers/upload-url?filename=test.pdf",
                headers=_auth(token),
            )
        assert r.status_code == 503, r.json()

    def test_02_non_pdf_rejected(self, client, db_session):
        user = _seed_user(db_session, email="pu02@si.com")
        token = _login(client, user.email)
        mock_boto = MagicMock()
        mock_boto.generate_presigned_url.return_value = "https://r2.example.com/presigned"
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto):
            r = client.get(
                "/api/papers/upload-url?filename=malware.exe",
                headers=_auth(token),
            )
        assert r.status_code == 400

    def test_03_r2_configured_returns_url_and_key(self, client, db_session):
        user = _seed_user(db_session, email="pu03@si.com")
        token = _login(client, user.email)
        mock_boto = MagicMock()
        mock_boto.generate_presigned_url.return_value = "https://r2.example.com/presigned-put"
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto):
            r = client.get(
                "/api/papers/upload-url?filename=paper.pdf",
                headers=_auth(token),
            )
        assert r.status_code == 200
        data = r.json()
        assert "url" in data
        assert "key" in data
        assert data["key"].startswith("papers/")
        assert data["expires_in"] > 0

    def test_04_unauthenticated_returns_401(self, client, db_session):
        r = client.get("/api/papers/upload-url?filename=paper.pdf")
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# TestConfirmUpload — POST /api/papers/confirm-upload
# ---------------------------------------------------------------------------

class TestConfirmUpload:
    def test_05_confirm_upload_creates_task(self, client, db_session):
        user = _seed_user(db_session, email="cu05@si.com")
        token = _login(client, user.email)
        with patch("backend.routers.papers_pipeline.generate_code_from_pdf_task") as mock_task:
            mock_task.delay = MagicMock()
            r = client.post(
                "/api/papers/confirm-upload",
                json={
                    "key": "papers/abc123_test.pdf",
                    "paper_name": "Attention Is All You Need",
                    "visibility": "public",
                    "terms_accepted": True,
                    "file_size_bytes": 1024 * 1024,  # 1 MB
                },
                headers=_auth(token),
            )
        assert r.status_code == 200
        data = r.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert "poll_url" in data

    def test_06_no_terms_rejected(self, client, db_session):
        user = _seed_user(db_session, email="cu06@si.com")
        token = _login(client, user.email)
        r = client.post(
            "/api/papers/confirm-upload",
            json={
                "key": "papers/abc123.pdf",
                "paper_name": "Test",
                "terms_accepted": False,
            },
            headers=_auth(token),
        )
        assert r.status_code == 400

    def test_07_invalid_key_rejected(self, client, db_session):
        user = _seed_user(db_session, email="cu07@si.com")
        token = _login(client, user.email)
        r = client.post(
            "/api/papers/confirm-upload",
            json={
                "key": "../../../etc/passwd",
                "paper_name": "Hack",
                "terms_accepted": True,
            },
            headers=_auth(token),
        )
        assert r.status_code == 400

    def test_08_storage_bytes_incremented(self, client, db_session):
        user = _seed_user(db_session, email="cu08@si.com")
        initial_bytes = user.storage_bytes_used
        token = _login(client, user.email)
        file_size = 2 * 1024 * 1024  # 2 MB
        with patch("backend.routers.papers_pipeline.generate_code_from_pdf_task") as mock_task:
            mock_task.delay = MagicMock()
            client.post(
                "/api/papers/confirm-upload",
                json={
                    "key": "papers/cu08_test.pdf",
                    "paper_name": "CU08 Paper",
                    "terms_accepted": True,
                    "file_size_bytes": file_size,
                },
                headers=_auth(token),
            )
        db_session.refresh(user)
        assert user.storage_bytes_used == initial_bytes + file_size

    def test_09_invalid_visibility_rejected(self, client, db_session):
        user = _seed_user(db_session, email="cu09@si.com")
        token = _login(client, user.email)
        r = client.post(
            "/api/papers/confirm-upload",
            json={
                "key": "papers/cu09.pdf",
                "paper_name": "Test",
                "visibility": "secret",
                "terms_accepted": True,
            },
            headers=_auth(token),
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# TestPresignedDownload — GET /api/papers/{id}/download
# ---------------------------------------------------------------------------

class TestPresignedDownload:
    def test_10_paper_without_r2_key_returns_404(self, client, db_session):
        paper = _seed_paper(db_session, title="No R2 Key SI10")
        r = client.get(f"/api/papers/{paper.id}/download")
        assert r.status_code == 404

    def test_11_public_paper_redirects_to_presigned_url(self, client, db_session):
        paper = _seed_paper(db_session, title="Public PDF SI11", r2_key="papers/test11.pdf")
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client") as mock_r2:
            mock_r2.return_value.generate_presigned_url.return_value = "https://r2.example.com/signed"
            r = client.get(f"/api/papers/{paper.id}/download", follow_redirects=False)
        assert r.status_code == 302
        assert "r2.example.com" in r.headers.get("location", "")

    def test_12_private_paper_requires_ownership(self, client, db_session):
        owner = _seed_user(db_session, email="owner12@si.com")
        other = _seed_user(db_session, email="other12@si.com")
        paper = _seed_paper(
            db_session, title="Private PDF SI12",
            r2_key="papers/priv12.pdf",
            uploaded_by=owner.id,
        )
        db_session.query(Paper).filter_by(id=paper.id).update({"visibility": "private"})
        db_session.commit()
        token = _login(client, other.email)
        r = client.get(f"/api/papers/{paper.id}/download", headers=_auth(token))
        assert r.status_code == 403

    def test_13_owner_can_download_private_paper(self, client, db_session):
        owner = _seed_user(db_session, email="owner13@si.com")
        paper = _seed_paper(
            db_session, title="Private PDF SI13",
            r2_key="papers/priv13.pdf",
            uploaded_by=owner.id,
        )
        db_session.query(Paper).filter_by(id=paper.id).update({"visibility": "private"})
        db_session.commit()
        token = _login(client, owner.email)
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client") as mock_r2:
            mock_r2.return_value.generate_presigned_url.return_value = "https://r2.example.com/signed"
            r = client.get(f"/api/papers/{paper.id}/download",
                           headers=_auth(token), follow_redirects=False)
        assert r.status_code == 302

    def test_14_nonexistent_paper_returns_404(self, client, db_session):
        r = client.get("/api/papers/99999/download")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# TestStorageQuota — User.storage_bytes_used enforcement
# ---------------------------------------------------------------------------

class TestStorageQuota:
    def test_15_upload_url_blocked_when_over_quota(self, client, db_session):
        over_quota_bytes = 501 * 1024 * 1024
        user = _seed_user(db_session, email="sq15@si.com",
                          storage_bytes_used=over_quota_bytes)
        token = _login(client, user.email)
        mock_boto = MagicMock()
        mock_boto.generate_presigned_url.return_value = "https://r2.example.com/put"
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto), \
             patch("backend.routers.papers_pipeline._STORAGE_QUOTA_BYTES", 500 * 1024 * 1024):
            r = client.get(
                "/api/papers/upload-url?filename=paper.pdf",
                headers=_auth(token),
            )
        assert r.status_code == 429

    def test_16_confirm_blocked_when_over_quota(self, client, db_session):
        user = _seed_user(db_session, email="sq16@si.com",
                          storage_bytes_used=490 * 1024 * 1024)
        token = _login(client, user.email)
        with patch("backend.routers.papers_pipeline._STORAGE_QUOTA_BYTES", 500 * 1024 * 1024):
            r = client.post(
                "/api/papers/confirm-upload",
                json={
                    "key": "papers/sq16.pdf",
                    "paper_name": "Over Quota",
                    "terms_accepted": True,
                    "file_size_bytes": 20 * 1024 * 1024,
                },
                headers=_auth(token),
            )
        assert r.status_code == 429

    def test_17_quota_zero_means_unlimited(self, client, db_session):
        user = _seed_user(db_session, email="sq17@si.com",
                          storage_bytes_used=999 * 1024 * 1024)
        token = _login(client, user.email)
        mock_boto = MagicMock()
        mock_boto.generate_presigned_url.return_value = "https://r2.example.com/put"
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto), \
             patch("backend.routers.papers_pipeline._STORAGE_QUOTA_BYTES", 0):
            r = client.get(
                "/api/papers/upload-url?filename=paper.pdf",
                headers=_auth(token),
            )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# TestPrometheusMetrics — /metrics endpoint
# ---------------------------------------------------------------------------

class TestPrometheusMetrics:
    def test_18_metrics_endpoint_exists(self, client, db_session):
        r = client.get("/metrics")
        # Returns 200 if prometheus_client installed, else 503
        assert r.status_code in (200, 503)

    def test_19_metrics_200_when_prometheus_available(self, client, db_session):
        try:
            import prometheus_client
            r = client.get("/metrics")
            assert r.status_code == 200
            body = r.text
            assert "http_requests_total" in body or "python_gc" in body
        except ImportError:
            pytest.skip("prometheus_client not installed")

    def test_20_metrics_503_when_prometheus_unavailable(self, client, db_session):
        with patch("backend.middleware.metrics._PROMETHEUS_AVAILABLE", False):
            r = client.get("/metrics")
        assert r.status_code == 503, r.json()


# ---------------------------------------------------------------------------
# TestSlackAlerting — unit tests for dedup + dispatch logic
# ---------------------------------------------------------------------------

class TestSlackAlerting:
    def test_21_no_op_when_webhook_not_set(self):
        from backend.middleware.alerting import alert_5xx
        with patch("backend.middleware.alerting.SLACK_WEBHOOK_URL", ""), \
             patch("backend.middleware.alerting._send_slack") as mock_send:
            alert_5xx("GET", "/api/fail", 500)
        mock_send.assert_not_called()

    def test_22_sends_when_webhook_set(self):
        from backend.middleware.alerting import alert_5xx, _last_alert
        _last_alert.clear()
        with patch("backend.middleware.alerting.SLACK_WEBHOOK_URL", "https://hooks.slack.com/test"), \
             patch("backend.middleware.alerting._send_slack") as mock_send:
            alert_5xx("POST", "/api/error", 502)
        mock_send.assert_called_once()

    def test_23_dedup_within_cooldown(self):
        from backend.middleware.alerting import alert_5xx, _last_alert
        _last_alert.clear()
        calls = []
        with patch("backend.middleware.alerting.SLACK_WEBHOOK_URL", "https://hooks.slack.com/test"), \
             patch("backend.middleware.alerting._send_slack", side_effect=calls.append):
            alert_5xx("GET", "/api/dedup", 500)
            alert_5xx("GET", "/api/dedup", 500)  # should be deduped
        assert len(calls) == 1

    def test_24_different_routes_both_alert(self):
        from backend.middleware.alerting import alert_5xx, _last_alert
        _last_alert.clear()
        calls = []
        with patch("backend.middleware.alerting.SLACK_WEBHOOK_URL", "https://hooks.slack.com/test"), \
             patch("backend.middleware.alerting._send_slack", side_effect=calls.append):
            alert_5xx("GET", "/api/route-a", 500)
            alert_5xx("GET", "/api/route-b", 500)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# TestStorageService — unit tests for new storage_service helpers
# ---------------------------------------------------------------------------

class TestStorageService:
    def test_25_generate_presigned_upload_raises_without_r2(self):
        from backend.services.storage_service import generate_presigned_upload_url
        with patch("backend.services.storage_service.R2_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="R2 not configured"):
                generate_presigned_upload_url("test.pdf")

    def test_26_generate_presigned_upload_returns_correct_shape(self):
        from backend.services.storage_service import generate_presigned_upload_url
        mock_boto = MagicMock()
        mock_boto.generate_presigned_url.return_value = "https://r2.example.com/upload"
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto):
            result = generate_presigned_upload_url("my paper.pdf", "application/pdf", 600)
        assert result["url"] == "https://r2.example.com/upload"
        assert result["key"].startswith("papers/")
        assert "my_paper.pdf" in result["key"]
        assert result["expires_in"] == 600

    def test_27_get_object_size_returns_zero_without_r2(self):
        from backend.services.storage_service import get_object_size
        with patch("backend.services.storage_service.R2_AVAILABLE", False):
            assert get_object_size("papers/test.pdf") == 0

    def test_28_get_object_size_returns_content_length(self):
        from backend.services.storage_service import get_object_size
        mock_boto = MagicMock()
        mock_boto.head_object.return_value = {"ContentLength": 1048576}
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto):
            size = get_object_size("papers/test.pdf")
        assert size == 1048576

    def test_29_get_object_size_returns_zero_on_error(self):
        from backend.services.storage_service import get_object_size
        mock_boto = MagicMock()
        mock_boto.head_object.side_effect = Exception("Not found")
        with patch("backend.services.storage_service.R2_AVAILABLE", True), \
             patch("backend.services.storage_service._r2_client", return_value=mock_boto):
            size = get_object_size("papers/missing.pdf")
        assert size == 0


# ---------------------------------------------------------------------------
# TestNginxConfig — nginx.conf structural checks
# ---------------------------------------------------------------------------

class TestNginxConfig:
    def _read_nginx_conf(self) -> str:
        import pathlib
        p = pathlib.Path(__file__).parent.parent / "nginx" / "nginx.conf"
        return p.read_text(encoding="utf-8")

    def test_30_nginx_conf_exists(self):
        import pathlib
        p = pathlib.Path(__file__).parent.parent / "nginx" / "nginx.conf"
        assert p.exists(), "nginx/nginx.conf must exist"

    def test_31_nginx_conf_has_rate_limiting(self):
        conf = self._read_nginx_conf()
        assert "limit_req_zone" in conf
        assert "limit_req" in conf

    def test_32_nginx_conf_has_gzip(self):
        conf = self._read_nginx_conf()
        assert "gzip on" in conf
        assert "application/json" in conf

    def test_33_nginx_conf_metrics_restricted(self):
        conf = self._read_nginx_conf()
        assert "/metrics" in conf
        assert "deny all" in conf

    def test_34_nginx_conf_has_tls_section(self):
        conf = self._read_nginx_conf()
        assert "ssl_certificate" in conf
        assert "TLSv1" in conf
