"""Sprint C — Discovery & Operations tests.

Covers:
  1. Full-text search (SQLite LIKE fallback)
  2. Leaderboard API (all / weekly / monthly)
  3. Notifications (create, list, read, read-all)
  4. Celery Beat tasks (zombie cleanup, backup skips on non-Postgres)
  5. Flower / Beat config (celery_app beat_schedule present)
  6. Usage quota enforcement (paper monthly limit)
"""

import datetime
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Problem, Paper, Task, Notification
from backend.modules.auth.security.hashing import hash_password
from backend.repositories.task_repository import TaskRepository

# ── Shared helpers ────────────────────────────────────────────────────────────

ADMIN_EMAIL = "sprint_c_admin@example.com"
USER_EMAIL  = "sprint_c_user@example.com"
USER2_EMAIL = "sprint_c_user2@example.com"
TEST_PASS   = "SecurePass123!"


def _create_user(db: Session, email: str, is_admin: bool = False, points: int = 0) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        existing.points = points
        db.commit()
        return existing
    user = User(
        email=email,
        name=email.split("@")[0],
        hashed_password=hash_password(TEST_PASS),
        is_verified=True,
        is_email_verified=True,
        is_admin=is_admin,
        points=points,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ── Task 1: Full-text search ──────────────────────────────────────────────────

class TestSearch:
    def test_search_returns_papers(self, client: TestClient, db_session: Session):
        p = Paper(title="Attention Is All You Need Sprint C", abstract="transformer architecture", visibility="public")
        db_session.add(p)
        db_session.commit()

        r = client.get("/api/search?q=transformer+architecture")
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert "total" in body
        titles = [item["title"] for item in body["results"] if item["type"] == "paper"]
        assert any("Sprint C" in t for t in titles)

    def test_search_returns_problems(self, client: TestClient, db_session: Session):
        p = Problem(
            id="sc-search-001", slug="sc-search-001", title="Sprint C Search Problem",
            difficulty="Easy", category="Attention", description="multi-head self-attention mechanism",
        )
        db_session.add(p)
        db_session.commit()

        r = client.get("/api/search?q=multi-head+self-attention")
        assert r.status_code == 200
        ids = [item["id"] for item in r.json()["results"] if item["type"] == "problem"]
        assert "sc-search-001" in ids

    def test_search_excludes_retired_problems(self, client: TestClient, db_session: Session):
        p = Problem(
            id="sc-ret-search-001", slug="sc-ret-search-001", title="Retired Sprint C Problem",
            difficulty="Hard", category="Retired", description="do not show me retired stuff",
            is_retired=True,
        )
        db_session.add(p)
        db_session.commit()

        r = client.get("/api/search?q=do+not+show+me+retired")
        assert r.status_code == 200
        ids = [item["id"] for item in r.json()["results"] if item["type"] == "problem"]
        assert "sc-ret-search-001" not in ids

    def test_search_requires_min_query_length(self, client: TestClient):
        r = client.get("/api/search?q=a")
        assert r.status_code == 422  # Pydantic validation

    def test_search_filter_types_papers_only(self, client: TestClient, db_session: Session):
        r = client.get("/api/search?q=attention&types=papers")
        assert r.status_code == 200
        types = {item["type"] for item in r.json()["results"]}
        assert "problem" not in types

    def test_search_filter_types_problems_only(self, client: TestClient, db_session: Session):
        r = client.get("/api/search?q=attention&types=problems")
        assert r.status_code == 200
        types = {item["type"] for item in r.json()["results"]}
        assert "paper" not in types


# ── Task 2: Leaderboard ───────────────────────────────────────────────────────

class TestLeaderboard:
    def test_leaderboard_all_time(self, client: TestClient, db_session: Session):
        u = _create_user(db_session, USER_EMAIL, points=500)
        u.last_active = datetime.datetime.utcnow()
        db_session.commit()

        r = client.get("/api/leaderboard?period=all")
        assert r.status_code == 200
        body = r.json()
        assert body["period"] == "all"
        assert "leaders" in body
        assert "generated_at" in body
        assert isinstance(body["leaders"], list)

    def test_leaderboard_weekly(self, client: TestClient, db_session: Session):
        u = _create_user(db_session, USER_EMAIL, points=200)
        u.last_active = datetime.datetime.utcnow() - datetime.timedelta(days=3)
        db_session.commit()

        r = client.get("/api/leaderboard?period=weekly")
        assert r.status_code == 200
        assert r.json()["period"] == "weekly"

    def test_leaderboard_monthly(self, client: TestClient, db_session: Session):
        r = client.get("/api/leaderboard?period=monthly")
        assert r.status_code == 200
        assert r.json()["period"] == "monthly"

    def test_leaderboard_invalid_period(self, client: TestClient):
        r = client.get("/api/leaderboard?period=yearly")
        assert r.status_code == 400

    def test_leaderboard_rank_order(self, client: TestClient, db_session: Session):
        u1 = _create_user(db_session, USER_EMAIL, points=1000)
        u2 = _create_user(db_session, USER2_EMAIL, points=500)
        u1.last_active = datetime.datetime.utcnow()
        u2.last_active = datetime.datetime.utcnow()
        db_session.commit()

        r = client.get("/api/leaderboard?period=all")
        assert r.status_code == 200
        leaders = r.json()["leaders"]
        if len(leaders) >= 2:
            # Ranks should be descending by points
            for i in range(len(leaders) - 1):
                assert leaders[i]["points"] >= leaders[i + 1]["points"]

    def test_leaderboard_response_shape(self, client: TestClient, db_session: Session):
        u = _create_user(db_session, USER_EMAIL, points=100)
        u.last_active = datetime.datetime.utcnow()
        db_session.commit()

        r = client.get("/api/leaderboard?period=all&limit=1")
        assert r.status_code == 200
        leaders = r.json()["leaders"]
        if leaders:
            leader = leaders[0]
            assert "rank" in leader
            assert "points" in leader
            assert "xp_level" in leader
            assert "problems_solved" in leader


# ── Task 3: Notifications ─────────────────────────────────────────────────────

class TestNotifications:
    def _make_notif(self, db: Session, user_id: int, title: str = "Test Notif") -> Notification:
        n = Notification(user_id=user_id, type="paper.done", title=title, body="body text")
        db.add(n)
        db.commit()
        db.refresh(n)
        return n

    def test_list_requires_auth(self, client: TestClient):
        r = client.get("/api/notifications")
        assert r.status_code == 401

    def test_list_own_notifications(self, client: TestClient, db_session: Session):
        user = _create_user(db_session, USER_EMAIL)
        self._make_notif(db_session, user.id, "Paper Ready")
        token = _login(client, USER_EMAIL)

        r = client.get("/api/notifications", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "unread" in body
        assert "notifications" in body
        assert any(n["title"] == "Paper Ready" for n in body["notifications"])

    def test_mark_single_read(self, client: TestClient, db_session: Session):
        user = _create_user(db_session, USER_EMAIL)
        n = self._make_notif(db_session, user.id)
        token = _login(client, USER_EMAIL)

        r = client.post(
            f"/api/notifications/{n.id}/read",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["is_read"] is True

    def test_mark_other_user_notification_forbidden(self, client: TestClient, db_session: Session):
        u1 = _create_user(db_session, USER_EMAIL)
        u2 = _create_user(db_session, USER2_EMAIL)
        n = self._make_notif(db_session, u2.id)
        token = _login(client, USER_EMAIL)

        r = client.post(
            f"/api/notifications/{n.id}/read",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 403

    def test_mark_all_read(self, client: TestClient, db_session: Session):
        user = _create_user(db_session, USER_EMAIL)
        self._make_notif(db_session, user.id, "N1")
        self._make_notif(db_session, user.id, "N2")
        token = _login(client, USER_EMAIL)

        r = client.post("/api/notifications/read-all", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["marked_read"] >= 2

        # Confirm all are read
        r2 = client.get("/api/notifications?unread_only=true", headers={"Authorization": f"Bearer {token}"})
        assert r2.json()["total"] == 0

    def test_unread_filter(self, client: TestClient, db_session: Session):
        user = _create_user(db_session, USER_EMAIL)
        n1 = self._make_notif(db_session, user.id, "Unread")
        n2 = self._make_notif(db_session, user.id, "Read")
        n2.is_read = True
        db_session.commit()
        token = _login(client, USER_EMAIL)

        r = client.get(
            "/api/notifications?unread_only=true",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        titles = [n["title"] for n in r.json()["notifications"]]
        assert "Unread" in titles
        assert "Read" not in titles

    def test_nonexistent_notification_404(self, client: TestClient, db_session: Session):
        _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        r = client.post(
            "/api/notifications/999999/read",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404


# ── Task 4: Celery Beat scheduled tasks ──────────────────────────────────────

class TestScheduledTasks:
    def test_cleanup_zombie_tasks(self, db_session: Session):
        from backend.tasks.scheduled_tasks import _do_cleanup_zombie_tasks

        # Create a task stuck in 'running' with old created_at
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "zombie-paper")
        repo.set_running(task.id)

        # Backdate created_at to 3 hours ago
        t = db_session.query(Task).filter_by(id=task.id).first()
        t.created_at = datetime.datetime.utcnow() - datetime.timedelta(hours=3)
        t.updated_at = None
        db_session.commit()

        # Call inner function directly with the test session
        result = _do_cleanup_zombie_tasks(db_session)
        assert result.get("cleaned", 0) >= 1

        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.status == "failed"
        assert "Zombie" in (fresh.error or "")

    def test_cleanup_does_not_touch_recent_running_tasks(self, db_session: Session):
        from backend.tasks.scheduled_tasks import _do_cleanup_zombie_tasks

        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "recent-paper")
        repo.set_running(task.id)
        # created_at is very recent (default) → should NOT be cleaned

        _do_cleanup_zombie_tasks(db_session)

        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.status == "running"

    def test_daily_db_backup_skips_non_postgres(self):
        from backend.tasks.scheduled_tasks import _do_daily_db_backup
        result = _do_daily_db_backup()  # SQLite in tests → should skip
        assert result.get("skipped") is True

    def test_beat_schedule_configured(self):
        from backend.celery_app import celery_app
        schedule = celery_app.conf.beat_schedule
        assert "cleanup-zombie-tasks-hourly" in schedule
        assert "daily-db-backup" in schedule
        assert schedule["cleanup-zombie-tasks-hourly"]["task"] == "backend.tasks.scheduled_tasks.cleanup_zombie_tasks"
        assert schedule["daily-db-backup"]["task"] == "backend.tasks.scheduled_tasks.daily_db_backup"


# ── Task 5: Flower / docker-compose ──────────────────────────────────────────

class TestDockerComposeConfig:
    def test_flower_service_in_compose(self):
        import yaml, pathlib
        compose = yaml.safe_load(
            pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
        )
        services = compose.get("services", {})
        assert "flower" in services, "Flower service missing from docker-compose.yml"
        assert "beat" in services, "Beat service missing from docker-compose.yml"

    def test_flower_binds_localhost_only(self):
        import yaml, pathlib
        compose = yaml.safe_load(
            pathlib.Path("docker-compose.yml").read_text(encoding="utf-8")
        )
        flower_ports = compose["services"]["flower"].get("ports", [])
        # All port mappings must bind to 127.0.0.1 (not 0.0.0.0)
        for p in flower_ports:
            assert str(p).startswith("127.0.0.1"), \
                f"Flower port {p} must bind to 127.0.0.1 to avoid public exposure"


# ── Task 6: Usage quota enforcement ──────────────────────────────────────────

class TestPaperQuota:
    def test_upload_allowed_when_no_limit(self, client: TestClient, db_session: Session):
        """Default PAPER_MONTHLY_LIMIT=0 means unlimited — no 429."""
        import os
        os.environ["PAPER_MONTHLY_LIMIT"] = "0"
        import importlib, backend.routers.papers as pr
        importlib.reload(pr)
        # Just verify the check function doesn't raise with limit=0
        from backend.routers.papers import _check_paper_quota
        _check_paper_quota(db_session, user_id=99999)  # must not raise

    def test_quota_check_raises_when_exceeded(self, db_session: Session):
        import os
        os.environ["PAPER_MONTHLY_LIMIT"] = "2"
        import importlib, backend.routers.papers as pr
        importlib.reload(pr)
        from backend.routers.papers import _check_paper_quota
        from fastapi import HTTPException

        user = _create_user(db_session, USER_EMAIL)
        # Create 2 paper.codegen tasks this month (at the limit)
        repo = TaskRepository(db_session)
        for _ in range(2):
            t = repo.create("paper.codegen", user.id, "paper")
        db_session.commit()

        with pytest.raises(HTTPException) as exc_info:
            _check_paper_quota(db_session, user.id)
        assert exc_info.value.status_code == 429

    def test_quota_not_exceeded_below_limit(self, db_session: Session):
        import os
        os.environ["PAPER_MONTHLY_LIMIT"] = "5"
        import importlib, backend.routers.papers as pr
        importlib.reload(pr)
        from backend.routers.papers import _check_paper_quota

        user = _create_user(db_session, USER2_EMAIL)
        # Only 1 task, limit is 5 → should not raise
        repo = TaskRepository(db_session)
        repo.create("paper.codegen", user.id, "one-paper")
        db_session.commit()

        _check_paper_quota(db_session, user.id)  # must not raise

    def test_upload_rejects_when_over_quota(self, client: TestClient, db_session: Session):
        import os
        os.environ["PAPER_MONTHLY_LIMIT"] = "1"
        import importlib, backend.routers.papers as pr
        importlib.reload(pr)

        user = _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)

        # Pre-fill quota
        repo = TaskRepository(db_session)
        repo.create("paper.codegen", user.id, "existing-paper")
        db_session.commit()

        # Patch the module-level limit
        import backend.routers.papers_pipeline as papers_mod
        papers_mod._PAPER_MONTHLY_LIMIT = 1

        r = client.post(
            "/api/papers/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"terms_accepted": "true", "visibility": "public"},
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert r.status_code == 429
        assert "limit" in r.json()["detail"].lower()

        # Restore
        papers_mod._PAPER_MONTHLY_LIMIT = 0
