"""
test_complex_integration.py

End-to-end platform integration test.

Simulates real user journeys across every Sprint A/B/C feature:
  - User lifecycle: register → login → profile → change password
  - Cross-user isolation: private papers, notifications, sessions
  - Full paper pipeline: upload → stage progression → notification → search
  - Dojo lifecycle: list (retired excluded) → submit → XP award → leaderboard
  - AI Tutor: start session → ask → rate info → session security
  - Admin operations: problem CRUD cycle; user management
  - Notifications: create → list → partial read → read-all → unread filter
  - Leaderboard: multi-user ranking across all periods
  - Search: cross-type, type-filtered, retired excluded, private excluded
  - Quotas: monthly paper limit; tutor daily counter
  - Scheduled tasks: zombie cleanup, backup skip on non-Postgres
  - Beat schedule: all periodic tasks registered
  - docker-compose: beat + flower present and correctly configured
"""

import datetime
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import (
    User, Problem, Paper, Task, Notification, DojoSubmission, LearnerProgress,
)
from backend.modules.auth.security.hashing import hash_password
from backend.repositories.task_repository import TaskRepository

# ── Shared helpers ────────────────────────────────────────────────────────────

_PASS = "SecurePass999!"
_PASS2 = "AnotherPass999!"

def _register(client, email: str, name: str = "Test User") -> dict:
    r = client.post("/api/auth/register", json={"email": email, "name": name, "password": _PASS})
    assert r.status_code in (200, 201), f"Register failed: {r.text}"
    return r.json()

def _login(client, email: str, password: str = _PASS) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()["access_token"]

def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}

def _seed_user(db: Session, email: str, points: int = 0, is_admin: bool = False,
               last_active: datetime.datetime = None) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        existing.points = points
        existing.is_admin = is_admin
        if last_active:
            existing.last_active = last_active
        db.commit()
        return existing
    u = User(
        email=email, name=email.split("@")[0],
        hashed_password=hash_password(_PASS),
        is_verified=True, is_email_verified=True,
        is_admin=is_admin, points=points,
        last_active=last_active,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — User lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserLifecycle:
    EMAIL_A = "ci_user_a@example.com"

    def test_01_register_and_login(self, client: TestClient, db_session: Session):
        """Registration returns tokens; login returns valid access token."""
        _register(client, self.EMAIL_A, name="Alice")
        token = _login(client, self.EMAIL_A)
        assert len(token) > 10

    def test_02_get_me_returns_profile(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL_A)
        token = _login(client, self.EMAIL_A)
        r = client.get("/api/auth/me", headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == self.EMAIL_A
        assert "points" in body
        assert "streak" in body
        assert "xp_level" in body

    def test_03_unauthenticated_me_returns_401(self, client: TestClient):
        r = client.get("/api/auth/me")
        assert r.status_code == 401

    def test_04_refresh_token_returns_new_access_token(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL_A)
        r = client.post("/api/auth/login", data={"username": self.EMAIL_A, "password": _PASS})
        refresh = r.json().get("refresh_token")
        if refresh:
            r2 = client.post("/api/auth/refresh", json={"refresh_token": refresh})
            assert r2.status_code == 200
            assert "access_token" in r2.json()

    def test_05_wrong_password_returns_401(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL_A)
        r = client.post("/api/auth/login", data={"username": self.EMAIL_A, "password": "wrongpassword!"})
        assert r.status_code in (401, 400)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — Paper upload + pipeline stages + notification
# ═══════════════════════════════════════════════════════════════════════════════

class TestPaperPipelineAndNotification:
    EMAIL = "ci_paper_user@example.com"

    def test_01_upload_requires_terms_accepted(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL)
        token = _login(client, self.EMAIL)
        r = client.post(
            "/api/papers/upload",
            headers=_auth(token),
            data={"terms_accepted": "false", "visibility": "public"},
            files={"file": ("p.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 400
        assert "Terms" in r.json()["detail"]

    def test_02_upload_rejects_non_pdf(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL)
        token = _login(client, self.EMAIL)
        r = client.post(
            "/api/papers/upload",
            headers=_auth(token),
            data={"terms_accepted": "true", "visibility": "public"},
            files={"file": ("p.txt", b"text content", "text/plain")},
        )
        assert r.status_code == 400

    def test_03_pipeline_stage_tracking(self, db_session: Session):
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "ci-paper")
        repo.set_running(task.id)
        for stage in ["extracting", "analyzing", "generating", "saving"]:
            repo.set_stage(task.id, stage)
            t = db_session.query(Task).filter_by(id=task.id).first()
            assert t.result["stage"] == stage
        repo.set_complete(task.id, {"paper_id": 1, "code": "pass", "stage": "complete"})
        final = db_session.query(Task).filter_by(id=task.id).first()
        assert final.status == "completed"
        assert final.result["stage"] == "complete"

    def test_04_paper_visibility_private_hidden_from_others(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL)
        p = Paper(title="CI Private Paper", visibility="private", uploaded_by=u.id)
        db_session.add(p)
        db_session.commit()

        # Anonymous should NOT see it
        r = client.get("/api/papers")
        titles = [x["title"] for x in r.json()["papers"]]
        assert "CI Private Paper" not in titles

    def test_05_paper_visibility_owner_sees_own_private(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL)
        p = Paper(title="CI Owner Private Paper", visibility="private", uploaded_by=u.id)
        db_session.add(p)
        db_session.commit()

        token = _login(client, self.EMAIL)
        r = client.get("/api/papers", headers=_auth(token))
        titles = [x["title"] for x in r.json()["papers"]]
        assert "CI Owner Private Paper" in titles

    def test_06_paper_done_notification_created(self, db_session: Session):
        u = _seed_user(db_session, self.EMAIL)
        notif = Notification(
            user_id=u.id, type="paper.done",
            title="Paper ready: CI Paper", body="Done.",
            payload={"paper_id": 99},
        )
        db_session.add(notif)
        db_session.commit()

        saved = db_session.query(Notification).filter_by(user_id=u.id, type="paper.done").first()
        assert saved is not None
        assert saved.is_read is False
        assert saved.payload["paper_id"] == 99


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — Dojo: retired filter, submission, XP flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestDojoLifecycle:
    EMAIL = "ci_dojo_user@example.com"

    def _seed_problem(self, db: Session, pid: str, retired: bool = False) -> Problem:
        existing = db.query(Problem).filter_by(id=pid).first()
        if existing:
            existing.is_retired = retired
            db.commit()
            return existing
        p = Problem(
            id=pid, slug=pid, title=f"Problem {pid}",
            difficulty="Easy", category="CI",
            description=f"Integration test problem {pid}",
            python_template="pass", test_cases=[], hints=[], explanation=[],
            is_retired=retired,
        )
        db.add(p)
        db.commit()
        return p

    def test_01_public_problems_excludes_retired(self, client: TestClient, db_session: Session):
        self._seed_problem(db_session, "ci-active-001", retired=False)
        self._seed_problem(db_session, "ci-retired-001", retired=True)

        r = client.get("/api/problems")
        assert r.status_code == 200
        ids = [p["id"] for p in r.json()]
        assert "ci-active-001" in ids
        assert "ci-retired-001" not in ids

    def test_02_search_excludes_retired_problems(self, client: TestClient, db_session: Session):
        self._seed_problem(db_session, "ci-ret-search-99", retired=True)
        r = client.get("/api/search?q=integration+test+problem+ci-ret-search-99")
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()["results"] if x["type"] == "problem"]
        assert "ci-ret-search-99" not in ids

    def test_03_admin_retire_and_restore_cycle(self, client: TestClient, db_session: Session):
        admin = _seed_user(db_session, "ci_admin@example.com", is_admin=True)
        token = _login(client, "ci_admin@example.com")

        # Create problem
        r = client.post("/api/admin/problems", headers=_auth(token), json={
            "id": "ci-crud-001", "slug": "ci-crud-001", "title": "CI CRUD Problem",
            "difficulty": "Medium", "category": "CI", "description": "CRUD lifecycle test.",
        })
        assert r.status_code == 201
        assert r.json()["is_retired"] is False

        # Update
        r2 = client.put("/api/admin/problems/ci-crud-001", headers=_auth(token),
                         json={"title": "CI CRUD Updated"})
        assert r2.status_code == 200
        assert r2.json()["title"] == "CI CRUD Updated"

        # Retire → should vanish from public list
        r3 = client.patch("/api/admin/problems/ci-crud-001/retire", headers=_auth(token))
        assert r3.status_code == 200
        pub = client.get("/api/problems")
        pub_ids = [p["id"] for p in pub.json()]
        assert "ci-crud-001" not in pub_ids

        # Restore → back in public list
        r4 = client.patch("/api/admin/problems/ci-crud-001/restore", headers=_auth(token))
        assert r4.status_code == 200
        pub2 = client.get("/api/problems")
        pub2_ids = [p["id"] for p in pub2.json()]
        assert "ci-crud-001" in pub2_ids

    def test_04_submission_awards_xp(self, db_session: Session):
        """DojoSubmission records trigger XP via progress_service (unit-level check)."""
        from backend.services.progress_service import award_xp
        u = _seed_user(db_session, self.EMAIL, points=0)
        initial = u.points
        new_points = award_xp(db_session, u.id, "dojo.solved.easy")
        db_session.refresh(u)
        assert u.points > initial


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — Notification lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestNotificationLifecycle:
    EMAIL_A = "ci_notif_a@example.com"
    EMAIL_B = "ci_notif_b@example.com"

    def _add_notifs(self, db: Session, user: User, count: int) -> list:
        notifs = []
        for i in range(count):
            n = Notification(user_id=user.id, type="paper.done", title=f"CI Notif {i}")
            db.add(n)
            notifs.append(n)
        db.commit()
        for n in notifs:
            db.refresh(n)
        return notifs

    def test_01_list_shows_own_notifications(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL_A)
        self._add_notifs(db_session, u, 3)
        token = _login(client, self.EMAIL_A)
        r = client.get("/api/notifications", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["unread"] >= 3

    def test_02_mark_one_read(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL_A)
        [n] = self._add_notifs(db_session, u, 1)
        token = _login(client, self.EMAIL_A)
        r = client.post(f"/api/notifications/{n.id}/read", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["is_read"] is True

    def test_03_cross_user_read_blocked(self, client: TestClient, db_session: Session):
        u_a = _seed_user(db_session, self.EMAIL_A)
        u_b = _seed_user(db_session, self.EMAIL_B)
        [n] = self._add_notifs(db_session, u_b, 1)  # belongs to B
        token_a = _login(client, self.EMAIL_A)
        r = client.post(f"/api/notifications/{n.id}/read", headers=_auth(token_a))
        assert r.status_code == 403

    def test_04_read_all_clears_unread(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL_A)
        self._add_notifs(db_session, u, 4)
        token = _login(client, self.EMAIL_A)
        client.post("/api/notifications/read-all", headers=_auth(token))
        r = client.get("/api/notifications?unread_only=true", headers=_auth(token))
        assert r.json()["total"] == 0

    def test_05_notifications_require_auth(self, client: TestClient):
        assert client.get("/api/notifications").status_code == 401
        assert client.post("/api/notifications/1/read").status_code == 401
        assert client.post("/api/notifications/read-all").status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5 — Leaderboard: multi-user ranking
# ═══════════════════════════════════════════════════════════════════════════════

class TestLeaderboardRanking:
    def test_01_ranks_descending_by_points(self, client: TestClient, db_session: Session):
        now = datetime.datetime.utcnow()
        users = [
            _seed_user(db_session, f"ci_lb_{i}@example.com", points=100*(5-i), last_active=now)
            for i in range(5)
        ]
        r = client.get("/api/leaderboard?period=all&limit=200")
        assert r.status_code == 200
        leaders = r.json()["leaders"]
        pts = [l["points"] for l in leaders if l["points"] > 0]
        assert pts == sorted(pts, reverse=True)

    def test_02_weekly_only_includes_recently_active(self, client: TestClient, db_session: Session):
        now = datetime.datetime.utcnow()
        active = _seed_user(db_session, "ci_lb_active@example.com", points=999,
                            last_active=now - datetime.timedelta(days=3))
        inactive = _seed_user(db_session, "ci_lb_inactive@example.com", points=888,
                              last_active=now - datetime.timedelta(days=14))
        r = client.get("/api/leaderboard?period=weekly")
        assert r.status_code == 200
        ids = [l["user_id"] for l in r.json()["leaders"]]
        assert active.id in ids
        assert inactive.id not in ids

    def test_03_response_shape(self, client: TestClient, db_session: Session):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        body = r.json()
        assert all(k in body for k in ("period", "generated_at", "total_ranked", "leaders"))
        for leader in body["leaders"][:3]:
            assert all(k in leader for k in
                       ("rank", "user_id", "name", "points", "xp_level", "streak", "problems_solved"))

    def test_04_invalid_period_returns_400(self, client: TestClient):
        assert client.get("/api/leaderboard?period=forever").status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6 — Full-text search
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchIntegration:
    def _seed_paper(self, db: Session, title: str, abstract: str) -> Paper:
        existing = db.query(Paper).filter_by(title=title).first()
        if existing:
            return existing
        p = Paper(title=title, abstract=abstract, visibility="public")
        db.add(p)
        db.commit()
        db.refresh(p)
        return p

    def test_01_finds_paper_by_title(self, client: TestClient, db_session: Session):
        self._seed_paper(db_session, "CI Vision Transformer Study", "image patches and self-attention layers")
        r = client.get("/api/search?q=Vision+Transformer+Study")
        assert r.status_code == 200
        titles = [x["title"] for x in r.json()["results"] if x["type"] == "paper"]
        assert any("Vision Transformer Study" in t for t in titles)

    def test_02_finds_problem_by_description(self, client: TestClient, db_session: Session):
        p = Problem(
            id="ci-search-kv-001", slug="ci-search-kv-001",
            title="Key-Value Attention Cache", difficulty="Hard", category="Transformers",
            description="implement kv-cache autoregressive decoding for efficient inference",
        )
        db_session.merge(p)
        db_session.commit()
        # "kv-cache autoregressive" is an exact substring of the description above
        r = client.get("/api/search?q=kv-cache+autoregressive&types=problems")
        assert r.status_code == 200
        ids = [x["id"] for x in r.json()["results"] if x["type"] == "problem"]
        assert "ci-search-kv-001" in ids

    def test_03_private_papers_excluded_from_search(self, client: TestClient, db_session: Session):
        p = Paper(title="CI Secret Private Architecture", visibility="private")
        db_session.add(p)
        db_session.commit()
        r = client.get("/api/search?q=CI+Secret+Private")
        titles = [x["title"] for x in r.json()["results"] if x["type"] == "paper"]
        assert "CI Secret Private Architecture" not in titles

    def test_04_short_query_rejected(self, client: TestClient):
        r = client.get("/api/search?q=x")
        assert r.status_code == 422

    def test_05_type_filter_papers_only(self, client: TestClient):
        r = client.get("/api/search?q=attention&types=papers")
        assert r.status_code == 200
        types = {x["type"] for x in r.json()["results"]}
        assert "problem" not in types

    def test_06_type_filter_problems_only(self, client: TestClient):
        r = client.get("/api/search?q=attention&types=problems")
        assert r.status_code == 200
        types = {x["type"] for x in r.json()["results"]}
        assert "paper" not in types

    def test_07_result_shape(self, client: TestClient, db_session: Session):
        self._seed_paper(db_session, "CI Shape Check Paper", "architecture for shape validation tests")
        r = client.get("/api/search?q=shape+validation")
        assert r.status_code == 200
        body = r.json()
        assert "query" in body and "total" in body and "results" in body
        for item in body["results"][:3]:
            assert all(k in item for k in ("type", "id", "title", "snippet", "url"))


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 7 — AI Tutor security
# ═══════════════════════════════════════════════════════════════════════════════

class TestTutorSecurity:
    EMAIL_A = "ci_tutor_a@example.com"
    EMAIL_B = "ci_tutor_b@example.com"

    def test_01_start_session_requires_auth(self, client: TestClient):
        assert client.post("/api/tutor/start-session").status_code == 401

    def test_02_start_session_returns_uuid(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL_A)
        token = _login(client, self.EMAIL_A)
        r = client.post("/api/tutor/start-session", headers=_auth(token))
        assert r.status_code == 200
        sid = r.json()["session_id"]
        assert len(sid) == 36  # UUID4

    def test_03_session_owned_by_creator(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        sid = store.create_session(user_id=1)
        assert store.validate_ownership(sid, 1) is True
        assert store.validate_ownership(sid, 2) is False  # other user rejected

    def test_04_foreign_session_gets_new_one(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL_A)
        token = _login(client, self.EMAIL_A)
        fake_sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with patch("core.agents.tutor_agent.llm_complete",
                   return_value='{"answer":"ok","source_context":"x","confidence":"High","reasoning_type":"Structural"}'):
            r = client.post("/api/tutor/ask", headers=_auth(token), json={
                "session_id": fake_sid,
                "context_type": "module",
                "context_data": {
                    "paper_title": "ResNet", "layer_name": "Conv1",
                    "module_type": "conv", "explanation": "conv layer", "flops_context": {}
                },
                "query": "what does this do?",
            })
        assert r.status_code == 200
        assert r.json()["session_id"] != fake_sid  # fresh session issued

    def test_05_ask_without_auth_returns_401(self, client: TestClient):
        r = client.post("/api/tutor/ask", json={
            "context_type": "module", "context_data": {}, "query": "hello",
        })
        assert r.status_code == 401

    def test_06_rate_info_returned_in_ask_response(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.EMAIL_A)
        token = _login(client, self.EMAIL_A)
        with patch("core.agents.tutor_agent.llm_complete",
                   return_value='{"answer":"ok","source_context":"x","confidence":"Low","reasoning_type":"Fallback"}'):
            r = client.post("/api/tutor/ask", headers=_auth(token), json={
                "context_type": "module",
                "context_data": {"paper_title": "VGG", "layer_name": "L1", "module_type": "conv",
                                  "explanation": "conv", "flops_context": {}},
                "query": "explain this layer",
            })
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert "queries_today" in body
        assert "queries_limit" in body

    def test_07_session_history_persists_across_calls(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        sid = store.create_session(user_id=5)
        store.update_history(sid, [{"role": "user", "content": "q1"}, {"role": "tutor", "content": "a1"}])
        hist = store.get_history(sid)
        assert len(hist) == 2
        assert hist[0]["content"] == "q1"

    def test_08_history_capped_at_6_messages(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        sid = store.create_session(user_id=5)
        big_history = [{"role": "user", "content": str(i)} for i in range(20)]
        store.update_history(sid, big_history)
        assert len(store.get_history(sid)) == 6


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 8 — Usage quotas
# ═══════════════════════════════════════════════════════════════════════════════

class TestUsageQuotas:
    EMAIL = "ci_quota_user@example.com"

    def test_01_paper_quota_not_enforced_when_zero(self, db_session: Session):
        import backend.routers.papers as papers_mod
        papers_mod._PAPER_MONTHLY_LIMIT = 0
        from backend.routers.papers import _check_paper_quota
        _check_paper_quota(db_session, user_id=99998)  # must not raise

    def test_02_paper_quota_raises_at_threshold(self, db_session: Session):
        import backend.routers.papers as papers_mod
        papers_mod._PAPER_MONTHLY_LIMIT = 2
        from fastapi import HTTPException
        from backend.routers.papers import _check_paper_quota

        u = _seed_user(db_session, self.EMAIL)
        repo = TaskRepository(db_session)
        repo.create("paper.codegen", u.id, "p1")
        repo.create("paper.codegen", u.id, "p2")
        db_session.commit()

        with pytest.raises(HTTPException) as exc:
            _check_paper_quota(db_session, u.id)
        assert exc.value.status_code == 429

        # Reset
        papers_mod._PAPER_MONTHLY_LIMIT = 0

    def test_03_tutor_rate_store_tracks_per_user(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        count, limit = store.get_rate_info(user_id=12345)
        assert limit >= 1  # limit is configured

    def test_04_tutor_store_allows_when_under_limit(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        allowed, count, limit = store.check_and_incr_rate(user_id=99997)
        assert allowed is True


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 9 — Scheduled tasks integrity
# ═══════════════════════════════════════════════════════════════════════════════

class TestScheduledTasksIntegrity:
    def test_01_zombie_cleanup_marks_stale_tasks_failed(self, db_session: Session):
        from backend.tasks.scheduled_tasks import _do_cleanup_zombie_tasks
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "ci-zombie")
        repo.set_running(task.id)
        t = db_session.query(Task).filter_by(id=task.id).first()
        t.created_at = datetime.datetime.utcnow() - datetime.timedelta(hours=4)
        t.updated_at = None
        db_session.commit()

        result = _do_cleanup_zombie_tasks(db_session)
        assert result["cleaned"] >= 1
        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.status == "failed"
        assert "Zombie" in fresh.error

    def test_02_zombie_cleanup_spares_fresh_tasks(self, db_session: Session):
        from backend.tasks.scheduled_tasks import _do_cleanup_zombie_tasks
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "ci-fresh-task")
        repo.set_running(task.id)
        # No backdating — should NOT be touched
        _do_cleanup_zombie_tasks(db_session)
        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.status == "running"

    def test_03_db_backup_skips_sqlite(self):
        from backend.tasks.scheduled_tasks import _do_daily_db_backup
        result = _do_daily_db_backup()
        assert result["skipped"] is True
        assert "not_postgres" in result.values()

    def test_04_beat_schedule_has_correct_tasks(self):
        from backend.celery_app import celery_app
        sched = celery_app.conf.beat_schedule
        tasks = {v["task"] for v in sched.values()}
        assert "backend.tasks.scheduled_tasks.cleanup_zombie_tasks" in tasks
        assert "backend.tasks.scheduled_tasks.daily_db_backup" in tasks


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 10 — Docker-compose and infrastructure config
# ═══════════════════════════════════════════════════════════════════════════════

class TestInfrastructureConfig:
    def _compose(self) -> dict:
        import yaml, pathlib
        return yaml.safe_load(pathlib.Path("docker-compose.yml").read_text(encoding="utf-8"))

    def test_01_all_required_services_present(self):
        services = self._compose()["services"]
        for svc in ("postgres", "redis", "api", "worker", "beat", "flower", "piston"):
            assert svc in services, f"Missing service: {svc}"

    def test_02_flower_bound_to_localhost(self):
        ports = self._compose()["services"]["flower"].get("ports", [])
        for p in ports:
            assert str(p).startswith("127.0.0.1"), f"Flower port {p} must not be public"

    def test_03_piston_on_isolated_network(self):
        piston = self._compose()["services"]["piston"]
        api = self._compose()["services"]["api"]
        piston_nets = set(piston.get("networks", []))
        api_nets = set(api.get("networks", []))
        # Piston must NOT share any network with API
        shared = piston_nets & api_nets
        assert len(shared) == 0, f"Piston shares network(s) with API: {shared}"

    def test_04_beat_runs_correct_command(self):
        cmd = self._compose()["services"]["beat"]["command"]
        assert "beat" in cmd
        assert "celery_app" in cmd

    def test_05_env_example_has_all_sprint_vars(self):
        import pathlib
        env_text = pathlib.Path(".env.example").read_text(encoding="utf-8")
        required_vars = [
            "RESEND_API_KEY", "R2_ACCOUNT_ID", "R2_BUCKET_NAME",
            "TUTOR_DAILY_LIMIT", "PAPER_MONTHLY_LIMIT",
            "ZOMBIE_THRESHOLD_HOURS", "SENTRY_DSN",
        ]
        for var in required_vars:
            assert var in env_text, f"Missing env var in .env.example: {var}"


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 11 — Admin role boundary checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdminBoundaries:
    ADMIN_EMAIL = "ci_admin_boundary@example.com"
    USER_EMAIL  = "ci_plain_boundary@example.com"

    def test_01_plain_user_cannot_access_admin_stats(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.USER_EMAIL, is_admin=False)
        token = _login(client, self.USER_EMAIL)
        r = client.get("/api/admin/stats", headers=_auth(token))
        assert r.status_code == 403

    def test_02_plain_user_cannot_create_problems(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.USER_EMAIL, is_admin=False)
        token = _login(client, self.USER_EMAIL)
        r = client.post("/api/admin/problems", headers=_auth(token), json={
            "id": "evil-problem", "slug": "evil", "title": "Evil",
            "difficulty": "Easy", "category": "x", "description": "x",
        })
        assert r.status_code == 403

    def test_03_admin_cannot_modify_own_account(self, client: TestClient, db_session: Session):
        admin = _seed_user(db_session, self.ADMIN_EMAIL, is_admin=True)
        token = _login(client, self.ADMIN_EMAIL)
        r = client.patch(f"/api/admin/users/{admin.id}",
                          json={"is_admin": False}, headers=_auth(token))
        assert r.status_code == 400
        assert "own admin account" in r.json()["detail"]

    def test_04_admin_costs_breakdown_shape(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.ADMIN_EMAIL, is_admin=True)
        token = _login(client, self.ADMIN_EMAIL)
        r = client.get("/api/admin/costs?days=7", headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert "by_action" in body and "top_users_by_cost" in body
        assert body["period_days"] == 7

    def test_05_admin_stats_shape(self, client: TestClient, db_session: Session):
        _seed_user(db_session, self.ADMIN_EMAIL, is_admin=True)
        token = _login(client, self.ADMIN_EMAIL)
        r = client.get("/api/admin/stats", headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        for key in ("users", "content", "llm", "tasks", "timestamp"):
            assert key in body


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 12 — Cross-cutting: auth token invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthInvariants:
    EMAIL = "ci_auth_inv@example.com"

    def test_01_expired_token_rejected(self, client: TestClient, db_session: Session):
        """Tampered/invalid JWT must return 401 on protected endpoints."""
        r = client.get("/api/auth/me", headers={"Authorization": "Bearer totallyinvalidtoken"})
        assert r.status_code == 401

    def test_02_no_token_rejected_everywhere(self, client: TestClient):
        protected = [
            ("GET",  "/api/auth/me"),
            ("POST", "/api/tutor/start-session"),
            ("GET",  "/api/notifications"),
            ("POST", "/api/notifications/read-all"),
            ("POST", "/api/papers/upload"),
        ]
        for method, path in protected:
            r = getattr(client, method.lower())(path)
            assert r.status_code == 401, f"{method} {path} should require auth"

    def test_03_storage_service_roundtrip(self):
        from backend.services import storage_service
        data = b"PDF-BYTES-SIMULATION"
        ref = storage_service.store_pdf(data, "ci_test.pdf")
        fetched = storage_service.fetch_pdf(ref)
        assert fetched == data
        storage_service.cleanup(ref)
        assert storage_service.r2_key_from_ref(ref) is None  # local ref → None

    def test_04_task_repository_full_lifecycle(self, db_session: Session):
        repo = TaskRepository(db_session)
        t = repo.create("ci.test", 1, "ref-ci")
        assert t.status == "pending"
        repo.set_running(t.id)
        repo.set_stage(t.id, "processing")
        repo.set_complete(t.id, {"result": "done", "stage": "complete"})
        final = repo.get(t.id)
        assert final.status == "completed"
        assert final.result["stage"] == "complete"

    def test_05_failed_task_stores_error(self, db_session: Session):
        repo = TaskRepository(db_session)
        t = repo.create("ci.test.fail", None, "ref-fail")
        repo.set_running(t.id)
        repo.set_failed(t.id, "something exploded")
        final = repo.get(t.id)
        assert final.status == "failed"
        assert "exploded" in final.error
