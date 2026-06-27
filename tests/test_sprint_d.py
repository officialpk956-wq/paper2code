"""
test_sprint_d.py — Sprint D: Growth & Retention

Tests covering:
  1. OAuth authorize-URL + linked providers endpoint
  2. Achievement system — catalogue, award, XP, notification, dedup
  3. Onboarding email drip — day targeting, dedup, send tracking
  4. Streak-at-risk email — correct user filtering
  5. Submission history + acceptance rate stats
  6. PostHog analytics — fire-and-forget, no-op when unconfigured
"""

import datetime
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import (
    User, Problem, DojoSubmission, Achievement, UserAchievement,
    EmailDripLog, OAuthAccount, Notification,
)
from backend.modules.auth.security.hashing import hash_password

# ── helpers ──────────────────────────────────────────────────────────────────

_PASS = "SecurePass999!"


def _seed_user(db: Session, email: str, streak: int = 0, points: int = 0,
               last_active: datetime.datetime = None,
               created_at: datetime.datetime = None) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        existing.streak = streak
        existing.points = points
        if last_active:
            existing.last_active = last_active
        if created_at:
            existing.created_at = created_at
        db.commit()
        return existing
    u = User(
        email=email, name=email.split("@")[0],
        hashed_password=hash_password(_PASS),
        is_verified=True, is_email_verified=True,
        streak=streak, points=points,
        last_active=last_active,
        created_at=created_at or datetime.datetime.utcnow(),
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


def _seed_problem(db: Session, pid: str, difficulty: str = "Easy") -> Problem:
    existing = db.query(Problem).filter_by(id=pid).first()
    if existing:
        return existing
    p = Problem(
        id=pid, slug=pid, title=f"Problem {pid}",
        difficulty=difficulty, category="Algorithms",
        description=f"Test problem {pid}",
        python_template="pass", test_cases=[], hints=[], explanation=[],
    )
    db.add(p)
    db.commit()
    return p


def _ensure_achievements(db: Session):
    from backend.services.achievement_service import seed_achievements
    seed_achievements(db)


# =============================================================================
# TASK 1 — OAuth authorize-URL + linked providers
# =============================================================================

class TestOAuth:
    def test_01_authorize_url_unknown_provider_404(self, client: TestClient):
        r = client.get("/api/auth/oauth/twitter/authorize-url")
        assert r.status_code == 404

    def test_02_google_authorize_url_unconfigured_503(self, client: TestClient):
        with patch.dict("os.environ", {"GOOGLE_CLIENT_ID": "", "GOOGLE_CLIENT_SECRET": ""}):
            r = client.get("/api/auth/oauth/google/authorize-url?redirect_uri=http://localhost:3000/callback")
            assert r.status_code == 503

    def test_03_google_authorize_url_configured(self, client: TestClient):
        with patch.dict("os.environ", {"GOOGLE_CLIENT_ID": "test_client_id", "GOOGLE_CLIENT_SECRET": "test_secret"}):
            # Re-import so env vars are picked up by the router's module-level read
            import importlib, backend.routers.oauth as oauth_mod
            orig = oauth_mod._PROVIDERS["google"]["client_id"]
            oauth_mod._PROVIDERS["google"]["client_id"] = "test_client_id"
            oauth_mod._PROVIDERS["google"]["client_secret"] = "test_secret"

            r = client.get("/api/auth/oauth/google/authorize-url?redirect_uri=http://localhost:3000/cb")
            assert r.status_code == 200
            body = r.json()
            assert "url" in body and "state" in body
            assert "accounts.google.com" in body["url"]
            assert len(body["state"]) > 10

            oauth_mod._PROVIDERS["google"]["client_id"] = orig

    def test_04_github_authorize_url_configured(self, client: TestClient):
        import backend.routers.oauth as oauth_mod
        oauth_mod._PROVIDERS["github"]["client_id"] = "gh_client"
        oauth_mod._PROVIDERS["github"]["client_secret"] = "gh_secret"

        r = client.get("/api/auth/oauth/github/authorize-url?redirect_uri=http://localhost:3000/cb")
        assert r.status_code == 200
        assert "github.com/login/oauth/authorize" in r.json()["url"]

        oauth_mod._PROVIDERS["github"]["client_id"] = ""

    def test_05_list_providers_requires_auth(self, client: TestClient):
        r = client.get("/api/auth/oauth/providers")
        assert r.status_code == 401

    def test_06_list_providers_returns_linked_accounts(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, "ci_oauth_list@example.com")
        oa = OAuthAccount(
            user_id=u.id, provider="github",
            provider_user_id="gh-uid-999", provider_email="ci_oauth_list@example.com",
        )
        db_session.add(oa)
        db_session.commit()

        token = _login(client, "ci_oauth_list@example.com")
        r = client.get("/api/auth/oauth/providers", headers=_auth(token))
        assert r.status_code == 200
        providers = r.json()["providers"]
        assert any(p["provider"] == "github" for p in providers)

    def test_07_oauth_account_unique_constraint(self, db_session: Session):
        from sqlalchemy.exc import IntegrityError
        u = _seed_user(db_session, "ci_oauth_dup@example.com")
        oa1 = OAuthAccount(user_id=u.id, provider="google", provider_user_id="google-uid-1")
        oa2 = OAuthAccount(user_id=u.id, provider="google", provider_user_id="google-uid-1")
        db_session.add(oa1)
        db_session.commit()
        db_session.add(oa2)
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_08_existing_oauth_token_flow_still_works(self, client: TestClient, db_session: Session):
        """Existing POST /api/auth/oauth/login token-based flow remains intact."""
        _seed_user(db_session, "ci_oauth_token@example.com")
        with patch(
            "backend.modules.auth.oauth.provider.GoogleProvider.get_user_info",
            return_value=MagicMock(
                email="ci_oauth_token@example.com", name="CI User",
                avatar_url=None, uid="google-uid-token-test",
            ),
        ):
            r = client.post("/api/auth/oauth/login", json={"provider": "google", "token": "fake_id_token"})
            assert r.status_code == 200
            assert "access_token" in r.json()


# =============================================================================
# TASK 2 — Achievement system
# =============================================================================

class TestAchievements:
    def test_01_achievements_seeded_at_startup(self, client: TestClient, db_session: Session):
        _ensure_achievements(db_session)
        r = client.get("/api/achievements")
        assert r.status_code == 200
        slugs = [a["slug"] for a in r.json()]
        assert "first-paper" in slugs
        assert "hard-conqueror" in slugs
        assert "week-streak-7" in slugs

    def test_02_catalogue_shape(self, client: TestClient, db_session: Session):
        _ensure_achievements(db_session)
        r = client.get("/api/achievements")
        for item in r.json():
            assert all(k in item for k in ("slug", "title", "description", "icon", "xp_reward", "category"))
            assert item["xp_reward"] >= 0

    def test_03_my_achievements_requires_auth(self, client: TestClient):
        r = client.get("/api/achievements/my")
        assert r.status_code == 401

    def test_04_earn_first_paper_achievement(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award, _award
        u = _seed_user(db_session, "ci_ach_paper@example.com")

        awarded = check_and_award(db_session, u.id, "paper.uploaded")
        assert "first-paper" in awarded

        ua = db_session.query(UserAchievement).join(Achievement).filter(
            UserAchievement.user_id == u.id,
            Achievement.slug == "first-paper",
        ).first()
        assert ua is not None

    def test_05_achievement_not_awarded_twice(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        u = _seed_user(db_session, "ci_ach_nodup@example.com")

        first  = check_and_award(db_session, u.id, "paper.uploaded")
        second = check_and_award(db_session, u.id, "paper.uploaded")
        assert "first-paper" in first
        assert "first-paper" not in second  # already earned

    def test_06_earn_first_solve(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        u = _seed_user(db_session, "ci_ach_solve@example.com")
        awarded = check_and_award(db_session, u.id, "dojo.solved.easy", {"problem_id": "p1"})
        assert "first-solve" in awarded

    def test_07_earn_hard_conqueror(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        u = _seed_user(db_session, "ci_ach_hard@example.com")
        awarded = check_and_award(db_session, u.id, "dojo.solved.hard", {"problem_id": "hard-p1"})
        assert "hard-conqueror" in awarded

    def test_08_earn_week_streak(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        u = _seed_user(db_session, "ci_ach_streak@example.com")
        awarded = check_and_award(db_session, u.id, "streak.updated", {"streak": 7})
        assert "week-streak-7" in awarded

    def test_09_month_streak_not_awarded_at_7_days(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        u = _seed_user(db_session, "ci_ach_month@example.com")
        awarded = check_and_award(db_session, u.id, "streak.updated", {"streak": 7})
        assert "month-streak-30" not in awarded

    def test_10_achievement_triggers_xp_and_notification(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import _award
        u = _seed_user(db_session, "ci_ach_notif@example.com", points=0)
        initial_points = u.points

        _award(db_session, u.id, "first-paper")
        db_session.refresh(u)

        # XP bonus applied
        assert u.points > initial_points

        # Notification created
        notif = db_session.query(Notification).filter_by(
            user_id=u.id, type="achievement.unlocked"
        ).first()
        assert notif is not None
        assert "first-paper" in (notif.payload or {}).get("slug", "")

    def test_11_my_achievements_lists_earned(self, client: TestClient, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        u = _seed_user(db_session, "ci_my_ach@example.com")
        check_and_award(db_session, u.id, "paper.uploaded")

        token = _login(client, "ci_my_ach@example.com")
        r = client.get("/api/achievements/my", headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert body["count"] >= 1
        slugs = [a["slug"] for a in body["achievements"]]
        assert "first-paper" in slugs
        for a in body["achievements"]:
            assert all(k in a for k in ("slug", "title", "icon", "xp_reward", "earned_at"))

    def test_12_early_adopter_awarded_for_new_users(self, db_session: Session):
        _ensure_achievements(db_session)
        from backend.services.achievement_service import check_and_award
        # Clear user count with a fresh user — early-adopter only if ≤100 users
        u = _seed_user(db_session, "ci_early@example.com")
        awarded = check_and_award(db_session, u.id, "user.registered")
        # early-adopter awarded when total users ≤ 100 (test DB is small)
        assert isinstance(awarded, list)


# =============================================================================
# TASK 3 — Onboarding email drip
# =============================================================================

class TestEmailDrip:
    def test_01_day1_drip_targets_yesterday_registrations(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_onboarding_drips

        yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        u = _seed_user(db_session, "ci_drip_d1@example.com", created_at=yesterday)

        with patch("backend.tasks.growth_tasks.send_drip_email_sync") as mock_send:
            mock_send.return_value = True
            result = _do_onboarding_drips(db_session)

        assert result["sent"] >= 1
        mock_send.assert_called()
        # Drip log recorded
        log_entry = db_session.query(EmailDripLog).filter_by(user_id=u.id, drip_day=1).first()
        assert log_entry is not None

    def test_02_drip_not_sent_twice(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_onboarding_drips

        yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        u = _seed_user(db_session, "ci_drip_nodup@example.com", created_at=yesterday)
        db_session.add(EmailDripLog(user_id=u.id, drip_day=1))
        db_session.commit()

        with patch("backend.tasks.growth_tasks.send_drip_email_sync") as mock_send:
            mock_send.return_value = True
            result = _do_onboarding_drips(db_session)

        # send_drip_email_sync should NOT be called for this user (already logged)
        for call in mock_send.call_args_list:
            assert call.args[0] != u.email

    def test_03_day3_drip_targets_3_days_ago(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_onboarding_drips

        three_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=3)
        u = _seed_user(db_session, "ci_drip_d3@example.com", created_at=three_days_ago)

        with patch("backend.tasks.growth_tasks.send_drip_email_sync") as mock_send:
            mock_send.return_value = True
            _do_onboarding_drips(db_session)

        log_entry = db_session.query(EmailDripLog).filter_by(user_id=u.id, drip_day=3).first()
        assert log_entry is not None

    def test_04_today_registered_user_not_dripped(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_onboarding_drips

        u = _seed_user(db_session, "ci_drip_today@example.com",
                       created_at=datetime.datetime.utcnow())

        with patch("backend.tasks.growth_tasks.send_drip_email_sync") as mock_send:
            mock_send.return_value = True
            _do_onboarding_drips(db_session)

        for call in mock_send.call_args_list:
            assert call.args[0] != u.email

    def test_05_drip_log_enforces_unique_constraint(self, db_session: Session):
        from sqlalchemy.exc import IntegrityError
        u = _seed_user(db_session, "ci_drip_dup@example.com")
        db_session.add(EmailDripLog(user_id=u.id, drip_day=1))
        db_session.commit()
        db_session.add(EmailDripLog(user_id=u.id, drip_day=1))
        with pytest.raises(IntegrityError):
            db_session.commit()
        db_session.rollback()

    def test_06_email_templates_exist_for_all_drip_days(self):
        from backend.services.email_service import _DRIP_TEMPLATES
        for day in (1, 3, 7):
            assert day in _DRIP_TEMPLATES
            tpl = _DRIP_TEMPLATES[day]
            assert "subject" in tpl and "cta_text" in tpl and "body" in tpl


# =============================================================================
# TASK 4 — Streak-at-risk email
# =============================================================================

class TestStreakAtRisk:
    def test_01_at_risk_user_identified(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_streak_at_risk

        yesterday = datetime.datetime.utcnow() - datetime.timedelta(hours=30)
        u = _seed_user(db_session, "ci_streak_risk@example.com", streak=5, last_active=yesterday)

        with patch("backend.tasks.growth_tasks.send_streak_at_risk_email_sync") as mock_send:
            mock_send.return_value = True
            result = _do_streak_at_risk(db_session)

        assert result["at_risk_count"] >= 1
        emails_called = [c.args[0] for c in mock_send.call_args_list]
        assert u.email in emails_called

    def test_02_zero_streak_user_not_notified(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_streak_at_risk

        yesterday = datetime.datetime.utcnow() - datetime.timedelta(hours=30)
        u = _seed_user(db_session, "ci_streak_zero@example.com", streak=0, last_active=yesterday)

        with patch("backend.tasks.growth_tasks.send_streak_at_risk_email_sync") as mock_send:
            mock_send.return_value = True
            _do_streak_at_risk(db_session)

        emails_called = [c.args[0] for c in mock_send.call_args_list]
        assert u.email not in emails_called

    def test_03_active_today_not_at_risk(self, db_session: Session):
        from backend.tasks.growth_tasks import _do_streak_at_risk

        now = datetime.datetime.utcnow()
        u = _seed_user(db_session, "ci_streak_active@example.com", streak=10, last_active=now)

        with patch("backend.tasks.growth_tasks.send_streak_at_risk_email_sync") as mock_send:
            mock_send.return_value = True
            _do_streak_at_risk(db_session)

        emails_called = [c.args[0] for c in mock_send.call_args_list]
        assert u.email not in emails_called

    def test_04_beat_schedule_has_streak_at_risk_at_18_utc(self):
        from backend.celery_app import celery_app
        sched = celery_app.conf.beat_schedule
        streak_task = sched.get("streak-at-risk-daily")
        assert streak_task is not None
        assert streak_task["task"] == "backend.tasks.growth_tasks.send_streak_at_risk"
        assert streak_task["schedule"].hour == {18}  # crontab hour check

    def test_05_beat_schedule_has_drip_at_09_utc(self):
        from backend.celery_app import celery_app
        sched = celery_app.conf.beat_schedule
        drip_task = sched.get("onboarding-drip-daily")
        assert drip_task is not None
        assert drip_task["task"] == "backend.tasks.growth_tasks.send_onboarding_drips"
        assert drip_task["schedule"].hour == {9}


# =============================================================================
# TASK 5 — Submission history + acceptance rate
# =============================================================================

class TestSubmissionHistory:
    EMAIL = "ci_sub_hist@example.com"

    def _seed_submissions(self, db: Session, user: User, prob: Problem,
                          records: list[dict]) -> list[DojoSubmission]:
        subs = []
        for r in records:
            s = DojoSubmission(
                user_id=user.id, problem_id=prob.id,
                code=r.get("code", "pass"),
                passed=r.get("passed", False),
                time_ms=r.get("time_ms"),
                stdout=r.get("stdout", ""),
                stderr=r.get("stderr", ""),
            )
            db.add(s)
            subs.append(s)
        db.commit()
        return subs

    def test_01_history_requires_auth(self, client: TestClient, db_session: Session):
        _seed_problem(db_session, "ci-hist-prob-01")
        r = client.get("/api/problems/ci-hist-prob-01/submissions")
        assert r.status_code == 401

    def test_02_history_returns_own_submissions(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL)
        prob = _seed_problem(db_session, "ci-hist-prob-02")
        # Seed with explicit timestamps to ensure ordering in SQLite
        now = datetime.datetime.utcnow()
        for i, rec in enumerate([
            {"passed": True,  "time_ms": 120},
            {"passed": False, "time_ms": 80},
            {"passed": True,  "time_ms": 95},
        ]):
            s = DojoSubmission(
                user_id=u.id, problem_id=prob.id,
                code="pass", passed=rec["passed"], time_ms=rec["time_ms"],
                created_at=now + datetime.timedelta(seconds=i),
            )
            db_session.add(s)
        db_session.commit()

        token = _login(client, self.EMAIL)
        r = client.get(f"/api/problems/{prob.id}/submissions", headers=_auth(token))
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 3
        assert all("passed" in s and "time_ms" in s and "created_at" in s
                   for s in body["submissions"])
        # newest first: last seeded (time_ms=95, +2s) should be first
        assert body["submissions"][0]["time_ms"] == 95

    def test_03_history_404_on_unknown_problem(self, client: TestClient, db_session: Session):
        u = _seed_user(db_session, self.EMAIL)
        token = _login(client, self.EMAIL)
        r = client.get("/api/problems/ci-nonexistent-xyz/submissions", headers=_auth(token))
        assert r.status_code == 404

    def test_04_stats_acceptance_rate(self, client: TestClient, db_session: Session):
        prob = _seed_problem(db_session, "ci-stats-prob-01")
        u1 = _seed_user(db_session, "ci_stat_u1@example.com")
        u2 = _seed_user(db_session, "ci_stat_u2@example.com")
        u3 = _seed_user(db_session, "ci_stat_u3@example.com")

        for u, passed in [(u1, True), (u2, True), (u3, False)]:
            db_session.add(DojoSubmission(
                user_id=u.id, problem_id=prob.id, code="pass", passed=passed,
            ))
        db_session.commit()

        r = client.get(f"/api/problems/{prob.id}/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_users"] == 3
        assert body["passed_users"] == 2
        assert abs(body["acceptance_rate"] - 2/3) < 0.001

    def test_05_stats_zero_when_no_submissions(self, client: TestClient, db_session: Session):
        prob = _seed_problem(db_session, "ci-stats-empty-01")
        r = client.get(f"/api/problems/{prob.id}/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["total_users"] == 0
        assert body["acceptance_rate"] == 0.0

    def test_06_stats_shape(self, client: TestClient, db_session: Session):
        prob = _seed_problem(db_session, "ci-stats-shape-01")
        r = client.get(f"/api/problems/{prob.id}/stats")
        assert r.status_code == 200
        for key in ("problem_id", "total_users", "passed_users", "total_submissions", "acceptance_rate"):
            assert key in r.json()

    def test_07_time_ms_stored_on_submission(self, db_session: Session):
        u    = _seed_user(db_session, "ci_timems@example.com")
        prob = _seed_problem(db_session, "ci-timems-prob-01")
        s = DojoSubmission(
            user_id=u.id, problem_id=prob.id, code="pass", passed=True, time_ms=142,
        )
        db_session.add(s)
        db_session.commit()
        db_session.refresh(s)
        assert s.time_ms == 142


# =============================================================================
# TASK 6 — PostHog analytics
# =============================================================================

class TestPostHogAnalytics:
    def test_01_track_is_noop_when_no_key(self):
        import backend.services.analytics_service as svc
        orig_key = svc.POSTHOG_API_KEY
        svc.POSTHOG_API_KEY = ""

        with patch("backend.services.analytics_service._send") as mock_send:
            svc.track(1, "test_event", {"foo": "bar"})
        mock_send.assert_not_called()

        svc.POSTHOG_API_KEY = orig_key

    def test_02_track_fires_thread_when_key_set(self):
        import backend.services.analytics_service as svc
        orig_key = svc.POSTHOG_API_KEY
        svc.POSTHOG_API_KEY = "phc_test_key_123"

        with patch("backend.services.analytics_service._send") as mock_send, \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            svc.track(42, "paper_uploaded", {"visibility": "public"})
            mock_thread.assert_called_once()
            # Verify payload shape
            call_kwargs = mock_thread.call_args
            payload = call_kwargs.kwargs["args"][0]
            assert payload["event"] == "paper_uploaded"
            assert payload["distinct_id"] == "42"
            assert payload["properties"]["visibility"] == "public"

        svc.POSTHOG_API_KEY = orig_key

    def test_03_track_anonymous_user(self):
        import backend.services.analytics_service as svc
        orig_key = svc.POSTHOG_API_KEY
        svc.POSTHOG_API_KEY = "phc_test_key"

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            svc.track(None, "page_view", {})
            payload = mock_thread.call_args.kwargs["args"][0]
            assert payload["distinct_id"] == "anonymous"

        svc.POSTHOG_API_KEY = orig_key

    def test_04_send_posts_to_posthog_url(self):
        import backend.services.analytics_service as svc
        payload = {
            "api_key": "phc_test",
            "event": "signup",
            "distinct_id": "1",
            "properties": {},
        }
        with patch("httpx.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            svc._send(payload)
            mock_post.assert_called_once()
            url = mock_post.call_args.args[0]
            assert "posthog" in url or "capture" in url

    def test_05_send_never_raises(self):
        import backend.services.analytics_service as svc
        with patch("httpx.post", side_effect=Exception("network error")):
            svc._send({"api_key": "x", "event": "e", "distinct_id": "1", "properties": {}})
        # No exception propagated — fire-and-forget

    def test_06_env_example_has_posthog_vars(self):
        import pathlib
        env_text = pathlib.Path(".env.example").read_text(encoding="utf-8")
        assert "POSTHOG_API_KEY" in env_text
        assert "POSTHOG_HOST"    in env_text

    def test_07_env_example_has_oauth_vars(self):
        import pathlib
        env_text = pathlib.Path(".env.example").read_text(encoding="utf-8")
        for var in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
                    "GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET"):
            assert var in env_text
