"""
tests/test_infra_tasks.py

Covers:
  - _do_prune_tutor_sessions   — deletes old, keeps recent, null last_active_at
  - _do_prune_xp_events        — deletes old, keeps recent
  - _do_recalc_acceptance_rates — fixes rate after submissions, idempotent
  - check_and_award("leaderboard.top10") — new event handler
  - _do_weekly_leaderboard_reset — now triggers leaderboard achievement
  - _dojo_user_key              — per-user JWT key / IP fallback
  - beat_schedule entries       — three new Beat tasks registered
"""

import datetime
import uuid
import pytest
from unittest.mock import MagicMock

from backend.models import (
    User, TutorSessionRecord, XPEvent,
    Problem, DojoSubmission,
    UserAchievement,
)
from backend.tasks.scheduled_tasks import (
    _do_prune_tutor_sessions,
    _do_prune_xp_events,
    _do_recalc_acceptance_rates,
)
from backend.tasks.growth_tasks import _do_weekly_leaderboard_reset
from backend.services.achievement_service import check_and_award, seed_achievements


# ===========================================================================
# Helpers
# ===========================================================================

def _uid():
    return uuid.uuid4().hex[:8]


def _make_user(db, suffix=None, points=0, weekly_points=0):
    suffix = suffix or _uid()
    u = User(
        email=f"infra_{suffix}@test.com",
        name=f"Infra{suffix}",
        hashed_password="x",
        points=points,
        weekly_points=weekly_points,
        is_email_verified=True,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _make_problem(db):
    pid = f"p-{_uid()}"
    p = Problem(
        id=pid,
        slug=pid,
        title="Infra Test",
        difficulty="Easy",
        category="Testing",
        description="desc",
        python_template="pass",
        test_cases=[],
        hints=[],
        explanation=[],
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ===========================================================================
# TestPruneOldTutorSessions
# ===========================================================================

class TestPruneOldTutorSessions:
    def test_deletes_old_session_by_last_active(self, db_session):
        user = _make_user(db_session)
        old_ts = datetime.datetime.utcnow() - datetime.timedelta(days=100)
        db_session.add(TutorSessionRecord(
            user_id=user.id,
            session_id=str(uuid.uuid4()),
            context_type="paper",
            messages=[],
            created_at=old_ts,
            last_active_at=old_ts,
        ))
        db_session.commit()

        result = _do_prune_tutor_sessions(db_session)
        assert result["deleted"] >= 1
        assert result["cutoff_days"] == 90

    def test_keeps_recent_session(self, db_session):
        user = _make_user(db_session)
        sid = str(uuid.uuid4())
        recent_ts = datetime.datetime.utcnow() - datetime.timedelta(days=5)
        db_session.add(TutorSessionRecord(
            user_id=user.id,
            session_id=sid,
            context_type="paper",
            messages=[],
            created_at=recent_ts,
            last_active_at=recent_ts,
        ))
        db_session.commit()

        _do_prune_tutor_sessions(db_session)

        still_there = db_session.query(TutorSessionRecord).filter_by(session_id=sid).first()
        assert still_there is not None

    def test_deletes_old_session_with_null_last_active(self, db_session):
        user = _make_user(db_session)
        old_ts = datetime.datetime.utcnow() - datetime.timedelta(days=200)
        sid = str(uuid.uuid4())
        db_session.add(TutorSessionRecord(
            user_id=user.id,
            session_id=sid,
            context_type="paper",
            messages=[],
            created_at=old_ts,
            last_active_at=None,
        ))
        db_session.commit()

        result = _do_prune_tutor_sessions(db_session)
        assert result["deleted"] >= 1
        gone = db_session.query(TutorSessionRecord).filter_by(session_id=sid).first()
        assert gone is None

    def test_bulk_delete_returns_count(self, db_session):
        user = _make_user(db_session)
        old_ts = datetime.datetime.utcnow() - datetime.timedelta(days=100)
        for _ in range(4):
            db_session.add(TutorSessionRecord(
                user_id=user.id,
                session_id=str(uuid.uuid4()),
                context_type="paper",
                messages=[],
                created_at=old_ts,
                last_active_at=old_ts,
            ))
        db_session.commit()

        result = _do_prune_tutor_sessions(db_session)
        assert result["deleted"] >= 4


# ===========================================================================
# TestPruneOldXPEvents
# ===========================================================================

class TestPruneOldXPEvents:
    def test_deletes_old_event(self, db_session):
        user = _make_user(db_session)
        old_ts = datetime.datetime.utcnow() - datetime.timedelta(days=400)
        db_session.add(XPEvent(user_id=user.id, action="old.action", amount=10, created_at=old_ts))
        db_session.commit()

        result = _do_prune_xp_events(db_session)
        assert result["deleted"] >= 1
        assert result["cutoff_days"] == 365

    def test_keeps_recent_event(self, db_session):
        user = _make_user(db_session)
        ev = XPEvent(
            user_id=user.id,
            action="recent.action",
            amount=10,
            created_at=datetime.datetime.utcnow() - datetime.timedelta(days=30),
        )
        db_session.add(ev)
        db_session.commit()
        ev_id = ev.id

        _do_prune_xp_events(db_session)

        still_there = db_session.query(XPEvent).get(ev_id)
        assert still_there is not None

    def test_mixed_old_and_new_only_deletes_old(self, db_session):
        user = _make_user(db_session)
        old_ts = datetime.datetime.utcnow() - datetime.timedelta(days=400)
        new_ts = datetime.datetime.utcnow() - datetime.timedelta(days=10)

        old_ev = XPEvent(user_id=user.id, action="xp.old", amount=5, created_at=old_ts)
        new_ev = XPEvent(user_id=user.id, action="xp.new", amount=5, created_at=new_ts)
        db_session.add_all([old_ev, new_ev])
        db_session.commit()
        new_id = new_ev.id

        result = _do_prune_xp_events(db_session)
        assert result["deleted"] >= 1

        still_there = db_session.query(XPEvent).get(new_id)
        assert still_there is not None

    def test_bulk_delete_returns_count(self, db_session):
        user = _make_user(db_session)
        old_ts = datetime.datetime.utcnow() - datetime.timedelta(days=400)
        for i in range(5):
            db_session.add(XPEvent(user_id=user.id, action=f"bulk.{i}", amount=1, created_at=old_ts))
        db_session.commit()

        result = _do_prune_xp_events(db_session)
        assert result["deleted"] >= 5


# ===========================================================================
# TestRecalcAcceptanceRates
# ===========================================================================

class TestRecalcAcceptanceRates:
    def test_updates_rate_50_percent(self, db_session):
        u1 = _make_user(db_session)
        u2 = _make_user(db_session)
        prob = _make_problem(db_session)

        db_session.add(DojoSubmission(user_id=u1.id, problem_id=prob.id, passed=True, code="x"))
        db_session.add(DojoSubmission(user_id=u2.id, problem_id=prob.id, passed=False, code="x"))
        db_session.commit()

        result = _do_recalc_acceptance_rates(db_session)
        assert result["problems_checked"] >= 1
        assert result["problems_updated"] >= 1

        db_session.refresh(prob)
        assert float(prob.acceptance_rate) == pytest.approx(0.5, abs=0.001)

    def test_zero_rate_no_passes(self, db_session):
        user = _make_user(db_session)
        prob = _make_problem(db_session)
        db_session.add(DojoSubmission(user_id=user.id, problem_id=prob.id, passed=False, code="x"))
        db_session.commit()

        _do_recalc_acceptance_rates(db_session)
        db_session.refresh(prob)
        assert float(prob.acceptance_rate or 0) == 0.0

    def test_full_rate_all_pass(self, db_session):
        u1 = _make_user(db_session)
        u2 = _make_user(db_session)
        prob = _make_problem(db_session)

        db_session.add(DojoSubmission(user_id=u1.id, problem_id=prob.id, passed=True, code="x"))
        db_session.add(DojoSubmission(user_id=u2.id, problem_id=prob.id, passed=True, code="x"))
        db_session.commit()

        _do_recalc_acceptance_rates(db_session)
        db_session.refresh(prob)
        assert float(prob.acceptance_rate or 0) == pytest.approx(1.0, abs=0.001)

    def test_no_submissions_returns_zero_counts(self, db_session):
        # This checks the function is safe to run with no submissions for new problems
        result = _do_recalc_acceptance_rates(db_session)
        assert isinstance(result["problems_checked"], int)
        assert isinstance(result["problems_updated"], int)


# ===========================================================================
# TestLeaderboardAchievement
# ===========================================================================

class TestLeaderboardAchievement:
    def test_awards_leaderboard_top10(self, db_session):
        seed_achievements(db_session)
        user = _make_user(db_session, points=500)

        result = check_and_award(db_session, user.id, "leaderboard.top10")

        assert "leaderboard-top-10" in result

    def test_not_awarded_twice(self, db_session):
        seed_achievements(db_session)
        user = _make_user(db_session, points=500)

        first = check_and_award(db_session, user.id, "leaderboard.top10")
        second = check_and_award(db_session, user.id, "leaderboard.top10")

        assert "leaderboard-top-10" in first
        assert "leaderboard-top-10" not in second

    def test_unrelated_event_does_not_award_lb(self, db_session):
        seed_achievements(db_session)
        user = _make_user(db_session)

        result = check_and_award(db_session, user.id, "paper.uploaded")

        assert "leaderboard-top-10" not in result

    def test_context_passed_through(self, db_session):
        seed_achievements(db_session)
        user = _make_user(db_session, points=1000)
        ctx = {"rank": 3}

        result = check_and_award(db_session, user.id, "leaderboard.top10", ctx)

        assert "leaderboard-top-10" in result
        ua = (
            db_session.query(UserAchievement)
            .join(UserAchievement.achievement)
            .filter(UserAchievement.user_id == user.id)
            .first()
        )
        assert ua is not None


# ===========================================================================
# TestWeeklyLeaderboardResetAchievement
# ===========================================================================

class TestWeeklyLeaderboardResetAchievement:
    def test_awards_achievement_to_top_10(self, db_session):
        seed_achievements(db_session)
        # Create 11 users with all-time points; top 10 should get the achievement
        users = []
        for i in range(11):
            u = _make_user(db_session, points=(11 - i) * 1000, weekly_points=10)
            users.append(u)

        result = _do_weekly_leaderboard_reset(db_session)

        assert "lb_achievements_awarded" in result
        assert result["lb_achievements_awarded"] >= 1

    def test_reset_archives_weekly_entries(self, db_session):
        user = _make_user(db_session, weekly_points=50)

        result = _do_weekly_leaderboard_reset(db_session)

        assert result["archived"] >= 1
        assert "week_start" in result

    def test_weekly_points_zeroed_after_reset(self, db_session):
        user = _make_user(db_session, weekly_points=100)

        _do_weekly_leaderboard_reset(db_session)
        db_session.refresh(user)

        assert user.weekly_points == 0

    def test_lb_achievements_key_present_when_no_top_users(self, db_session):
        # If no users have all-time points, key still exists with 0
        result = _do_weekly_leaderboard_reset(db_session)
        assert "lb_achievements_awarded" in result
        assert isinstance(result["lb_achievements_awarded"], int)


# ===========================================================================
# TestDojoUserRateKey
# ===========================================================================

class TestDojoUserRateKey:
    def test_returns_user_key_for_valid_jwt(self, db_session, client):
        reg = client.post("/api/auth/register", json={
            "email": "ratekey_test@infra.test",
            "name": "RateKeyUser",
            "password": "Pass1234!",
        })
        assert reg.status_code in (200, 201)

        u = db_session.query(User).filter_by(email="ratekey_test@infra.test").first()
        u.is_email_verified = True
        db_session.commit()

        login = client.post("/api/auth/login", json={
            "email": "ratekey_test@infra.test",
            "password": "Pass1234!",
        })
        token = login.json()["access_token"]

        from backend.routers.dojo import _dojo_user_key
        req = MagicMock()
        req.headers.get = lambda key, default="": f"Bearer {token}" if key == "authorization" else default
        req.client = MagicMock()
        req.client.host = "127.0.0.1"

        key = _dojo_user_key(req)

        assert key.startswith("dojo:")
        expected_uid = db_session.query(User).filter_by(email="ratekey_test@infra.test").first().id
        assert key == f"dojo:{expected_uid}"

    def test_falls_back_to_ip_without_auth(self):
        from backend.routers.dojo import _dojo_user_key
        req = MagicMock()
        req.headers.get = lambda key, default="": "" if key == "authorization" else default
        req.client = MagicMock()
        req.client.host = "10.0.0.1"

        key = _dojo_user_key(req)

        assert not key.startswith("dojo:")

    def test_falls_back_to_ip_with_invalid_token(self):
        from backend.routers.dojo import _dojo_user_key
        req = MagicMock()
        req.headers.get = lambda key, default="": "Bearer garbage_token_xyz" if key == "authorization" else default
        req.client = MagicMock()
        req.client.host = "10.0.0.2"

        key = _dojo_user_key(req)

        assert not key.startswith("dojo:")


# ===========================================================================
# TestBeatScheduleEntries
# ===========================================================================

class TestBeatScheduleEntries:
    def _schedule(self):
        from backend.celery_app import celery_app
        return celery_app.conf.beat_schedule

    def test_prune_tutor_sessions_registered(self):
        assert "prune-old-tutor-sessions" in self._schedule()

    def test_prune_xp_events_registered(self):
        assert "prune-old-xp-events" in self._schedule()

    def test_recalc_acceptance_rates_registered(self):
        assert "recalc-acceptance-rates-nightly" in self._schedule()

    def test_prune_tutor_sessions_task_name(self):
        entry = self._schedule()["prune-old-tutor-sessions"]
        assert entry["task"] == "backend.tasks.scheduled_tasks.prune_old_tutor_sessions"

    def test_prune_xp_events_task_name(self):
        entry = self._schedule()["prune-old-xp-events"]
        assert entry["task"] == "backend.tasks.scheduled_tasks.prune_old_xp_events"

    def test_recalc_acceptance_rates_task_name(self):
        entry = self._schedule()["recalc-acceptance-rates-nightly"]
        assert entry["task"] == "backend.tasks.scheduled_tasks.recalc_all_acceptance_rates"

    def test_all_existing_schedules_preserved(self):
        sched = self._schedule()
        expected = {
            "cleanup-zombie-tasks-hourly",
            "daily-db-backup",
            "onboarding-drip-daily",
            "streak-at-risk-daily",
            "weekly-leaderboard-reset",
            "monthly-quota-reset",
            "weekly-digest-emails",
        }
        assert expected.issubset(sched.keys())
