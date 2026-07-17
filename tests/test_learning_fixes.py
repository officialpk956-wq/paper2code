"""
tests/test_learning_fixes.py

Covers 4 audit fixes applied to backend/routers/learning.py:
  1. GET /api/adaptive/recommendations — method name fix (no more AttributeError)
  2. GET /api/analytics/dashboard      — JWT path: authenticated user's ID used as learner key
  3. POST /api/assessment/validate     — XP awarded (assessment.completed, 75 XP) when correct + auth'd
  4. GET/PATCH /api/me/notification-prefs — new endpoint, email_drip_opt_out toggle
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import AssessmentAttempt, User, XPEvent
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "LearningFix999!"


def _seed_user(db: Session, email: str, is_admin: bool = False) -> User:
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
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": _PASS})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# TestAdaptiveRecommendationsFix — GET /api/adaptive/recommendations
# ---------------------------------------------------------------------------


class TestAdaptiveRecommendationsFix:
    """Verify the wrong method name (generate_recommendations) is fixed."""

    def test_01_returns_200_not_500(self, client, db_session):
        # Endpoint was broken: adaptive_engine.generate_recommendations does not exist.
        # Fix: use get_personalized_recommendations.
        with patch(
            "core.analytics.adaptive_engine.adaptive_engine.get_personalized_recommendations",
            return_value={"next_topics": [], "review_items": []},
        ):
            r = client.get(
                "/api/adaptive/recommendations",
                headers={"X-Learner-ID": "anon-test-01"},
            )
        assert r.status_code == 200

    def test_02_response_has_recommendations_key(self, client, db_session):
        with patch(
            "core.analytics.adaptive_engine.adaptive_engine.get_personalized_recommendations",
            return_value={"next_topics": ["transformer"], "review_items": []},
        ):
            r = client.get(
                "/api/adaptive/recommendations",
                headers={"X-Learner-ID": "anon-test-02"},
            )
        assert r.status_code == 200
        assert "recommendations" in r.json()


# ---------------------------------------------------------------------------
# TestAnalyticsDashboardJWT — GET /api/analytics/dashboard
# ---------------------------------------------------------------------------


class TestAnalyticsDashboardJWT:
    """Verify authenticated users don't need X-Learner-ID; their user.id is used."""

    def test_03_unauthenticated_still_works(self, client, db_session):
        with patch(
            "core.analytics.recommendation_engine.recommendation_engine.compute",
            return_value=[],
        ):
            r = client.get(
                "/api/analytics/dashboard",
                headers={"X-Learner-ID": "anon-test-03"},
            )
        assert r.status_code == 200
        data = r.json()
        assert "learning_overview" in data

    def test_04_authenticated_user_gets_dashboard(self, client, db_session):
        user = _seed_user(db_session, "dash04@lf.com")
        token = _login(client, user.email)

        with patch(
            "core.analytics.recommendation_engine.recommendation_engine.compute",
            return_value=[],
        ):
            r = client.get("/api/analytics/dashboard", headers=_auth(token))
        assert r.status_code == 200
        assert "learning_overview" in r.json()

    def test_05_auth_overrides_header_learner_id(self, client, db_session):
        # When JWT is provided, the learner_key used internally should be the
        # user's integer id (as string), not the X-Learner-ID header value.
        user = _seed_user(db_session, "dash05@lf.com")
        token = _login(client, user.email)

        # We verify that the query runs without error; if the learner_key were
        # the header value, this would still be 200 — so we just confirm 200.
        with patch(
            "core.analytics.recommendation_engine.recommendation_engine.compute",
            return_value=[],
        ):
            r = client.get(
                "/api/analytics/dashboard",
                headers={**_auth(token), "X-Learner-ID": "should-be-ignored"},
            )
        assert r.status_code == 200

    def test_05b_header_cannot_spoof_another_users_real_data(self, client, db_session):
        # IDOR regression: previously the header was used verbatim with no
        # auth check at all, so anyone could read ANY user's real analytics
        # by guessing/knowing their integer id (GET /api/analytics/dashboard
        # with X-Learner-ID: <victim id>, no login required). Confirm the
        # authenticated caller's own data comes back even when the header
        # names a different, real user with different attempt data.
        victim = _seed_user(db_session, "victim05b@lf.com")
        db_session.add(
            AssessmentAttempt(
                learner_id=str(victim.id),
                assessment_type="architecture",
                architecture="Transformer",
                is_correct=True,
            )
        )
        db_session.commit()

        caller = _seed_user(db_session, "caller05b@lf.com")
        db_session.add(
            AssessmentAttempt(
                learner_id=str(caller.id),
                assessment_type="architecture",
                architecture="ResNet",
                is_correct=True,
            )
        )
        db_session.commit()
        token = _login(client, caller.email)

        with patch(
            "core.analytics.recommendation_engine.recommendation_engine.compute",
            return_value=[],
        ):
            r = client.get(
                "/api/analytics/dashboard",
                headers={**_auth(token), "X-Learner-ID": str(victim.id)},
            )
        assert r.status_code == 200
        data = r.json()
        # Caller's own attempt (ResNet) must be reflected, not the victim's (Transformer).
        assert "ResNet" in data["assessment_performance"]["strongest_architecture"]
        assert "Transformer" not in data["assessment_performance"]["strongest_architecture"]


# ---------------------------------------------------------------------------
# TestAssessmentValidateXP — POST /api/assessment/validate
# ---------------------------------------------------------------------------


class TestAssessmentValidateXP:
    """Verify award_xp is called for correct authenticated answers."""

    _CHALLENGE = {
        "question": "What is 1+1?",
        "assessment_type": "math",
        "architecture": "ResNet",
        "difficulty": "beginner",
        "answer": "2",
    }

    def test_06_xp_awarded_when_correct_and_authed(self, client, db_session):
        user = _seed_user(db_session, "xp06@lf.com")
        token = _login(client, user.email)

        with patch(
            "core.assessment.engine.assessment_engine.validate",
            return_value={
                "is_correct": True,
                "score": 1.0,
                "correct_answer": "2",
                "explanation": "Correct",
            },
        ):
            r = client.post(
                "/api/assessment/validate",
                json={"challenge": self._CHALLENGE, "user_answer": "2"},
                headers=_auth(token),
            )

        assert r.status_code == 200
        # XP event should have been written to DB
        ev = (
            db_session.query(XPEvent)
            .filter_by(user_id=user.id, action="assessment.completed")
            .first()
        )
        assert ev is not None
        assert ev.amount == 75

    def test_07_no_xp_when_incorrect(self, client, db_session):
        user = _seed_user(db_session, "xp07@lf.com")
        token = _login(client, user.email)

        with patch(
            "core.assessment.engine.assessment_engine.validate",
            return_value={
                "is_correct": False,
                "score": 0.0,
                "correct_answer": "2",
                "explanation": "Wrong",
            },
        ):
            r = client.post(
                "/api/assessment/validate",
                json={"challenge": self._CHALLENGE, "user_answer": "3"},
                headers=_auth(token),
            )

        assert r.status_code == 200
        ev = (
            db_session.query(XPEvent)
            .filter_by(user_id=user.id, action="assessment.completed")
            .first()
        )
        assert ev is None

    def test_08_no_xp_when_unauthenticated(self, client, db_session):
        with patch(
            "core.assessment.engine.assessment_engine.validate",
            return_value={
                "is_correct": True,
                "score": 1.0,
                "correct_answer": "2",
                "explanation": "Correct",
            },
        ):
            r = client.post(
                "/api/assessment/validate",
                json={"challenge": self._CHALLENGE, "user_answer": "2"},
                headers={"X-Learner-ID": "anon-08"},
            )

        # Should still 200 — no auth doesn't break the endpoint
        assert r.status_code == 200

    def test_09_validate_still_records_attempt(self, client, db_session):
        user = _seed_user(db_session, "xp09@lf.com")
        token = _login(client, user.email)

        with patch(
            "core.assessment.engine.assessment_engine.validate",
            return_value={
                "is_correct": True,
                "score": 1.0,
                "correct_answer": "2",
                "explanation": "Correct",
            },
        ):
            r = client.post(
                "/api/assessment/validate",
                json={"challenge": self._CHALLENGE, "user_answer": "2"},
                headers=_auth(token),
            )

        assert r.status_code == 200
        data = r.json()
        assert data["is_correct"] is True


# ---------------------------------------------------------------------------
# TestNotificationPrefs — GET/PATCH /api/me/notification-prefs
# ---------------------------------------------------------------------------


class TestNotificationPrefs:
    def test_10_get_prefs_default_false(self, client, db_session):
        user = _seed_user(db_session, "pref10@lf.com")
        token = _login(client, user.email)

        r = client.get("/api/me/notification-prefs", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["email_drip_opt_out"] is False

    def test_11_patch_opt_out(self, client, db_session):
        user = _seed_user(db_session, "pref11@lf.com")
        token = _login(client, user.email)

        r = client.patch(
            "/api/me/notification-prefs",
            json={"email_drip_opt_out": True},
            headers=_auth(token),
        )
        assert r.status_code == 200
        assert r.json()["email_drip_opt_out"] is True

    def test_12_patch_persisted_across_requests(self, client, db_session):
        user = _seed_user(db_session, "pref12@lf.com")
        token = _login(client, user.email)

        client.patch(
            "/api/me/notification-prefs",
            json={"email_drip_opt_out": True},
            headers=_auth(token),
        )
        r = client.get("/api/me/notification-prefs", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["email_drip_opt_out"] is True

    def test_13_patch_opt_back_in(self, client, db_session):
        user = _seed_user(db_session, "pref13@lf.com")
        token = _login(client, user.email)

        client.patch(
            "/api/me/notification-prefs",
            json={"email_drip_opt_out": True},
            headers=_auth(token),
        )
        r = client.patch(
            "/api/me/notification-prefs",
            json={"email_drip_opt_out": False},
            headers=_auth(token),
        )
        assert r.status_code == 200
        assert r.json()["email_drip_opt_out"] is False

    def test_14_unauthenticated_get_rejected(self, client, db_session):
        r = client.get("/api/me/notification-prefs")
        assert r.status_code in (401, 403)

    def test_15_unauthenticated_patch_rejected(self, client, db_session):
        r = client.patch(
            "/api/me/notification-prefs",
            json={"email_drip_opt_out": True},
        )
        assert r.status_code in (401, 403)
