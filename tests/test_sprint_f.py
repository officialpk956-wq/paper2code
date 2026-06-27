"""
tests/test_sprint_f.py

Sprint F — Progress & XP, Search filters, AI Tutor, Adaptive Engine  (~40 tests)

  TestXPEvents            — XPEvent logged on award_xp
  TestStreakMaintenance   — streak increment / reset / achievement trigger
  TestDomainCompletion    — domain detection + XP bonus via progress endpoint
  TestProblemFilters      — GET /api/problems?difficulty=&category=&q=
  TestPaperFilters        — GET /api/papers?q=&domain=
  TestRecommendations     — GET /api/recommendations
  TestLearningPaths       — GET /api/learning-paths
  TestTutorFeedback       — POST /api/tutor/feedback
  TestTutorSessions       — GET /api/tutor/sessions
"""

import datetime
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import (
    User, Problem, LearnerProgress, XPEvent,
    TutorFeedback, TutorSessionRecord,
)
from backend.modules.auth.security.hashing import hash_password


# ---------------------------------------------------------------------------
# Rate-limiter reset (same pattern as test_storage_infra.py)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_rate_limiter():
    try:
        from backend.server import limiter
        limiter.reset()
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "SprintF999!"


def _seed_user(db: Session, email: str, is_admin: bool = False) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        return existing
    u = User(
        email=email,
        name=email.split("@")[0],
        hashed_password=hash_password(_PASS),
        is_admin=is_admin,
        is_email_verified=True,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": _PASS})
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()["access_token"]


def _seed_problem(db: Session, pid: str = "prob-f-1", difficulty: str = "Easy", category: str = "Transformers") -> Problem:
    existing = db.query(Problem).filter_by(id=pid).first()
    if existing:
        return existing
    p = Problem(
        id=pid, slug=f"slug-{pid}", title=f"Implement {category}",
        difficulty=difficulty, category=category,
        description=f"A {difficulty} problem about {category}",
        is_retired=False,
        test_cases=[{"input": "x=1", "expected": "1"}],
        python_template="def solve(): pass",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ===========================================================================
# TestXPEvents — award_xp logs XPEvent rows
# ===========================================================================

class TestXPEvents:
    def test_01_award_xp_creates_event(self, db_session, client):
        u = _seed_user(db_session, "xplog1@test.com")
        from backend.services.progress_service import award_xp
        initial = u.points or 0
        new_pts = award_xp(db_session, u.id, "topic.completed", entity_id="transformer")
        assert new_pts == initial + 50

        events = db_session.query(XPEvent).filter_by(user_id=u.id, action="topic.completed").all()
        assert len(events) >= 1
        assert events[-1].amount == 50
        assert events[-1].entity_id == "transformer"

    def test_02_unknown_event_logs_nothing(self, db_session, client):
        u = _seed_user(db_session, "xplog2@test.com")
        from backend.services.progress_service import award_xp
        result = award_xp(db_session, u.id, "nonexistent.event")
        assert result == 0
        events = db_session.query(XPEvent).filter_by(user_id=u.id, action="nonexistent.event").all()
        assert events == []

    def test_03_weekly_points_incremented(self, db_session, client):
        u = _seed_user(db_session, "xplog3@test.com")
        from backend.services.progress_service import award_xp
        award_xp(db_session, u.id, "paper.uploaded")
        db_session.refresh(u)
        assert (u.weekly_points or 0) >= 25

    def test_04_multiple_events_accumulate(self, db_session, client):
        u = _seed_user(db_session, "xplog4@test.com")
        from backend.services.progress_service import award_xp
        award_xp(db_session, u.id, "dojo.solved.easy")
        award_xp(db_session, u.id, "topic.completed", entity_id="resnet")
        db_session.refresh(u)
        assert (u.points or 0) >= 100   # 50 + 50
        events = db_session.query(XPEvent).filter_by(user_id=u.id).all()
        assert len(events) >= 2


# ===========================================================================
# TestStreakMaintenance
# ===========================================================================

class TestStreakMaintenance:
    def test_05_first_activity_sets_streak_1(self, db_session, client):
        u = _seed_user(db_session, "streak1@test.com")
        from backend.services.progress_service import update_user_activity
        update_user_activity(db_session, u.id)
        db_session.refresh(u)
        assert u.streak == 1

    def test_06_same_day_no_increment(self, db_session, client):
        u = _seed_user(db_session, "streak2@test.com")
        from backend.services.progress_service import update_user_activity
        update_user_activity(db_session, u.id)
        update_user_activity(db_session, u.id)
        db_session.refresh(u)
        assert u.streak == 1   # called twice on same day

    def test_07_consecutive_day_increments(self, db_session, client):
        u = _seed_user(db_session, "streak3@test.com")
        yesterday = datetime.datetime.utcnow() - datetime.timedelta(days=1)
        u.last_active = yesterday
        u.streak = 3
        db_session.commit()
        from backend.services.progress_service import update_user_activity
        update_user_activity(db_session, u.id)
        db_session.refresh(u)
        assert u.streak == 4

    def test_08_gap_resets_streak(self, db_session, client):
        u = _seed_user(db_session, "streak4@test.com")
        three_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=3)
        u.last_active = three_days_ago
        u.streak = 10
        db_session.commit()
        from backend.services.progress_service import update_user_activity
        update_user_activity(db_session, u.id)
        db_session.refresh(u)
        assert u.streak == 1


# ===========================================================================
# TestDomainCompletion
# ===========================================================================

class TestDomainCompletion:
    def test_09_domain_not_complete_partially_done(self, db_session, client):
        u = _seed_user(db_session, "domain1@test.com")
        # Complete only one architecture from "transformers"
        db_session.add(LearnerProgress(
            learner_id=str(u.id), entity_type="architecture",
            entity_id="transformer", status="completed",
            completed_at=datetime.datetime.utcnow(),
        ))
        db_session.commit()
        from backend.services.progress_service import check_domain_completion
        result = check_domain_completion(db_session, u.id, "transformer")
        assert result is None   # domain has more required architectures

    def test_10_domain_completion_detected(self, db_session, client):
        u = _seed_user(db_session, "domain2@test.com")
        from backend.services.progress_service import DOMAIN_ARCHITECTURES
        for slug in DOMAIN_ARCHITECTURES["generative"]:
            db_session.add(LearnerProgress(
                learner_id=str(u.id), entity_type="architecture",
                entity_id=slug, status="completed",
                completed_at=datetime.datetime.utcnow(),
            ))
        db_session.commit()
        from backend.services.progress_service import check_domain_completion
        last_slug = DOMAIN_ARCHITECTURES["generative"][-1]
        result = check_domain_completion(db_session, u.id, last_slug)
        assert result == "generative"

    def test_11_progress_endpoint_awards_xp_on_topic(self, db_session, client):
        u = _seed_user(db_session, "domain3@test.com")
        tok = _login(client, "domain3@test.com")
        initial = u.points or 0
        r = client.post(
            "/api/progress/architecture/vae",
            json={"status": "completed", "time_spent_seconds": 120},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        db_session.refresh(u)
        assert (u.points or 0) >= initial + 50   # topic.completed = 50 XP

    def test_12_domain_completion_fires_bonus_xp(self, db_session, client):
        u = _seed_user(db_session, "domain4@test.com")
        from backend.services.progress_service import DOMAIN_ARCHITECTURES
        # Pre-complete all but the last "generative" architecture
        generative = DOMAIN_ARCHITECTURES["generative"]
        for slug in generative[:-1]:
            db_session.add(LearnerProgress(
                learner_id=str(u.id), entity_type="architecture",
                entity_id=slug, status="completed",
                completed_at=datetime.datetime.utcnow(),
            ))
        db_session.commit()

        tok = _login(client, "domain4@test.com")
        initial_pts = u.points or 0
        # Complete the last one via API
        last = generative[-1]
        r = client.post(
            f"/api/progress/architecture/{last}",
            json={"status": "completed", "time_spent_seconds": 60},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        db_session.refresh(u)
        # Should have topic XP (50) + domain bonus (500)
        assert (u.points or 0) >= initial_pts + 550


# ===========================================================================
# TestProblemFilters  — GET /api/problems
# ===========================================================================

class TestProblemFilters:
    def test_13_no_filter_returns_all(self, db_session, client):
        _seed_problem(db_session, "pf-easy-1", "Easy", "Transformers")
        _seed_problem(db_session, "pf-hard-1", "Hard", "CNNs")
        r = client.get("/api/problems")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_14_filter_by_difficulty(self, db_session, client):
        _seed_problem(db_session, "pf-easy-2", "Easy", "Diffusion")
        _seed_problem(db_session, "pf-hard-2", "Hard", "RL")
        r = client.get("/api/problems?difficulty=Easy")
        assert r.status_code == 200
        data = r.json()
        assert all(p["difficulty"].lower() == "easy" for p in data)

    def test_15_filter_by_category(self, db_session, client):
        _seed_problem(db_session, "pf-cat-1", "Medium", "GraphNN")
        r = client.get("/api/problems?category=GraphNN")
        assert r.status_code == 200
        data = r.json()
        assert any("GraphNN" in (p.get("category") or "") for p in data)

    def test_16_filter_by_q_matches_title(self, db_session, client):
        _seed_problem(db_session, "pf-q-1", "Easy", "Attention")
        r = client.get("/api/problems?q=Attention")
        assert r.status_code == 200
        data = r.json()
        assert any("Attention" in (p.get("category") or "") or "Attention" in (p.get("title") or "") for p in data)

    def test_17_q_too_short_returns_422(self, db_session, client):
        r = client.get("/api/problems?q=x")
        assert r.status_code == 422

    def test_18_retired_problems_excluded(self, db_session, client):
        existing = db_session.query(Problem).filter_by(id="pf-retired").first()
        if not existing:
            db_session.add(Problem(
                id="pf-retired", slug="slug-pf-retired", title="Retired Problem",
                difficulty="Hard", category="Misc", description="retired",
                is_retired=True, test_cases=[], python_template="",
            ))
            db_session.commit()
        r = client.get("/api/problems")
        assert r.status_code == 200
        assert all(not p.get("is_retired", False) for p in r.json())

    def test_19_combined_filters(self, db_session, client):
        _seed_problem(db_session, "pf-combo-1", "Hard", "Transformers")
        r = client.get("/api/problems?difficulty=Hard&category=Transformers")
        assert r.status_code == 200
        data = r.json()
        for p in data:
            assert p["difficulty"].lower() == "hard"
            assert "transformers" in p["category"].lower()


# ===========================================================================
# TestPaperFilters  — GET /api/papers
# ===========================================================================

class TestPaperFilters:
    def test_20_no_filter_returns_all(self, db_session, client):
        r = client.get("/api/papers")
        assert r.status_code == 200
        assert "papers" in r.json()

    def test_21_q_filter_short_422(self, db_session, client):
        r = client.get("/api/papers?q=x")
        assert r.status_code == 422

    def test_22_q_filter_accepted(self, db_session, client):
        r = client.get("/api/papers?q=attention")
        assert r.status_code == 200
        assert "papers" in r.json()

    def test_23_domain_filter_accepted(self, db_session, client):
        r = client.get("/api/papers?domain=transformer")
        assert r.status_code == 200
        assert "papers" in r.json()

    def test_24_combined_q_and_domain(self, db_session, client):
        r = client.get("/api/papers?q=attention&domain=transformer")
        assert r.status_code == 200


# ===========================================================================
# TestRecommendations  — GET /api/recommendations
# ===========================================================================

class TestRecommendations:
    def test_25_requires_auth(self, db_session, client):
        r = client.get("/api/recommendations")
        assert r.status_code == 401

    def test_26_returns_structure(self, db_session, client):
        u = _seed_user(db_session, "recs1@test.com")
        tok = _login(client, "recs1@test.com")
        r = client.get("/api/recommendations", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert "user_id" in body
        assert "recommendations" in body
        assert "adaptive" in body
        assert body["user_id"] == u.id


# ===========================================================================
# TestLearningPaths  — GET /api/learning-paths
# ===========================================================================

class TestLearningPaths:
    def test_27_requires_auth(self, db_session, client):
        r = client.get("/api/learning-paths")
        assert r.status_code == 401

    def test_28_returns_paths_list(self, db_session, client):
        _seed_user(db_session, "lp1@test.com")
        tok = _login(client, "lp1@test.com")
        r = client.get("/api/learning-paths", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert "paths" in body
        assert len(body["paths"]) >= 3

    def test_29_paths_have_required_fields(self, db_session, client):
        _seed_user(db_session, "lp2@test.com")
        tok = _login(client, "lp2@test.com")
        r = client.get("/api/learning-paths", headers={"Authorization": f"Bearer {tok}"})
        for path in r.json()["paths"]:
            assert "id" in path
            assert "title" in path
            assert "steps" in path
            assert "progress_pct" in path
            assert 0 <= path["progress_pct"] <= 100

    def test_30_steps_have_completion_flag(self, db_session, client):
        _seed_user(db_session, "lp3@test.com")
        tok = _login(client, "lp3@test.com")
        r = client.get("/api/learning-paths", headers={"Authorization": f"Bearer {tok}"})
        for path in r.json()["paths"]:
            for step in path["steps"]:
                assert "completed" in step
                assert isinstance(step["completed"], bool)

    def test_31_completed_steps_reflected(self, db_session, client):
        u = _seed_user(db_session, "lp4@test.com")
        db_session.add(LearnerProgress(
            learner_id=str(u.id), entity_type="architecture",
            entity_id="transformer", status="completed",
            completed_at=datetime.datetime.utcnow(),
        ))
        db_session.commit()
        tok = _login(client, "lp4@test.com")
        r = client.get("/api/learning-paths", headers={"Authorization": f"Bearer {tok}"})
        paths = r.json()["paths"]
        # ai-engineer path includes "transformer"
        ai_path = next(p for p in paths if p["id"] == "ai-engineer")
        transformer_step = next(s for s in ai_path["steps"] if s["id"] == "transformer")
        assert transformer_step["completed"] is True
        assert ai_path["progress_pct"] > 0


# ===========================================================================
# TestTutorFeedback  — POST /api/tutor/feedback
# ===========================================================================

class TestTutorFeedback:
    def _make_session(self, client, tok):
        r = client.post("/api/tutor/start-session", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        return r.json()["session_id"]

    def test_32_requires_auth(self, db_session, client):
        r = client.post("/api/tutor/feedback", json={"session_id": "x", "message_index": 0, "rating": 1})
        assert r.status_code == 401

    def test_33_invalid_rating_rejected(self, db_session, client):
        _seed_user(db_session, "tfb1@test.com")
        tok = _login(client, "tfb1@test.com")
        sess = self._make_session(client, tok)
        r = client.post(
            "/api/tutor/feedback",
            json={"session_id": sess, "message_index": 0, "rating": 0},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 400

    def test_34_thumbs_up_recorded(self, db_session, client):
        _seed_user(db_session, "tfb2@test.com")
        tok = _login(client, "tfb2@test.com")
        sess = self._make_session(client, tok)
        r = client.post(
            "/api/tutor/feedback",
            json={"session_id": sess, "message_index": 0, "rating": 1},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["recorded"] is True
        assert body["rating"] == 1

    def test_35_thumbs_down_recorded(self, db_session, client):
        _seed_user(db_session, "tfb3@test.com")
        tok = _login(client, "tfb3@test.com")
        sess = self._make_session(client, tok)
        r = client.post(
            "/api/tutor/feedback",
            json={"session_id": sess, "message_index": 1, "rating": -1},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 201
        assert r.json()["rating"] == -1

    def test_36_feedback_persisted_in_db(self, db_session, client):
        u = _seed_user(db_session, "tfb4@test.com")
        tok = _login(client, "tfb4@test.com")
        sess = self._make_session(client, tok)
        client.post(
            "/api/tutor/feedback",
            json={"session_id": sess, "message_index": 2, "rating": 1},
            headers={"Authorization": f"Bearer {tok}"},
        )
        row = db_session.query(TutorFeedback).filter_by(
            user_id=u.id, session_id=sess, message_index=2,
        ).first()
        assert row is not None
        assert row.rating == 1

    def test_37_rating_update_idempotent(self, db_session, client):
        u = _seed_user(db_session, "tfb5@test.com")
        tok = _login(client, "tfb5@test.com")
        sess = self._make_session(client, tok)
        client.post(
            "/api/tutor/feedback",
            json={"session_id": sess, "message_index": 0, "rating": 1},
            headers={"Authorization": f"Bearer {tok}"},
        )
        r = client.post(
            "/api/tutor/feedback",
            json={"session_id": sess, "message_index": 0, "rating": -1},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 201
        row = db_session.query(TutorFeedback).filter_by(
            user_id=u.id, session_id=sess, message_index=0,
        ).first()
        assert row.rating == -1   # updated

    def test_38_foreign_session_rejected(self, db_session, client):
        _seed_user(db_session, "tfb6@test.com")
        tok = _login(client, "tfb6@test.com")
        r = client.post(
            "/api/tutor/feedback",
            json={"session_id": "00000000-0000-0000-0000-000000000000", "message_index": 0, "rating": 1},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 403


# ===========================================================================
# TestTutorSessions  — GET /api/tutor/sessions
# ===========================================================================

class TestTutorSessions:
    def test_39_requires_auth(self, db_session, client):
        r = client.get("/api/tutor/sessions")
        assert r.status_code == 401

    def test_40_empty_initially(self, db_session, client):
        _seed_user(db_session, "ts1@test.com")
        tok = _login(client, "ts1@test.com")
        r = client.get("/api/tutor/sessions", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert "sessions" in body
        assert "total" in body

    def test_41_session_record_appears_after_ask(self, db_session, client):
        u = _seed_user(db_session, "ts2@test.com")
        tok = _login(client, "ts2@test.com")

        # Inject a TutorSessionRecord directly (bypass LLM call)
        sess_id = "aaaabbbb-0000-0000-0000-000000000001"
        db_session.add(TutorSessionRecord(
            user_id=u.id,
            session_id=sess_id,
            context_type="architecture",
            messages=[{"role": "user", "content": "What is attention?"}, {"role": "assistant", "content": "..."}],
            last_active_at=datetime.datetime.utcnow(),
        ))
        db_session.commit()

        r = client.get("/api/tutor/sessions", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        ids = [s["session_id"] for s in body["sessions"]]
        assert sess_id in ids

    def test_42_session_shows_message_count(self, db_session, client):
        u = _seed_user(db_session, "ts3@test.com")
        tok = _login(client, "ts3@test.com")
        sess_id = "aaaabbbb-0000-0000-0000-000000000002"
        db_session.add(TutorSessionRecord(
            user_id=u.id,
            session_id=sess_id,
            context_type="architecture",
            messages=[{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
            last_active_at=datetime.datetime.utcnow(),
        ))
        db_session.commit()
        r = client.get("/api/tutor/sessions", headers={"Authorization": f"Bearer {tok}"})
        sess_data = next(s for s in r.json()["sessions"] if s["session_id"] == sess_id)
        assert sess_data["message_count"] == 2

    def test_43_sessions_respect_limit(self, db_session, client):
        u = _seed_user(db_session, "ts4@test.com")
        tok = _login(client, "ts4@test.com")
        for i in range(5):
            sid = f"aaaabbbb-0000-0000-0000-{i:012d}"
            if not db_session.query(TutorSessionRecord).filter_by(session_id=sid).first():
                db_session.add(TutorSessionRecord(
                    user_id=u.id, session_id=sid, context_type="test",
                    messages=[], last_active_at=datetime.datetime.utcnow(),
                ))
        db_session.commit()
        r = client.get("/api/tutor/sessions?limit=2", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        assert len(r.json()["sessions"]) <= 2
