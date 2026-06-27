"""
tests/test_user_profile.py

Covers 6 audit items for auth/user profile endpoints:
  1. GET /api/auth/me        Partial fix — now returns full profile incl. problems_solved
  2. PATCH /api/me           Missing P0 — update name / avatar_url
  3. GET /api/users/{id}     Missing P0 — public profile with rank, XP, streak, achievements
  4. GET /api/me/xp-history  Missing P1 — paginated XPEvents for current user
  5. GET /api/me/papers      Missing P1 — own papers including private
  (Dead stub /api/users/leaderboard removed — verified via separate route check)
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, XPEvent, Paper, Achievement, UserAchievement, DojoSubmission, Problem
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "UserProfile99!"


def _seed_user(db: Session, email: str, points: int = 0, name: str = None) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        return existing
    u = User(
        email=email,
        name=name or email.split("@")[0],
        hashed_password=hash_password(_PASS),
        is_verified=True,
        is_email_verified=True,
        points=points,
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


def _seed_paper(db: Session, user_id: int, suffix: str, visibility: str = "public") -> Paper:
    p = Paper(
        title=f"My Paper {suffix}",
        authors="Test Author",
        abstract="Abstract",
        uploaded_by=user_id,
        visibility=visibility,
        r2_key=f"papers/test-{suffix}.pdf",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _seed_xp_event(db: Session, user_id: int, action: str = "test.action", amount: int = 10) -> XPEvent:
    e = XPEvent(user_id=user_id, action=action, amount=amount)
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


def _seed_problem(db: Session, pid: str) -> Problem:
    existing = db.query(Problem).filter_by(id=pid).first()
    if existing:
        return existing
    p = Problem(
        id=pid,
        slug=f"slug-{pid}",
        title=f"Problem {pid}",
        difficulty="Easy",
        category="Testing",
        description="desc",
        python_template="# t",
        test_cases=[],
        version=1,
        is_retired=False,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _seed_submission(db: Session, user_id: int, problem_id: str, passed: bool = True, is_best: bool = True) -> DojoSubmission:
    s = DojoSubmission(
        user_id=user_id,
        problem_id=problem_id,
        code="pass",
        passed=passed,
        stdout="",
        stderr="",
        time_ms=100,
        is_best=is_best,
        problem_version=1,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ---------------------------------------------------------------------------
# TestGetMe — GET /api/auth/me full fields
# ---------------------------------------------------------------------------

class TestGetMe:

    def test_01_returns_all_required_fields(self, client, db_session):
        user = _seed_user(db_session, "me01@up.com")
        token = _login(client, user.email)

        r = client.get("/api/auth/me", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()

        required = [
            "id", "email", "name", "avatar_url", "points", "weekly_points",
            "storage_bytes_used", "streak", "xp_level", "last_active",
            "is_admin", "is_verified", "mfa_enabled", "is_email_verified",
            "problems_solved",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_02_problems_solved_counts_best_passing(self, client, db_session):
        user = _seed_user(db_session, "me02@up.com")
        prob = _seed_problem(db_session, "up-me02")
        _seed_submission(db_session, user.id, prob.id, passed=True, is_best=True)
        token = _login(client, user.email)

        r = client.get("/api/auth/me", headers=_auth(token))
        assert r.json()["problems_solved"] >= 1

    def test_03_problems_solved_excludes_non_best(self, client, db_session):
        user = _seed_user(db_session, "me03@up.com")
        prob = _seed_problem(db_session, "up-me03")
        _seed_submission(db_session, user.id, prob.id, passed=True, is_best=False)
        token = _login(client, user.email)

        r = client.get("/api/auth/me", headers=_auth(token))
        assert r.json()["problems_solved"] == 0

    def test_04_unauthenticated_returns_401(self, client, db_session):
        r = client.get("/api/auth/me")
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# TestPatchMe — PATCH /api/me
# ---------------------------------------------------------------------------

class TestPatchMe:

    def test_05_update_name(self, client, db_session):
        user = _seed_user(db_session, "patch05@up.com")
        token = _login(client, user.email)

        r = client.patch("/api/me", json={"name": "New Name"}, headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["name"] == "New Name"

    def test_06_name_persisted_in_db(self, client, db_session):
        user = _seed_user(db_session, "patch06@up.com")
        token = _login(client, user.email)

        client.patch("/api/me", json={"name": "Persisted"}, headers=_auth(token))
        db_session.refresh(user)
        assert user.name == "Persisted"

    def test_07_update_avatar_url(self, client, db_session):
        user = _seed_user(db_session, "patch07@up.com")
        token = _login(client, user.email)

        r = client.patch("/api/me", json={"avatar_url": "https://example.com/avatar.png"}, headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["avatar_url"] == "https://example.com/avatar.png"

    def test_08_partial_update_leaves_other_fields(self, client, db_session):
        user = _seed_user(db_session, "patch08@up.com", name="Original")
        token = _login(client, user.email)

        client.patch("/api/me", json={"avatar_url": "https://x.com/img.png"}, headers=_auth(token))
        db_session.refresh(user)
        assert user.name == "Original"

    def test_09_unauthenticated_returns_401(self, client, db_session):
        r = client.patch("/api/me", json={"name": "hack"})
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# TestPublicProfile — GET /api/users/{id}
# ---------------------------------------------------------------------------

class TestPublicProfile:

    def test_10_returns_200_for_existing_user(self, client, db_session):
        user = _seed_user(db_session, "pub10@up.com")
        r = client.get(f"/api/users/{user.id}")
        assert r.status_code == 200

    def test_11_returns_expected_fields(self, client, db_session):
        user = _seed_user(db_session, "pub11@up.com")
        r = client.get(f"/api/users/{user.id}")
        data = r.json()
        for field in ["id", "name", "points", "weekly_points", "xp_level", "streak", "rank", "problems_solved", "achievements"]:
            assert field in data, f"Missing field: {field}"

    def test_12_rank_is_positive_int(self, client, db_session):
        user = _seed_user(db_session, "pub12@up.com")
        r = client.get(f"/api/users/{user.id}")
        rank = r.json()["rank"]
        assert isinstance(rank, int) and rank >= 1

    def test_13_achievements_is_list(self, client, db_session):
        user = _seed_user(db_session, "pub13@up.com")
        r = client.get(f"/api/users/{user.id}")
        assert isinstance(r.json()["achievements"], list)

    def test_14_problems_solved_reflects_submissions(self, client, db_session):
        user = _seed_user(db_session, "pub14@up.com")
        prob = _seed_problem(db_session, "up-pub14")
        _seed_submission(db_session, user.id, prob.id, passed=True, is_best=True)

        r = client.get(f"/api/users/{user.id}")
        assert r.json()["problems_solved"] >= 1

    def test_15_missing_user_returns_404(self, client, db_session):
        r = client.get("/api/users/9999999")
        assert r.status_code == 404

    def test_16_rank_lower_for_higher_points_user(self, client, db_session):
        top = _seed_user(db_session, "pub16top@up.com", points=10000)
        low = _seed_user(db_session, "pub16low@up.com", points=1)

        r_top = client.get(f"/api/users/{top.id}")
        r_low = client.get(f"/api/users/{low.id}")
        assert r_top.json()["rank"] <= r_low.json()["rank"]


# ---------------------------------------------------------------------------
# TestXPHistory — GET /api/me/xp-history
# ---------------------------------------------------------------------------

class TestXPHistory:

    def test_17_returns_200(self, client, db_session):
        user = _seed_user(db_session, "xp17@up.com")
        token = _login(client, user.email)
        r = client.get("/api/me/xp-history", headers=_auth(token))
        assert r.status_code == 200

    def test_18_returns_expected_fields(self, client, db_session):
        user = _seed_user(db_session, "xp18@up.com")
        token = _login(client, user.email)
        r = client.get("/api/me/xp-history", headers=_auth(token))
        data = r.json()
        assert "total" in data
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_19_events_contain_user_events(self, client, db_session):
        user = _seed_user(db_session, "xp19@up.com")
        _seed_xp_event(db_session, user.id, action="problem.solved", amount=50)
        token = _login(client, user.email)

        r = client.get("/api/me/xp-history", headers=_auth(token))
        data = r.json()
        assert data["total"] >= 1
        assert any(e["action"] == "problem.solved" for e in data["events"])

    def test_20_events_ordered_newest_first(self, client, db_session):
        user = _seed_user(db_session, "xp20@up.com")
        _seed_xp_event(db_session, user.id, action="first.event", amount=10)
        _seed_xp_event(db_session, user.id, action="second.event", amount=20)
        token = _login(client, user.email)

        r = client.get("/api/me/xp-history", headers=_auth(token))
        events = r.json()["events"]
        if len(events) >= 2:
            # Newest first: second event (higher id) should come before first
            ids = [e["id"] for e in events]
            assert ids == sorted(ids, reverse=True)

    def test_21_does_not_return_other_users_events(self, client, db_session):
        user_a = _seed_user(db_session, "xp21a@up.com")
        user_b = _seed_user(db_session, "xp21b@up.com")
        _seed_xp_event(db_session, user_b.id, action="other.event", amount=99)
        token = _login(client, user_a.email)

        r = client.get("/api/me/xp-history", headers=_auth(token))
        assert all(e["action"] != "other.event" for e in r.json()["events"])

    def test_22_unauthenticated_returns_401(self, client, db_session):
        r = client.get("/api/me/xp-history")
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# TestMyPapers — GET /api/me/papers
# ---------------------------------------------------------------------------

class TestMyPapers:

    def test_23_returns_200(self, client, db_session):
        user = _seed_user(db_session, "mp23@up.com")
        token = _login(client, user.email)
        r = client.get("/api/me/papers", headers=_auth(token))
        assert r.status_code == 200

    def test_24_returns_expected_structure(self, client, db_session):
        user = _seed_user(db_session, "mp24@up.com")
        token = _login(client, user.email)
        r = client.get("/api/me/papers", headers=_auth(token))
        data = r.json()
        assert "total" in data
        assert "papers" in data
        assert isinstance(data["papers"], list)

    def test_25_includes_own_papers(self, client, db_session):
        user = _seed_user(db_session, "mp25@up.com")
        _seed_paper(db_session, user.id, "mp25pub", visibility="public")
        token = _login(client, user.email)

        r = client.get("/api/me/papers", headers=_auth(token))
        assert r.json()["total"] >= 1

    def test_26_includes_private_papers(self, client, db_session):
        user = _seed_user(db_session, "mp26@up.com")
        _seed_paper(db_session, user.id, "mp26priv", visibility="private")
        token = _login(client, user.email)

        r = client.get("/api/me/papers", headers=_auth(token))
        titles = [p["title"] for p in r.json()["papers"]]
        assert any("mp26priv" in t for t in titles)

    def test_27_excludes_other_users_papers(self, client, db_session):
        owner = _seed_user(db_session, "mp27owner@up.com")
        other = _seed_user(db_session, "mp27other@up.com")
        _seed_paper(db_session, owner.id, "mp27other-paper", visibility="public")
        token = _login(client, other.email)

        r = client.get("/api/me/papers", headers=_auth(token))
        assert all(p["title"] != "My Paper mp27other-paper" for p in r.json()["papers"])

    def test_28_paper_has_visibility_field(self, client, db_session):
        user = _seed_user(db_session, "mp28@up.com")
        _seed_paper(db_session, user.id, "mp28", visibility="unlisted")
        token = _login(client, user.email)

        r = client.get("/api/me/papers", headers=_auth(token))
        papers = r.json()["papers"]
        assert len(papers) >= 1
        assert "visibility" in papers[0]

    def test_29_unauthenticated_returns_401(self, client, db_session):
        r = client.get("/api/me/papers")
        assert r.status_code in (401, 403)
