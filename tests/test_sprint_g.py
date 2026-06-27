"""
tests/test_sprint_g.py

Sprint G — Problem Management, Execution Engine, Submissions, Leaderboard  (~40 tests)

  TestProblemVersioning     — version bump on test_case update, problem_version on submissions
  TestPerProblemTimeout     — time_limit_ms stored, passed to executor
  TestBestSubmission        — is_best tracking across multiple submissions
  TestAcceptanceRate        — acceptance_rate updated after each submission
  TestSubmissionHistory     — GET /api/dojo/submissions (user history, paginated)
  TestAdminProblemCRUD      — time_limit_ms in create/update, version in response
  TestLeaderboardCategory   — GET /api/leaderboard?category= filter
  TestStdoutTruncation      — 64 KB output_limit in Piston payload
"""

import datetime
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Problem, DojoSubmission
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Rate-limiter reset
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

_PASS = "SprintG999!"


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


def _seed_problem(
    db: Session,
    pid: str = "sg-prob-1",
    difficulty: str = "Easy",
    category: str = "Transformers",
    time_limit_ms: int = None,
) -> Problem:
    existing = db.query(Problem).filter_by(id=pid).first()
    if existing:
        return existing
    p = Problem(
        id=pid,
        slug=f"slug-{pid}",
        title=f"Implement {category}",
        difficulty=difficulty,
        category=category,
        description=f"A {difficulty} problem about {category}",
        is_retired=False,
        version=1,
        time_limit_ms=time_limit_ms,
        test_cases=[{"input": "1", "expected": "1"}],
        python_template="def solve(): pass",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _seed_submission(
    db: Session,
    user_id: int,
    problem_id: str,
    passed: bool = True,
    time_ms: int = 100,
    is_best: bool = False,
    problem_version: int = 1,
) -> DojoSubmission:
    s = DojoSubmission(
        user_id=user_id,
        problem_id=problem_id,
        code="print('ok')",
        passed=passed,
        time_ms=time_ms,
        is_best=is_best,
        problem_version=problem_version,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ===========================================================================
# TestProblemVersioning
# ===========================================================================

class TestProblemVersioning:
    def test_01_new_problem_starts_at_version_1(self, db_session, client):
        p = _seed_problem(db_session, "ver-1")
        assert p.version == 1

    def test_02_admin_update_non_test_no_version_bump(self, db_session, client):
        _seed_problem(db_session, "ver-2")
        admin = _seed_user(db_session, "ver-admin2@test.com", is_admin=True)
        tok = _login(client, "ver-admin2@test.com")
        r = client.put(
            "/api/admin/problems/ver-2",
            json={"title": "Updated Title"},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        assert r.json()["version"] == 1   # title change → no bump

    def test_03_admin_update_test_cases_bumps_version(self, db_session, client):
        _seed_problem(db_session, "ver-3")
        _seed_user(db_session, "ver-admin3@test.com", is_admin=True)
        tok = _login(client, "ver-admin3@test.com")
        r = client.put(
            "/api/admin/problems/ver-3",
            json={"test_cases": [{"input": "2", "expected": "2"}]},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        assert r.json()["version"] == 2   # test_cases changed → bump to 2

    def test_04_submission_records_problem_version(self, db_session, client):
        p = _seed_problem(db_session, "ver-4")
        p.version = 3
        db_session.commit()
        u = _seed_user(db_session, "ver-user4@test.com")
        s = _seed_submission(db_session, u.id, "ver-4", problem_version=3)
        assert s.problem_version == 3

    def test_05_version_included_in_admin_problem_response(self, db_session, client):
        _seed_problem(db_session, "ver-5")
        _seed_user(db_session, "ver-admin5@test.com", is_admin=True)
        tok = _login(client, "ver-admin5@test.com")
        r = client.get("/api/admin/problems", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        probs = r.json()["problems"]
        ver5 = next((p for p in probs if p["id"] == "ver-5"), None)
        assert ver5 is not None
        assert "version" in ver5


# ===========================================================================
# TestPerProblemTimeout
# ===========================================================================

class TestPerProblemTimeout:
    def test_06_default_time_limit_is_null(self, db_session, client):
        p = _seed_problem(db_session, "tl-1")
        assert p.time_limit_ms is None

    def test_07_admin_create_with_time_limit(self, db_session, client):
        _seed_user(db_session, "tl-admin1@test.com", is_admin=True)
        tok = _login(client, "tl-admin1@test.com")
        r = client.post(
            "/api/admin/problems",
            json={
                "id": "tl-new-1", "slug": "tl-new-slug-1",
                "title": "TL Test", "difficulty": "Hard",
                "category": "ML", "description": "...",
                "time_limit_ms": 25000,
            },
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 201
        assert r.json()["time_limit_ms"] == 25000

    def test_08_admin_update_time_limit(self, db_session, client):
        _seed_problem(db_session, "tl-2")
        _seed_user(db_session, "tl-admin2@test.com", is_admin=True)
        tok = _login(client, "tl-admin2@test.com")
        r = client.put(
            "/api/admin/problems/tl-2",
            json={"time_limit_ms": 20000},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        assert r.json()["time_limit_ms"] == 20000

    def test_09_problem_list_includes_time_limit_ms(self, db_session, client):
        """GET /api/problems response includes time_limit_ms for each problem."""
        _seed_problem(db_session, "tl-3", time_limit_ms=25000)
        r = client.get("/api/problems")
        assert r.status_code == 200
        tl3 = next((p for p in r.json() if p["id"] == "tl-3"), None)
        assert tl3 is not None
        assert tl3["time_limit_ms"] == 25000

    def test_10_filter_problems_by_difficulty(self, db_session, client):
        """GET /api/problems?difficulty=Easy returns only Easy problems."""
        _seed_problem(db_session, "tl-easy", difficulty="Easy", category="ML")
        _seed_problem(db_session, "tl-hard", difficulty="Hard", category="ML")
        r = client.get("/api/problems?difficulty=Easy")
        assert r.status_code == 200
        ids = [p["id"] for p in r.json()]
        assert "tl-easy" in ids
        assert "tl-hard" not in ids


# ===========================================================================
# TestBestSubmission
# ===========================================================================

class TestBestSubmission:
    def test_11_first_passing_submission_is_best(self, db_session, client):
        u = _seed_user(db_session, "best1@test.com")
        p = _seed_problem(db_session, "best-1")
        from backend.tasks.dojo_tasks import _mark_best_submission
        s = _seed_submission(db_session, u.id, "best-1", passed=True, time_ms=200)
        _mark_best_submission(db_session, u.id, "best-1", s.id, 200)
        db_session.refresh(s)
        assert s.is_best is True

    def test_12_faster_submission_replaces_best(self, db_session, client):
        u = _seed_user(db_session, "best2@test.com")
        _seed_problem(db_session, "best-2")
        from backend.tasks.dojo_tasks import _mark_best_submission
        s1 = _seed_submission(db_session, u.id, "best-2", passed=True, time_ms=500, is_best=True)
        s2 = _seed_submission(db_session, u.id, "best-2", passed=True, time_ms=100)
        _mark_best_submission(db_session, u.id, "best-2", s2.id, 100)
        db_session.refresh(s1)
        db_session.refresh(s2)
        assert s1.is_best is False
        assert s2.is_best is True

    def test_13_slower_submission_does_not_replace_best(self, db_session, client):
        u = _seed_user(db_session, "best3@test.com")
        _seed_problem(db_session, "best-3")
        from backend.tasks.dojo_tasks import _mark_best_submission
        s1 = _seed_submission(db_session, u.id, "best-3", passed=True, time_ms=50, is_best=True)
        s2 = _seed_submission(db_session, u.id, "best-3", passed=True, time_ms=999)
        _mark_best_submission(db_session, u.id, "best-3", s2.id, 999)
        db_session.refresh(s1)
        db_session.refresh(s2)
        assert s1.is_best is True   # original still best
        assert s2.is_best is False

    def test_14_is_best_visible_in_submission_history(self, db_session, client):
        u = _seed_user(db_session, "best4@test.com")
        _seed_problem(db_session, "best-4")
        _seed_submission(db_session, u.id, "best-4", passed=True, time_ms=100, is_best=True)
        tok = _login(client, "best4@test.com")
        r = client.get("/api/dojo/submissions?problem_id=best-4", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        subs = r.json()["submissions"]
        assert any(s["is_best"] for s in subs)


# ===========================================================================
# TestAcceptanceRate
# ===========================================================================

class TestAcceptanceRate:
    def test_15_acceptance_rate_none_initially(self, db_session, client):
        p = _seed_problem(db_session, "ar-1")
        assert p.acceptance_rate is None

    def test_16_acceptance_rate_updated_after_submission(self, db_session, client):
        u = _seed_user(db_session, "ar2@test.com")
        _seed_problem(db_session, "ar-2")
        _seed_submission(db_session, u.id, "ar-2", passed=True)
        from backend.tasks.dojo_tasks import _update_acceptance_rate
        _update_acceptance_rate(db_session, "ar-2")
        p = db_session.query(Problem).filter_by(id="ar-2").first()
        assert float(p.acceptance_rate) == 1.0

    def test_17_partial_acceptance_rate(self, db_session, client):
        u1 = _seed_user(db_session, "ar3a@test.com")
        u2 = _seed_user(db_session, "ar3b@test.com")
        _seed_problem(db_session, "ar-3")
        _seed_submission(db_session, u1.id, "ar-3", passed=True)
        _seed_submission(db_session, u2.id, "ar-3", passed=False)
        from backend.tasks.dojo_tasks import _update_acceptance_rate
        _update_acceptance_rate(db_session, "ar-3")
        p = db_session.query(Problem).filter_by(id="ar-3").first()
        assert float(p.acceptance_rate) == 0.5

    def test_18_acceptance_rate_in_admin_response(self, db_session, client):
        u = _seed_user(db_session, "ar4u@test.com")
        _seed_user(db_session, "ar4a@test.com", is_admin=True)
        _seed_problem(db_session, "ar-4")
        _seed_submission(db_session, u.id, "ar-4", passed=True)
        from backend.tasks.dojo_tasks import _update_acceptance_rate
        _update_acceptance_rate(db_session, "ar-4")
        tok = _login(client, "ar4a@test.com")
        r = client.get("/api/admin/problems", headers={"Authorization": f"Bearer {tok}"})
        probs = r.json()["problems"]
        ar4 = next((p for p in probs if p["id"] == "ar-4"), None)
        assert ar4 is not None
        assert ar4["acceptance_rate"] == 1.0

    def test_19_acceptance_rate_in_public_problem_list(self, db_session, client):
        u = _seed_user(db_session, "ar5u@test.com")
        _seed_problem(db_session, "ar-5")
        _seed_submission(db_session, u.id, "ar-5", passed=True)
        from backend.tasks.dojo_tasks import _update_acceptance_rate
        _update_acceptance_rate(db_session, "ar-5")
        r = client.get("/api/problems")
        assert r.status_code == 200
        probs = r.json()
        ar5 = next((p for p in probs if p["id"] == "ar-5"), None)
        assert ar5 is not None
        assert ar5["acceptance_rate"] == 1.0


# ===========================================================================
# TestSubmissionHistory
# ===========================================================================

class TestSubmissionHistory:
    def test_20_requires_auth(self, db_session, client):
        r = client.get("/api/dojo/submissions")
        assert r.status_code == 401

    def test_21_returns_own_submissions(self, db_session, client):
        u = _seed_user(db_session, "sh1@test.com")
        _seed_problem(db_session, "sh-1")
        _seed_submission(db_session, u.id, "sh-1")
        tok = _login(client, "sh1@test.com")
        r = client.get("/api/dojo/submissions", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "submissions" in body
        assert body["total"] >= 1

    def test_22_other_users_submissions_hidden(self, db_session, client):
        u1 = _seed_user(db_session, "sh2a@test.com")
        u2 = _seed_user(db_session, "sh2b@test.com")
        _seed_problem(db_session, "sh-2")
        _seed_submission(db_session, u1.id, "sh-2")  # u1's submission
        tok = _login(client, "sh2b@test.com")  # log in as u2
        r = client.get("/api/dojo/submissions", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        ids = [s["problem_id"] for s in r.json()["submissions"]]
        # u2 has no submissions for sh-2 (u1 does)
        # We can't assert empty list since u2 may have others, but total for sh-2 should be 0
        r2 = client.get("/api/dojo/submissions?problem_id=sh-2", headers={"Authorization": f"Bearer {tok}"})
        assert r2.json()["total"] == 0

    def test_23_filter_by_problem_id(self, db_session, client):
        u = _seed_user(db_session, "sh3@test.com")
        _seed_problem(db_session, "sh-3a")
        _seed_problem(db_session, "sh-3b")
        _seed_submission(db_session, u.id, "sh-3a")
        _seed_submission(db_session, u.id, "sh-3b")
        tok = _login(client, "sh3@test.com")
        r = client.get("/api/dojo/submissions?problem_id=sh-3a", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] >= 1
        assert all(s["problem_id"] == "sh-3a" for s in body["submissions"])

    def test_24_pagination_limit(self, db_session, client):
        u = _seed_user(db_session, "sh4@test.com")
        _seed_problem(db_session, "sh-4")
        for _ in range(5):
            _seed_submission(db_session, u.id, "sh-4")
        tok = _login(client, "sh4@test.com")
        r = client.get("/api/dojo/submissions?limit=2", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        assert len(r.json()["submissions"]) <= 2

    def test_25_pagination_offset(self, db_session, client):
        u = _seed_user(db_session, "sh5@test.com")
        _seed_problem(db_session, "sh-5")
        for _ in range(4):
            _seed_submission(db_session, u.id, "sh-5")
        tok = _login(client, "sh5@test.com")
        r_all = client.get("/api/dojo/submissions?problem_id=sh-5", headers={"Authorization": f"Bearer {tok}"})
        r_off = client.get("/api/dojo/submissions?problem_id=sh-5&offset=2", headers={"Authorization": f"Bearer {tok}"})
        assert r_off.status_code == 200
        total = r_all.json()["total"]
        remaining = len(r_off.json()["submissions"])
        assert remaining == max(0, min(total - 2, 20))

    def test_26_submissions_include_is_best_and_version(self, db_session, client):
        u = _seed_user(db_session, "sh6@test.com")
        _seed_problem(db_session, "sh-6")
        _seed_submission(db_session, u.id, "sh-6", is_best=True, problem_version=2)
        tok = _login(client, "sh6@test.com")
        r = client.get("/api/dojo/submissions?problem_id=sh-6", headers={"Authorization": f"Bearer {tok}"})
        s = r.json()["submissions"][0]
        assert "is_best" in s
        assert "problem_version" in s
        assert s["problem_version"] == 2


# ===========================================================================
# TestAdminProblemCRUD  — Sprint B already done; test the new Sprint G fields
# ===========================================================================

class TestAdminProblemCRUD:
    def test_27_create_problem_returns_version_and_timeout(self, db_session, client):
        _seed_user(db_session, "crud1@test.com", is_admin=True)
        tok = _login(client, "crud1@test.com")
        r = client.post(
            "/api/admin/problems",
            json={
                "id": "crud-g-1", "slug": "crud-g-slug-1",
                "title": "Attention", "difficulty": "Hard",
                "category": "Transformers", "description": "Impl attention",
                "time_limit_ms": 20000,
            },
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["version"] == 1
        assert body["time_limit_ms"] == 20000
        assert body["acceptance_rate"] is None

    def test_28_update_non_test_case_no_version_bump(self, db_session, client):
        _seed_problem(db_session, "crud-g-2")
        _seed_user(db_session, "crud2@test.com", is_admin=True)
        tok = _login(client, "crud2@test.com")
        r = client.put(
            "/api/admin/problems/crud-g-2",
            json={"difficulty": "Medium"},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        assert r.json()["version"] == 1

    def test_29_update_test_cases_bumps_version(self, db_session, client):
        _seed_problem(db_session, "crud-g-3")
        _seed_user(db_session, "crud3@test.com", is_admin=True)
        tok = _login(client, "crud3@test.com")
        r = client.put(
            "/api/admin/problems/crud-g-3",
            json={"test_cases": [{"input": "x", "expected": "x"}]},
            headers={"Authorization": f"Bearer {tok}"},
        )
        assert r.status_code == 200
        assert r.json()["version"] == 2


# ===========================================================================
# TestLeaderboardCategory
# ===========================================================================

class TestLeaderboardCategory:
    def test_30_no_category_returns_all(self, db_session, client):
        r = client.get("/api/leaderboard")
        assert r.status_code == 200
        body = r.json()
        assert "leaders" in body
        assert body["category"] is None

    def test_31_invalid_period_rejected(self, db_session, client):
        r = client.get("/api/leaderboard?period=decade")
        assert r.status_code == 400

    def test_32_category_filter_empty_when_none_solved(self, db_session, client):
        r = client.get("/api/leaderboard?category=NonExistentCategory999")
        assert r.status_code == 200
        assert r.json()["total_ranked"] == 0

    def test_33_category_filter_includes_solvers(self, db_session, client):
        u = _seed_user(db_session, "lb-cat1@test.com")
        u.points = 500
        db_session.commit()
        _seed_problem(db_session, "lb-cat-prob-1", category="SpecialCategory")
        _seed_submission(db_session, u.id, "lb-cat-prob-1", passed=True)
        r = client.get("/api/leaderboard?category=SpecialCategory")
        assert r.status_code == 200
        body = r.json()
        user_ids = [l["user_id"] for l in body["leaders"]]
        assert u.id in user_ids

    def test_34_category_excludes_non_solvers(self, db_session, client):
        u_solver = _seed_user(db_session, "lb-cat2a@test.com")
        u_other  = _seed_user(db_session, "lb-cat2b@test.com")
        u_solver.points = 300
        u_other.points  = 300
        db_session.commit()
        _seed_problem(db_session, "lb-cat-prob-2", category="RareCategoryXYZ")
        _seed_submission(db_session, u_solver.id, "lb-cat-prob-2", passed=True)
        # u_other has NOT solved any RareCategoryXYZ problem
        r = client.get("/api/leaderboard?category=RareCategoryXYZ")
        user_ids = [l["user_id"] for l in r.json()["leaders"]]
        assert u_solver.id in user_ids
        assert u_other.id not in user_ids

    def test_35_category_response_has_category_field(self, db_session, client):
        r = client.get("/api/leaderboard?category=Transformers")
        assert r.status_code == 200
        assert r.json()["category"] == "Transformers"


# ===========================================================================
# TestStdoutTruncation
# ===========================================================================

class TestStdoutTruncation:
    def test_36_output_limit_in_piston_payload(self):
        """Verify execute_python includes output_limit in the Piston payload."""
        from backend.services import dojo_execution_service as svc
        captured_payload = {}

        async def fake_call_piston(payload):
            captured_payload.update(payload)
            return {"run": {"stdout": "ok", "stderr": "", "code": 0, "time": 10}}

        import asyncio
        with patch.object(svc, "_call_piston", side_effect=fake_call_piston):
            asyncio.run(svc.execute_python("print('ok')"))

        assert "output_limit" in captured_payload
        assert captured_payload["output_limit"] == svc.OUTPUT_LIMIT_BYTES  # 65536

    def test_37_custom_timeout_in_piston_payload(self):
        """Verify per-problem time_limit_ms overrides global run_timeout."""
        from backend.services import dojo_execution_service as svc
        captured_payload = {}

        async def fake_call_piston(payload):
            captured_payload.update(payload)
            return {"run": {"stdout": "", "stderr": "", "code": 0, "time": 5}}

        import asyncio
        with patch.object(svc, "_call_piston", side_effect=fake_call_piston):
            asyncio.run(svc.execute_python("pass", run_timeout_ms=25000))

        assert captured_payload["run_timeout"] == 25000
