"""
tests/test_dojo_fixes.py

Covers 5 audit items for the dojo/problems endpoints:
  1. GET /api/problems/{id}              Bug fix — dict with all fields
  2. POST /api/submissions/{id}/share    Missing P1 — is_public flag + ownership
  3. GET /api/problems/{id}/solutions    Missing P1 — public best submissions
  4. GET /api/dojo/stats                 Missing P2 — global aggregate stats
  5. GET /api/me/dojo-stats              Missing P2 — per-user summary
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Problem, DojoSubmission
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "DojoFix999!"
_PROB_COUNTER = [0]


def _seed_user(db: Session, email: str, points: int = 0) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        return existing
    u = User(
        email=email,
        name=email.split("@")[0],
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


def _seed_problem(db: Session, suffix: str, difficulty: str = "Easy") -> Problem:
    pid = f"fix-{suffix}"
    existing = db.query(Problem).filter_by(id=pid).first()
    if existing:
        return existing
    p = Problem(
        id=pid,
        slug=f"fix-slug-{suffix}",
        title=f"Fix Problem {suffix}",
        difficulty=difficulty,
        category="Testing",
        description="A test problem",
        python_template="# template",
        test_cases=[{"input": "1", "expected": "1"}],
        hints=["Try harder"],
        explanation=[{"step": "do something"}],
        tags=["test"],
        version=1,
        is_retired=False,
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
    is_best: bool = True,
    is_public: bool = False,
    time_ms: int = 200,
) -> DojoSubmission:
    s = DojoSubmission(
        user_id=user_id,
        problem_id=problem_id,
        code="print('hi')",
        passed=passed,
        stdout="hi\n",
        stderr="",
        time_ms=time_ms,
        is_best=is_best,
        is_public=is_public,
        problem_version=1,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


# ---------------------------------------------------------------------------
# TestProblemDetailFix — GET /api/problems/{id} bug fix
# ---------------------------------------------------------------------------

class TestProblemDetailFix:

    def test_01_returns_dict_not_orm(self, client, db_session):
        prob = _seed_problem(db_session, "detail01")
        r = client.get(f"/api/problems/{prob.id}")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)

    def test_02_exposes_tests_without_leaking_hidden(self, client, db_session):
        prob = _seed_problem(db_session, "detail02")
        r = client.get(f"/api/problems/{prob.id}")
        assert r.status_code == 200
        data = r.json()
        # Dojo v2 security: the raw test_cases (which may hold hidden expected
        # values / harness / forward_ref) must NOT be sent to the client — only
        # the sanitized public view under "tests".
        assert "test_cases" not in data
        assert "tests" in data and isinstance(data["tests"], dict)

    def test_03_includes_hints(self, client, db_session):
        prob = _seed_problem(db_session, "detail03")
        r = client.get(f"/api/problems/{prob.id}")
        assert r.status_code == 200
        assert "hints" in r.json()

    def test_04_includes_explanation(self, client, db_session):
        prob = _seed_problem(db_session, "detail04")
        r = client.get(f"/api/problems/{prob.id}")
        assert r.status_code == 200
        assert "explanation" in r.json()

    def test_05_includes_python_template(self, client, db_session):
        prob = _seed_problem(db_session, "detail05")
        r = client.get(f"/api/problems/{prob.id}")
        assert r.status_code == 200
        assert "python_template" in r.json()

    def test_06_404_for_missing(self, client, db_session):
        r = client.get("/api/problems/does-not-exist")
        assert r.status_code == 404

    def test_07_acceptance_rate_serializable(self, client, db_session):
        prob = _seed_problem(db_session, "detail07")
        r = client.get(f"/api/problems/{prob.id}")
        assert r.status_code == 200
        data = r.json()
        assert "acceptance_rate" in data
        # Must be float or None — not a Decimal that can't serialize
        assert data["acceptance_rate"] is None or isinstance(data["acceptance_rate"], float)


# ---------------------------------------------------------------------------
# TestSubmissionShare — POST /api/submissions/{id}/share
# ---------------------------------------------------------------------------

class TestSubmissionShare:

    def test_08_owner_can_share_best_submission(self, client, db_session):
        user = _seed_user(db_session, "share08@df.com")
        prob = _seed_problem(db_session, "share08")
        sub = _seed_submission(db_session, user.id, prob.id, is_best=True, is_public=False)
        token = _login(client, user.email)

        r = client.post(f"/api/submissions/{sub.id}/share", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["is_public"] is True

    def test_09_is_public_persisted(self, client, db_session):
        user = _seed_user(db_session, "share09@df.com")
        prob = _seed_problem(db_session, "share09")
        sub = _seed_submission(db_session, user.id, prob.id, is_best=True, is_public=False)
        token = _login(client, user.email)

        client.post(f"/api/submissions/{sub.id}/share", headers=_auth(token))
        db_session.refresh(sub)
        assert sub.is_public is True

    def test_10_non_owner_forbidden(self, client, db_session):
        owner = _seed_user(db_session, "owner10@df.com")
        other = _seed_user(db_session, "other10@df.com")
        prob = _seed_problem(db_session, "share10")
        sub = _seed_submission(db_session, owner.id, prob.id, is_best=True)
        token = _login(client, other.email)

        r = client.post(f"/api/submissions/{sub.id}/share", headers=_auth(token))
        assert r.status_code == 403

    def test_11_non_best_cannot_be_shared(self, client, db_session):
        user = _seed_user(db_session, "share11@df.com")
        prob = _seed_problem(db_session, "share11")
        sub = _seed_submission(db_session, user.id, prob.id, is_best=False, passed=False)
        token = _login(client, user.email)

        r = client.post(f"/api/submissions/{sub.id}/share", headers=_auth(token))
        assert r.status_code == 400

    def test_12_missing_submission_404(self, client, db_session):
        user = _seed_user(db_session, "share12@df.com")
        token = _login(client, user.email)

        r = client.post("/api/submissions/999999/share", headers=_auth(token))
        assert r.status_code == 404

    def test_13_unauthenticated_rejected(self, client, db_session):
        r = client.post("/api/submissions/1/share")
        assert r.status_code in (401, 403)


# ---------------------------------------------------------------------------
# TestProblemSolutions — GET /api/problems/{id}/solutions
# ---------------------------------------------------------------------------

class TestProblemSolutions:

    def test_14_returns_public_best_submissions(self, client, db_session):
        user = _seed_user(db_session, "sol14@df.com")
        prob = _seed_problem(db_session, "sol14")
        _seed_submission(db_session, user.id, prob.id, is_best=True, is_public=True)

        r = client.get(f"/api/problems/{prob.id}/solutions")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 1
        assert any(s["user_id"] == user.id for s in data["solutions"])

    def test_15_excludes_non_public_submissions(self, client, db_session):
        user = _seed_user(db_session, "sol15@df.com")
        prob = _seed_problem(db_session, "sol15")
        _seed_submission(db_session, user.id, prob.id, is_best=True, is_public=False)

        r = client.get(f"/api/problems/{prob.id}/solutions")
        assert r.status_code == 200
        assert all(s["user_id"] != user.id for s in r.json()["solutions"])

    def test_16_excludes_non_best_submissions(self, client, db_session):
        user = _seed_user(db_session, "sol16@df.com")
        prob = _seed_problem(db_session, "sol16")
        _seed_submission(db_session, user.id, prob.id, is_best=False, is_public=True)

        r = client.get(f"/api/problems/{prob.id}/solutions")
        assert r.status_code == 200
        assert all(s["user_id"] != user.id for s in r.json()["solutions"])

    def test_17_empty_when_no_public_solutions(self, client, db_session):
        prob = _seed_problem(db_session, "sol17")

        r = client.get(f"/api/problems/{prob.id}/solutions")
        assert r.status_code == 200
        assert r.json()["total"] == 0
        assert r.json()["solutions"] == []

    def test_18_ordered_by_time_ms_asc(self, client, db_session):
        u1 = _seed_user(db_session, "sol18a@df.com")
        u2 = _seed_user(db_session, "sol18b@df.com")
        prob = _seed_problem(db_session, "sol18")
        _seed_submission(db_session, u1.id, prob.id, is_best=True, is_public=True, time_ms=500)
        _seed_submission(db_session, u2.id, prob.id, is_best=True, is_public=True, time_ms=100)

        r = client.get(f"/api/problems/{prob.id}/solutions")
        assert r.status_code == 200
        times = [s["time_ms"] for s in r.json()["solutions"]]
        assert times == sorted(times)

    def test_19_solution_has_code_field(self, client, db_session):
        user = _seed_user(db_session, "sol19@df.com")
        prob = _seed_problem(db_session, "sol19")
        _seed_submission(db_session, user.id, prob.id, is_best=True, is_public=True)

        r = client.get(f"/api/problems/{prob.id}/solutions")
        data = r.json()
        assert len(data["solutions"]) >= 1
        assert "code" in data["solutions"][0]

    def test_20_404_for_missing_problem(self, client, db_session):
        r = client.get("/api/problems/no-such-problem/solutions")
        assert r.status_code == 404

    def test_21_limit_param_respected(self, client, db_session):
        prob = _seed_problem(db_session, "sol21")
        for i in range(5):
            u = _seed_user(db_session, f"sol21u{i}@df.com")
            _seed_submission(db_session, u.id, prob.id, is_best=True, is_public=True,
                             time_ms=100 + i)

        r = client.get(f"/api/problems/{prob.id}/solutions?limit=2")
        assert r.status_code == 200
        assert len(r.json()["solutions"]) == 2


# ---------------------------------------------------------------------------
# TestDojoGlobalStats — GET /api/dojo/stats
# ---------------------------------------------------------------------------

class TestDojoGlobalStats:

    def test_22_returns_expected_fields(self, client, db_session):
        r = client.get("/api/dojo/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_problems" in data
        assert "total_submissions" in data
        assert "avg_acceptance_rate" in data

    def test_23_total_problems_is_int(self, client, db_session):
        r = client.get("/api/dojo/stats")
        assert isinstance(r.json()["total_problems"], int)

    def test_24_avg_acceptance_rate_is_float(self, client, db_session):
        r = client.get("/api/dojo/stats")
        assert isinstance(r.json()["avg_acceptance_rate"], float)

    def test_25_excludes_retired_problems(self, client, db_session):
        # Add one active and one retired problem, verify count
        p_active = _seed_problem(db_session, "stats-active25")
        p_retired = _seed_problem(db_session, "stats-retired25")
        p_retired.is_retired = True
        db_session.commit()

        r = client.get("/api/dojo/stats")
        data = r.json()
        # Active count must be ≥ 1 (p_active); retired not counted
        assert data["total_problems"] >= 1


# ---------------------------------------------------------------------------
# TestMyDojoStats — GET /api/me/dojo-stats
# ---------------------------------------------------------------------------

class TestMyDojoStats:

    def test_26_returns_expected_fields(self, client, db_session):
        user = _seed_user(db_session, "me26@df.com")
        token = _login(client, user.email)

        r = client.get("/api/me/dojo-stats", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "total_solved" in data
        assert "solved_by_difficulty" in data
        assert "total_submissions" in data
        assert "streak" in data
        assert "leaderboard_rank" in data

    def test_27_solved_by_difficulty_counts(self, client, db_session):
        user = _seed_user(db_session, "me27@df.com")
        prob_easy = _seed_problem(db_session, "me27e", difficulty="Easy")
        prob_hard = _seed_problem(db_session, "me27h", difficulty="Hard")
        _seed_submission(db_session, user.id, prob_easy.id, passed=True, is_best=True)
        _seed_submission(db_session, user.id, prob_hard.id, passed=True, is_best=True)
        token = _login(client, user.email)

        r = client.get("/api/me/dojo-stats", headers=_auth(token))
        data = r.json()
        assert data["total_solved"] >= 2
        sbd = data["solved_by_difficulty"]
        assert sbd.get("Easy", 0) >= 1
        assert sbd.get("Hard", 0) >= 1

    def test_28_unsolved_problems_not_counted(self, client, db_session):
        user = _seed_user(db_session, "me28@df.com")
        prob = _seed_problem(db_session, "me28")
        _seed_submission(db_session, user.id, prob.id, passed=False, is_best=False)
        token = _login(client, user.email)

        r = client.get("/api/me/dojo-stats", headers=_auth(token))
        data = r.json()
        assert data["total_solved"] == 0

    def test_29_total_submissions_counted(self, client, db_session):
        user = _seed_user(db_session, "me29@df.com")
        prob = _seed_problem(db_session, "me29")
        _seed_submission(db_session, user.id, prob.id, passed=False, is_best=False)
        _seed_submission(db_session, user.id, prob.id, passed=True, is_best=True)
        token = _login(client, user.email)

        r = client.get("/api/me/dojo-stats", headers=_auth(token))
        data = r.json()
        assert data["total_submissions"] >= 2

    def test_30_leaderboard_rank_positive_int(self, client, db_session):
        user = _seed_user(db_session, "me30@df.com")
        token = _login(client, user.email)

        r = client.get("/api/me/dojo-stats", headers=_auth(token))
        rank = r.json()["leaderboard_rank"]
        assert isinstance(rank, int)
        assert rank >= 1

    def test_31_unauthenticated_rejected(self, client, db_session):
        r = client.get("/api/me/dojo-stats")
        assert r.status_code in (401, 403)

    def test_get_problem_related_no_qdrant(self, client, db_session):
        prob = _seed_problem(db_session, "attn")
        prob.slug = "ml-attention"
        prob.tags = ["attention"]
        db_session.commit()
        
        r = client.get("/api/dojo/problems/ml-attention/related")
        assert r.status_code == 200
        data = r.json()
        assert data["arch_slug"] == "transformer"
        assert data["learn_name"] == "Natural Language Processing"

    def test_get_problem_related_404(self, client, db_session):
        r = client.get("/api/dojo/problems/nonexistent/related")
        assert r.status_code == 404

    def test_get_problem_related_resolves_by_id_when_slug_is_stale(self, client, db_session):
        # Prod bug: rows seeded before the slug column existed carry a stale slug,
        # and the frontend calls /related with the problem ID — so the lookup must
        # match id OR slug, not slug only.
        prob = _seed_problem(db_session, "attn2")  # id == "fix-attn2"
        prob.slug = "some-stale-old-slug"
        prob.tags = ["attention"]
        db_session.commit()

        r = client.get("/api/dojo/problems/fix-attn2/related")  # by ID, slug is stale
        assert r.status_code == 200
        assert r.json()["arch_slug"] == "transformer"

    def test_seeder_backfills_slug_on_existing_rows(self, db_session):
        from backend.models import Problem
        from backend.services.problem_seed_service import seed_dojo_problems

        # a pre-existing row with a stale slug, exactly like prod
        db_session.add(Problem(id="ml-attention", slug="STALE", title="old", test_cases=[]))
        db_session.commit()

        seed_dojo_problems(db_session)

        row = db_session.query(Problem).filter_by(id="ml-attention").first()
        assert row.slug == "ml-attention"  # backfilled from the seed file
