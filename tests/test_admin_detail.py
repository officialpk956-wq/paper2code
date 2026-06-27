"""
tests/test_admin_detail.py

Tests for 3 missing admin endpoints:
  GET /api/admin/users/{id}  — rich user detail (xp_events, papers, submissions lists)
  GET /api/admin/papers/moderation-queue
  GET /api/admin/xp-events
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Paper, DojoSubmission, XPEvent
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "DetailPass999!"


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


def _seed_paper(db: Session, title: str, uploaded_by: int = None,
                is_flagged: bool = False, flag_reason: str = None) -> Paper:
    existing = db.query(Paper).filter_by(title=title).first()
    if existing:
        return existing
    p = Paper(
        title=title,
        visibility="public",
        uploaded_by=uploaded_by,
        is_flagged=is_flagged,
        flag_reason=flag_reason,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _seed_submission(db: Session, user_id: int, problem_id: str = "p-test",
                     passed: bool = True) -> DojoSubmission:
    s = DojoSubmission(
        user_id=user_id,
        problem_id=problem_id,
        code="print('hi')",
        passed=passed,
        stdout="hi\n",
        stderr="",
        time_ms=100,
        is_best=passed,
        problem_version=1,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _seed_xp_event(db: Session, user_id: int, action: str = "submission",
                   amount: int = 50) -> XPEvent:
    e = XPEvent(user_id=user_id, action=action, amount=amount)
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


# ---------------------------------------------------------------------------
# TestAdminUserDetailRich — GET /api/admin/users/{id}
# ---------------------------------------------------------------------------

class TestAdminUserDetailRich:

    def test_01_xp_events_included(self, client, db_session):
        admin = _seed_user(db_session, "ad01rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt01rich@t.com")
        _seed_xp_event(db_session, target.id, action="submission", amount=50)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "xp_events" in data
        assert isinstance(data["xp_events"], list)
        assert any(e["action"] == "submission" for e in data["xp_events"])

    def test_02_papers_included(self, client, db_session):
        admin = _seed_user(db_session, "ad02rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt02rich@t.com")
        _seed_paper(db_session, "Rich Paper 02", uploaded_by=target.id)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "papers" in data
        assert any(p["title"] == "Rich Paper 02" for p in data["papers"])

    def test_03_submissions_included(self, client, db_session):
        admin = _seed_user(db_session, "ad03rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt03rich@t.com")
        _seed_submission(db_session, target.id, problem_id="p-rich-03")
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "submissions" in data
        assert any(s["problem_id"] == "p-rich-03" for s in data["submissions"])

    def test_04_empty_lists_when_no_activity(self, client, db_session):
        admin = _seed_user(db_session, "ad04rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt04rich@t.com")
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert data["xp_events"] == []
        assert data["papers"] == []
        assert data["submissions"] == []

    def test_05_stats_still_present(self, client, db_session):
        admin = _seed_user(db_session, "ad05rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt05rich@t.com")
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "stats" in data
        assert "papers_uploaded" in data["stats"]
        assert "dojo_submissions" in data["stats"]

    def test_06_xp_event_fields(self, client, db_session):
        admin = _seed_user(db_session, "ad06rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt06rich@t.com")
        _seed_xp_event(db_session, target.id, action="domain_complete", amount=500)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        data = r.json()
        ev = next(e for e in data["xp_events"] if e["action"] == "domain_complete")
        assert ev["amount"] == 500
        assert "id" in ev
        assert "created_at" in ev

    def test_07_submission_fields(self, client, db_session):
        admin = _seed_user(db_session, "ad07rich@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt07rich@t.com")
        _seed_submission(db_session, target.id, problem_id="p-fields-07", passed=False)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        data = r.json()
        sub = next(s for s in data["submissions"] if s["problem_id"] == "p-fields-07")
        assert sub["passed"] is False
        assert "time_ms" in sub
        assert "is_best" in sub

    def test_08_non_admin_forbidden(self, client, db_session):
        _seed_user(db_session, "ad08rich@t.com", is_admin=True)
        user = _seed_user(db_session, "usr08rich@t.com")
        target = _seed_user(db_session, "tgt08rich@t.com")
        token = _login(client, user.email)

        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 403

    def test_09_missing_user_404(self, client, db_session):
        admin = _seed_user(db_session, "ad09rich@t.com", is_admin=True)
        token = _login(client, admin.email)

        r = client.get("/api/admin/users/99998", headers=_auth(token))
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# TestAdminModerationQueue — GET /api/admin/papers/moderation-queue
# ---------------------------------------------------------------------------

class TestAdminModerationQueue:

    def test_10_returns_flagged_papers(self, client, db_session):
        admin = _seed_user(db_session, "ad10mq@t.com", is_admin=True)
        _seed_paper(db_session, "Flagged MQ 10", is_flagged=True,
                    flag_reason="spam")
        token = _login(client, admin.email)

        r = client.get("/api/admin/papers/moderation-queue", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "papers" in data
        titles = [p["title"] for p in data["papers"]]
        assert "Flagged MQ 10" in titles

    def test_11_excludes_clean_papers(self, client, db_session):
        admin = _seed_user(db_session, "ad11mq@t.com", is_admin=True)
        _seed_paper(db_session, "Clean Paper MQ 11", is_flagged=False)
        token = _login(client, admin.email)

        r = client.get("/api/admin/papers/moderation-queue", headers=_auth(token))
        assert r.status_code == 200
        titles = [p["title"] for p in r.json()["papers"]]
        assert "Clean Paper MQ 11" not in titles

    def test_12_empty_when_nothing_flagged(self, client, db_session):
        admin = _seed_user(db_session, "ad12mq@t.com", is_admin=True)
        token = _login(client, admin.email)

        r = client.get("/api/admin/papers/moderation-queue", headers=_auth(token))
        assert r.status_code == 200
        # total may be > 0 from other tests but endpoint must work
        assert isinstance(r.json()["papers"], list)

    def test_13_non_admin_forbidden(self, client, db_session):
        _seed_user(db_session, "ad13mq@t.com", is_admin=True)
        user = _seed_user(db_session, "usr13mq@t.com")
        token = _login(client, user.email)

        r = client.get("/api/admin/papers/moderation-queue", headers=_auth(token))
        assert r.status_code == 403

    def test_14_paper_fields_in_queue(self, client, db_session):
        admin = _seed_user(db_session, "ad14mq@t.com", is_admin=True)
        _seed_paper(db_session, "Fields Test MQ 14", is_flagged=True,
                    flag_reason="copyright")
        token = _login(client, admin.email)

        r = client.get("/api/admin/papers/moderation-queue", headers=_auth(token))
        data = r.json()
        paper = next((p for p in data["papers"] if p["title"] == "Fields Test MQ 14"), None)
        assert paper is not None
        assert paper["flag_reason"] == "copyright"
        assert "visibility" in paper
        assert "uploaded_by" in paper

    def test_15_pagination_returns_subset(self, client, db_session):
        admin = _seed_user(db_session, "ad15mq@t.com", is_admin=True)
        for i in range(3):
            _seed_paper(db_session, f"Paged MQ 15-{i}", is_flagged=True)
        token = _login(client, admin.email)

        r = client.get("/api/admin/papers/moderation-queue?limit=1&page=1",
                       headers=_auth(token))
        assert r.status_code == 200
        assert len(r.json()["papers"]) == 1


# ---------------------------------------------------------------------------
# TestAdminXPEvents — GET /api/admin/xp-events
# ---------------------------------------------------------------------------

class TestAdminXPEvents:

    def test_16_returns_events(self, client, db_session):
        admin = _seed_user(db_session, "ad16xp@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt16xp@t.com")
        _seed_xp_event(db_session, target.id, action="xp_test_16", amount=10)
        token = _login(client, admin.email)

        r = client.get("/api/admin/xp-events", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "events" in data
        actions = [e["action"] for e in data["events"]]
        assert "xp_test_16" in actions

    def test_17_filter_by_user_id(self, client, db_session):
        admin = _seed_user(db_session, "ad17xp@t.com", is_admin=True)
        u1 = _seed_user(db_session, "u1_17xp@t.com")
        u2 = _seed_user(db_session, "u2_17xp@t.com")
        _seed_xp_event(db_session, u1.id, action="u1_action_17", amount=10)
        _seed_xp_event(db_session, u2.id, action="u2_action_17", amount=20)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/xp-events?user_id={u1.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        uids = {e["user_id"] for e in data["events"]}
        assert uids == {u1.id}
        actions = [e["action"] for e in data["events"]]
        assert "u1_action_17" in actions
        assert "u2_action_17" not in actions

    def test_18_non_admin_forbidden(self, client, db_session):
        _seed_user(db_session, "ad18xp@t.com", is_admin=True)
        user = _seed_user(db_session, "usr18xp@t.com")
        token = _login(client, user.email)

        r = client.get("/api/admin/xp-events", headers=_auth(token))
        assert r.status_code == 403

    def test_19_event_fields(self, client, db_session):
        admin = _seed_user(db_session, "ad19xp@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt19xp@t.com")
        _seed_xp_event(db_session, target.id, action="fields_test_19", amount=75)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/xp-events?user_id={target.id}",
                       headers=_auth(token))
        data = r.json()
        ev = next(e for e in data["events"] if e["action"] == "fields_test_19")
        assert ev["amount"] == 75
        assert ev["user_id"] == target.id
        assert "id" in ev
        assert "entity_id" in ev
        assert "created_at" in ev

    def test_20_pagination(self, client, db_session):
        admin = _seed_user(db_session, "ad20xp@t.com", is_admin=True)
        target = _seed_user(db_session, "tgt20xp@t.com")
        for i in range(5):
            _seed_xp_event(db_session, target.id, action=f"page_test_20_{i}", amount=1)
        token = _login(client, admin.email)

        r = client.get(f"/api/admin/xp-events?user_id={target.id}&limit=2&page=1",
                       headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert len(data["events"]) == 2
        assert data["total"] >= 5

    def test_21_unauthenticated_rejected(self, client, db_session):
        r = client.get("/api/admin/xp-events")
        assert r.status_code in (401, 403)
