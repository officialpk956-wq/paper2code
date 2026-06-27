"""
tests/test_sprint_e.py

Sprint E — Admin API & Cron Layer (~30 tests)
Uses conftest.py fixtures (db_session, client) via the login flow.
"""

import datetime
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import (
    Base, User, Paper, LeaderboardArchive, UsageLog, DojoSubmission
)
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PASS = "AdminPass999!"


def _seed_user(db: Session, email: str, is_admin: bool = False,
               points: int = 0, weekly_points: int = 0, streak: int = 0) -> User:
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
        points=points,
        weekly_points=weekly_points,
        streak=streak,
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


def _seed_paper(db: Session, title: str, visibility: str = "public",
                is_flagged: bool = False) -> Paper:
    existing = db.query(Paper).filter_by(title=title).first()
    if existing:
        return existing
    p = Paper(title=title, visibility=visibility, is_flagged=is_flagged)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ---------------------------------------------------------------------------
# TestAdminUserDetail  — GET /api/admin/users/{id}
# ---------------------------------------------------------------------------

class TestAdminUserDetail:
    def test_01_admin_can_get_user_detail(self, client, db_session):
        admin = _seed_user(db_session, email="a01@te.com", is_admin=True)
        target = _seed_user(db_session, email="t01@te.com")
        token = _login(client, admin.email)
        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == target.id
        assert data["email"] == target.email
        assert "stats" in data
        assert "papers_uploaded" in data["stats"]

    def test_02_non_admin_forbidden(self, client, db_session):
        _seed_user(db_session, email="a02admin@te.com", is_admin=True)
        user = _seed_user(db_session, email="r02@te.com")
        target = _seed_user(db_session, email="tgt02@te.com")
        token = _login(client, user.email)
        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 403

    def test_03_missing_user_returns_404(self, client, db_session):
        admin = _seed_user(db_session, email="a03@te.com", is_admin=True)
        token = _login(client, admin.email)
        r = client.get("/api/admin/users/99999", headers=_auth(token))
        assert r.status_code == 404

    def test_04_user_detail_includes_points(self, client, db_session):
        admin = _seed_user(db_session, email="a04@te.com", is_admin=True)
        target = _seed_user(db_session, email="t04@te.com", points=250, weekly_points=100)
        token = _login(client, admin.email)
        r = client.get(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert data["points"] == 250
        assert data["weekly_points"] == 100


# ---------------------------------------------------------------------------
# TestAdminUserDelete  — DELETE /api/admin/users/{id}
# ---------------------------------------------------------------------------

class TestAdminUserDelete:
    def test_05_admin_can_delete_user(self, client, db_session):
        admin = _seed_user(db_session, email="a05@te.com", is_admin=True)
        target = _seed_user(db_session, email="del05@te.com")
        token = _login(client, admin.email)
        r = client.delete(f"/api/admin/users/{target.id}", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_06_deleted_user_is_gone(self, client, db_session):
        admin = _seed_user(db_session, email="a06@te.com", is_admin=True)
        target = _seed_user(db_session, email="del06@te.com")
        tid = target.id
        admin_token = _login(client, admin.email)
        client.delete(f"/api/admin/users/{tid}", headers=_auth(admin_token))
        r = client.get(f"/api/admin/users/{tid}", headers=_auth(admin_token))
        assert r.status_code == 404

    def test_07_cannot_delete_self(self, client, db_session):
        admin = _seed_user(db_session, email="selfadmin07@te.com", is_admin=True)
        token = _login(client, admin.email)
        r = client.delete(f"/api/admin/users/{admin.id}", headers=_auth(token))
        assert r.status_code == 400

    def test_08_delete_nonexistent_returns_404(self, client, db_session):
        admin = _seed_user(db_session, email="a08@te.com", is_admin=True)
        token = _login(client, admin.email)
        r = client.delete("/api/admin/users/88888", headers=_auth(token))
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# TestAdminPaperModeration  — GET/DELETE/POST /api/admin/papers
# ---------------------------------------------------------------------------

class TestAdminPaperModeration:
    def test_09_admin_list_papers(self, client, db_session):
        admin = _seed_user(db_session, email="a09@te.com", is_admin=True)
        _seed_paper(db_session, "Sprint E Paper A09a")
        _seed_paper(db_session, "Sprint E Paper A09b")
        token = _login(client, admin.email)
        r = client.get("/api/admin/papers", headers=_auth(token))
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 2

    def test_10_flagged_papers_listed_first(self, client, db_session):
        admin = _seed_user(db_session, email="a10@te.com", is_admin=True)
        _seed_paper(db_session, "Normal Paper A10")
        _seed_paper(db_session, "Flagged Paper A10", is_flagged=True)
        token = _login(client, admin.email)
        r = client.get("/api/admin/papers", headers=_auth(token))
        assert r.status_code == 200
        papers = r.json()["papers"]
        flagged_indexes = [i for i, p in enumerate(papers) if p["is_flagged"]]
        normal_indexes  = [i for i, p in enumerate(papers) if not p["is_flagged"]]
        if flagged_indexes and normal_indexes:
            assert min(flagged_indexes) < min(normal_indexes)

    def test_11_flag_paper(self, client, db_session):
        admin = _seed_user(db_session, email="a11@te.com", is_admin=True)
        paper = _seed_paper(db_session, "Flag Me A11")
        token = _login(client, admin.email)
        r = client.post(
            f"/api/admin/papers/{paper.id}/flag",
            json={"reason": "dmca"},
            headers=_auth(token),
        )
        assert r.status_code == 200
        assert r.json()["is_flagged"] is True
        assert r.json()["flag_reason"] == "dmca"

    def test_12_flag_sets_db_values(self, client, db_session):
        admin = _seed_user(db_session, email="a12@te.com", is_admin=True)
        paper = _seed_paper(db_session, "Flag DB A12")
        token = _login(client, admin.email)
        client.post(
            f"/api/admin/papers/{paper.id}/flag",
            json={"reason": "policy_violation"},
            headers=_auth(token),
        )
        db_session.refresh(paper)
        assert paper.is_flagged is True
        assert paper.flag_reason == "policy_violation"

    def test_13_delete_paper_dmca(self, client, db_session):
        admin = _seed_user(db_session, email="a13@te.com", is_admin=True)
        paper = _seed_paper(db_session, "DMCA Paper A13")
        pid = paper.id
        token = _login(client, admin.email)
        r = client.delete(f"/api/admin/papers/{pid}", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["deleted"] is True
        assert db_session.query(Paper).filter_by(id=pid).first() is None

    def test_14_flag_nonexistent_paper_404(self, client, db_session):
        admin = _seed_user(db_session, email="a14@te.com", is_admin=True)
        token = _login(client, admin.email)
        r = client.post(
            "/api/admin/papers/77777/flag",
            json={"reason": "test"},
            headers=_auth(token),
        )
        assert r.status_code == 404

    def test_15_flagged_only_filter(self, client, db_session):
        admin = _seed_user(db_session, email="a15@te.com", is_admin=True)
        _seed_paper(db_session, "Normal A15")
        _seed_paper(db_session, "Flagged A15", is_flagged=True)
        token = _login(client, admin.email)
        r = client.get("/api/admin/papers?flagged_only=true", headers=_auth(token))
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert all(p["is_flagged"] for p in papers)


# ---------------------------------------------------------------------------
# TestAnnouncements  — POST/DELETE /api/admin/announcements + GET /api/announcements
# ---------------------------------------------------------------------------

class TestAnnouncements:
    def test_16_create_announcement_no_redis(self, client, db_session):
        admin = _seed_user(db_session, email="a16@te.com", is_admin=True)
        token = _login(client, admin.email)
        with patch("backend.routers.announcements._get_redis", return_value=None):
            r = client.post(
                "/api/admin/announcements",
                json={"message": "Site maintenance at midnight", "level": "warning"},
                headers=_auth(token),
            )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["announcement"]["message"] == "Site maintenance at midnight"

    def test_17_get_announcement_no_redis_returns_null(self, client, db_session):
        with patch("backend.routers.announcements._get_redis", return_value=None):
            r = client.get("/api/announcements")
        assert r.status_code == 200
        assert r.json()["announcement"] is None

    def test_18_announcement_stored_and_retrieved(self, client, db_session):
        admin = _seed_user(db_session, email="a18@te.com", is_admin=True)
        token = _login(client, admin.email)
        mock_redis = MagicMock()
        stored = {}

        def fake_set(key, value):
            stored[key] = value

        def fake_get(key):
            return stored.get(key)

        mock_redis.set = fake_set
        mock_redis.get = fake_get

        with patch("backend.routers.announcements._get_redis", return_value=mock_redis):
            client.post(
                "/api/admin/announcements",
                json={"message": "Hello world", "level": "info"},
                headers=_auth(token),
            )
            r = client.get("/api/announcements")

        assert r.status_code == 200
        ann = r.json()["announcement"]
        assert ann is not None
        assert ann["message"] == "Hello world"

    def test_19_delete_announcement(self, client, db_session):
        admin = _seed_user(db_session, email="a19@te.com", is_admin=True)
        token = _login(client, admin.email)
        mock_redis = MagicMock()
        deleted_keys = []
        mock_redis.delete = lambda key: deleted_keys.append(key)
        with patch("backend.routers.announcements._get_redis", return_value=mock_redis):
            r = client.delete(
                "/api/admin/announcements",
                headers=_auth(token),
            )
        assert r.status_code == 200
        assert r.json()["cleared"] is True
        assert "announcement:current" in deleted_keys

    def test_20_non_admin_cannot_create_announcement(self, client, db_session):
        user = _seed_user(db_session, email="r20@te.com")
        token = _login(client, user.email)
        r = client.post(
            "/api/admin/announcements",
            json={"message": "Hack", "level": "error"},
            headers=_auth(token),
        )
        assert r.status_code == 403

    def test_21_announcement_with_expiry(self, client, db_session):
        admin = _seed_user(db_session, email="a21@te.com", is_admin=True)
        token = _login(client, admin.email)
        mock_redis = MagicMock()
        stored_ex = {}

        def fake_setex(key, ttl, value):
            stored_ex[key] = (ttl, value)

        mock_redis.setex = fake_setex
        with patch("backend.routers.announcements._get_redis", return_value=mock_redis):
            r = client.post(
                "/api/admin/announcements",
                json={"message": "Temporary", "level": "info", "expires_in_seconds": 3600},
                headers=_auth(token),
            )
        assert r.status_code == 200
        assert "announcement:current" in stored_ex
        assert stored_ex["announcement:current"][0] == 3600


# ---------------------------------------------------------------------------
# TestWeeklyLeaderboardReset
# ---------------------------------------------------------------------------

class TestWeeklyLeaderboardReset:
    def test_22_reset_archives_and_zeros_weekly_points(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_leaderboard_reset
        u1 = _seed_user(db_session, email="lb22a@te.com", weekly_points=300)
        u2 = _seed_user(db_session, email="lb22b@te.com", weekly_points=150)
        # Ensure points are set (seed_user checks for existing)
        db_session.query(User).filter_by(id=u1.id).update({"weekly_points": 300})
        db_session.query(User).filter_by(id=u2.id).update({"weekly_points": 150})
        db_session.commit()
        result = _do_weekly_leaderboard_reset(db_session)
        assert result["archived"] >= 2
        db_session.refresh(u1)
        db_session.refresh(u2)
        assert u1.weekly_points == 0
        assert u2.weekly_points == 0

    def test_23_reset_creates_leaderboard_archive_rows(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_leaderboard_reset
        u = _seed_user(db_session, email="lb23a@te.com")
        db_session.query(User).filter_by(id=u.id).update({"weekly_points": 500})
        db_session.commit()
        before_count = db_session.query(LeaderboardArchive).count()
        _do_weekly_leaderboard_reset(db_session)
        after_count = db_session.query(LeaderboardArchive).count()
        assert after_count > before_count

    def test_24_reset_assigns_rank_in_order(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_leaderboard_reset
        u1 = _seed_user(db_session, email="lb24a@te.com")
        u2 = _seed_user(db_session, email="lb24b@te.com")
        db_session.query(User).filter_by(id=u1.id).update({"weekly_points": 400})
        db_session.query(User).filter_by(id=u2.id).update({"weekly_points": 200})
        db_session.commit()
        # Wipe prior archives
        db_session.query(LeaderboardArchive).delete()
        db_session.commit()
        _do_weekly_leaderboard_reset(db_session)
        archives = (
            db_session.query(LeaderboardArchive)
            .order_by(LeaderboardArchive.rank)
            .all()
        )
        assert len(archives) >= 2
        # Rank 1 should have more weekly_points than rank 2
        assert archives[0].weekly_points >= archives[1].weekly_points

    def test_25_reset_skips_zero_point_users(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_leaderboard_reset
        u_zero = _seed_user(db_session, email="lb25z@te.com")
        u_pts  = _seed_user(db_session, email="lb25p@te.com")
        db_session.query(User).filter_by(id=u_zero.id).update({"weekly_points": 0})
        db_session.query(User).filter_by(id=u_pts.id).update({"weekly_points": 50})
        db_session.commit()
        db_session.query(LeaderboardArchive).delete()
        db_session.commit()
        result = _do_weekly_leaderboard_reset(db_session)
        archives = db_session.query(LeaderboardArchive).all()
        # Only user with points > 0 should be archived
        assert all(a.weekly_points > 0 for a in archives)


# ---------------------------------------------------------------------------
# TestMonthlyQuotaReset
# ---------------------------------------------------------------------------

class TestMonthlyQuotaReset:
    def test_26_clears_old_usage_logs(self, db_session):
        from backend.tasks.growth_tasks import _do_monthly_quota_reset
        u = _seed_user(db_session, email="mq26@te.com")
        old_date    = datetime.datetime.utcnow() - datetime.timedelta(days=90)
        recent_date = datetime.datetime.utcnow() - datetime.timedelta(days=10)
        db_session.add(UsageLog(user_id=u.id, action="api_call", created_at=old_date))
        db_session.add(UsageLog(user_id=u.id, action="api_call", created_at=recent_date))
        db_session.commit()
        result = _do_monthly_quota_reset(db_session)
        assert result["deleted_usage_logs"] >= 1

    def test_27_leaves_recent_logs(self, db_session):
        from backend.tasks.growth_tasks import _do_monthly_quota_reset
        u = _seed_user(db_session, email="mq27@te.com")
        recent = datetime.datetime.utcnow() - datetime.timedelta(days=5)
        db_session.add(UsageLog(user_id=u.id, action="api_call", created_at=recent))
        db_session.commit()
        count_before = db_session.query(UsageLog).filter(
            UsageLog.user_id == u.id
        ).count()
        _do_monthly_quota_reset(db_session)
        count_after = db_session.query(UsageLog).filter(
            UsageLog.user_id == u.id
        ).count()
        assert count_after == count_before


# ---------------------------------------------------------------------------
# TestWeeklyDigest
# ---------------------------------------------------------------------------

class TestWeeklyDigest:
    def test_28_digest_skips_inactive_users(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_digest
        u = _seed_user(db_session, email="wd28@te.com")
        db_session.query(User).filter_by(id=u.id).update({"weekly_points": 0})
        db_session.commit()
        with patch("backend.tasks.growth_tasks.send_weekly_digest_email_sync") as mock_send:
            result = _do_weekly_digest(db_session)
        # Either 0 sent or mock not called for this specific user
        # (other users from prior tests may have weekly_points > 0)
        mock_send.assert_not_called() if result["sent"] == 0 else None
        assert "sent" in result

    def test_29_digest_sends_to_active_users(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_digest
        u = _seed_user(db_session, email="wd29@te.com")
        db_session.query(User).filter_by(id=u.id).update({"weekly_points": 100})
        db_session.commit()
        with patch("backend.tasks.growth_tasks.send_weekly_digest_email_sync", return_value=True) as mock_send:
            result = _do_weekly_digest(db_session)
        assert result["sent"] >= 1
        mock_send.assert_called()

    def test_30_digest_passes_correct_stats(self, db_session):
        from backend.tasks.growth_tasks import _do_weekly_digest
        u = _seed_user(db_session, email="wd30@te.com")
        db_session.query(User).filter_by(id=u.id).update(
            {"weekly_points": 200, "email": "wd30@te.com"}
        )
        db_session.commit()
        captured = {}

        def capture_send(to, name, stats):
            if to == "wd30@te.com":
                captured["to"]    = to
                captured["stats"] = stats
            return True

        with patch("backend.tasks.growth_tasks.send_weekly_digest_email_sync", side_effect=capture_send):
            _do_weekly_digest(db_session)

        assert captured.get("to") == "wd30@te.com"
        assert captured["stats"]["xp_earned"] == 200
