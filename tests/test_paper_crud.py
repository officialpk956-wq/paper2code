"""
tests/test_paper_crud.py

Covers paper CRUD audit fixes:
  1. GET /api/papers              — Partial: authors, abstract, created_at, visibility, uploaded_by
  2. GET /api/papers/{id}         — Security bug: visibility gate + response fields
  3. GET /api/papers/{id}/modules — Security bug: visibility gate
  4. GET /api/papers/{id}/blueprint — Security bug: visibility gate
  5. POST /api/papers/{id}/publish — Security bug: auth + ownership
  6. PATCH /api/papers/{id}/visibility — Missing P0: ownership, valid values
  7. DELETE /api/papers/{id}       — Missing P1: ownership, quota free
  8. POST /api/papers/{id}/flag    — Missing P1: user report endpoint
  9. GET /api/papers/{id}/similar  — Missing P2: arch-type similarity
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Paper
from backend.modules.auth.security.hashing import hash_password

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "PaperCrud999!"
_PAPER_CTR = [0]


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


def _seed_paper(
    db: Session,
    suffix: str,
    visibility: str = "public",
    uploaded_by: int = None,
    authors: str = "Test Author",
    abstract: str = "Test abstract",
) -> Paper:
    title = f"Test Paper {suffix}"
    existing = db.query(Paper).filter_by(title=title).first()
    if existing:
        return existing
    p = Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        visibility=visibility,
        uploaded_by=uploaded_by,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# ---------------------------------------------------------------------------
# 1. TestListPapersFields — GET /api/papers
# ---------------------------------------------------------------------------

class TestListPapersFields:

    def test_01_authors_present(self, client, db_session):
        _seed_paper(db_session, "list01", authors="John Doe")
        r = client.get("/api/papers")
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert len(papers) >= 1
        assert all("authors" in p for p in papers)

    def test_02_abstract_present(self, client, db_session):
        _seed_paper(db_session, "list02", abstract="An important abstract")
        r = client.get("/api/papers")
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert all("abstract" in p for p in papers)

    def test_03_visibility_present(self, client, db_session):
        _seed_paper(db_session, "list03")
        r = client.get("/api/papers")
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert all("visibility" in p for p in papers)

    def test_04_uploaded_by_present(self, client, db_session):
        user = _seed_user(db_session, "list04@pc.com")
        _seed_paper(db_session, "list04", uploaded_by=user.id)
        r = client.get("/api/papers")
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert all("uploaded_by" in p for p in papers)

    def test_05_created_at_present(self, client, db_session):
        _seed_paper(db_session, "list05")
        r = client.get("/api/papers")
        assert r.status_code == 200
        papers = r.json()["papers"]
        assert all("created_at" in p for p in papers)

    def test_06_private_paper_hidden_from_anon(self, client, db_session):
        user = _seed_user(db_session, "list06@pc.com")
        p = _seed_paper(db_session, "list06-priv", visibility="private", uploaded_by=user.id)
        r = client.get("/api/papers")
        ids = [x["id"] for x in r.json()["papers"]]
        assert p.id not in ids

    def test_07_own_private_paper_visible_to_owner(self, client, db_session):
        user = _seed_user(db_session, "list07@pc.com")
        p = _seed_paper(db_session, "list07-priv", visibility="private", uploaded_by=user.id)
        token = _login(client, user.email)
        r = client.get("/api/papers", headers=_auth(token))
        ids = [x["id"] for x in r.json()["papers"]]
        assert p.id in ids


# ---------------------------------------------------------------------------
# 2. TestPaperDetailVisibility — GET /api/papers/{id}
# ---------------------------------------------------------------------------

class TestPaperDetailVisibility:

    def test_08_public_paper_accessible_anon(self, client, db_session):
        p = _seed_paper(db_session, "det08", visibility="public")
        r = client.get(f"/api/papers/{p.id}")
        assert r.status_code == 200

    def test_09_private_paper_blocked_for_anon(self, client, db_session):
        user = _seed_user(db_session, "det09@pc.com")
        p = _seed_paper(db_session, "det09-priv", visibility="private", uploaded_by=user.id)
        r = client.get(f"/api/papers/{p.id}")
        assert r.status_code == 403

    def test_10_private_paper_accessible_to_owner(self, client, db_session):
        user = _seed_user(db_session, "det10@pc.com")
        p = _seed_paper(db_session, "det10-priv", visibility="private", uploaded_by=user.id)
        token = _login(client, user.email)
        r = client.get(f"/api/papers/{p.id}", headers=_auth(token))
        assert r.status_code == 200

    def test_11_private_paper_blocked_for_other_user(self, client, db_session):
        owner = _seed_user(db_session, "det11o@pc.com")
        other = _seed_user(db_session, "det11u@pc.com")
        p = _seed_paper(db_session, "det11-priv", visibility="private", uploaded_by=owner.id)
        token = _login(client, other.email)
        r = client.get(f"/api/papers/{p.id}", headers=_auth(token))
        assert r.status_code == 403

    def test_12_detail_includes_visibility_field(self, client, db_session):
        p = _seed_paper(db_session, "det12", visibility="public")
        r = client.get(f"/api/papers/{p.id}")
        assert r.status_code == 200
        assert "visibility" in r.json()["metadata"]

    def test_13_detail_includes_uploaded_by(self, client, db_session):
        user = _seed_user(db_session, "det13@pc.com")
        p = _seed_paper(db_session, "det13", uploaded_by=user.id)
        r = client.get(f"/api/papers/{p.id}")
        assert r.status_code == 200
        assert r.json()["metadata"]["uploaded_by"] == user.id

    def test_14_detail_includes_created_at(self, client, db_session):
        p = _seed_paper(db_session, "det14")
        r = client.get(f"/api/papers/{p.id}")
        assert r.status_code == 200
        assert "created_at" in r.json()["metadata"]


# ---------------------------------------------------------------------------
# 3. TestModulesVisibility — GET /api/papers/{id}/modules
# ---------------------------------------------------------------------------

class TestModulesVisibility:

    def test_15_private_modules_blocked_for_anon(self, client, db_session):
        user = _seed_user(db_session, "mod15@pc.com")
        p = _seed_paper(db_session, "mod15-priv", visibility="private", uploaded_by=user.id)
        r = client.get(f"/api/papers/{p.id}/modules")
        assert r.status_code == 403

    def test_16_private_modules_accessible_to_owner(self, client, db_session):
        user = _seed_user(db_session, "mod16@pc.com")
        p = _seed_paper(db_session, "mod16-priv", visibility="private", uploaded_by=user.id)
        token = _login(client, user.email)
        r = client.get(f"/api/papers/{p.id}/modules", headers=_auth(token))
        assert r.status_code == 200

    def test_17_public_modules_accessible_anon(self, client, db_session):
        p = _seed_paper(db_session, "mod17", visibility="public")
        r = client.get(f"/api/papers/{p.id}/modules")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 4. TestBlueprintVisibility — GET /api/papers/{id}/blueprint
# ---------------------------------------------------------------------------

class TestBlueprintVisibility:

    def test_18_private_blueprint_blocked_for_anon(self, client, db_session):
        user = _seed_user(db_session, "bp18@pc.com")
        p = _seed_paper(db_session, "bp18-priv", visibility="private", uploaded_by=user.id)
        r = client.get(f"/api/papers/{p.id}/blueprint")
        assert r.status_code == 403

    def test_19_private_blueprint_accessible_to_owner(self, client, db_session):
        user = _seed_user(db_session, "bp19@pc.com")
        p = _seed_paper(db_session, "bp19-priv", visibility="private", uploaded_by=user.id)
        token = _login(client, user.email)
        # No blueprint yet → 404 (not 403) means the auth check passed
        r = client.get(f"/api/papers/{p.id}/blueprint", headers=_auth(token))
        assert r.status_code in (200, 404)
        assert r.status_code != 403

    def test_20_public_blueprint_not_blocked(self, client, db_session):
        p = _seed_paper(db_session, "bp20", visibility="public")
        r = client.get(f"/api/papers/{p.id}/blueprint")
        # 404 (no blueprint) is fine — 403 is not
        assert r.status_code in (200, 404)
        assert r.status_code != 403


# ---------------------------------------------------------------------------
# 5. TestPublishOwnership — POST /api/papers/{id}/publish
# ---------------------------------------------------------------------------

class TestPublishOwnership:

    def test_21_unauthenticated_publish_rejected(self, client, db_session):
        p = _seed_paper(db_session, "pub21", visibility="public")
        r = client.post(f"/api/papers/{p.id}/publish")
        assert r.status_code in (401, 403)

    def test_22_non_owner_publish_rejected(self, client, db_session):
        owner = _seed_user(db_session, "pub22o@pc.com")
        other = _seed_user(db_session, "pub22u@pc.com")
        p = _seed_paper(db_session, "pub22", uploaded_by=owner.id)
        token = _login(client, other.email)
        r = client.post(f"/api/papers/{p.id}/publish", headers=_auth(token))
        assert r.status_code == 403

    def test_23_owner_can_publish(self, client, db_session):
        user = _seed_user(db_session, "pub23@pc.com")
        p = _seed_paper(db_session, "pub23", uploaded_by=user.id)
        token = _login(client, user.email)
        r = client.post(f"/api/papers/{p.id}/publish", headers=_auth(token))
        assert r.status_code == 200

    def test_24_admin_can_publish_any(self, client, db_session):
        admin = _seed_user(db_session, "pub24a@pc.com", is_admin=True)
        owner = _seed_user(db_session, "pub24o@pc.com")
        p = _seed_paper(db_session, "pub24", uploaded_by=owner.id)
        token = _login(client, admin.email)
        r = client.post(f"/api/papers/{p.id}/publish", headers=_auth(token))
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 6. TestPaperVisibilityChange — PATCH /api/papers/{id}/visibility
# ---------------------------------------------------------------------------

class TestPaperVisibilityChange:

    def test_25_owner_can_change_to_private(self, client, db_session):
        user = _seed_user(db_session, "vis25@pc.com")
        p = _seed_paper(db_session, "vis25", visibility="public", uploaded_by=user.id)
        token = _login(client, user.email)
        r = client.patch(
            f"/api/papers/{p.id}/visibility",
            json={"visibility": "private"},
            headers=_auth(token),
        )
        assert r.status_code == 200
        assert r.json()["visibility"] == "private"

    def test_26_change_persisted(self, client, db_session):
        user = _seed_user(db_session, "vis26@pc.com")
        p = _seed_paper(db_session, "vis26", visibility="public", uploaded_by=user.id)
        token = _login(client, user.email)
        client.patch(
            f"/api/papers/{p.id}/visibility",
            json={"visibility": "unlisted"},
            headers=_auth(token),
        )
        db_session.refresh(p)
        assert p.visibility == "unlisted"

    def test_27_non_owner_rejected(self, client, db_session):
        owner = _seed_user(db_session, "vis27o@pc.com")
        other = _seed_user(db_session, "vis27u@pc.com")
        p = _seed_paper(db_session, "vis27", visibility="public", uploaded_by=owner.id)
        token = _login(client, other.email)
        r = client.patch(
            f"/api/papers/{p.id}/visibility",
            json={"visibility": "private"},
            headers=_auth(token),
        )
        assert r.status_code == 403

    def test_28_invalid_value_rejected(self, client, db_session):
        user = _seed_user(db_session, "vis28@pc.com")
        p = _seed_paper(db_session, "vis28", uploaded_by=user.id)
        token = _login(client, user.email)
        r = client.patch(
            f"/api/papers/{p.id}/visibility",
            json={"visibility": "secret"},
            headers=_auth(token),
        )
        assert r.status_code == 400

    def test_29_unauthenticated_rejected(self, client, db_session):
        p = _seed_paper(db_session, "vis29")
        r = client.patch(
            f"/api/papers/{p.id}/visibility",
            json={"visibility": "private"},
        )
        assert r.status_code in (401, 403)

    def test_30_all_three_valid_values_accepted(self, client, db_session):
        user = _seed_user(db_session, "vis30@pc.com")
        for vis in ("public", "unlisted", "private"):
            p = _seed_paper(db_session, f"vis30-{vis}", uploaded_by=user.id)
            token = _login(client, user.email)
            r = client.patch(
                f"/api/papers/{p.id}/visibility",
                json={"visibility": vis},
                headers=_auth(token),
            )
            assert r.status_code == 200, f"Failed for visibility={vis}: {r.text}"


# ---------------------------------------------------------------------------
# 7. TestPaperDelete — DELETE /api/papers/{id}
# ---------------------------------------------------------------------------

class TestPaperDelete:

    def test_31_owner_can_delete(self, client, db_session):
        user = _seed_user(db_session, "del31@pc.com")
        p = _seed_paper(db_session, "del31", uploaded_by=user.id)
        pid = p.id
        token = _login(client, user.email)
        r = client.delete(f"/api/papers/{pid}", headers=_auth(token))
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_32_paper_gone_after_delete(self, client, db_session):
        user = _seed_user(db_session, "del32@pc.com")
        p = _seed_paper(db_session, "del32", uploaded_by=user.id)
        pid = p.id
        token = _login(client, user.email)
        client.delete(f"/api/papers/{pid}", headers=_auth(token))
        assert db_session.query(Paper).filter_by(id=pid).first() is None

    def test_33_non_owner_cannot_delete(self, client, db_session):
        owner = _seed_user(db_session, "del33o@pc.com")
        other = _seed_user(db_session, "del33u@pc.com")
        p = _seed_paper(db_session, "del33", uploaded_by=owner.id)
        token = _login(client, other.email)
        r = client.delete(f"/api/papers/{p.id}", headers=_auth(token))
        assert r.status_code == 403

    def test_34_missing_paper_404(self, client, db_session):
        user = _seed_user(db_session, "del34@pc.com")
        token = _login(client, user.email)
        r = client.delete("/api/papers/999999", headers=_auth(token))
        assert r.status_code == 404

    def test_35_unauthenticated_rejected(self, client, db_session):
        p = _seed_paper(db_session, "del35")
        r = client.delete(f"/api/papers/{p.id}")
        assert r.status_code in (401, 403)

    def test_36_storage_quota_freed(self, client, db_session):
        user = _seed_user(db_session, "del36@pc.com")
        p = _seed_paper(db_session, "del36", uploaded_by=user.id)
        p.file_size_bytes = 1_000_000
        user.storage_bytes_used = 1_000_000
        db_session.commit()

        token = _login(client, user.email)
        client.delete(f"/api/papers/{p.id}", headers=_auth(token))
        db_session.refresh(user)
        assert user.storage_bytes_used == 0

    def test_37_admin_can_delete_any(self, client, db_session):
        admin = _seed_user(db_session, "del37a@pc.com", is_admin=True)
        owner = _seed_user(db_session, "del37o@pc.com")
        p = _seed_paper(db_session, "del37", uploaded_by=owner.id)
        token = _login(client, admin.email)
        r = client.delete(f"/api/papers/{p.id}", headers=_auth(token))
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# 8. TestPaperFlag — POST /api/papers/{id}/flag
# ---------------------------------------------------------------------------

class TestPaperFlag:

    def test_38_user_can_flag(self, client, db_session):
        user = _seed_user(db_session, "flag38@pc.com")
        p = _seed_paper(db_session, "flag38")
        token = _login(client, user.email)
        r = client.post(
            f"/api/papers/{p.id}/flag",
            json={"reason": "spam"},
            headers=_auth(token),
        )
        assert r.status_code == 200
        assert r.json()["flagged"] is True

    def test_39_is_flagged_persisted(self, client, db_session):
        user = _seed_user(db_session, "flag39@pc.com")
        p = _seed_paper(db_session, "flag39")
        token = _login(client, user.email)
        client.post(
            f"/api/papers/{p.id}/flag",
            json={"reason": "copyright"},
            headers=_auth(token),
        )
        db_session.refresh(p)
        assert p.is_flagged is True
        assert p.flag_reason == "copyright"

    def test_40_default_reason_accepted(self, client, db_session):
        user = _seed_user(db_session, "flag40@pc.com")
        p = _seed_paper(db_session, "flag40")
        token = _login(client, user.email)
        r = client.post(
            f"/api/papers/{p.id}/flag",
            json={},
            headers=_auth(token),
        )
        assert r.status_code == 200

    def test_41_unauthenticated_rejected(self, client, db_session):
        p = _seed_paper(db_session, "flag41")
        r = client.post(f"/api/papers/{p.id}/flag", json={"reason": "spam"})
        assert r.status_code in (401, 403)

    def test_42_missing_paper_404(self, client, db_session):
        user = _seed_user(db_session, "flag42@pc.com")
        token = _login(client, user.email)
        r = client.post(
            "/api/papers/999999/flag",
            json={"reason": "spam"},
            headers=_auth(token),
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 9. TestPaperSimilar — GET /api/papers/{id}/similar
# ---------------------------------------------------------------------------

class TestPaperSimilar:

    def test_43_returns_200_for_public_paper(self, client, db_session):
        p = _seed_paper(db_session, "sim43", visibility="public")
        r = client.get(f"/api/papers/{p.id}/similar")
        assert r.status_code == 200
        assert "similar" in r.json()

    def test_44_private_paper_blocked_for_anon(self, client, db_session):
        user = _seed_user(db_session, "sim44@pc.com")
        p = _seed_paper(db_session, "sim44-priv", visibility="private", uploaded_by=user.id)
        r = client.get(f"/api/papers/{p.id}/similar")
        assert r.status_code == 403

    def test_45_excludes_private_papers_from_results(self, client, db_session):
        owner = _seed_user(db_session, "sim45o@pc.com")
        viewer = _seed_user(db_session, "sim45v@pc.com")
        base = _seed_paper(db_session, "sim45-base", visibility="public")
        priv = _seed_paper(db_session, "sim45-priv", visibility="private", uploaded_by=owner.id)
        token = _login(client, viewer.email)
        r = client.get(f"/api/papers/{base.id}/similar", headers=_auth(token))
        assert r.status_code == 200
        ids = [s["id"] for s in r.json()["similar"]]
        assert priv.id not in ids

    def test_46_missing_paper_404(self, client, db_session):
        r = client.get("/api/papers/999999/similar")
        assert r.status_code == 404

    def test_47_result_has_expected_fields(self, client, db_session):
        _seed_paper(db_session, "sim47a", visibility="public")
        p2 = _seed_paper(db_session, "sim47b", visibility="public")
        r = client.get(f"/api/papers/{p2.id}/similar")
        assert r.status_code == 200
        data = r.json()
        assert "paper_id" in data
        assert isinstance(data["similar"], list)
