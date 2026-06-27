"""Sprint B — Content & Storage integration tests.

Covers:
  1. R2 storage service (local fallback mode)
  2. Paper visibility field + ToS acceptance on upload
  3. Admin CRUD API for Dojo problems
  4. Pipeline stage tracking via TaskRepository.set_stage
  5. Tutor session security (server-generated IDs, ownership validation, rate info)
"""
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User, Problem, Task
from backend.modules.auth.security.hashing import hash_password
from backend.repositories.task_repository import TaskRepository

# ── Shared fixtures ──────────────────────────────────────────────────────────

ADMIN_EMAIL = "sprint_b_admin@example.com"
USER_EMAIL  = "sprint_b_user@example.com"
TEST_PASS   = "SecurePass123!"


def _create_user(db: Session, email: str, is_admin: bool = False) -> User:
    existing = db.query(User).filter_by(email=email).first()
    if existing:
        return existing
    user = User(
        email=email,
        name="Sprint B Test",
        hashed_password=hash_password(TEST_PASS),
        is_verified=True,
        is_email_verified=True,
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _login(client: TestClient, email: str) -> str:
    r = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ── Task 1: storage_service local fallback ───────────────────────────────────

class TestStorageServiceLocalFallback:
    def test_store_and_fetch_roundtrip(self):
        from backend.services.storage_service import store_pdf, fetch_pdf, cleanup
        payload = b"%PDF-1.4 fake content"
        ref = store_pdf(payload, "test.pdf")
        assert ref  # non-empty storage ref
        fetched = fetch_pdf(ref)
        assert fetched == payload
        cleanup(ref)

    def test_cleanup_does_not_raise_on_missing(self):
        from backend.services.storage_service import cleanup
        cleanup("/tmp/does_not_exist_xyz.pdf")  # must not raise

    def test_r2_key_from_ref_local_returns_none(self):
        from backend.services.storage_service import r2_key_from_ref
        assert r2_key_from_ref("/tmp/foo.pdf") is None

    def test_r2_key_from_ref_r2_returns_key(self):
        from backend.services.storage_service import r2_key_from_ref
        key = r2_key_from_ref("r2://papers/abc.pdf")
        assert key == "papers/abc.pdf"

    def test_presigned_url_local_returns_empty(self):
        from backend.services.storage_service import presigned_download_url
        assert presigned_download_url("/tmp/foo.pdf") == ""


# ── Task 2: Paper visibility + ToS ───────────────────────────────────────────

class TestPaperVisibility:
    def test_upload_requires_terms_accepted(self, client: TestClient, db_session: Session):
        _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        data = {"terms_accepted": "false", "visibility": "public"}
        r = client.post(
            "/api/papers/upload",
            headers={"Authorization": f"Bearer {token}"},
            data=data,
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert r.status_code == 400
        assert "Terms of Service" in r.json()["detail"]

    def test_upload_rejects_invalid_visibility(self, client: TestClient, db_session: Session):
        _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        data = {"terms_accepted": "true", "visibility": "invisible"}
        r = client.post(
            "/api/papers/upload",
            headers={"Authorization": f"Bearer {token}"},
            data=data,
            files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
        assert r.status_code == 400
        assert "visibility" in r.json()["detail"].lower()

    def test_list_papers_anonymous_excludes_private(self, client: TestClient, db_session: Session):
        from backend.models import Paper
        # Create a private paper directly
        p = Paper(title="Private Sprint B Paper", visibility="private")
        db_session.add(p)
        db_session.commit()

        r = client.get("/api/papers")
        assert r.status_code == 200
        titles = [p["title"] for p in r.json()["papers"]]
        assert "Private Sprint B Paper" not in titles

    def test_list_papers_anonymous_sees_public(self, client: TestClient, db_session: Session):
        from backend.models import Paper
        p = Paper(title="Public Sprint B Paper", visibility="public")
        db_session.add(p)
        db_session.commit()

        r = client.get("/api/papers")
        assert r.status_code == 200
        titles = [p["title"] for p in r.json()["papers"]]
        assert "Public Sprint B Paper" in titles

    def test_list_papers_owner_sees_own_private(self, client: TestClient, db_session: Session):
        from backend.models import Paper
        user = _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        p = Paper(title="My Private Sprint B Paper", visibility="private", uploaded_by=user.id)
        db_session.add(p)
        db_session.commit()

        r = client.get("/api/papers", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        titles = [p["title"] for p in r.json()["papers"]]
        assert "My Private Sprint B Paper" in titles


# ── Task 3: Admin Problem CRUD ────────────────────────────────────────────────

class TestAdminProblemCRUD:
    def _admin_token(self, client, db_session):
        _create_user(db_session, ADMIN_EMAIL, is_admin=True)
        return _login(client, ADMIN_EMAIL)

    def test_create_problem(self, client: TestClient, db_session: Session):
        token = self._admin_token(client, db_session)
        body = {
            "id": "sb-test-001",
            "slug": "sb-test-001",
            "title": "Sprint B Test Problem",
            "difficulty": "Easy",
            "category": "Testing",
            "description": "A test problem for Sprint B.",
        }
        r = client.post(
            "/api/admin/problems",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["id"] == "sb-test-001"
        assert data["is_retired"] is False

    def test_create_problem_duplicate_id_returns_409(self, client: TestClient, db_session: Session):
        token = self._admin_token(client, db_session)
        body = {
            "id": "sb-dup-001",
            "slug": "sb-dup-001",
            "title": "Dup Problem",
            "difficulty": "Easy",
            "category": "Testing",
            "description": "Duplicate.",
        }
        client.post("/api/admin/problems", json=body, headers={"Authorization": f"Bearer {token}"})
        r = client.post("/api/admin/problems", json=body, headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 409

    def test_update_problem(self, client: TestClient, db_session: Session):
        token = self._admin_token(client, db_session)
        p = Problem(
            id="sb-upd-001", slug="sb-upd-001", title="Old Title",
            difficulty="Easy", category="Testing", description="Old",
        )
        db_session.add(p)
        db_session.commit()

        r = client.put(
            "/api/admin/problems/sb-upd-001",
            json={"title": "New Title"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["title"] == "New Title"

    def test_update_nonexistent_problem_returns_404(self, client: TestClient, db_session: Session):
        token = self._admin_token(client, db_session)
        r = client.put(
            "/api/admin/problems/nonexistent-xxx",
            json={"title": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 404

    def test_retire_and_restore_problem(self, client: TestClient, db_session: Session):
        token = self._admin_token(client, db_session)
        p = Problem(
            id="sb-ret-001", slug="sb-ret-001", title="Retirable",
            difficulty="Medium", category="Testing", description="Retire me",
        )
        db_session.add(p)
        db_session.commit()

        r = client.patch(
            "/api/admin/problems/sb-ret-001/retire",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["is_retired"] is True

        r2 = client.patch(
            "/api/admin/problems/sb-ret-001/restore",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 200
        assert r2.json()["is_retired"] is False

    def test_list_problems_includes_retired(self, client: TestClient, db_session: Session):
        token = self._admin_token(client, db_session)
        p = Problem(
            id="sb-list-001", slug="sb-list-001", title="Retired One",
            difficulty="Hard", category="Testing", description="x",
            is_retired=True,
        )
        db_session.add(p)
        db_session.commit()

        r = client.get(
            "/api/admin/problems?include_retired=true",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        ids = [pr["id"] for pr in r.json()["problems"]]
        assert "sb-list-001" in ids

    def test_problem_crud_requires_admin(self, client: TestClient, db_session: Session):
        _create_user(db_session, USER_EMAIL, is_admin=False)
        token = _login(client, USER_EMAIL)
        r = client.post(
            "/api/admin/problems",
            json={"id": "x", "slug": "x", "title": "x", "difficulty": "Easy", "category": "x", "description": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 403


# ── Task 4: Pipeline stage tracking ──────────────────────────────────────────

class TestPipelineStageTracking:
    def test_set_stage_updates_result(self, db_session: Session):
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "test-paper")
        repo.set_running(task.id)
        repo.set_stage(task.id, "extracting")

        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.result is not None
        assert fresh.result["stage"] == "extracting"

    def test_set_stage_preserves_existing_result_keys(self, db_session: Session):
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "test-paper2")
        repo.set_running(task.id)
        # Manually set an existing result key
        t = db_session.query(Task).filter_by(id=task.id).first()
        t.result = {"some_key": "value"}
        db_session.commit()

        repo.set_stage(task.id, "analyzing")
        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.result["stage"] == "analyzing"
        assert fresh.result["some_key"] == "value"

    def test_set_complete_includes_stage(self, db_session: Session):
        repo = TaskRepository(db_session)
        task = repo.create("paper.codegen", None, "test-paper3")
        repo.set_running(task.id)
        repo.set_complete(task.id, {"paper_id": 1, "stage": "complete"})

        fresh = db_session.query(Task).filter_by(id=task.id).first()
        assert fresh.status == "completed"
        assert fresh.result["stage"] == "complete"


# ── Task 5: Tutor session security ───────────────────────────────────────────

class TestTutorSessionSecurity:
    def test_start_session_requires_auth(self, client: TestClient):
        r = client.post("/api/tutor/start-session")
        assert r.status_code == 401

    def test_start_session_returns_uuid(self, client: TestClient, db_session: Session):
        _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        r = client.post(
            "/api/tutor/start-session",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        session_id = r.json()["session_id"]
        assert len(session_id) == 36  # UUID format

    def test_tutor_ask_requires_auth(self, client: TestClient):
        r = client.post(
            "/api/tutor/ask",
            json={"context_type": "module", "context_data": {}, "query": "what?"},
        )
        assert r.status_code == 401

    def test_tutor_ask_auto_creates_session_when_none_provided(
        self, client: TestClient, db_session: Session
    ):
        _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        with patch("core.agents.tutor_agent.llm_complete", return_value='{"answer":"ok","source_context":"ctx","confidence":"High","reasoning_type":"Structural"}'):
            r = client.post(
                "/api/tutor/ask",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "session_id": None,
                    "context_type": "module",
                    "context_data": {"paper_title": "ResNet", "layer_name": "Conv1", "module_type": "conv", "explanation": "x", "flops_context": {}},
                    "query": "What does this layer do?",
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert "session_id" in body
        assert len(body["session_id"]) == 36
        assert "queries_today" in body
        assert "queries_limit" in body

    def test_session_store_ownership_validation(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()  # uses in-memory in test environment
        sid = store.create_session(user_id=42)
        assert store.validate_ownership(sid, 42) is True
        assert store.validate_ownership(sid, 99) is False

    def test_session_store_history_roundtrip(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        sid = store.create_session(user_id=1)
        assert store.get_history(sid) == []
        store.update_history(sid, [{"role": "user", "content": "hello"}])
        hist = store.get_history(sid)
        assert len(hist) == 1
        assert hist[0]["content"] == "hello"

    def test_session_store_caps_history_at_6(self):
        from backend.services.tutor_session_store import TutorSessionStore
        store = TutorSessionStore()
        sid = store.create_session(user_id=1)
        messages = [{"role": "user", "content": str(i)} for i in range(10)]
        store.update_history(sid, messages)
        assert len(store.get_history(sid)) == 6

    def test_invalid_session_gets_new_one(self, client: TestClient, db_session: Session):
        _create_user(db_session, USER_EMAIL)
        token = _login(client, USER_EMAIL)
        with patch("core.agents.tutor_agent.llm_complete", return_value='{"answer":"ok","source_context":"ctx","confidence":"High","reasoning_type":"Structural"}'):
            r = client.post(
                "/api/tutor/ask",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "session_id": "00000000-fake-fake-fake-000000000000",
                    "context_type": "module",
                    "context_data": {"paper_title": "VGG", "layer_name": "Conv1", "module_type": "conv", "explanation": "x", "flops_context": {}},
                    "query": "explain",
                },
            )
        assert r.status_code == 200
        body = r.json()
        # Should issue a fresh session, not use the fake one
        assert body["session_id"] != "00000000-fake-fake-fake-000000000000"
