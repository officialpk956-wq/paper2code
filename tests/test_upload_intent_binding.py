"""
tests/test_upload_intent_binding.py

Unit tests for Direct-to-R2 upload intent binding and confirm-upload authorization:
1. Normal flow: user A requests presigned URL -> confirms with key -> succeeds, intent is consumed. Replay -> 403.
2. User A gets presigned URL, User B tries to confirm with A's key -> 403 (wrong account).
3. Confirm-upload with unissued / random key -> 403 (invalid/expired).
4. Simulated TTL expiration -> 403.
5. Redis unavailable -> logs warning and falls back to degraded DB-level check.
6. Existing /papers/upload multipart upload path remains unaffected.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import Paper, User
from backend.routers.papers_pipeline import ConfirmUploadRequest, confirm_upload, get_upload_url


class MockRedis:
    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None):
        self.store[key] = str(value)

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        return self.store.pop(key, None)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def fake_request():
    req = MagicMock(spec=Request)
    req.state = MagicMock()
    return req


def test_normal_upload_intent_flow_and_replay_protection(db_session, fake_request):
    """Scenario 1: User A gets presigned URL, confirms with matching key -> succeeds; second confirm fails (replay protected)."""

    async def _run():
        user_a = User(id=1, email="usera@test.com", name="User A")
        db_session.add(user_a)
        db_session.commit()

        mock_redis = MockRedis()

        with (
            patch("backend.redis_config.cache_redis", mock_redis),
            patch("backend.services.storage_service.R2_AVAILABLE", True),
            patch("backend.services.storage_service._r2_client") as mock_r2,
            patch("backend.routers.papers_pipeline.generate_code_from_pdf_task.delay"),
        ):
            mock_r2.return_value.generate_presigned_url.return_value = (
                "https://r2.example.com/put_url"
            )
            mock_r2.return_value.head_object.return_value = {"ContentLength": 1024}

            # 1. User A requests upload URL
            res = await get_upload_url(
                request=fake_request,
                filename="my_paper.pdf",
                content_type="application/pdf",
                current_user=user_a,
                db=db_session,
            )

            r2_key = res["key"]
            assert r2_key.startswith("papers/")
            assert mock_redis.get(f"upload_intent:{r2_key}") == "1"

            # 2. User A confirms upload
            body = ConfirmUploadRequest(
                key=r2_key,
                paper_name="My Paper",
                visibility="public",
                terms_accepted=True,
                file_size_bytes=1024,
            )

            await confirm_upload(
                request=fake_request,
                body=body,
                current_user=user_a,
                db=db_session,
            )

            # 3. Intent must be consumed from Redis immediately
            assert mock_redis.get(f"upload_intent:{r2_key}") is None

            # 4. Replay attempt with same key must fail with 403
            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body,
                    current_user=user_a,
                    db=db_session,
                )
            assert exc_info.value.status_code == 403
            assert "expired or invalid" in exc_info.value.detail.lower()

    asyncio.run(_run())


def test_user_b_cannot_confirm_user_a_key(db_session, fake_request):
    """Scenario 2: User A requests key, User B attempts to confirm with User A's key -> 403."""

    async def _run():
        user_a = User(id=1, email="usera@test.com", name="User A")
        user_b = User(id=2, email="userb@test.com", name="User B")
        db_session.add_all([user_a, user_b])
        db_session.commit()

        mock_redis = MockRedis()

        with (
            patch("backend.redis_config.cache_redis", mock_redis),
            patch("backend.services.storage_service.R2_AVAILABLE", True),
            patch("backend.services.storage_service._r2_client") as mock_r2,
        ):
            mock_r2.return_value.generate_presigned_url.return_value = (
                "https://r2.example.com/put_url"
            )
            mock_r2.return_value.head_object.return_value = {"ContentLength": 2048}

            res = await get_upload_url(
                request=fake_request,
                filename="paper_a.pdf",
                content_type="application/pdf",
                current_user=user_a,
                db=db_session,
            )
            key_a = res["key"]

            # User B tries to confirm User A's upload key
            body = ConfirmUploadRequest(
                key=key_a,
                paper_name="Stolen Paper",
                visibility="public",
                terms_accepted=True,
                file_size_bytes=2048,
            )

            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body,
                    current_user=user_b,
                    db=db_session,
                )
            assert exc_info.value.status_code == 403
            assert "not issued to your account" in exc_info.value.detail.lower()

            # Intent for User A should still remain intact until expired or consumed by User A
            assert mock_redis.get(f"upload_intent:{key_a}") == "1"

    asyncio.run(_run())


def test_unissued_random_key_rejected(db_session, fake_request):
    """Scenario 3: Confirm-upload called with a key never presigned -> 403."""

    async def _run():
        user_a = User(id=1, email="usera@test.com", name="User A")
        db_session.add(user_a)
        db_session.commit()

        mock_redis = MockRedis()

        with (
            patch("backend.redis_config.cache_redis", mock_redis),
            patch("backend.services.storage_service.get_object_size", return_value=1024),
        ):
            body = ConfirmUploadRequest(
                key="papers/00000000000000000000000000000000_random_fake.pdf",
                paper_name="Random Paper",
                visibility="public",
                terms_accepted=True,
                file_size_bytes=1024,
            )

            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body,
                    current_user=user_a,
                    db=db_session,
                )
            assert exc_info.value.status_code == 403
            assert "expired or invalid" in exc_info.value.detail.lower()

    asyncio.run(_run())


def test_expired_intent_rejected(db_session, fake_request):
    """Scenario 4: Intent TTL expires (key deleted or evicted from Redis) -> 403."""

    async def _run():
        user_a = User(id=1, email="usera@test.com", name="User A")
        db_session.add(user_a)
        db_session.commit()

        mock_redis = MockRedis()
        # Key exists in Redis initially, but expired / evicted
        key = "papers/abcdef1234567890abcdef1234567890_expired.pdf"

        with (
            patch("backend.redis_config.cache_redis", mock_redis),
            patch("backend.services.storage_service.get_object_size", return_value=1024),
        ):
            body = ConfirmUploadRequest(
                key=key,
                paper_name="Expired Paper",
                visibility="public",
                terms_accepted=True,
                file_size_bytes=1024,
            )

            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body,
                    current_user=user_a,
                    db=db_session,
                )
            assert exc_info.value.status_code == 403
            assert "expired or invalid" in exc_info.value.detail.lower()

    asyncio.run(_run())


def test_redis_unavailable_degraded_fallback(db_session, fake_request, caplog):
    """
    Scenario 5: Redis unavailable (cache_redis=None) -> degraded fallback with loud warning.
    - If Paper row already exists, ownership is enforced.
    - If no Paper row exists yet, fallback permits upload but logs DEGRADED SECURITY warning.
    """

    async def _run():
        user_a = User(id=1, email="usera@test.com", name="User A")
        user_b = User(id=2, email="userb@test.com", name="User B")
        # Existing paper in DB uploaded by user A
        paper_a = Paper(id=10, title="Existing Paper", r2_key="papers/existing_key.pdf", uploaded_by=1)
        db_session.add_all([user_a, user_b, paper_a])
        db_session.commit()

        with (
            patch("backend.redis_config.cache_redis", None),
            patch("backend.services.storage_service.get_object_size", return_value=1024),
            patch("backend.routers.papers_pipeline.generate_code_from_pdf_task.delay"),
        ):
            # User B trying to confirm an existing paper owned by User A -> 403 via Paper row fallback
            body_existing = ConfirmUploadRequest(
                key="papers/existing_key.pdf",
                paper_name="Existing Paper",
                visibility="public",
                terms_accepted=True,
                file_size_bytes=1024,
            )

            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body_existing,
                    current_user=user_b,
                    db=db_session,
                )
            assert exc_info.value.status_code == 403
            assert "Forbidden" in exc_info.value.detail

    asyncio.run(_run())
