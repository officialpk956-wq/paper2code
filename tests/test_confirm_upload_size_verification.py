"""
tests/test_confirm_upload_size_verification.py

Tests verifying that POST /api/papers/confirm-upload enforces storage quota
and increments user usage based exclusively on verified R2 object size (get_object_size),
and rejects non-existent or zero-byte uploads.

Scenarios:
1. Normal confirm: get_object_size returns verified size -> quota check & increment use verified size.
2. Quota bypass attempt: Client sends file_size_bytes=1, but real R2 object is 50 MB -> user storage is charged 50 MB.
3. Non-existent / premature confirm: get_object_size returns 0 -> 400 HTTPException raised, quota untouched, task not queued.
4. Quota exceeded via verified size: User has 490 MB used / 500 MB limit, uploads 20 MB object (claims file_size_bytes=100) -> 429 HTTPException raised.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import User
from backend.routers.papers_pipeline import ConfirmUploadRequest, confirm_upload


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


def test_normal_confirm_uses_verified_size(db_session, fake_request):
    """Scenario 1: Verified R2 object size is used for storage increment."""

    async def _run():
        user = User(id=1, email="user1@example.com", name="User 1", storage_bytes_used=1000)
        db_session.add(user)
        db_session.commit()

        real_size = 5 * 1024 * 1024  # 5 MB
        body = ConfirmUploadRequest(
            key="papers/uuid1_test.pdf",
            paper_name="Verified Paper",
            visibility="public",
            terms_accepted=True,
            file_size_bytes=100,  # Client claimed 100 bytes
        )

        with (
            patch("backend.services.storage_service.get_object_size", return_value=real_size),
            patch("backend.redis_config.cache_redis", None),
            patch(
                "backend.routers.papers_pipeline.generate_code_from_pdf_task.delay"
            ) as mock_delay,
        ):
            resp = await confirm_upload(
                request=fake_request,
                body=body,
                current_user=user,
                db=db_session,
            )

            assert resp["status"] == "pending"
            mock_delay.assert_called_once()

            db_session.refresh(user)
            # Usage must have incremented by the real 5MB, not client-reported 100 bytes
            assert user.storage_bytes_used == 1000 + real_size

    asyncio.run(_run())


def test_quota_bypass_attempt_defeated(db_session, fake_request):
    """Scenario 2: Attacker claims file_size_bytes=1 for a 50MB file -> full 50MB is charged."""

    async def _run():
        user = User(id=2, email="attacker@example.com", name="Attacker", storage_bytes_used=0)
        db_session.add(user)
        db_session.commit()

        real_size = 50 * 1024 * 1024  # 50 MB
        body = ConfirmUploadRequest(
            key="papers/uuid2_large.pdf",
            paper_name="Large Paper",
            visibility="public",
            terms_accepted=True,
            file_size_bytes=1,  # Attacker attempts to bypass quota with 1 byte
        )

        with (
            patch("backend.services.storage_service.get_object_size", return_value=real_size),
            patch("backend.redis_config.cache_redis", None),
            patch("backend.routers.papers_pipeline.generate_code_from_pdf_task.delay"),
        ):
            await confirm_upload(
                request=fake_request,
                body=body,
                current_user=user,
                db=db_session,
            )

            db_session.refresh(user)
            # Verified 50MB is recorded in DB
            assert user.storage_bytes_used == real_size

    asyncio.run(_run())


def test_non_existent_r2_object_rejected(db_session, fake_request):
    """Scenario 3: get_object_size returns 0 (upload not in R2) -> 400 error."""

    async def _run():
        user = User(id=3, email="premature@example.com", name="Premature", storage_bytes_used=0)
        db_session.add(user)
        db_session.commit()

        body = ConfirmUploadRequest(
            key="papers/uuid3_missing.pdf",
            paper_name="Missing Paper",
            visibility="public",
            terms_accepted=True,
            file_size_bytes=1024,
        )

        with (
            patch("backend.services.storage_service.get_object_size", return_value=0),
            patch("backend.redis_config.cache_redis", None),
            patch(
                "backend.routers.papers_pipeline.generate_code_from_pdf_task.delay"
            ) as mock_delay,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body,
                    current_user=user,
                    db=db_session,
                )

            assert exc_info.value.status_code == 400
            assert "Upload not found in storage" in exc_info.value.detail
            # Task was not queued and storage counter was not modified
            mock_delay.assert_not_called()
            db_session.refresh(user)
            assert user.storage_bytes_used == 0

    asyncio.run(_run())


def test_quota_exceeded_based_on_verified_size(db_session, fake_request):
    """Scenario 4: User near quota limit uploads file whose verified size exceeds quota -> 429."""

    async def _run():
        user = User(
            id=4,
            email="near_limit@example.com",
            name="Near Limit",
            storage_bytes_used=490 * 1024 * 1024,  # 490 MB used of 500 MB limit
        )
        db_session.add(user)
        db_session.commit()

        real_size = 20 * 1024 * 1024  # 20 MB (would bring user to 510 MB > 500 MB limit)
        body = ConfirmUploadRequest(
            key="papers/uuid4_over.pdf",
            paper_name="Over Limit Paper",
            visibility="public",
            terms_accepted=True,
            file_size_bytes=100,  # Client lies about size
        )

        with (
            patch("backend.services.storage_service.get_object_size", return_value=real_size),
            patch("backend.redis_config.cache_redis", None),
            patch("backend.routers.papers_pipeline._STORAGE_QUOTA_BYTES", 500 * 1024 * 1024),
            patch(
                "backend.routers.papers_pipeline.generate_code_from_pdf_task.delay"
            ) as mock_delay,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await confirm_upload(
                    request=fake_request,
                    body=body,
                    current_user=user,
                    db=db_session,
                )

            assert exc_info.value.status_code == 429
            assert "Storage quota exceeded" in exc_info.value.detail
            mock_delay.assert_not_called()
            db_session.refresh(user)
            assert user.storage_bytes_used == 490 * 1024 * 1024

    asyncio.run(_run())
