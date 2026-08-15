"""
tests/test_auth_registration_fix.py

Tests verifying that AuthService registration creates exactly one verification
token, dispatches a single email, updates both is_verified and is_email_verified
flags upon token verification, and correctly handles duplicate/expired tokens.
"""

import datetime
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models import User
from backend.modules.auth.models import VerificationToken, AuditLog
from backend.modules.auth.repositories.verification_repository import hash_token
from backend.modules.auth.services.auth_service import AuthService
from backend.modules.auth.services.verification_service import VerificationService


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


def test_registration_single_token_and_email(db_session):
    with patch("backend.modules.auth.services.email_service.EmailService.send_verification_email") as mock_email:
        auth_service = AuthService(db_session)
        user = auth_service.register(
            email="single_dispatch@example.com",
            name="Test User",
            password="StrongPassword123!@",
            ip_address="127.0.0.1",
            user_agent="pytest",
        )

        assert user.id is not None
        assert user.email == "single_dispatch@example.com"
        assert user.is_verified is False
        assert user.is_email_verified is False

        # Verify exactly ONE email was sent
        assert mock_email.call_count == 1
        call_email, call_token = mock_email.call_args[0]
        assert call_email == "single_dispatch@example.com"
        assert len(call_token) > 20

        # Verify exactly ONE token was created in DB
        tokens = db_session.query(VerificationToken).filter_by(user_id=user.id).all()
        assert len(tokens) == 1
        assert tokens[0].token_hash == hash_token(call_token)
        assert tokens[0].used is False


def test_verification_updates_both_flags(db_session):
    with patch("backend.modules.auth.services.email_service.EmailService.send_verification_email") as mock_email:
        auth_service = AuthService(db_session)
        user = auth_service.register(
            email="verify_flags@example.com",
            name="Verify Flags",
            password="StrongPassword123!@",
            ip_address="127.0.0.1",
            user_agent="pytest",
        )
        token = mock_email.call_args[0][1]

        verification_service = VerificationService(db_session)
        result = verification_service.verify_email(token, ip_address="127.0.0.1", user_agent="pytest")
        assert result is True

        # Refresh user from DB
        db_session.refresh(user)
        assert user.is_verified is True
        assert user.is_email_verified is True
        assert user.email_verified_at is not None


def test_token_reuse_rejected(db_session):
    with patch("backend.modules.auth.services.email_service.EmailService.send_verification_email") as mock_email:
        auth_service = AuthService(db_session)
        user = auth_service.register(
            email="token_reuse@example.com",
            name="Reuse User",
            password="StrongPassword123!@",
            ip_address="127.0.0.1",
            user_agent="pytest",
        )
        token = mock_email.call_args[0][1]

        verification_service = VerificationService(db_session)
        verification_service.verify_email(token, ip_address="127.0.0.1", user_agent="pytest")

        # Second verification attempt with same token must fail
        with pytest.raises(HTTPException) as exc:
            verification_service.verify_email(token, ip_address="127.0.0.1", user_agent="pytest")
        assert exc.value.status_code == 400


def test_duplicate_registration_rejected(db_session):
    with patch("backend.modules.auth.services.email_service.EmailService.send_verification_email"):
        auth_service = AuthService(db_session)
        auth_service.register(
            email="dup@example.com",
            name="First User",
            password="StrongPassword123!@",
            ip_address="127.0.0.1",
            user_agent="pytest",
        )

        with pytest.raises(HTTPException) as exc:
            auth_service.register(
                email="dup@example.com",
                name="Second User",
                password="StrongPassword123!@",
                ip_address="127.0.0.1",
                user_agent="pytest",
            )
        assert exc.value.status_code == 400
        assert "already registered" in exc.value.detail.lower()


def test_expired_token_rejected(db_session):
    with patch("backend.modules.auth.services.email_service.EmailService.send_verification_email") as mock_email:
        auth_service = AuthService(db_session)
        user = auth_service.register(
            email="expired@example.com",
            name="Expired User",
            password="StrongPassword123!@",
            ip_address="127.0.0.1",
            user_agent="pytest",
        )
        token = mock_email.call_args[0][1]

        # Manually expire token
        vt = db_session.query(VerificationToken).filter_by(token_hash=hash_token(token)).first()
        vt.expires_at = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        db_session.commit()

        verification_service = VerificationService(db_session)
        with pytest.raises(HTTPException) as exc:
            verification_service.verify_email(token, ip_address="127.0.0.1", user_agent="pytest")
        assert exc.value.status_code == 400
        assert "expired" in exc.value.detail.lower()
