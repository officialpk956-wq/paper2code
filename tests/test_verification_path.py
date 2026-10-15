import datetime
from unittest.mock import patch

import pytest

from backend.models import EmailVerificationToken, User
from backend.modules.auth.models import AuditLog, VerificationToken
from backend.modules.auth.services.auth_service import AuthService
from backend.modules.auth.services.email_service import EmailService
from backend.repositories.token_repository import TokenRepository


@pytest.mark.parametrize("issuer", ["signup", "legacy"])
@pytest.mark.parametrize("state", ["valid_then_reused", "expired"])
def test_one_post_path_redeems_both_issuers_truthfully(client, db_session, issuer, state):
    with patch.object(EmailService, "send_verification_email") as send:
        user = AuthService(db_session).register(
            "verification@example.com", "Verify", "VerifyPass2026!", None, None
        )
    token = send.call_args.args[1]
    if issuer == "legacy":
        token = TokenRepository(db_session).create_email_verification(user.id)
    if state == "expired":
        model = VerificationToken if issuer == "signup" else EmailVerificationToken
        row = db_session.query(model).filter_by(user_id=user.id).one()
        row.expires_at = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
        db_session.commit()
        response = client.post("/api/auth/verify-email", json={"token": token})
        assert response.status_code == 400
        assert response.json()["detail"] == "Verification token has expired"
        db_session.refresh(user)
        assert not user.is_email_verified
    else:
        response = client.post("/api/auth/verify-email", json={"token": token})
        assert response.status_code == 200, response.text
        db_session.refresh(user)
        first_timestamp = user.email_verified_at
        assert user.is_verified and user.is_email_verified and first_timestamp
        response = client.post("/api/auth/verify-email", json={"token": token})
        assert response.status_code == 400
        assert response.json()["detail"] == "Verification link already used"
        db_session.refresh(user)
        assert user.email_verified_at == first_timestamp
        assert (
            db_session.query(AuditLog).filter_by(user_id=user.id, action="email_verified").count()
            == 1
        )


def test_get_signup_link_redirects_to_real_page_without_consuming_token(client, db_session):
    with patch.object(EmailService, "send_verification_email") as send:
        user = AuthService(db_session).register(
            "get@example.com", "Verify", "VerifyPass2026!", None, None
        )
    token = send.call_args.args[1]
    response = client.get("/api/auth/verify-email", params={"token": token}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].endswith("/auth/verified?token=" + token)
    db_session.refresh(user)
    assert not user.is_email_verified


def test_malformed_token_is_distinct_from_expiry(client):
    response = client.post("/api/auth/verify-email", json={"token": "bad-token"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid verification token"


def test_both_senders_emit_the_surviving_page():
    from backend.services.email_service import send_verification_email_sync

    with (
        patch("backend.modules.auth.services.email_service._send") as new,
        patch("backend.services.email_service.send_email_sync") as old,
    ):
        EmailService.send_verification_email("test@example.com", "fixture")
        send_verification_email_sync("test@example.com", "fixture")
    for html in (new.call_args.kwargs["html"], old.call_args.args[2]):
        assert "/auth/verified?token=fixture" in html
        assert "/verify-email?" not in html
