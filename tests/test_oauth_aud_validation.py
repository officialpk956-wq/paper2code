"""
tests/test_oauth_aud_validation.py

Unit tests for OAuth ID token validation in GoogleProvider, specifically verifying:
1. Valid token (aud matches GOOGLE_CLIENT_ID, valid iss, email_verified=True) -> returns OAuthUserInfo
2. Token issued for a DIFFERENT client (aud mismatch) -> returns None (prevents cross-app replay)
3. Token with invalid issuer (iss mismatch) -> returns None
4. GOOGLE_CLIENT_ID unset / empty in environment -> fails closed (returns None)
5. email_verified=False with matching aud -> returns None
6. GitHubProvider unaffected and still functions as expected
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.modules.auth.oauth.provider import GitHubProvider, GoogleProvider, OAuthUserInfo


@pytest.fixture
def mock_google_client_id(monkeypatch):
    test_client_id = "1234567890-testapps.googleusercontent.com"
    monkeypatch.setenv("GOOGLE_CLIENT_ID", test_client_id)
    return test_client_id


def test_google_oauth_valid_token(mock_google_client_id):
    """Scenario 1: aud matches GOOGLE_CLIENT_ID, iss valid, email_verified=true -> returns OAuthUserInfo."""
    provider = GoogleProvider()
    fake_token = "valid_google_id_token"

    mock_response_data = {
        "iss": "https://accounts.google.com",
        "aud": mock_google_client_id,
        "sub": "google-user-123",
        "email": "user@example.com",
        "email_verified": "true",
        "name": "Jane Doe",
        "picture": "https://lh3.googleusercontent.com/a/avatar.png",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info(fake_token))

        assert user_info is not None
        assert isinstance(user_info, OAuthUserInfo)
        assert user_info.provider == "google"
        assert user_info.uid == "google-user-123"
        assert user_info.email == "user@example.com"
        assert user_info.name == "Jane Doe"
        assert user_info.avatar_url == "https://lh3.googleusercontent.com/a/avatar.png"

        # Verify Google TokenInfo endpoint was called with the token query parameter
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert f"id_token={fake_token}" in called_url


def test_google_oauth_valid_token_alternate_issuer(mock_google_client_id):
    """Valid token with 'accounts.google.com' (without https://) is also accepted."""
    provider = GoogleProvider()
    mock_response_data = {
        "iss": "accounts.google.com",
        "aud": mock_google_client_id,
        "sub": "google-user-456",
        "email": "user2@example.com",
        "email_verified": True,
        "name": "John Doe",
        "picture": None,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info("alternate_iss_token"))

        assert user_info is not None
        assert user_info.uid == "google-user-456"
        assert user_info.email == "user2@example.com"


def test_google_oauth_aud_mismatch_rejected(mock_google_client_id, caplog):
    """Scenario 2: Token issued for a DIFFERENT client (aud mismatch) -> returns None and logs warning without leaking token."""
    provider = GoogleProvider()
    secret_token_value = "SECRET_VICTIM_TOKEN_DO_NOT_LOG"

    # Token issued for an attacker app or third-party app
    mock_response_data = {
        "iss": "https://accounts.google.com",
        "aud": "attacker-client-id.apps.googleusercontent.com",
        "sub": "victim-user-999",
        "email": "victim@example.com",
        "email_verified": "true",
        "name": "Victim User",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info(secret_token_value))

        # Mismatched audience must be rejected
        assert user_info is None

        # Verify token value is NEVER logged
        assert secret_token_value not in caplog.text


def test_google_oauth_invalid_issuer_rejected(mock_google_client_id):
    """Scenario 3: Token with invalid iss (fake issuer) -> returns None."""
    provider = GoogleProvider()

    mock_response_data = {
        "iss": "https://fake-google-accounts.evil.com",
        "aud": mock_google_client_id,
        "sub": "user-789",
        "email": "attacker@evil.com",
        "email_verified": True,
        "name": "Evil User",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info("invalid_iss_token"))

        assert user_info is None


def test_google_oauth_missing_env_fails_closed(monkeypatch):
    """Scenario 4: GOOGLE_CLIENT_ID unset in environment -> fails closed (returns None)."""
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    provider = GoogleProvider()

    mock_response_data = {
        "iss": "https://accounts.google.com",
        "aud": "some-client-id.apps.googleusercontent.com",
        "sub": "user-111",
        "email": "user@example.com",
        "email_verified": True,
        "name": "User",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info("token_without_env_config"))

        assert user_info is None


def test_google_oauth_empty_env_fails_closed(monkeypatch):
    """Scenario 4b: GOOGLE_CLIENT_ID set to empty string -> fails closed (returns None)."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "   ")
    provider = GoogleProvider()

    mock_response_data = {
        "iss": "https://accounts.google.com",
        "aud": "some-client-id.apps.googleusercontent.com",
        "sub": "user-111",
        "email": "user@example.com",
        "email_verified": True,
        "name": "User",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info("token_with_whitespace_env"))

        assert user_info is None


def test_google_oauth_unverified_email_rejected(mock_google_client_id):
    """Scenario 5: email_verified=false with correct aud -> returns None."""
    provider = GoogleProvider()

    mock_response_data = {
        "iss": "https://accounts.google.com",
        "aud": mock_google_client_id,
        "sub": "unverified-user",
        "email": "unverified@example.com",
        "email_verified": "false",
        "name": "Unverified User",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_data

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info("unverified_email_token"))

        assert user_info is None


def test_google_oauth_tokeninfo_http_error(mock_google_client_id):
    """Tokeninfo endpoint returning non-200 (e.g. 400 Bad Request) returns None."""
    provider = GoogleProvider()

    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = "Invalid Value"

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        user_info = asyncio.run(provider.get_user_info("bad_token"))

        assert user_info is None


def test_github_provider_untouched():
    """Scenario 6: GitHubProvider remains operational and untouched."""
    provider = GitHubProvider()
    token = "gho_valid_github_token"

    mock_user_resp = MagicMock()
    mock_user_resp.status_code = 200
    mock_user_resp.json.return_value = {
        "id": 98765,
        "email": "githubuser@example.com",
        "name": "GitHub Dev",
        "avatar_url": "https://avatars.githubusercontent.com/u/98765",
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_user_resp
        user_info = asyncio.run(provider.get_user_info(token))

        assert user_info is not None
        assert user_info.provider == "github"
        assert user_info.uid == "98765"
        assert user_info.email == "githubuser@example.com"
        assert user_info.name == "GitHub Dev"
