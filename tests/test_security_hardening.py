import pytest
import os
import json
import jwt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.security.jwt_rotation import encode_rotated_jwt, decode_rotated_jwt, get_key_ring
from backend.modules.security.brute_force import is_locked_out, record_failed_attempt, reset_failed_attempts, MAX_ATTEMPTS
from backend.modules.security.cors import get_allowed_origins
from backend.modules.security.startup_validation import validate_production_security_config

TEST_EMAIL = "security_test@example.com"
TEST_PASS = "SecurePass123!"

def test_security_headers_present(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in resp.headers

def test_cors_validation_and_methods():
    # Dev configurations
    os.environ["ENVIRONMENT"] = "development"
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000,http://localhost:8000"
    origins = get_allowed_origins()
    assert "http://localhost:3000" in origins
    assert "http://localhost:8000" in origins

    # Production validation wildcard check
    os.environ["ENVIRONMENT"] = "production"
    os.environ["ALLOWED_ORIGINS"] = "*,http://localhost:3000"
    with pytest.raises(ValueError, match="Wildcard '\\*' origin is forbidden"):
        get_allowed_origins()

    # Reset
    os.environ["ENVIRONMENT"] = "development"
    os.environ["ALLOWED_ORIGINS"] = ""

def test_brute_force_lockout(db_session: Session):
    reset_failed_attempts(TEST_EMAIL)
    assert is_locked_out(TEST_EMAIL) is False

    # Simulate failed attempts up to limit
    for _ in range(MAX_ATTEMPTS):
        record_failed_attempt(TEST_EMAIL, db_session)

    assert is_locked_out(TEST_EMAIL) is True

    # Reset and verify unlocked
    reset_failed_attempts(TEST_EMAIL)
    assert is_locked_out(TEST_EMAIL) is False

def test_jwt_rotation_and_fallback():
    # Setup key ring
    os.environ["JWT_KEY_RING"] = json.dumps({
        "key_v1": "secret_version_1_key_ring_test",
        "key_v2": "secret_version_2_key_ring_test"
    })
    
    # 1. Sign with active key (v1)
    os.environ["JWT_ACTIVE_KEY_ID"] = "key_v1"
    token_v1 = encode_rotated_jwt({"sub": "user@example.com", "user_id": 1, "token_version": 1, "iss": "paper2code-auth", "aud": "paper2code-app", "iat": 1234567})
    
    decoded = decode_rotated_jwt(token_v1, audience="paper2code-app", issuer="paper2code-auth")
    assert decoded["sub"] == "user@example.com"

    # 2. Rotate to v2 and sign new token
    os.environ["JWT_ACTIVE_KEY_ID"] = "key_v2"
    token_v2 = encode_rotated_jwt({"sub": "user@example.com", "user_id": 1, "token_version": 1, "iss": "paper2code-auth", "aud": "paper2code-app", "iat": 1234567})
    
    # Check decode still accepts token signed with v1 because it resides in key ring!
    decoded_old = decode_rotated_jwt(token_v1, audience="paper2code-app", issuer="paper2code-auth")
    assert decoded_old["sub"] == "user@example.com"
    
    decoded_new = decode_rotated_jwt(token_v2, audience="paper2code-app", issuer="paper2code-auth")
    assert decoded_new["sub"] == "user@example.com"

    # Clean up
    os.environ.pop("JWT_KEY_RING", None)
    os.environ.pop("JWT_ACTIVE_KEY_ID", None)

def test_startup_validation_failures():
    # Trigger CORS validation failure on wildcard in production
    os.environ["ENVIRONMENT"] = "production"
    os.environ["ALLOWED_ORIGINS"] = "*"
    os.environ["CONTENT_SECURITY_POLICY"] = "default-src 'self'"
    
    with pytest.raises(ValueError, match="Wildcard '\\*' origin is forbidden"):
        validate_production_security_config()

    # Cleanup
    os.environ["ENVIRONMENT"] = "development"
    os.environ["ALLOWED_ORIGINS"] = ""
    os.environ.pop("CONTENT_SECURITY_POLICY", None)
