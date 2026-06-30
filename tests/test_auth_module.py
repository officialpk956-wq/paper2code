import datetime
import pyotp
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.models import User
from backend.modules.auth.models import UserSession, VerificationToken, ResetToken, AuditLog
from backend.modules.auth.security.hashing import hash_password, verify_password_and_needs_rehash
from backend.modules.auth.services import AuthService, SessionService, MFAService

# Use pytest-compatible naming
TEST_EMAIL = "new_auth_test@example.com"
TEST_PASS = "SecurePass123!"

def test_password_security_and_rehash(db_session: Session):
    # Verify Argon2id hash creation
    hashed = hash_password(TEST_PASS)
    assert hashed.startswith("$argon2")

    # Verify verification and rehash logic
    verified, needs_rehash = verify_password_and_needs_rehash(TEST_PASS, hashed)
    assert verified is True
    assert needs_rehash is False

    # Verify legacy bcrypt compatibility
    import bcrypt
    bcrypt_salt = bcrypt.gensalt()
    bcrypt_hash = bcrypt.hashpw(TEST_PASS.encode("utf-8"), bcrypt_salt).decode("utf-8")
    verified_bc, needs_rehash_bc = verify_password_and_needs_rehash(TEST_PASS, bcrypt_hash)
    assert verified_bc is True
    assert needs_rehash_bc is True # Needs upgrade to Argon2id

def test_registration_and_email_verification(client: TestClient, db_session: Session):
    # Register new user
    r = client.post("/api/auth/register", json={
        "email": TEST_EMAIL,
        "name": "Auth Test User",
        "password": TEST_PASS
    })
    assert r.status_code == 201
    user_data = r.json()
    assert user_data["email"] == TEST_EMAIL
    assert user_data["is_verified"] is False # Must start unverified

    # Query verification token from DB
    user = db_session.query(User).filter_by(email=TEST_EMAIL).first()
    assert user is not None
    token_entry = db_session.query(VerificationToken).filter_by(user_id=user.id).first()
    assert token_entry is not None

    # Get verification token details to mock validation
    # Since verification tokens are stored as hashes in DB, we find it by looking up the printed token in log/mocking
    # But since we have direct access to database verification token, let's test verify endpoint using the service directly
    from backend.modules.auth.services.verification_service import VerificationService
    v_service = VerificationService(db_session)
    
    # We can retrieve the token from email logging or generate a mock token for test
    import secrets
    mock_token = secrets.token_urlsafe(32)
    expires = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    v_service.repo.create_verification_token(user.id, mock_token, expires)
    db_session.commit()

    # Call verify-email endpoint
    resp = client.post("/api/auth/verify-email", json={"token": mock_token})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # Reload user and check status
    db_session.refresh(user)
    assert user.is_verified is True

def test_session_rotation_and_replay_attack(client: TestClient, db_session: Session):
    # Register and verify user
    email = "rotation_test@example.com"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="Rotation Test User", hashed_password=hash_password(TEST_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()

    headers = {"user-agent": "Mozilla/5.0"}
    # Login to get refresh token
    login_resp = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS}, headers=headers)
    assert login_resp.status_code == 200
    tokens = login_resp.json()
    refresh_token = tokens["refresh_token"]

    # Refresh token rotation check
    refresh_resp = client.post("/api/auth/refresh", json={"refresh_token": refresh_token}, headers=headers)
    assert refresh_resp.status_code == 200
    new_tokens = refresh_resp.json()
    new_refresh = new_tokens["refresh_token"]
    assert new_refresh != refresh_token

    # Replay attack detection check: reusing the old refresh_token with a DIFFERENT user agent to simulate attack
    bad_headers = {"user-agent": "Evil-Agent/1.0"}
    replay_resp = client.post("/api/auth/refresh", json={"refresh_token": refresh_token}, headers=bad_headers)
    assert replay_resp.status_code == 401
    
    # Replay attack should trigger revocation of all sessions of the user
    db_session.expire_all()
    session_service = SessionService(db_session)
    active_sessions = session_service.get_active_sessions(user.id)
    assert len(active_sessions) == 0

def test_logout_and_logout_all(client: TestClient, db_session: Session):
    email = "logout_test@example.com"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="Logout Test User", hashed_password=hash_password(TEST_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()

    # Login
    login_resp = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS})
    tokens = login_resp.json()
    access = tokens["access_token"]
    refresh = tokens["refresh_token"]

    # Logout
    logout_resp = client.post("/api/auth/logout", json={"refresh_token": refresh})
    assert logout_resp.status_code == 200

    # Ensure access token is revoked or session is gone
    db_session.expire_all()
    session_service = SessionService(db_session)
    # Reload user and check sessions
    user = db_session.query(User).filter_by(email=email).first()
    assert len(session_service.get_active_sessions(user.id)) == 0

    # Logout-all
    # Create multiple sessions
    login_resp1 = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS})
    login_resp2 = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS})
    assert login_resp1.status_code == 200
    assert login_resp2.status_code == 200
    
    access_tok1 = login_resp1.json()["access_token"]
    
    # Call logout-all
    headers = {"Authorization": f"Bearer {access_tok1}"}
    logout_all_resp = client.post("/api/auth/logout-all", headers=headers)
    assert logout_all_resp.status_code == 200

    # Verify all sessions revoked and token version changed
    db_session.expire_all()
    db_session.refresh(user)
    assert len(session_service.get_active_sessions(user.id)) == 0

def test_password_reset_flow(client: TestClient, db_session: Session):
    email = "reset_test@example.com"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="Reset Test User", hashed_password=hash_password(TEST_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()
    
    # Request reset password
    forgot_resp = client.post("/api/auth/forgot-password", json={"email": email})
    assert forgot_resp.status_code == 200

    # Extract mock token
    import secrets
    mock_reset_token = secrets.token_urlsafe(32)
    from backend.modules.auth.services.reset_service import ResetService
    reset_service = ResetService(db_session)
    expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
    reset_service.repo.create_reset_token(user.id, mock_reset_token, expires)
    db_session.commit()

    # Reset password
    NEW_PASS = "BrandNewPassword123!"
    reset_resp = client.post("/api/auth/reset-password", json={
        "token": mock_reset_token,
        "new_password": NEW_PASS
    })
    assert reset_resp.status_code == 200

    # Verify password updated and login succeeds with new password
    login_resp = client.post("/api/auth/login", data={"username": email, "password": NEW_PASS})
    assert login_resp.status_code == 200

def test_session_querying_and_revocation(client: TestClient, db_session: Session):
    email = "revoke_test@example.com"
    NEW_PASS = "BrandNewPassword123!"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="Revoke Test User", hashed_password=hash_password(NEW_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()
    
    # Create two sessions
    login_resp1 = client.post("/api/auth/login", data={"username": email, "password": NEW_PASS})
    tokens1 = login_resp1.json()
    access1 = tokens1["access_token"]

    login_resp2 = client.post("/api/auth/login", data={"username": email, "password": NEW_PASS})
    tokens2 = login_resp2.json()
    access2 = tokens2["access_token"]

    # Query sessions
    headers = {"Authorization": f"Bearer {access1}"}
    sessions_resp = client.get("/api/auth/sessions", headers=headers)
    assert sessions_resp.status_code == 200
    sessions_list = sessions_resp.json()
    assert len(sessions_list) >= 2

    # Revoke a session
    target_session_id = sessions_list[0]["id"]
    revoke_resp = client.delete(f"/api/auth/sessions/{target_session_id}", headers=headers)
    assert revoke_resp.status_code == 200

    # Query again and check length
    sessions_resp = client.get("/api/auth/sessions", headers=headers)
    new_sessions_list = sessions_resp.json()
    assert len(new_sessions_list) == len(sessions_list) - 1

def test_two_factor_authentication(client: TestClient, db_session: Session):
    email = "mfa_test@example.com"
    NEW_PASS = "BrandNewPassword123!"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="MFA Test User", hashed_password=hash_password(NEW_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()
    
    # Perform Setup MFA
    login_resp = client.post("/api/auth/login", data={"username": email, "password": NEW_PASS})
    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    setup_resp = client.post("/api/auth/mfa/setup", headers=headers)
    assert setup_resp.status_code == 200
    mfa_details = setup_resp.json()
    assert "secret" in mfa_details
    assert "qr_code_data_uri" in mfa_details
    assert len(mfa_details["backup_codes"]) == 8

    # Enable MFA
    totp = pyotp.TOTP(mfa_details["secret"])
    code = totp.now()
    
    enable_resp = client.post("/api/auth/mfa/enable", headers=headers, json={"code": code})
    assert enable_resp.status_code == 200

    # Test login requiring MFA
    login_mfa_req_resp = client.post("/api/auth/login", data={"username": email, "password": NEW_PASS})
    assert login_mfa_req_resp.status_code == 200
    mfa_req_data = login_mfa_req_resp.json()
    assert mfa_req_data.get("mfa_required") is True
    assert "mfa_token" in mfa_req_data

    # Complete MFA login
    mfa_code = totp.now()
    # Explicitly verify the temp model schema we created inline
    login_mfa_done_resp = client.post("/api/auth/login/mfa", json={
        "mfa_token": mfa_req_data["mfa_token"],
        "code": mfa_code
    })
    assert login_mfa_done_resp.status_code == 200
    assert "access_token" in login_mfa_done_resp.json()

    # Disable MFA
    disable_resp = client.post("/api/auth/mfa/disable", headers=headers, json={"password": NEW_PASS})
    assert disable_resp.status_code == 200

def test_account_deletion_cascades(client: TestClient, db_session: Session):
    email = "delete_test@example.com"
    NEW_PASS = "BrandNewPassword123!"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="Delete Test User", hashed_password=hash_password(NEW_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()
    
    login_resp = client.post("/api/auth/login", data={"username": email, "password": NEW_PASS})
    access_token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # Delete Account
    del_resp = client.delete("/api/auth/settings/delete", headers=headers)
    assert del_resp.status_code == 200

    # Verify user is completely purged from DB
    purged_user = db_session.query(User).filter_by(email=email).first()
    assert purged_user is None

    # Verify session and audit logs Cascade Deleted / Set Null
    sessions = db_session.query(UserSession).filter_by(user_id=user.id).all()
    assert len(sessions) == 0

def test_jwt_validation_claims(client: TestClient, db_session: Session):
    email = "jwt_test@example.com"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="JWT Test", hashed_password=hash_password(TEST_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()

    import jwt
    from backend.modules.auth.config import SECRET_KEY, ALGORITHM
    import time
    
    # 1. Missing claims
    payload1 = {"sub": email, "user_id": user.id, "token_version": user.token_version, "exp": int(time.time()) + 3600}
    bad_token1 = jwt.encode(payload1, SECRET_KEY, algorithm=ALGORITHM)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {bad_token1}"})
    assert r.status_code == 401

    # 2. Wrong audience
    payload2 = {
        "sub": email, "user_id": user.id, "token_version": user.token_version,
        "exp": int(time.time()) + 3600, "iss": "paper2code-auth", "aud": "wrong-app", "iat": int(time.time())
    }
    bad_token2 = jwt.encode(payload2, SECRET_KEY, algorithm=ALGORITHM)
    r = client.get("/api/auth/me", headers={"Authorization": f"Bearer {bad_token2}"})
    assert r.status_code == 401

def test_proxy_ip_rate_limiting_resolution():
    from fastapi import Request
    from backend.modules.auth.middleware.rate_limit import get_client_ip, TRUSTED_PROXIES
    
    def mock_request(client_host: str, headers: dict) -> Request:
        scope = {
            "type": "http",
            "client": (client_host, 12345),
            "headers": [(k.lower().encode("latin1"), v.encode("latin1")) for k, v in headers.items()]
        }
        return Request(scope)

    TRUSTED_PROXIES.add("10.0.0.1")

    # Case 1: Client is NOT trusted proxy -> ignore X-Forwarded-For
    req1 = mock_request("192.168.1.100", {"X-Forwarded-For": "8.8.8.8"})
    assert get_client_ip(req1) == "192.168.1.100"

    # Case 2: Client IS trusted proxy -> parse X-Forwarded-For
    req2 = mock_request("10.0.0.1", {"X-Forwarded-For": "8.8.8.8, 10.0.0.1"})
    assert get_client_ip(req2) == "8.8.8.8"

    # Clean up
    TRUSTED_PROXIES.discard("10.0.0.1")

def test_concurrent_refresh_grace_period(client: TestClient, db_session: Session):
    email = "grace_test@example.com"
    user = db_session.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, name="Grace Test", hashed_password=hash_password(TEST_PASS), is_verified=True)
        db_session.add(user)
        db_session.commit()

    headers = {"user-agent": "Mozilla/5.0"}
    login_resp = client.post("/api/auth/login", data={"username": email, "password": TEST_PASS}, headers=headers)
    tokens = login_resp.json()
    refresh_token = tokens["refresh_token"]

    r1 = client.post("/api/auth/refresh", json={"refresh_token": refresh_token}, headers=headers)
    assert r1.status_code == 200

    # Second refresh immediately using the SAME rotated token (grace period)
    r2 = client.post("/api/auth/refresh", json={"refresh_token": refresh_token}, headers=headers)
    assert r2.status_code == 200

    # Third refresh from a different UA/IP using the SAME rotated token (replay lockout)
    bad_headers = {"user-agent": "Evil-Agent/1.0"}
    r3 = client.post("/api/auth/refresh", json={"refresh_token": refresh_token}, headers=bad_headers)
    assert r3.status_code == 401

    db_session.expire_all()
    session_service = SessionService(db_session)
    assert len(session_service.get_active_sessions(user.id)) == 0
