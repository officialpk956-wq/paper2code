import os
import json
import jwt
import datetime
from typing import Dict, Any, Optional

# Env configuration
JWT_ACTIVE_KEY_ID = os.getenv("JWT_ACTIVE_KEY_ID", "key_v1")
JWT_KEY_RING_RAW = os.getenv("JWT_KEY_RING")

_WEAK_KEY_FRAGMENT = "change_in_production"

def get_key_ring() -> Dict[str, str]:
    if not JWT_KEY_RING_RAW:
        if os.getenv("ENVIRONMENT", "development") == "production":
            raise RuntimeError(
                "JWT_KEY_RING env var is not set. "
                "Refusing to start with no signing key in production."
            )
        return {"key_v1": "dev_only_key_not_for_production"}
    try:
        ring = json.loads(JWT_KEY_RING_RAW)
    except Exception as exc:
        raise RuntimeError(f"JWT_KEY_RING is not valid JSON: {exc}") from exc
    for kid, secret in ring.items():
        if _WEAK_KEY_FRAGMENT in secret:
            raise RuntimeError(
                f"JWT key ring contains the default placeholder key (kid={kid!r}). "
                "Set a strong random key before deploying to production."
            )
    return ring


def get_active_secret() -> str:
    ring = get_key_ring()
    if JWT_ACTIVE_KEY_ID not in ring:
        raise ValueError(f"Active key ID {JWT_ACTIVE_KEY_ID} is not present in the key ring")
    return ring[JWT_ACTIVE_KEY_ID]

def encode_rotated_jwt(payload: Dict[str, Any], expires_delta: Optional[datetime.timedelta] = None) -> str:
    """Encode JWT inserting 'kid' in headers and signing with active secret."""
    ring = get_key_ring()
    secret = get_active_secret()
    
    # Set expiration
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        
    payload["exp"] = expire
    headers = {"kid": JWT_ACTIVE_KEY_ID}
    
    return jwt.encode(payload, secret, algorithm="HS256", headers=headers)

def decode_rotated_jwt(token: str, audience: str, issuer: str) -> Dict[str, Any]:
    """Decode JWT resolving secret by 'kid' in headers, supporting legacy fallbacks."""
    ring = get_key_ring()
    
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
    except Exception:
        kid = None

    if kid:
        if kid not in ring:
            raise jwt.InvalidKeyError(f"Key ID {kid} not found in configured key ring")
        secret = ring[kid]
    else:
        # Fallback for legacy tokens: check overlap window or try active key
        secret = get_active_secret()

    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        audience=audience,
        issuer=issuer,
        options={"require": ["exp", "iss", "aud", "sub", "iat"]}
    )
