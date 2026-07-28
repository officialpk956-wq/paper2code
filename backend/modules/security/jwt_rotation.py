import datetime
import json
import logging
import os
from typing import Any

import jwt

logger = logging.getLogger(__name__)

# Env configuration
JWT_ACTIVE_KEY_ID = os.getenv("JWT_ACTIVE_KEY_ID", "key_v1")

_WEAK_KEY_FRAGMENT = "change_in_production"


def get_key_ring() -> dict[str, str]:
    """Return {kid: secret}. Prefers a JWT_KEY_RING JSON object; if that's unset or
    malformed, falls back to the single SECRET_KEY so a misconfigured env var can't
    brick every deploy. Weak/placeholder keys are still rejected outright."""
    raw = (os.getenv("JWT_KEY_RING") or "").strip()
    ring: dict[str, str] | None = None
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("JWT_KEY_RING is not valid JSON (%s); falling back to SECRET_KEY.", exc)
            parsed = None
        if isinstance(parsed, dict) and parsed:
            ring = {str(k): str(v) for k, v in parsed.items()}
        elif parsed is not None:
            logger.error("JWT_KEY_RING must be a non-empty JSON object {kid: secret}; ignoring it.")

    if ring:
        for kid, secret in ring.items():
            if _WEAK_KEY_FRAGMENT in secret:
                raise RuntimeError(
                    f"JWT key ring contains the default placeholder key (kid={kid!r}). "
                    "Set a strong random key before deploying to production."
                )
        return ring

    # No usable key ring — fall back to the single SECRET_KEY (already required and
    # set in every environment). This keeps the service bootable.
    secret = os.getenv("SECRET_KEY", "")
    if secret and _WEAK_KEY_FRAGMENT not in secret and len(secret) >= 32:
        logger.warning(
            "JWT_KEY_RING not configured; signing JWTs with SECRET_KEY (kid=%s).", JWT_ACTIVE_KEY_ID
        )
        return {JWT_ACTIVE_KEY_ID: secret}

    if os.getenv("ENVIRONMENT", "development") == "production":
        raise RuntimeError(
            "No JWT signing key configured. Set JWT_KEY_RING to a JSON object like "
            '{"key_v1":"<strong 32+ char secret>"} or set a strong SECRET_KEY.'
        )
    return {"key_v1": "dev_only_key_not_for_production"}


def get_active_secret() -> str:
    ring = get_key_ring()
    if JWT_ACTIVE_KEY_ID not in ring:
        raise ValueError(f"Active key ID {JWT_ACTIVE_KEY_ID} is not present in the key ring")
    return ring[JWT_ACTIVE_KEY_ID]


def encode_rotated_jwt(
    payload: dict[str, Any], expires_delta: datetime.timedelta | None = None
) -> str:
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


def resolve_signing_secret(token: str) -> str:
    """Resolve the secret that signed `token` by its 'kid' header, falling back
    to the active key for legacy tokens with no kid."""
    ring = get_key_ring()

    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
    except Exception:
        kid = None

    if kid:
        if kid not in ring:
            raise jwt.InvalidKeyError(f"Key ID {kid} not found in configured key ring")
        return ring[kid]
    return get_active_secret()


def decode_rotated_jwt(token: str, audience: str, issuer: str) -> dict[str, Any]:
    """Decode JWT resolving secret by 'kid' in headers, supporting legacy fallbacks."""
    secret = resolve_signing_secret(token)

    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        audience=audience,
        issuer=issuer,
        options={"require": ["exp", "iss", "aud", "sub", "iat"]},
    )
