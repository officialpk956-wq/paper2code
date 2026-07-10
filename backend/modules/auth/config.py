import os

SECRET_KEY: str = os.getenv("SECRET_KEY", "")
_UNSAFE_KEYS = {
    "",
    "demo",
    "secret",
    "changeme",
    "supersecretkey",
    "supersecretkey_please_change_in_production",
}
if SECRET_KEY in _UNSAFE_KEYS:
    if os.getenv("ENVIRONMENT", "development") == "production":
        raise RuntimeError(
            "SECRET_KEY is not set or is using an unsafe default. "
            'Generate one: python -c "import secrets; print(secrets.token_hex(32))"'
        )
    # Give a default in development so it doesn't crash if unset
    SECRET_KEY = "supersecretkey_please_change_in_production"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 30
