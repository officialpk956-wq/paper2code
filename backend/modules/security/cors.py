import os

# Load configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_METHODS_RAW = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS")
ALLOWED_HEADERS_RAW = os.getenv("ALLOWED_HEADERS", "Content-Type,Authorization")
ALLOW_CREDENTIALS_RAW = os.getenv("ALLOW_CREDENTIALS", "true")


def get_allowed_origins() -> list[str]:
    """Parse and sanitize allowed origins, handling development loopback domains."""
    env = os.getenv("ENVIRONMENT", "development")
    allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "")
    if not allowed_origins_raw:
        # Development fallback
        if env == "production":
            raise ValueError("ALLOWED_ORIGINS environment variable is required in production")
        return ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"]

    origins = [origin.strip() for origin in allowed_origins_raw.split(",") if origin.strip()]

    # Validation
    if env == "production":
        if "*" in origins:
            raise ValueError("Wildcard '*' origin is forbidden in production environment")

    return origins


def get_allowed_methods() -> list[str]:
    methods_raw = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS")
    return [m.strip() for m in methods_raw.split(",") if m.strip()]


def get_allowed_headers() -> list[str]:
    headers_raw = os.getenv("ALLOWED_HEADERS", "Content-Type,Authorization")
    return [h.strip() for h in headers_raw.split(",") if h.strip()]


def get_allow_credentials() -> bool:
    allow_credentials_raw = os.getenv("ALLOW_CREDENTIALS", "true")
    return allow_credentials_raw.lower() in ("true", "1", "yes")
