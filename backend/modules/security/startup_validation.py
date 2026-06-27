import os
from backend.modules.security.cors import get_allowed_origins
from backend.modules.security.jwt_rotation import get_key_ring, JWT_ACTIVE_KEY_ID

def validate_production_security_config() -> None:
    """Enforce strict configurations verification to prevent insecure deployment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    # 1. CORS Validation
    origins = get_allowed_origins()
    if env == "production":
        if "*" in origins:
            raise ValueError("SECURITY FAILURE: Wildcard '*' origin is forbidden in production environment")

    # 2. JWT Configuration
    ring = get_key_ring()
    if not ring:
        raise ValueError("SECURITY FAILURE: JWT signing keys are completely missing from the key ring")
    if JWT_ACTIVE_KEY_ID not in ring:
        raise ValueError(f"SECURITY FAILURE: Active signing key ID {JWT_ACTIVE_KEY_ID} not found in configured key ring")
        
    # 3. Content Security Policy (CSP)
    csp = os.getenv("CONTENT_SECURITY_POLICY")
    if env == "production" and not csp:
        raise ValueError("SECURITY FAILURE: Content Security Policy (CSP) header configuration is missing in production")
        
    # 4. Redis Check if required
    if os.getenv("REDIS_REQUIRED", "false").lower() in ("true", "1", "yes"):
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            r = redis.Redis.from_url(url)
            r.ping()
        except Exception as e:
            raise ValueError(f"STARTUP FAILURE: Redis is required but unavailable at {url}: {e}")
            
    print("Security configuration validation passed successfully.")
