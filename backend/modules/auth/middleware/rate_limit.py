import os
import time
import logging
import redis
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Redis Connection Setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_client = None
try:
    _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Ping to check if alive
    _redis_client.ping()
except Exception as e:
    logger.warning(f"Could not connect to Redis at {REDIS_URL}. Falling back to in-memory rate limiting: {e}")
    _redis_client = None

# Simple in-memory fallback store
_in_memory_store = {}

def check_rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    """
    Returns True if the request is within limits, False otherwise.
    Uses sliding window log / counter via Redis or fallback memory.
    """
    now = int(time.time())
    
    if _redis_client is not None:
        try:
            # We use a pipeline for transactions
            pipe = _redis_client.pipeline()
            # Clear old entries in sorted set
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            # Count remaining entries
            pipe.zcard(key)
            # Add current entry
            pipe.zadd(key, {str(now): now})
            # Set TTL on key
            pipe.expire(key, window_seconds)
            
            # Execute
            _, current_count, _, _ = pipe.execute()
            
            return current_count <= limit
        except Exception as e:
            logger.exception(f"Redis rate limit check error: {e}", exc_info=True)
            # Fail open or fallback to in-memory
    
    # In-memory fallback
    if key not in _in_memory_store:
        _in_memory_store[key] = []
    
    # Filter expired timestamps
    history = [t for t in _in_memory_store[key] if t > now - window_seconds]
    _in_memory_store[key] = history
    
    if len(history) < limit:
        _in_memory_store[key].append(now)
        return True
    return False

# Trusted proxies configuration
_trusted_proxies_raw = os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1")
TRUSTED_PROXIES = {ip.strip() for ip in _trusted_proxies_raw.split(",") if ip.strip()}

def get_client_ip(request: Request) -> str:
    """
    Safely extract client IP. Trusts X-Forwarded-For only if immediate client
    host is a configured trusted proxy, preventing client IP spoofing.
    """
    client_ip = request.client.host if request.client else "unknown"
    
    if client_ip in TRUSTED_PROXIES:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            ips = [ip.strip() for ip in forwarded.split(",")]
            # Traverse from right to left, returning first untrusted IP
            for ip in reversed(ips):
                if ip not in TRUSTED_PROXIES:
                    return ip
            return ips[0]
            
    return client_ip

def rate_limiter(limit: int, window_seconds: int):
    """
    FastAPI dependency creator for rate limiting.
    """
    def dependency(request: Request):
        # Resolve client IP safely
        ip = get_client_ip(request)
        # Create a unique key per endpoint + IP
        path = request.url.path
        key = f"rate_limit:{path}:{ip}"
        
        if not check_rate_limit(key, limit, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )
        return True
    return dependency
