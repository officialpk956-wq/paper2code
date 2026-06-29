import os
import time
import redis
import logging
from typing import Optional
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

_redis_client = None
try:
    _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    _redis_client.ping()
except Exception as e:
    logger.warning(f"Could not connect to Redis at {REDIS_URL}. Falling back to in-memory: {e}")
    _redis_client = None

_in_memory_store = {}

def check_sliding_window_rate_limit(key: str, limit: int, window_seconds: int) -> bool:
    """Redis or in-memory sliding window rate limiter."""
    if os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "false":
        return True

    now = int(time.time())
    
    if _redis_client is not None:
        try:
            pipe = _redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, window_seconds)
            _, current_count, _, _ = pipe.execute()
            return current_count <= limit
        except Exception as e:
            logger.exception(f"Redis rate limit check error: {e}")
            
    # In-memory fallback
    if key not in _in_memory_store:
        _in_memory_store[key] = []
    
    history = [t for t in _in_memory_store[key] if t > now - window_seconds]
    _in_memory_store[key] = history
    
    if len(history) < limit:
        _in_memory_store[key].append(now)
        return True
    return False

def get_rate_limit_key(request: Request) -> str:
    """
    Resolve rate limit identity with priority:
    API Key -> Authenticated User -> Organization Member -> Client IP
    """
    # 1. API Key ID
    api_key_id = request.state.api_key_id if hasattr(request.state, "api_key_id") else None
    if api_key_id:
        return f"rl:apikey:{api_key_id}"
        
    # 2. User ID
    user_id = request.state.user_id if hasattr(request.state, "user_id") else None
    if user_id:
        return f"rl:user:{user_id}"
        
    # 3. Org ID
    org_id = request.state.org_id if hasattr(request.state, "org_id") else None
    if org_id:
        return f"rl:org:{org_id}"
        
    # 4. Client IP address (using safe proxy resolution)
    from backend.modules.auth.middleware.rate_limit import get_client_ip
    ip = get_client_ip(request)
    return f"rl:ip:{ip}"

def multi_key_rate_limiter(limit: int, window_seconds: int):
    """FastAPI dependency for scoped rate limiting."""
    def dependency(request: Request):
        key_prefix = get_rate_limit_key(request)
        path = request.url.path
        key = f"{key_prefix}:{path}"
        
        if not check_sliding_window_rate_limit(key, limit, window_seconds):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )
        return True
    return dependency
