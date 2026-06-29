import os
import redis

REDIS_CACHE_URL = os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/0")
REDIS_SESSIONS_URL = os.getenv("REDIS_SESSIONS_URL", "redis://localhost:6379/1")
REDIS_RATE_LIMIT_URL = os.getenv("REDIS_RATE_LIMIT_URL", "redis://localhost:6379/2")

try:
    cache_redis = redis.from_url(REDIS_CACHE_URL, decode_responses=True, socket_connect_timeout=2)
    cache_redis.ping()
except Exception:
    cache_redis = None

try:
    session_redis = redis.from_url(REDIS_SESSIONS_URL, decode_responses=True, socket_connect_timeout=2)
    session_redis.ping()
except Exception:
    session_redis = None

try:
    rate_limit_redis = redis.from_url(REDIS_RATE_LIMIT_URL, decode_responses=True, socket_connect_timeout=2)
    rate_limit_redis.ping()
except Exception:
    rate_limit_redis = None
