from fastapi import APIRouter, HTTPException
from backend.database import ping_db

router = APIRouter(tags=["Health"])

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/api/health/db")
def health_db():
    status = ping_db()
    if not status["ok"]:
        raise HTTPException(status_code=503, detail=status)
    return status

@router.get("/api/health/redis")
def health_redis():
    import os
    import redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis.Redis.from_url(redis_url, decode_responses=True)
        r.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/api/health/piston")
def health_piston():
    import os
    import httpx
    piston_url = os.environ.get("PISTON_URL", "http://localhost:2000")
    try:
        r = httpx.get(f"{piston_url}/api/v2/runtimes", timeout=3.0)
        r.raise_for_status()
        return {"status": "ok", "piston": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
