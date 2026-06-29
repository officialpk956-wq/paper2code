from fastapi import APIRouter, HTTPException
from backend.database import ping_db

router = APIRouter(tags=["Health"])

from fastapi import Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.database import get_db

@router.get("/health")
@router.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    checks = {}
    overall = "healthy"
    
    # Check DB
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
        overall = "unhealthy"
    
    # Check Redis
    try:
        import os
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.Redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=1)
        r.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
        overall = "degraded"
    
    status_code = 200 if overall in ["healthy", "degraded"] else 503
    return JSONResponse(
        {"status": overall, "checks": checks},
        status_code=status_code
    )

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

from backend.dependencies import get_current_user
from backend.models import User
from backend import metrics

def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@router.get("/api/metrics")
def get_metrics(current_user: User = Depends(get_current_admin)):
    return metrics.get_metrics()
