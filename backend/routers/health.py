import httpx
from fastapi import APIRouter, HTTPException

from backend.database import ping_db

router = APIRouter(tags=["Health"])

from fastapi import Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db


@router.get("/health")
@router.get("/api/health")
def health_check(db: Session = Depends(get_db)):
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
        from backend.redis_config import session_redis

        if session_redis is None:
            raise Exception("Redis connection is None")
        session_redis.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {str(e)}"
        if overall != "unhealthy":
            overall = "degraded"

    status_code = 200 if overall in ["healthy", "degraded"] else 503
    return JSONResponse({"status": overall, "checks": checks}, status_code=status_code)


@router.get("/api/health/db")
def health_db():
    status = ping_db()
    if not status["ok"]:
        raise HTTPException(status_code=503, detail=status)
    return status


@router.get("/api/health/redis")
def health_redis():
    from backend.redis_config import session_redis

    try:
        if session_redis is None:
            raise Exception("Redis connection is None")
        session_redis.ping()
        return {"status": "ok", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/api/health/celery")
def health_celery():
    from backend.celery_app import celery_app

    broker_reachable = False
    worker_responding = False
    try:
        with celery_app.connection_for_read(connect_timeout=1) as connection:
            connection.ensure_connection(max_retries=0)
            broker_reachable = True
            replies = celery_app.control.ping(timeout=1, connection=connection)
            worker_responding = any(
                reply.get("ok") == "pong" for worker in replies for reply in worker.values()
            )
    except Exception:
        # Connection errors may contain credential-bearing URLs; keep them private.
        pass
    ready = broker_reachable and worker_responding
    return JSONResponse(
        {
            "ready": ready,
            "broker_reachable": broker_reachable,
            "worker_responding": worker_responding,
        },
        status_code=200 if ready else 503,
    )


@router.get("/api/health/e2b")
async def health_e2b():
    from fastapi.concurrency import run_in_threadpool

    from backend.services.e2b_service import run_code_in_sandbox

    try:
        res = await run_in_threadpool(
            run_code_in_sandbox, user_code='print("ok")', run_timeout_ms=5000
        )
        if res.get("passed"):
            return {"status": "ok", "e2b": "connected"}
        else:
            raise Exception(res.get("stderr") or "Trivial run failed")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


from backend import metrics
from backend.dependencies import get_current_user
from backend.models import User


def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.get("/api/health/celery/details")
def health_celery_details(current_user: User = Depends(get_current_admin)):
    import os

    from backend.celery_app import celery_app

    try:
        backend_url = celery_app.backend.url
    except Exception as e:
        backend_url = f"ERROR: {type(e).__name__}: {str(e)}"

    return {
        "celery_backend_url": backend_url,
        "broker_use_ssl": getattr(celery_app.conf, "broker_use_ssl", None),
        "redis_backend_use_ssl": getattr(celery_app.conf, "redis_backend_use_ssl", None),
        "REDIS_URL": os.getenv("REDIS_URL"),
        "CELERY_RESULT_BACKEND": os.getenv("CELERY_RESULT_BACKEND"),
        "CELERY_BROKER_URL": os.getenv("CELERY_BROKER_URL"),
    }


@router.get("/api/metrics")
def get_metrics(current_user: User = Depends(get_current_admin)):
    return metrics.get_metrics()
