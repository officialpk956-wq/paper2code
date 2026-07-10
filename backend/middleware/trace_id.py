import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)


class TraceIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        request.state.trace_id = trace_id

        logger.info(
            "request_start",
            extra={"trace_id": trace_id, "method": request.method, "path": request.url.path},
        )

        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id

        logger.info(
            "request_end", extra={"trace_id": trace_id, "status_code": response.status_code}
        )
        return response
