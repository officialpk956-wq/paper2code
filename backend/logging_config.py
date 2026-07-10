import contextvars
import json
import logging
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx = contextvars.ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    def format(self, record):
        req_id = getattr(record, "request_id", None) or request_id_ctx.get()
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "time": self.formatTime(record),
                "request_id": req_id,
            }
        )


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)
