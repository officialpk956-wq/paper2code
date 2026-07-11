"""
backend/services/dojo_execution_service.py

Sandboxed Python execution via the Piston API (self-hosted Docker).
Replaces the previous subprocess.run() which allowed full RCE — a user could
read os.environ and exfiltrate API keys.

Piston runs on an isolated Docker network with no access to the API server's
environment variables, filesystem, or host network. Set PISTON_URL in .env.

Local dev: add the piston service from docker-compose.yml, then:
  docker compose up piston
"""

import os
from typing import Any

import httpx
from fastapi.concurrency import run_in_threadpool

from backend.services.e2b_service import run_code_in_sandbox

PISTON_URL = os.getenv("PISTON_URL", "http://localhost:2000")
EXECUTION_TIMEOUT_S = 40.0  # httpx client timeout (must exceed max run_timeout_ms)
RUN_TIMEOUT_MS = 10_000  # 10 s — global default; can be overridden per-problem
MEMORY_LIMIT_BYTES = 67_108_864  # 64 MB per execution
OUTPUT_LIMIT_BYTES = 65_536  # 64 KB stdout cap — prevents infinite-print DoS


async def execute_python(
    code: str,
    stdin: str = "",
    run_timeout_ms: int = RUN_TIMEOUT_MS,
) -> dict[str, Any]:
    """
    Execute user-submitted Python code in the E2B sandbox.

    Args:
        code:  Python source code submitted by the user.
        stdin: Optional stdin to feed to the program.

    Returns:
        {
          "passed":    bool   — True if exit code == 0,
          "stdout":    str    — captured stdout,
          "stderr":    str    — captured stderr,
          "time_ms":   int|None — wall-clock ms,
          "exit_code": int   — process exit code,
        }
    """
    return await run_in_threadpool(
        run_code_in_sandbox,
        user_code=code,
        stdin=stdin,
        run_timeout_ms=run_timeout_ms,
    )
