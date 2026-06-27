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

PISTON_URL          = os.getenv("PISTON_URL", "http://localhost:2000")
EXECUTION_TIMEOUT_S = 20.0       # httpx client timeout (must exceed run_timeout_ms)
RUN_TIMEOUT_MS      = 10_000     # 10 s — matches the old subprocess timeout=10
MEMORY_LIMIT_BYTES  = 67_108_864  # 64 MB per execution


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

@retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=False,
)
async def _call_piston(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=EXECUTION_TIMEOUT_S) as client:
        resp = await client.post(f"{PISTON_URL}/api/v2/execute", json=payload)
        resp.raise_for_status()
    return resp.json()

async def execute_python(code: str, stdin: str = "") -> dict[str, Any]:
    """
    Execute user-submitted Python code in the Piston sandbox.

    Args:
        code:  Python source code submitted by the user.
        stdin: Optional stdin to feed to the program.

    Returns:
        {
          "passed":    bool   — True if exit code == 0,
          "stdout":    str    — captured stdout,
          "stderr":    str    — captured stderr,
          "time_ms":   int|None — wall-clock ms reported by Piston,
          "exit_code": int   — process exit code,
        }
    """
    payload = {
        "language": "python",
        "version": "3.10",
        "files": [{"name": "solution.py", "content": code}],
        "stdin": stdin,
        "args": [],
        "compile_timeout": 5_000,
        "run_timeout": RUN_TIMEOUT_MS,
        "compile_memory_limit": -1,
        "run_memory_limit": MEMORY_LIMIT_BYTES,
    }

    try:
        data = await _call_piston(payload)
    except (RetryError, httpx.HTTPError, Exception) as e:
        return {
            "passed": False,
            "stdout": "",
            "stderr": f"Execution service unavailable: {e}",
            "time_ms": None,
            "exit_code": -1,
        }

    run  = data.get("run", {})

    return {
        "passed":    run.get("code", 1) == 0,
        "stdout":    run.get("stdout", ""),
        "stderr":    run.get("stderr", ""),
        "time_ms":   run.get("time"),
        "exit_code": run.get("code", 1),
    }
