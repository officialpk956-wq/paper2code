"""
backend/services/dojo_execution_service.py

Sandboxed Python execution via E2B Code Interpreter.
"""

from typing import Any

from fastapi.concurrency import run_in_threadpool

from backend.services.e2b_service import run_code_in_sandbox

RUN_TIMEOUT_MS = 10_000  # 10 s — global default; can be overridden per-problem


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
