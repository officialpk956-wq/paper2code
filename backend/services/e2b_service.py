import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

E2B_API_KEY       = os.getenv("E2B_API_KEY", "")
SANDBOX_TEMPLATE  = os.getenv("E2B_SANDBOX_TEMPLATE", "base")
EXECUTION_TIMEOUT = 20  # seconds

def run_code_in_sandbox(
    setup_code: str,
    user_code: str,
    test_code: str,
) -> dict:
    """
    Runs: setup_code + user_code + test_code in an E2B sandbox.
    Returns: { passed: bool, stdout: str, stderr: str, time_ms: int }
    Falls back to a clear error if E2B_API_KEY is not set.
    """
    if not E2B_API_KEY:
        return {
            "passed": False,
            "stdout": "",
            "stderr": "E2B_API_KEY not configured. Cannot run data science code.",
            "time_ms": 0,
        }

    full_code = "\n\n".join(filter(None, [setup_code, user_code, test_code]))

    try:
        import time
        from e2b_code_interpreter import Sandbox
        start = time.monotonic()
        with Sandbox(
            template=SANDBOX_TEMPLATE,
            api_key=E2B_API_KEY,
            timeout=EXECUTION_TIMEOUT,
        ) as sandbox:
            execution = sandbox.run_code(full_code)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
        stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""

        passed = (
            execution.error is None
            and "AssertionError" not in stderr
            and "Error" not in stderr
        )
        return {"passed": passed, "stdout": stdout, "stderr": stderr, "time_ms": elapsed_ms}

    except Exception as e:
        log.exception("E2B execution failed")
        return {"passed": False, "stdout": "", "stderr": str(e), "time_ms": 0}
