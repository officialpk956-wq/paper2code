import logging
import os

log = logging.getLogger(__name__)

E2B_API_KEY = os.getenv("E2B_API_KEY", "")
# No default template override — e2b_code_interpreter's own default template
# (used when template=None) already has numpy/pandas/scipy preinstalled.
# Forcing "base" here previously broke every dojo problem (they all import
# numpy) with ModuleNotFoundError. Only set E2B_SANDBOX_TEMPLATE if you have
# a specific custom template you actually want.
SANDBOX_TEMPLATE = os.getenv("E2B_SANDBOX_TEMPLATE") or None
EXECUTION_TIMEOUT = 20  # seconds


def run_code_in_sandbox(
    user_code: str,
    setup_code: str = "",
    test_code: str = "",
    stdin: str = "",
    run_timeout_ms: int = 10_000,
) -> dict:
    """
    Runs: setup_code + user_code + test_code in an E2B sandbox.
    Returns: { passed: bool, stdout: str, stderr: str, time_ms: int, exit_code: int }
    Falls back to a clear error if E2B_API_KEY is not set.
    """
    if not E2B_API_KEY:
        return {
            "passed": False,
            "stdout": "",
            "stderr": "E2B_API_KEY not configured. Cannot run data science code.",
            "time_ms": 0,
            "exit_code": -1,
        }

    full_code = "\n\n".join(filter(None, [setup_code, user_code, test_code]))

    try:
        import time

        from e2b import CommandExitException
        from e2b_code_interpreter import Sandbox

        start = time.monotonic()
        with Sandbox.create(
            template=SANDBOX_TEMPLATE,
            api_key=E2B_API_KEY,
            timeout=max(5, int(run_timeout_ms / 1000) + 5),
        ) as sandbox:
            sandbox.files.write("/home/user/solution.py", full_code)
            sandbox.files.write("/home/user/stdin.txt", stdin)
            
            try:
                result = sandbox.commands.run(
                    "bash -c 'python3 /home/user/solution.py < /home/user/stdin.txt'"
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                exit_code = result.exit_code
            except CommandExitException as e:
                stdout = getattr(e, "stdout", "") or ""
                stderr = getattr(e, "stderr", "") or ""
                exit_code = getattr(e, "exit_code", 1) or 1
            
        elapsed_ms = int((time.monotonic() - start) * 1000)
        passed = exit_code == 0

        return {
            "passed": passed,
            "stdout": stdout,
            "stderr": stderr,
            "time_ms": elapsed_ms,
            "exit_code": exit_code,
        }

    except Exception as e:
        log.exception("E2B execution failed")
        return {
            "passed": False,
            "stdout": "",
            "stderr": str(e),
            "time_ms": 0,
            "exit_code": -1,
        }

