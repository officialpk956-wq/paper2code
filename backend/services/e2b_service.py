import logging
import os
import re

log = logging.getLogger(__name__)

# The default E2B template has numpy/scipy but not torch, so generated PyTorch
# code died with ModuleNotFoundError. Install the CPU wheel on demand (~25s cold).
# A custom template with torch preinstalled (e2b.Dockerfile) makes this a no-op.
_IMPORTS_TORCH = re.compile(r"^\s*(?:import|from)\s+torch\b", re.MULTILINE)
_TORCH_INSTALL_CMD = (
    "python3 -c 'import torch' 2>/dev/null || "
    "pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu"
)
_TORCH_INSTALL_BUDGET_S = 240
_TORCH_MIN_RUN_MS = 30_000  # cold `import torch` alone takes several seconds

E2B_API_KEY = os.getenv("E2B_API_KEY", "")
# No default template override — e2b_code_interpreter's own default template
# (used when template=None) already has numpy/pandas/scipy preinstalled.
# Forcing "base" here previously broke every dojo problem (they all import
# numpy) with ModuleNotFoundError. Only set E2B_SANDBOX_TEMPLATE if you have
# a specific custom template you actually want.
SANDBOX_TEMPLATE = os.getenv("E2B_SANDBOX_TEMPLATE") or None
OUTPUT_LIMIT_BYTES = 65_536


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
    needs_torch = bool(_IMPORTS_TORCH.search(full_code))
    if needs_torch:
        run_timeout_ms = max(run_timeout_ms, _TORCH_MIN_RUN_MS)

    try:
        import time

        from e2b import CommandExitException
        from e2b.exceptions import TimeoutException
        from e2b_code_interpreter import Sandbox

        start = time.monotonic()
        with Sandbox.create(
            template=SANDBOX_TEMPLATE,
            api_key=E2B_API_KEY,
            # Sandbox lifetime must also cover the install, or it is killed mid-pip.
            timeout=max(5, int(run_timeout_ms / 1000) + 5)
            + (_TORCH_INSTALL_BUDGET_S if needs_torch else 0),
        ) as sandbox:
            sandbox.files.write("/home/user/solution.py", full_code)
            sandbox.files.write("/home/user/stdin.txt", stdin)
            if needs_torch:
                sandbox.commands.run(_TORCH_INSTALL_CMD, timeout=_TORCH_INSTALL_BUDGET_S)

            try:
                result = sandbox.commands.run(
                    "bash -c 'python3 /home/user/solution.py < /home/user/stdin.txt'",
                    timeout=run_timeout_ms / 1000.0,
                )
                stdout = result.stdout or ""
                stderr = result.stderr or ""
                exit_code = result.exit_code
            except TimeoutException:
                return {
                    "passed": False,
                    "stdout": "",
                    "stderr": "Time limit exceeded",
                    "time_ms": run_timeout_ms,
                    "exit_code": -1,
                }
            except CommandExitException as e:
                stdout = getattr(e, "stdout", "") or ""
                stderr = getattr(e, "stderr", "") or ""
                exit_code = getattr(e, "exit_code", 1) or 1

        if len(stdout) > OUTPUT_LIMIT_BYTES:
            stdout = stdout[:OUTPUT_LIMIT_BYTES] + "\n... [output truncated at 64KB]"
        if len(stderr) > OUTPUT_LIMIT_BYTES:
            stderr = stderr[:OUTPUT_LIMIT_BYTES] + "\n... [output truncated at 64KB]"

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
