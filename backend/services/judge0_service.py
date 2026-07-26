"""
backend/services/judge0_service.py

Execution backends for the Dojo. The active engine is chosen by the DOJO_ENGINE
env var (read per-call):

  judge0 -> self-hosted Judge0 REST API (needs a dedicated VM; see docs)
  e2b    -> E2B sandbox (backend/services/e2b_service.py)
  local  -> plain subprocess python3  (DEV ONLY — no sandbox, do not use in prod)

Every backend returns, per submission, a uniform dict:
  { "stdout": str, "stderr": str, "exit_code": int, "time_ms": int, "memory_kb": int|None }

The grader (backend/services/dojo_grading.py) sits on top of this and never lets a
backend see the expected answer — it only ships `args` in and reads the printed
result back out.
"""

import os
import subprocess
import sys
import tempfile
import time
from typing import Any


def _engine() -> str:
    return (os.getenv("DOJO_ENGINE") or "e2b").lower()


def run_batch(
    sources: list[str],
    stdins: list[str],
    cpu_time_ms: int = 10_000,
    memory_mb: int = 256,
) -> list[dict[str, Any]]:
    """Run N (source, stdin) pairs and return N uniform result dicts."""
    engine = _engine()
    if engine == "judge0" and os.getenv("JUDGE0_URL"):
        return _run_judge0(sources, stdins, cpu_time_ms, memory_mb)
    if engine == "local":
        return _run_local(sources, stdins, cpu_time_ms)
    return _run_e2b(sources, stdins, cpu_time_ms)


# ── local subprocess (dev only) ───────────────────────────────────────────────


def _run_local(sources: list[str], stdins: list[str], cpu_time_ms: int) -> list[dict]:
    out: list[dict] = []
    for src, stdin in zip(sources, stdins):
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(src)
                path = f.name
            t0 = time.monotonic()
            p = subprocess.run(
                [sys.executable, path],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=max(1.0, cpu_time_ms / 1000) + 3,
            )
            out.append(
                {
                    "stdout": p.stdout or "",
                    "stderr": p.stderr or "",
                    "exit_code": p.returncode,
                    "time_ms": int((time.monotonic() - t0) * 1000),
                    "memory_kb": None,
                }
            )
        except subprocess.TimeoutExpired:
            out.append(
                {"stdout": "", "stderr": "Time limit exceeded", "exit_code": 124,
                 "time_ms": cpu_time_ms, "memory_kb": None}
            )
        except Exception as e:  # pragma: no cover - defensive
            out.append(
                {"stdout": "", "stderr": str(e), "exit_code": 1, "time_ms": 0, "memory_kb": None}
            )
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
    return out


# ── E2B sandbox ───────────────────────────────────────────────────────────────


def _run_e2b(sources: list[str], stdins: list[str], cpu_time_ms: int) -> list[dict]:
    from backend.services.e2b_service import run_code_in_sandbox

    out: list[dict] = []
    for src, stdin in zip(sources, stdins):
        r = run_code_in_sandbox(user_code=src, stdin=stdin, run_timeout_ms=cpu_time_ms)
        out.append(
            {
                "stdout": r.get("stdout", ""),
                "stderr": r.get("stderr", ""),
                "exit_code": r.get("exit_code", 1),
                "time_ms": r.get("time_ms", 0),
                "memory_kb": None,
            }
        )
    return out


# ── Judge0 (self-hosted) ──────────────────────────────────────────────────────

# status_id meanings: 3 = Accepted (exit 0), 5 = TLE, 6 = Compile Error, others = RE/WA.
_JUDGE0_ACCEPTED = 3


def _run_judge0(sources: list[str], stdins: list[str], cpu_time_ms: int, memory_mb: int) -> list[dict]:
    import httpx

    base = os.getenv("JUDGE0_URL", "").rstrip("/")
    lang = int(os.getenv("JUDGE0_PYTHON_LANG_ID", "71"))
    headers = {"Content-Type": "application/json"}
    token = os.getenv("JUDGE0_AUTH_TOKEN")
    if token:
        headers["X-Auth-Token"] = token

    out: list[dict] = []
    # Per-submission wait=true: simplest correct path. Batch is a later optimization.
    with httpx.Client(timeout=max(15.0, cpu_time_ms / 1000 + 10)) as client:
        for src, stdin in zip(sources, stdins):
            try:
                r = client.post(
                    f"{base}/submissions?base64_encoded=false&wait=true",
                    json={
                        "language_id": lang,
                        "source_code": src,
                        "stdin": stdin,
                        "cpu_time_limit": max(1.0, cpu_time_ms / 1000),
                        "memory_limit": memory_mb * 1024,
                    },
                    headers=headers,
                )
                r.raise_for_status()
                item = r.json()
                st = (item.get("status") or {}).get("id", 0)
                out.append(
                    {
                        "stdout": item.get("stdout") or "",
                        "stderr": (item.get("stderr") or "") + (item.get("compile_output") or ""),
                        "exit_code": 0 if st == _JUDGE0_ACCEPTED else 1,
                        "time_ms": int(float(item.get("time") or 0) * 1000),
                        "memory_kb": item.get("memory"),
                    }
                )
            except Exception as e:
                out.append(
                    {"stdout": "", "stderr": f"judge0 error: {e}", "exit_code": 1,
                     "time_ms": 0, "memory_kb": None}
                )
    return out
