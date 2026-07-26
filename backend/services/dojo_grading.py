"""
backend/services/dojo_grading.py

Grade a Dojo submission against structured v2 test cases (or a legacy harness).

Core rule: the SANDBOX COMPUTES, the SERVER COMPARES. The runner appended to the
user's code only ever receives `args` on stdin and prints the result behind a
sentinel. The expected value / gradient reference never enters the sandbox, so
hidden answers cannot be exfiltrated and float outputs grade with tolerance.

test_cases v2 shape (stored on Problem.test_cases):
  {
    "version": 2,
    "entry_kind": "function" | "class",
    "entry": "sigmoid",
    "call": "forward",            # class only
    "init": {},                   # class only
    "checker": "allclose" | "exact" | "shape" | "grad_check",
    "tolerance": 1e-6,
    "forward_ref": "def L(x, *rest): ...",   # grad_check only (trusted, server-side)
    "cases": [
      {"name":"s1","kind":"sample","args":[...],"expected":...,"explain":"..."},
      {"name":"h1","kind":"hidden","args":[...],"expected":...,"feedback":"..."}
    ]
  }

Legacy shape (still supported):  [{"type": "harness", "code": "<asserts>"}]
"""

import json
from typing import Any

from backend.services.judge0_service import run_batch

SENTINEL = "__P2C_RESULT__"
_MISSING = object()

_RUNNER = '''

# ── judge runner (appended by the grader) ──
import sys as _p2c_sys, json as _p2c_json


def _p2c_ser(o):
    try:
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.bool_):
            return bool(o)
    except Exception:
        pass
    return o


_p2c_c = _p2c_json.loads(_p2c_sys.stdin.read())
_p2c_args = _p2c_c.get("args", [])
_p2c_kwargs = _p2c_c.get("kwargs", {})
__CALL_BLOCK__
_p2c_sys.stdout.write("__SENTINEL__" + _p2c_json.dumps(_p2c_result, default=_p2c_ser))
'''


def _build_runner(spec: dict) -> str:
    entry = spec["entry"]
    if spec.get("entry_kind") == "class":
        init = spec.get("init", {}) or {}
        call = spec.get("call") or "__call__"
        call_block = (
            f"_p2c_obj = {entry}(**{init!r})\n"
            f"_p2c_result = getattr(_p2c_obj, {call!r})(*_p2c_args, **_p2c_kwargs)"
        )
    else:
        call_block = f"_p2c_result = {entry}(*_p2c_args, **_p2c_kwargs)"
    return _RUNNER.replace("__CALL_BLOCK__", call_block).replace("__SENTINEL__", SENTINEL)


# ── public / grade entry points ───────────────────────────────────────────────


def grade(test_spec: Any, code: str, cpu_time_ms: int = 10_000, memory_mb: int = 256) -> dict:
    """Grade `code` against `test_spec`. Returns the per-case result dict."""
    if isinstance(test_spec, list):
        return _grade_legacy(test_spec, code, cpu_time_ms)
    if not isinstance(test_spec, dict):
        return _err("Problem has no test cases configured.")

    entry = test_spec.get("entry")
    cases = test_spec.get("cases") or []
    checker = test_spec.get("checker", "allclose")
    tol = float(test_spec.get("tolerance", 1e-6))
    forward_ref = test_spec.get("forward_ref")
    if not entry or not cases:
        return _err("Problem has no entry point or cases.")

    # Catch syntax errors up front (compile doesn't execute) so the client gets a
    # clean compile_error banner instead of N identical runtime failures — and we
    # skip spawning N subprocesses that would all fail the same way.
    try:
        compile(code, "<submission>", "exec")
    except SyntaxError as e:
        msg = f"{type(e).__name__}: {e.msg}" + (f" (line {e.lineno})" if e.lineno else "")
        out = _err(msg)
        out["total"] = len(cases)
        return out

    runner = _build_runner(test_spec)
    sources = [code + runner for _ in cases]
    stdins = [
        json.dumps({"args": c.get("args", []), "kwargs": c.get("kwargs", {})}) for c in cases
    ]

    results = run_batch(sources, stdins, cpu_time_ms, memory_mb)

    out_cases: list[dict] = []
    num_passed = 0
    total_time = 0
    max_mem = 0
    for c, r in zip(cases, results):
        got, run_err = _extract_result(r)
        passed = False
        if run_err is None and got is not _MISSING:
            if checker == "grad_check":
                passed = _check_grad(got, c.get("args", []), forward_ref, tol)
            else:
                passed = _run_checker(checker, got, c.get("expected"), tol)

        kind = c.get("kind", "hidden")
        row: dict[str, Any] = {
            "name": c.get("name"),
            "kind": kind,
            "passed": bool(passed),
            "time_ms": r.get("time_ms"),
        }
        if kind == "sample":
            row["got"] = _short(got if got is not _MISSING else (run_err or r.get("stderr")))
            if checker != "grad_check":
                row["expected"] = _short(c.get("expected"))
            if c.get("explain"):
                row["explain"] = c["explain"]
        elif not passed and c.get("feedback"):
            # Hidden case: leak the teaching label ONLY on failure, never the data.
            row["feedback"] = c["feedback"]

        if passed:
            num_passed += 1
        total_time += r.get("time_ms") or 0
        if r.get("memory_kb"):
            max_mem = max(max_mem, r["memory_kb"])
        out_cases.append(row)

    n = len(cases)
    return {
        "passed": num_passed == n and n > 0,
        "total": n,
        "num_passed": num_passed,
        "time_ms": total_time,
        "max_memory_kb": max_mem or None,
        "compile_error": None,
        "stdout": "",
        "stderr": "",
        "cases": out_cases,
    }


def public_test_view(test_spec: Any) -> dict:
    """Client-safe view of a problem's tests — no expected values, no hidden data,
    no harness code, no forward_ref. Used by GET /api/problems/{id}."""
    if isinstance(test_spec, list):
        return {"mode": "harness"}
    if not isinstance(test_spec, dict):
        return {}
    view: dict[str, Any] = {
        k: test_spec[k]
        for k in ("entry_kind", "entry", "checker", "depth", "statement", "think_prompts")
        if k in test_spec
    }
    samples = [
        {"name": c.get("name"), "args": c.get("args"), "expected": c.get("expected"),
         "explain": c.get("explain")}
        for c in test_spec.get("cases", [])
        if c.get("kind") == "sample"
    ]
    view["sample_cases"] = samples
    view["num_hidden"] = sum(1 for c in test_spec.get("cases", []) if c.get("kind") != "sample")
    return view


# ── checkers (server-side, trusted) ───────────────────────────────────────────


def _run_checker(checker: str, got: Any, expected: Any, tol: float) -> bool:
    import numpy as np

    try:
        if checker == "exact":
            try:
                a, b = np.asarray(got), np.asarray(expected)
                if a.shape != b.shape:
                    return False
                return bool(np.array_equal(a, b))
            except Exception:
                return got == expected
        if checker == "shape":
            return tuple(np.asarray(got).shape) == tuple(expected)
        # allclose (default) — shape-aware value comparison
        a = np.asarray(got, dtype=float)
        b = np.asarray(expected, dtype=float)
        if a.shape != b.shape:
            return False
        return bool(np.allclose(a, b, atol=tol, rtol=0))
    except Exception:
        return False


def _check_grad(user_grad: Any, args: list, forward_ref: str | None, tol: float) -> bool:
    """Compare the user's analytic gradient to a numerical gradient of forward_ref
    (a trusted scalar function L(x, *rest)) computed via central differences."""
    import numpy as np

    if not forward_ref:
        return False
    ns: dict[str, Any] = {}
    try:
        exec(compile(forward_ref, "<forward_ref>", "exec"), {"np": np, "numpy": np}, ns)
    except Exception:
        return False
    loss_fn = ns.get("L")
    if not callable(loss_fn):
        return False

    try:
        x = np.asarray(args[0], dtype=float)
        rest = list(args[1:])
    except Exception:
        return False

    def loss(v):
        return float(loss_fn(v, *rest))

    eps = 1e-5
    num = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        xp = x.copy()
        xm = x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        try:
            num[idx] = (loss(xp) - loss(xm)) / (2 * eps)
        except Exception:
            return False
        it.iternext()

    try:
        ug = np.asarray(user_grad, dtype=float)
        if ug.shape != num.shape:
            return False
        return bool(np.allclose(ug, num, atol=max(tol, 1e-4), rtol=1e-3))
    except Exception:
        return False


# ── legacy harness ────────────────────────────────────────────────────────────


def _grade_legacy(test_spec: list, code: str, cpu_time_ms: int) -> dict:
    harness = None
    for tc in test_spec:
        if isinstance(tc, dict) and tc.get("type") == "harness" and tc.get("code"):
            harness = tc["code"]
            break
    src = code + ("\n\n# ── judge harness ──\n" + harness if harness else "")
    r = run_batch([src], [""], cpu_time_ms)[0]
    passed = r.get("exit_code", 1) == 0
    return {
        "passed": passed,
        "total": 1,
        "num_passed": 1 if passed else 0,
        "time_ms": r.get("time_ms", 0),
        "max_memory_kb": None,
        "compile_error": None,
        "stdout": r.get("stdout", ""),
        "stderr": r.get("stderr", ""),
        "cases": [
            {"name": "tests", "kind": "sample", "passed": passed,
             "got": _short(r.get("stdout") or r.get("stderr"))}
        ],
    }


# ── helpers ───────────────────────────────────────────────────────────────────


def _extract_result(r: dict):
    if r.get("exit_code", 1) != 0:
        return _MISSING, _last_err(r.get("stderr", "") or "runtime error")
    stdout = r.get("stdout", "") or ""
    idx = stdout.rfind(SENTINEL)
    if idx == -1:
        return _MISSING, _last_err(r.get("stderr", "") or "no result emitted")
    payload = stdout[idx + len(SENTINEL):].strip()
    try:
        return json.loads(payload), None
    except Exception as e:
        return _MISSING, f"could not parse result: {e}"


def _last_err(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else "error"


def _short(x: Any, limit: int = 240) -> str:
    try:
        s = x if isinstance(x, str) else json.dumps(x)
    except Exception:
        s = str(x)
    return s if len(s) <= limit else s[:limit] + "…"


def _err(message: str) -> dict:
    return {
        "passed": False,
        "total": 0,
        "num_passed": 0,
        "time_ms": 0,
        "max_memory_kb": None,
        "compile_error": message,
        "stdout": "",
        "stderr": message,
        "cases": [],
    }
