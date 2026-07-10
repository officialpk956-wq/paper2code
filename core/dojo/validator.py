"""
core/dojo/validator.py — Reference-solution runner + deep matcher (Phase 12 / DS expansion)

Runs OUR OWN reference solutions against an exercise's fixed test inputs to
precompute the expected outputs shipped to the browser. The browser (Pyodide)
runs the LEARNER's code against the same inputs and deep-matches to these.

Deep matching supports the full range of data-science answer types:
  - numbers / numpy arrays  → np.allclose (atol + rtol)
  - lists / tuples          → same length, element-wise deep match
  - dicts                   → same keys, deep-matched values (order-independent)
  - strings / bools / None  → exact equality

This module never executes learner-supplied code. It only execs trusted
reference-solution strings defined in the problem catalog.
"""

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _to_jsonable(x: Any) -> Any:
    """Convert numpy/Python values into plain JSON-serialisable structures."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(e) for e in x]
    return x


def _is_numeric_list(value: list) -> bool:
    flat = value
    while isinstance(flat, list) and flat:
        flat = flat[0]
    return isinstance(flat, (int, float)) and not isinstance(flat, bool)


def _coerce_inputs(raw_inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a test case's inputs into kwargs for the reference function.
    Numeric lists become numpy arrays; lists of strings/mixed stay as Python
    lists; scalars pass through. Mirrors the Pyodide harness exactly.
    """
    kwargs: dict[str, Any] = {}
    for key, value in raw_inputs.items():
        if isinstance(value, list) and _is_numeric_list(value):
            kwargs[key] = np.array(value, dtype=float)
        else:
            kwargs[key] = value
    return kwargs


def run_reference(
    reference_solution: str, fn_name: str, test_inputs: list[dict[str, Any]]
) -> list[Any]:
    """Execute a reference solution; return its JSON-serialisable output per test case."""
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    exec(reference_solution, namespace)  # noqa: S102 — trusted, our own code only
    if fn_name not in namespace:
        raise ValueError(f"Reference solution does not define '{fn_name}'")
    fn = namespace[fn_name]

    outputs: list[Any] = []
    for case in test_inputs:
        kwargs = _coerce_inputs(case)
        outputs.append(_to_jsonable(fn(**kwargs)))
    return outputs


# ---------------------------------------------------------------------------
# Deep matching (shared semantics with the client-side Pyodide harness)
# ---------------------------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def deep_match(expected: Any, got: Any, tol: float = 1e-6) -> bool:
    """Structural comparison: numbers within tolerance, everything else exact."""
    # numbers (incl. float/int mix)
    if _is_number(expected) and _is_number(got):
        if expected != expected and got != got:  # NaN == NaN treated equal
            return True
        return abs(float(expected) - float(got)) <= tol + 1e-9 * max(1.0, abs(float(expected)))
    # bool must match bool exactly
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected == got
    # lists / tuples
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)):
        if len(expected) != len(got):
            return False
        return all(deep_match(e, g, tol) for e, g in zip(expected, got))
    # dicts (order-independent)
    if isinstance(expected, dict) and isinstance(got, dict):
        if set(map(str, expected.keys())) != set(map(str, got.keys())):
            return False
        gnorm = {str(k): v for k, v in got.items()}
        return all(deep_match(v, gnorm[str(k)], tol) for k, v in expected.items())
    # strings / None / fallthrough
    return expected == got


def compare(expected: Any, got: Any, tol: float) -> bool:
    """
    Comparison used by tests. Tries fast numeric allclose first (handles ragged
    float arrays + rtol), then falls back to structural deep_match.
    """
    try:
        e = np.array(expected, dtype=float)
        g = np.array(got, dtype=float)
        if e.shape == g.shape:
            return bool(np.allclose(g, e, atol=tol, rtol=1e-5, equal_nan=True))
    except Exception:
        pass
    return deep_match(expected, got, tol)
