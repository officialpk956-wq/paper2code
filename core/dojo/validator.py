"""
core/dojo/validator.py — Reference-solution runner (Phase 12, M1)

Runs OUR OWN reference solutions against an exercise's fixed test inputs to
precompute the expected outputs that get shipped to the browser. The browser
(Pyodide) runs the LEARNER's code against the same inputs and compares to
these expected outputs with np.allclose.

This module never executes learner-supplied code. It only execs trusted
reference-solution strings defined in exercises.py.
"""

from typing import Any, Dict, List
import numpy as np


def _to_jsonable(x: Any) -> Any:
    """Convert numpy arrays/scalars to plain JSON-serialisable Python values."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(e) for e in x]
    return x


def _coerce_inputs(raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a test case's inputs into kwargs for the reference function.
    Lists become numpy arrays; scalars (hyperparameters like lr, eps, t)
    are passed through unchanged. This mirrors the Pyodide harness exactly.
    """
    kwargs: Dict[str, Any] = {}
    for key, value in raw_inputs.items():
        if isinstance(value, list):
            kwargs[key] = np.array(value, dtype=float) if _is_numeric_list(value) else np.array(value)
        else:
            kwargs[key] = value
    return kwargs


def _is_numeric_list(value: list) -> bool:
    """True if a (possibly nested) list contains only numbers."""
    flat = value
    while isinstance(flat, list) and flat:
        flat = flat[0]
    return isinstance(flat, (int, float))


def run_reference(reference_solution: str, fn_name: str, test_inputs: List[Dict[str, Any]]) -> List[Any]:
    """
    Execute a reference solution and return its outputs for each test case
    as JSON-serialisable values.

    Args:
        reference_solution: Python source defining `fn_name`.
        fn_name: the function the exercise asks the learner to implement.
        test_inputs: list of {arg_name: value} dicts (values are lists or scalars).

    Returns:
        List of expected outputs (one per test case), JSON-serialisable.
    """
    namespace: Dict[str, Any] = {"np": np, "numpy": np}
    exec(reference_solution, namespace)  # noqa: S102 — trusted, our own code only
    if fn_name not in namespace:
        raise ValueError(f"Reference solution does not define '{fn_name}'")
    fn = namespace[fn_name]

    outputs: List[Any] = []
    for case in test_inputs:
        kwargs = _coerce_inputs(case)
        result = fn(**kwargs)
        outputs.append(_to_jsonable(result))
    return outputs


def compare(expected: Any, got: Any, tol: float) -> bool:
    """
    Deterministic comparison used by tests (mirrors the Pyodide np.allclose check).
    """
    try:
        return bool(np.allclose(np.array(got, dtype=float), np.array(expected, dtype=float),
                                atol=tol, rtol=1e-5))
    except Exception:
        return False
