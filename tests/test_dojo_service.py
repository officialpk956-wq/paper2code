"""
Tests for the DS Coding Dojo service logic.
These tests run the same Python script that the API route generates,
verifying that the execution engine works correctly for all problem types.
"""

import json
import math
import sys
import os
import tempfile
import subprocess
import unittest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_solution(code: str, test_cases: list, function_name: str) -> list:
    """Execute user code against test cases; returns list of result dicts."""
    script_lines = [
        "import json, time, math",
        "try:",
        "    import numpy as np",
        "    HAS_NUMPY = True",
        "except ImportError:",
        "    HAS_NUMPY = False",
        "",
        "def _to_comparable(v):",
        "    if HAS_NUMPY and hasattr(v, 'tolist'): return v.tolist()",
        "    if isinstance(v, (list, tuple)): return [_to_comparable(x) for x in v]",
        "    return v",
        "",
        "def _deep_equal(a, b, tol=1e-5):",
        "    a = _to_comparable(a); b = _to_comparable(b)",
        "    if isinstance(a, list) and isinstance(b, list):",
        "        if len(a) != len(b): return False",
        "        return all(_deep_equal(x, y, tol) for x, y in zip(a, b))",
        "    if isinstance(a, (int, float)) and isinstance(b, (int, float)):",
        "        return abs(float(a) - float(b)) < tol or a == b",
        "    return a == b",
        "",
        "# USER CODE",
        code,
        "# END USER CODE",
        "",
        f'test_cases = json.loads("""{json.dumps(test_cases)}""")',
        "results = []",
        "for i, tc in enumerate(test_cases):",
        "    start = time.perf_counter()",
        "    try:",
        "        raw_inputs = tc['input']",
        "        if HAS_NUMPY:",
        "            inputs = {k: (__import__('numpy').array(v) if isinstance(v, list) else v) for k, v in raw_inputs.items()}",
        "        else:",
        "            inputs = raw_inputs",
        f"        actual = {function_name}(**inputs)",
        "        elapsed = (time.perf_counter() - start) * 1000",
        "        actual_cmp = _to_comparable(actual)",
        "        passed = _deep_equal(actual_cmp, tc['output'])",
        "        results.append({'index': i, 'passed': passed, 'actual': actual_cmp, 'error': None})",
        "    except Exception as e:",
        "        results.append({'index': i, 'passed': False, 'actual': None, 'error': str(e)})",
        "",
        "print(json.dumps(results))",
    ]

    script = "\n".join(script_lines)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 and not result.stdout:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout.strip())
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestMatrixMultiplication(unittest.TestCase):
    CODE = """
import numpy as np

def matrix_multiply(A, B):
    return np.dot(A, B)
"""
    TEST_CASES = [
        {"input": {"A": [[1, 2], [3, 4]], "B": [[5, 6], [7, 8]]}, "output": [[19, 22], [43, 50]], "visible": True},
        {"input": {"A": [[1, 0], [0, 1]], "B": [[2, 3], [4, 5]]}, "output": [[2, 3], [4, 5]], "visible": True},
    ]

    def test_correct_solution_passes_all(self):
        results = run_solution(self.CODE, self.TEST_CASES, "matrix_multiply")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["passed"] for r in results))

    def test_wrong_solution_fails(self):
        wrong_code = """
import numpy as np
def matrix_multiply(A, B):
    return np.zeros((2, 2)).tolist()
"""
        results = run_solution(wrong_code, self.TEST_CASES, "matrix_multiply")
        self.assertFalse(results[0]["passed"])

    def test_runtime_error_captured(self):
        error_code = """
def matrix_multiply(A, B):
    raise ValueError("deliberate error")
"""
        results = run_solution(error_code, self.TEST_CASES, "matrix_multiply")
        self.assertFalse(results[0]["passed"])
        self.assertIn("deliberate error", results[0]["error"])


class TestSoftmax(unittest.TestCase):
    CODE = """
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
"""
    TEST_CASES = [
        {"input": {"x": [1.0, 2.0, 3.0]}, "output": [0.09003057, 0.24472848, 0.66524094], "visible": True},
    ]

    def test_softmax_sums_to_one(self):
        import numpy as np
        results = run_solution(self.CODE, self.TEST_CASES, "softmax")
        self.assertTrue(results[0]["passed"], f"Failed: {results[0].get('actual')}")


class TestSigmoid(unittest.TestCase):
    CODE = """
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""
    TEST_CASES = [
        {"input": {"x": 0.0}, "output": 0.5, "visible": True},
        {"input": {"x": 1000.0}, "output": 1.0, "visible": True},
        {"input": {"x": -1000.0}, "output": 0.0, "visible": True},
    ]

    def test_sigmoid(self):
        results = run_solution(self.CODE, self.TEST_CASES, "sigmoid")
        self.assertTrue(all(r["passed"] for r in results))


class TestVisibleOnlyFilter(unittest.TestCase):
    """Verify that run mode only tests visible cases."""
    CODE = """
def solution(x):
    return x * 2
"""
    ALL_CASES = [
        {"input": {"x": 1}, "output": 2, "visible": True},
        {"input": {"x": 5}, "output": 10, "visible": True},
        {"input": {"x": 99}, "output": 198, "visible": False},  # hidden
    ]

    def test_all_cases_run_on_submit(self):
        results = run_solution(self.CODE, self.ALL_CASES, "solution")
        self.assertEqual(len(results), 3)

    def test_visible_only_on_run(self):
        visible = [tc for tc in self.ALL_CASES if tc["visible"]]
        results = run_solution(self.CODE, visible, "solution")
        self.assertEqual(len(results), 2)


class TestSubmissionStatus(unittest.TestCase):
    """Verify status computation: accepted / wrong_answer / error."""

    def _run_and_grade(self, code, test_cases, fn_name):
        results = run_solution(code, test_cases, fn_name)
        passed_count = sum(1 for r in results if r["passed"])
        has_error = any(r.get("error") for r in results)
        if passed_count == len(results):
            return "accepted"
        if has_error:
            return "error"
        return "wrong_answer"

    def test_accepted(self):
        code = "def f(x): return x + 1"
        cases = [{"input": {"x": 1}, "output": 2, "visible": True}]
        self.assertEqual(self._run_and_grade(code, cases, "f"), "accepted")

    def test_wrong_answer(self):
        code = "def f(x): return x + 99"
        cases = [{"input": {"x": 1}, "output": 2, "visible": True}]
        self.assertEqual(self._run_and_grade(code, cases, "f"), "wrong_answer")

    def test_runtime_error(self):
        code = "def f(x): raise RuntimeError('boom')"
        cases = [{"input": {"x": 1}, "output": 2, "visible": True}]
        self.assertEqual(self._run_and_grade(code, cases, "f"), "error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
