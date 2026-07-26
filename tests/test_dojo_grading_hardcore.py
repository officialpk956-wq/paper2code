"""Hardcore regression tests for the Dojo v2 grader (backend/services/dojo_grading).

Runs the real grading path against the local subprocess engine. Guards the two
things that must never break: (1) the security contract — a submission can't fake
a full pass and hidden expecteds never enter the sandbox; (2) grader correctness
across the 4 checkers and the ways user code goes wrong.
"""
import pytest

from backend.services.dojo_grading import _build_runner, grade, public_test_view


@pytest.fixture(autouse=True)
def _local_engine(monkeypatch):
    # Force the subprocess engine so tests need no Judge0/E2B and stay hermetic.
    monkeypatch.setenv("DOJO_ENGINE", "local")


def _spec(entry, cases, checker="allclose", tol=1e-6, forward_ref=None, **kw):
    s = {"version": 2, "entry_kind": "function", "entry": entry,
         "checker": checker, "tolerance": tol, "cases": cases}
    if forward_ref:
        s["forward_ref"] = forward_ref
    s.update(kw)
    return s


CASES = [
    {"name": "s", "kind": "sample", "args": [[1.0, 2.0]], "expected": [2.0, 4.0], "explain": "x*2"},
    {"name": "h1", "kind": "hidden", "args": [[3.0, 4.0]], "expected": [6.0, 8.0],
     "feedback": "multiply every element by two"},
    {"name": "h2", "kind": "hidden", "args": [[0.0, -5.0]], "expected": [0.0, -10.0]},
]
DOUBLE = "import numpy as np\ndef f(x):\n    return np.asarray(x, float) * 2\n"


# ── correctness ────────────────────────────────────────────────────────────
def test_reference_passes_all():
    r = grade(_spec("f", CASES), DOUBLE)
    assert r["passed"] and r["num_passed"] == 3 and r["total"] == 3


def test_wrong_answer_fails_but_all_cases_run():
    r = grade(_spec("f", CASES), "def f(x):\n    return [i*3 for i in x]\n")
    assert not r["passed"] and r["num_passed"] == 0 and len(r["cases"]) == 3


def test_exception_does_not_crash_grader():
    r = grade(_spec("f", CASES), "def f(x):\n    raise ValueError('kaboom')\n")
    assert not r["passed"] and len(r["cases"]) == 3
    assert "kaboom" in str(r["cases"][0].get("got", ""))


def test_syntax_error_surfaces_compile_error():
    # regression: a SyntaxError must set compile_error (drives the client banner)
    r = grade(_spec("f", CASES), "def f(x)\n    return x\n")
    assert not r["passed"] and r["num_passed"] == 0
    assert r.get("compile_error") and "SyntaxError" in r["compile_error"]


def test_infinite_loop_times_out():
    r = grade(_spec("f", CASES), "def f(x):\n    while True:\n        pass\n", cpu_time_ms=1200)
    assert not r["passed"]


def test_nan_output_fails_against_finite_expected():
    r = grade(_spec("f", CASES), "import numpy as np\ndef f(x):\n    return np.asarray(x,float)*np.nan\n")
    assert not r["passed"]


def test_debug_prints_do_not_break_grading():
    noisy = ("import numpy as np\ndef f(x):\n    print('noise', x)\n"
             "    return np.asarray(x, float) * 2\n")
    assert grade(_spec("f", CASES), noisy)["passed"]


# ── checkers ───────────────────────────────────────────────────────────────
def test_allclose_tolerance_boundary():
    spec = _spec("f", [
        {"name": "s", "kind": "sample", "args": [[1.0]], "expected": [2.0]},
        {"name": "h", "kind": "hidden", "args": [[2.0]], "expected": [4.0]},
    ], tol=1e-6)
    assert grade(spec, "def f(x):\n    return [i*2 + 5e-7 for i in x]\n")["passed"]
    assert not grade(spec, "def f(x):\n    return [i*2 + 5e-5 for i in x]\n")["passed"]


def test_exact_rejects_float_drift():
    spec = _spec("g", [
        {"name": "s", "kind": "sample", "args": [[1, 2]], "expected": [2, 4]},
        {"name": "h", "kind": "hidden", "args": [[3, 4]], "expected": [6, 8]},
    ], checker="exact")
    assert grade(spec, "def g(x):\n    return [i*2 for i in x]\n")["passed"]
    assert not grade(spec, "def g(x):\n    return [i*2 + 1e-9 for i in x]\n")["passed"]


def test_shape_checker():
    spec = _spec("z", [
        {"name": "s", "kind": "sample", "args": [2, 3], "expected": [2, 3]},
        {"name": "h", "kind": "hidden", "args": [4, 5], "expected": [4, 5]},
    ], checker="shape")
    assert grade(spec, "import numpy as np\ndef z(a,b):\n    return np.zeros((a,b))\n")["passed"]
    assert not grade(spec, "import numpy as np\ndef z(a,b):\n    return np.zeros((b,a))\n")["passed"]


def test_grad_check_correct_vs_wrong():
    spec = _spec("grad", [
        {"name": "s", "kind": "sample", "args": [[1.0, 2.0, 3.0]]},
        {"name": "h", "kind": "hidden", "args": [[-4.0, 0.5]]},
    ], checker="grad_check", forward_ref="def L(x):\n    return float((x**2).sum())\n")
    assert grade(spec, "import numpy as np\ndef grad(x):\n    return 2*np.asarray(x,float)\n")["passed"]
    assert not grade(spec, "import numpy as np\ndef grad(x):\n    return np.asarray(x,float)\n")["passed"]


def test_class_entry_with_init_and_call():
    spec = {"version": 2, "entry_kind": "class", "entry": "Scaler", "init": {"factor": 3.0},
            "call": "apply", "checker": "allclose", "tolerance": 1e-6,
            "cases": [
                {"name": "s", "kind": "sample", "args": [[1.0, 2.0]], "expected": [3.0, 6.0]},
                {"name": "h", "kind": "hidden", "args": [[4.0]], "expected": [12.0]},
            ]}
    code = ("import numpy as np\nclass Scaler:\n    def __init__(self, factor=1.0):\n"
            "        self.factor = factor\n    def apply(self, x):\n"
            "        return np.asarray(x, float) * self.factor\n")
    assert grade(spec, code)["passed"]


# ── security ───────────────────────────────────────────────────────────────
def test_sentinel_spoof_cannot_fake_full_pass():
    from backend.services.dojo_grading import SENTINEL
    spoof = ("import sys\ndef f(x):\n"
             f"    sys.stdout.write({SENTINEL!r} + '[2.0, 4.0]')\n"
             "    sys.stdout.flush()\n    sys.exit(0)\n")
    r = grade(_spec("f", CASES), spoof)
    assert not r["passed"]
    assert all(not c["passed"] for c in r["cases"] if c["kind"] == "hidden")


def test_expected_and_args_never_enter_shipped_source():
    shipped = DOUBLE + _build_runner(_spec("f", CASES))
    for tok in ("6.0", "8.0", "-10.0", "3.0", "-5.0", "expected"):
        assert tok not in shipped, f"leak: {tok}"


def test_public_view_hides_hidden_and_forward_ref():
    spec = _spec("f", CASES, forward_ref="def L(x): return 0.0", depth="L3",
                 statement={"task": "double it"}, think_prompts=["hmm"])
    view = public_test_view(spec)
    assert view["num_hidden"] == 2 and len(view["sample_cases"]) == 1
    assert "forward_ref" not in view and "cases" not in view
    assert "expected" not in str({k: view[k] for k in view if k != "sample_cases"})


# ── competitive I/O mode ────────────────────────────────────────────────────
IO_SPEC = {
    "version": 2, "entry_kind": "io",
    "cases": [
        {"name": "s", "kind": "sample", "stdin": "2 3\n", "expected_stdout": "5", "explain": "sum"},
        {"name": "h", "kind": "hidden", "stdin": "10 20\n", "expected_stdout": "30",
         "feedback": "read the two ints and print their sum"},
    ],
}
IO_OK = "a, b = map(int, input().split())\nprint(a + b)\n"


def test_io_correct_program_passes():
    r = grade(IO_SPEC, IO_OK)
    assert r["passed"] and r["num_passed"] == 2 and r["total"] == 2


def test_io_wrong_program_fails():
    assert not grade(IO_SPEC, "a, b = map(int, input().split())\nprint(a - b)\n")["passed"]


def test_io_normalizes_trailing_whitespace_and_newlines():
    # program emits extra trailing spaces + blank line; must still match "5"/"30"
    prog = "a, b = map(int, input().split())\nprint(str(a + b) + '   ')\nprint()\n"
    assert grade(IO_SPEC, prog)["passed"]


def test_io_runtime_error_fails_gracefully():
    r = grade(IO_SPEC, "raise SystemExit(1)\n")
    assert not r["passed"] and len(r["cases"]) == 2


def test_io_public_view_hides_hidden_expected_stdout():
    view = public_test_view(IO_SPEC)
    assert view["num_hidden"] == 1 and len(view["sample_cases"]) == 1
    assert view["sample_cases"][0]["expected_stdout"] == "5"   # sample is public
    assert "30" not in str(view)                                # hidden expected never leaks
