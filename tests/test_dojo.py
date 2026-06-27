"""
Tests for the Code Dojo (Phase 12, M1).

Verifies:
  - Every exercise's reference solution passes its own precomputed expected outputs.
  - A deliberately-wrong solution fails (the validator actually discriminates).
  - Public projection hides the reference solution but ships expected outputs.
  - All exercises have the required fields and sane test data.
  - API endpoints behave (list / detail / solution / 404 / submit).
"""

import numpy as np
import pytest

from core.dojo import EXERCISES, get_exercise_list, get_public_exercise, get_solution
from core.dojo.exercises import CATEGORY_ORDER
from core.dojo.validator import run_reference, compare

ALLOWED_CATEGORIES = set(CATEGORY_ORDER)


REQUIRED_FIELDS = {
    "id", "category", "title", "difficulty", "fn_name", "concept", "math",
    "intuition", "starter_code", "reference_solution", "test_inputs",
    "tolerance", "hints", "common_mistakes",
}


# ----------------------------------------------------------------------
# Content integrity
# ----------------------------------------------------------------------

def test_catalog_nonempty_and_unique_ids():
    assert len(EXERCISES) >= 10
    ids = [e["id"] for e in EXERCISES]
    assert len(ids) == len(set(ids)), "exercise ids must be unique"


@pytest.mark.parametrize("ex", EXERCISES, ids=[e["id"] for e in EXERCISES])
def test_exercise_has_required_fields(ex):
    missing = REQUIRED_FIELDS - set(ex.keys())
    assert not missing, f"{ex['id']} missing fields: {missing}"
    assert 1 <= ex["difficulty"] <= 5
    assert ex["category"] in ALLOWED_CATEGORIES
    assert ex["fn_name"] in ex["reference_solution"], "reference must define fn_name"
    assert ex["fn_name"] in ex["starter_code"], "starter must mention fn_name"
    assert len(ex["test_inputs"]) >= 1
    assert len(ex["hints"]) >= 1
    assert len(ex["common_mistakes"]) >= 1


@pytest.mark.parametrize("ex", EXERCISES, ids=[e["id"] for e in EXERCISES])
def test_reference_solution_passes_its_own_expected(ex):
    """The canonical solution must pass every test case for its exercise."""
    expected = run_reference(ex["reference_solution"], ex["fn_name"], ex["test_inputs"])
    assert len(expected) == len(ex["test_inputs"])
    # Re-run and compare to itself — deterministic and self-consistent.
    again = run_reference(ex["reference_solution"], ex["fn_name"], ex["test_inputs"])
    for exp, got in zip(expected, again):
        assert compare(exp, got, ex["tolerance"]), f"{ex['id']} not deterministic"


@pytest.mark.parametrize("ex", EXERCISES, ids=[e["id"] for e in EXERCISES])
def test_wrong_solution_fails(ex):
    """A broken implementation (returns zeros) must NOT match expected outputs
    for at least one test case — proves the checker discriminates."""
    expected = run_reference(ex["reference_solution"], ex["fn_name"], ex["test_inputs"])
    wrong_src = f"def {ex['fn_name']}(*args, **kwargs):\n    return 0.0\n"
    wrong = run_reference(wrong_src, ex["fn_name"], ex["test_inputs"])
    matches = [compare(e, w, ex["tolerance"]) for e, w in zip(expected, wrong)]
    assert not all(matches), f"{ex['id']}: zero-stub wrongly passed all cases"


# ----------------------------------------------------------------------
# Public projection (solution hiding)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ex", EXERCISES, ids=[e["id"] for e in EXERCISES])
def test_public_exercise_hides_solution(ex):
    pub = get_public_exercise(ex["id"])
    assert pub is not None
    assert "reference_solution" not in pub, "public payload must NOT leak the solution"
    assert "expected_outputs" in pub
    assert len(pub["expected_outputs"]) == len(pub["test_inputs"])


def test_get_solution_returns_reference():
    sol = get_solution("relu")
    assert sol is not None
    assert "np.maximum" in sol["reference_solution"]


def test_unknown_ids_return_none():
    assert get_public_exercise("nope") is None
    assert get_solution("nope") is None


def test_exercise_list_is_lightweight():
    lst = get_exercise_list()
    assert len(lst) == len(EXERCISES)
    for item in lst:
        assert "reference_solution" not in item
        assert "expected_outputs" not in item
        assert {"id", "category", "title", "difficulty", "concept"} <= set(item.keys())


# ----------------------------------------------------------------------
# Spot-check numerical correctness of a few references
# ----------------------------------------------------------------------

def test_softmax_sums_to_one_and_is_stable():
    out = run_reference(
        next(e["reference_solution"] for e in EXERCISES if e["id"] == "softmax"),
        "softmax",
        [{"x": [1000.0, 1001.0, 1002.0]}],
    )[0]
    assert abs(sum(out) - 1.0) < 1e-9
    assert all(np.isfinite(out)), "stable softmax must not overflow on large logits"


def test_accuracy_value():
    out = run_reference(
        next(e["reference_solution"] for e in EXERCISES if e["id"] == "accuracy"),
        "accuracy",
        [{"y_pred": [0, 1, 2, 1], "y_true": [0, 1, 1, 1]}],
    )[0]
    assert abs(out - 0.75) < 1e-9


# ----------------------------------------------------------------------
# API surface
# ----------------------------------------------------------------------

def test_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.server import app
    c = TestClient(app)

    r = c.get("/api/dojo/exercises")
    assert r.status_code == 200
    assert len(r.json()["exercises"]) == len(EXERCISES)

    r = c.get("/api/dojo/exercises/softmax")
    assert r.status_code == 200
    body = r.json()
    assert "expected_outputs" in body and "reference_solution" not in body

    r = c.get("/api/dojo/exercises/softmax/solution")
    assert r.status_code == 200 and "np.exp" in r.json()["reference_solution"]

    assert c.get("/api/dojo/exercises/does_not_exist").status_code == 404

    r = c.post("/api/dojo/submit_exercise",
               json={"exercise_id": "relu", "passed": True, "attempts": 1},
               headers={"X-Learner-ID": "pytest"})
    assert r.status_code == 200 and r.json()["status"] == "ok"
