"""CI gate: every v2 Dojo problem's shipped reference_solution must pass all of
its own test cases. If a harness or an expected value is wrong, the reference
fails here and the seed can't ship — the "reference-solution validation at seed
time, enforced in CI" from the audit. Runs in the backend-tests job.
"""
import json
import pathlib

import pytest

from backend.services.dojo_grading import grade

SEED = (
    pathlib.Path(__file__).resolve().parents[1]
    / "backend"
    / "scripts"
    / "json_dump"
    / "dojo_problems_seed.json"
)


def _v2_problems():
    """Seed entries that are v2 structured tests AND ship a reference solution."""
    data = json.loads(SEED.read_text(encoding="utf-8"))
    params = []
    for e in data:
        tc = e.get("test_cases")
        ref = e.get("reference_solution")
        if isinstance(tc, dict) and tc.get("cases") and ref:
            params.append(pytest.param(tc, ref, id=e["id"]))
    return params


V2 = _v2_problems()


@pytest.fixture(autouse=True)
def _local_engine(monkeypatch):
    monkeypatch.setenv("DOJO_ENGINE", "local")


def test_seed_has_migrated_v2_problems():
    # Guard so the parametrized test below can't silently collapse to zero coverage
    # (an empty parametrize list would make it vacuously "pass").
    assert len(V2) >= 3, f"expected >= 3 migrated v2 problems with references, found {len(V2)}"


@pytest.mark.parametrize("test_cases,reference", V2)
def test_reference_solution_passes_all_cases(test_cases, reference):
    r = grade(test_cases, reference)
    assert r["passed"], (
        f"reference solution failed its own tests ({r['num_passed']}/{r['total']}) — "
        f"the harness or expected values are broken. cases={r['cases']}"
    )
