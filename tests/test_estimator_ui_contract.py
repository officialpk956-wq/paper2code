import json
from pathlib import Path

from backend.routers.lab import CostEstimatorRequest, training_cost_estimator


def test_api_matches_browser_estimate_contract():
    fixture = json.loads(
        (Path(__file__).parents[1] / "src/__tests__/fixtures/training-estimates.json").read_text()
    )
    for family, expected in fixture["cases"].items():
        actual = training_cost_estimator(
            CostEstimatorRequest(architecture=family, **fixture["request"])
        )
        assert {key: actual[key] for key in expected} == expected
