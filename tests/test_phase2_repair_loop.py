"""
Phase 2.4: Tests for bounded repair loop and attempt tracking.
"""

from unittest.mock import MagicMock, patch
import pytest

from core.paper_to_code_generator import PaperToCodeGenerator
from core.architecture_graph import ArchitectureGraph, GraphNode


def test_repair_loop_improves_code_to_success_on_second_attempt():
    """Verify repair loop retries a failing verification and reaches success on attempt 2."""
    generator = PaperToCodeGenerator()
    generator.groq_available = True

    failing_report = {
        "passed": False,
        "status": "needs_review",
        "error": "RuntimeError: shape mismatch",
        "checks": {"syntax": True, "exec": False, "forward": False},
    }
    success_report = {
        "passed": True,
        "status": "success",
        "entrypoint_class": "ResNetBuilder",
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, 1000],
        "checks": {"syntax": True, "exec": True, "forward": True},
    }

    with (
        patch.object(generator.config_extractor, "extract_from_text", return_value={"name": "TestModel", "layers": [{"type": "conv2d", "params": {}}, {"type": "linear", "params": {}}], "connections": [["layer_0", "layer_1"]]}),
        patch.object(generator, "_generate_code", return_value=("broken_code_v1", "builder")),
        patch.object(generator, "validate_generated_code", side_effect=[failing_report, success_report]),
        patch.object(generator, "_repair_code", return_value="repaired_code_v2"),
    ):
        result = generator._run_pipeline("test excerpt text", "test_paper")

    assert result["generation_status"] == "success"
    assert result["verification_report"]["passed"] is True
    assert result["verification_report"]["total_attempts"] == 2
    assert result["verification_report"]["final_attempt"] == 2
    assert len(result["verification_report"]["attempts"]) == 2
    assert result["verification_report"]["attempts"][0]["passed"] is False
    assert result["verification_report"]["attempts"][1]["passed"] is True
    assert result["verification_report"]["attempts"][0]["report"]["error"] == (
        "RuntimeError: shape mismatch"
    )


def test_repair_loop_stops_at_exactly_3_attempts_for_unfixable_code():
    """Verify repair loop halts after max 3 attempts when code remains unfixable."""
    generator = PaperToCodeGenerator()
    generator.groq_available = True

    failing_report_1 = {
        "passed": False,
        "status": "needs_review",
        "error": "RuntimeError: attempt 1",
        "checks": {"syntax": True, "exec": False},
    }
    failing_report_2 = {
        "passed": False,
        "status": "needs_review",
        "error": "RuntimeError: attempt 2",
        "checks": {"syntax": True, "exec": False},
    }
    failing_report_3 = {
        "passed": False,
        "status": "needs_review",
        "error": "RuntimeError: attempt 3",
        "checks": {"syntax": True, "exec": False},
    }

    with (
        patch.object(generator.config_extractor, "extract_from_text", return_value={"name": "UnfixableModel", "layers": [{"type": "conv2d", "params": {}}, {"type": "linear", "params": {}}], "connections": [["layer_0", "layer_1"]]}),
        patch.object(generator, "_generate_code", return_value=("broken_v1", "builder")),
        patch.object(generator, "validate_generated_code", side_effect=[failing_report_1, failing_report_2, failing_report_3]),
        patch.object(generator, "_repair_code", side_effect=["broken_v2", "broken_v3", "broken_v4"]),
    ):
        result = generator._run_pipeline("test unfixable text", "test_paper")

    assert result["generation_status"] == "needs_review"
    assert result["verification_report"]["passed"] is False
    assert result["verification_report"]["total_attempts"] == 3
    assert result["verification_report"]["final_attempt"] == 3
    assert len(result["verification_report"]["attempts"]) == 3


def test_repair_code_returns_none_instead_of_raising_on_llm_failure():
    """
    Regression test: _repair_code called llm_complete() with no error
    handling. A transient failure (rate limit, timeout, network error --
    all realistic given core/llm_client.py's own retry-then-fallback
    behavior can still exhaust) would raise an unhandled exception straight
    out of _run_pipeline's repair while-loop, crashing the whole upload
    instead of gracefully stopping at the last known verification_report.
    """
    generator = PaperToCodeGenerator()

    with patch(
        "core.paper_to_code_generator.llm_complete",
        side_effect=RuntimeError("LLM circuit breaker tripped"),
    ):
        result = generator._repair_code(
            code="some code",
            verification_report={"error": "boom", "stage": "sandbox"},
            spec={"model_family": "gan"},
            graph=None,
            attempt=2,
        )

    assert result is None


def test_run_pipeline_stops_gracefully_when_repair_call_raises():
    """
    End-to-end version of the above: a repair LLM call that raises must not
    propagate out of _run_pipeline -- the loop's existing
    `if not repaired_code: break` guard handles a None return from
    _repair_code, so _run_pipeline should return needs_review with
    whatever attempt succeeded before the raise, not crash.
    """
    generator = PaperToCodeGenerator()
    generator.groq_available = True

    failing_report = {
        "passed": False,
        "status": "needs_review",
        "error": "RuntimeError: shape mismatch",
        "checks": {"syntax": True, "exec": False, "forward": False},
    }

    with (
        patch.object(
            generator.config_extractor,
            "extract_from_text",
            return_value={
                "name": "TestModel",
                "layers": [{"type": "conv2d", "params": {}}, {"type": "linear", "params": {}}],
                "connections": [["layer_0", "layer_1"]],
            },
        ),
        patch.object(generator, "_generate_code", return_value=("broken_code_v1", "builder")),
        patch.object(generator, "validate_generated_code", return_value=failing_report),
        patch.object(generator, "_repair_code", return_value=None),  # simulates LLM failure
    ):
        result = generator._run_pipeline("test excerpt text", "test_paper")

    assert result["generation_status"] == "needs_review"
    assert result["verification_report"]["total_attempts"] == 1
    assert result["verification_report"]["final_attempt"] == 1
