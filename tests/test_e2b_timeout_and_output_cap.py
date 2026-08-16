"""Tests for E2B execution timeout enforcement and output size cap."""
import pytest
from unittest.mock import patch, MagicMock, ANY

import backend.services.e2b_service as e2b_mod


# Ensure E2B_API_KEY is set so the function doesn't short-circuit
e2b_mod.E2B_API_KEY = "test_key"


def _make_sandbox_mock(*, stdout="", stderr="", exit_code=0, run_side_effect=None):
    """Build a mock Sandbox context manager with configurable commands.run behavior."""
    mock_sandbox = MagicMock()
    if run_side_effect:
        mock_sandbox.commands.run.side_effect = run_side_effect
    else:
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.stderr = stderr
        mock_result.exit_code = exit_code
        mock_sandbox.commands.run.return_value = mock_result

    mock_sandbox_cls = MagicMock()
    mock_sandbox_cls.create.return_value.__enter__ = MagicMock(return_value=mock_sandbox)
    mock_sandbox_cls.create.return_value.__exit__ = MagicMock(return_value=False)
    return mock_sandbox_cls, mock_sandbox


def test_timeout_exception_returns_clean_verdict():
    """run_timeout_ms=2000 — timeout fires → clean 'Time limit exceeded', not raw exception."""
    from e2b.exceptions import TimeoutException

    mock_cls, mock_sbx = _make_sandbox_mock(run_side_effect=TimeoutException("timeout"))

    with patch("e2b_code_interpreter.Sandbox", mock_cls):
        result = e2b_mod.run_code_in_sandbox("print('hello')", run_timeout_ms=2000)

    assert result["passed"] is False
    assert result["stderr"] == "Time limit exceeded"
    assert result["time_ms"] == 2000
    assert result["exit_code"] == -1


def test_command_run_receives_correct_timeout():
    """run_timeout_ms=10000 → commands.run called with timeout=10.0, not 60."""
    mock_cls, mock_sbx = _make_sandbox_mock(stdout="hello", exit_code=0)

    with patch("e2b_code_interpreter.Sandbox", mock_cls):
        result = e2b_mod.run_code_in_sandbox("print('hello')", run_timeout_ms=10_000)

    assert result["passed"] is True
    mock_sbx.commands.run.assert_called_once()
    call_kwargs = mock_sbx.commands.run.call_args
    # timeout should be passed as a keyword arg
    assert call_kwargs.kwargs.get("timeout") == pytest.approx(10.0)


def test_stdout_truncated_at_64kb():
    """200KB stdout → truncated to OUTPUT_LIMIT_BYTES with marker appended."""
    long_output = "x" * 200_000
    mock_cls, mock_sbx = _make_sandbox_mock(stdout=long_output, exit_code=0)

    with patch("e2b_code_interpreter.Sandbox", mock_cls):
        result = e2b_mod.run_code_in_sandbox("print('lots')")

    truncation_marker = "\n... [output truncated at 64KB]"
    expected_len = e2b_mod.OUTPUT_LIMIT_BYTES + len(truncation_marker)
    assert len(result["stdout"]) == expected_len
    assert result["stdout"].endswith(truncation_marker)
    assert result["passed"] is True


def test_stdout_under_limit_passes_through_unmodified():
    """Short stdout under 64KB → returned verbatim, no truncation marker."""
    short_output = "a" * 1000
    mock_cls, mock_sbx = _make_sandbox_mock(stdout=short_output, exit_code=0)

    with patch("e2b_code_interpreter.Sandbox", mock_cls):
        result = e2b_mod.run_code_in_sandbox("print('short')")

    assert result["stdout"] == short_output
    assert "truncated" not in result["stdout"]


def test_normal_successful_execution_no_regression():
    """Normal execution → passed=True, correct stdout/exit_code."""
    mock_cls, mock_sbx = _make_sandbox_mock(stdout="42\n", stderr="", exit_code=0)

    with patch("e2b_code_interpreter.Sandbox", mock_cls):
        result = e2b_mod.run_code_in_sandbox("print(42)", run_timeout_ms=5000)

    assert result["passed"] is True
    assert result["stdout"] == "42\n"
    assert result["stderr"] == ""
    assert result["exit_code"] == 0
    assert result["time_ms"] >= 0
