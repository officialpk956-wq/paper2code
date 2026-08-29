"""
Phase 2.3: Tests for E2B sandboxed execution validation of LLM-generated code.
"""

import os
from unittest.mock import patch
import pytest

from core.paper_to_code_generator import PaperToCodeGenerator


_LIVE_PHASE2_ENABLED = os.getenv("RUN_LIVE_PHASE2") == "1"


BROKEN_LLM_CODE = """
import torch
import torch.nn as nn

class CustomBrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Intentional dimension mismatch error on forward pass
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        return self.fc2(self.fc1(x))
"""

VALID_LLM_CODE = """
import torch
import torch.nn as nn

class CustomValidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
"""


def test_broken_llm_code_fails_with_structured_e2b_diagnostic():
    """
    Verify broken LLM code submitted to E2B sandbox fails with a structured
    error report. Mocks the __PAPER2CODE_RESULT__ JSON line the real
    harness prints -- exec succeeded (torch installed, class instantiated)
    but the forward pass itself failed, which is what a genuine shape
    mismatch looks like now that the harness actually runs forward().
    """
    generator = PaperToCodeGenerator()
    spec = {"model_family": "custom_gan", "output": {"num_classes": 5}}

    result_json = (
        '{"ok": false, "error": "RuntimeError: mat1 and mat2 shapes cannot be '
        'multiplied (1x20 and 50x5)", "output_shape": null, '
        '"class_name": "CustomBrokenModel", "input_used": null}'
    )
    with patch("backend.services.e2b_service.run_code_in_sandbox") as mock_sandbox:
        mock_sandbox.return_value = {
            "passed": False,
            "stdout": f"__PAPER2CODE_RESULT__{result_json}\n",
            "stderr": "",
            "time_ms": 120,
            "exit_code": 1,
        }

        report = generator.validate_generated_code(BROKEN_LLM_CODE, "llm", spec)

    assert report["passed"] is False
    assert report["status"] == "needs_review"
    assert report["stage"] == "sandbox"
    assert "RuntimeError" in report["error"]
    assert report["checks"]["syntax"] is True
    assert report["checks"]["exec"] is True  # ran; forward pass is what failed
    assert report["checks"]["forward"] is False


def test_valid_llm_code_passes_e2b_sandboxed_validation():
    """
    Verify correct LLM-generated code reaches success in E2B sandbox. Mocks
    the __PAPER2CODE_RESULT__ JSON line the real harness prints after
    genuinely running forward() and checking the output shape.
    """
    generator = PaperToCodeGenerator()
    spec = {"model_family": "custom_gan", "output": {"num_classes": 5}}

    result_json = (
        '{"ok": true, "error": null, "output_shape": [1, 5], '
        '"class_name": "CustomValidModel", "input_used": "torch.randn(1, 3, 224, 224)"}'
    )
    with patch("backend.services.e2b_service.run_code_in_sandbox") as mock_sandbox:
        mock_sandbox.return_value = {
            "passed": True,
            "stdout": f"__PAPER2CODE_RESULT__{result_json}\n",
            "stderr": "",
            "time_ms": 150,
            "exit_code": 0,
        }

        report = generator.validate_generated_code(VALID_LLM_CODE, "llm", spec)

    assert report["passed"] is True
    assert report["status"] == "success"
    assert report["stage"] == "sandbox"
    assert report["checks"]["syntax"] is True
    assert report["checks"]["exec"] is True
    assert report["checks"]["forward"] is True
    assert report["output_shape"] == [1, 5]


def test_sandbox_timeout_is_a_structured_diagnostic():
    generator = PaperToCodeGenerator()
    spec = {"model_family": "custom", "output": {"num_classes": 5}}

    with patch("backend.services.e2b_service.run_code_in_sandbox") as mock_sandbox:
        mock_sandbox.return_value = {
            "passed": False,
            "stdout": "",
            "stderr": "Time limit exceeded",
            "time_ms": 300_000,
            "exit_code": -1,
        }
        report = generator.validate_generated_code(VALID_LLM_CODE, "llm", spec)

    assert report["stage"] == "sandbox"
    assert report["status"] == "needs_review"
    assert report["sandbox"] == {
        "exit_code": -1,
        "time_ms": 300_000,
        "failure_kind": "timeout",
    }


_LIVE_BROKEN_CODE = """
import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)
"""

_LIVE_VALID_CODE = """
import torch
import torch.nn as nn

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
"""


@pytest.mark.live
@pytest.mark.skipif(
    not (_LIVE_PHASE2_ENABLED and os.getenv("E2B_API_KEY")),
    reason="requires RUN_LIVE_PHASE2=1 and a real E2B_API_KEY",
)
def test_real_e2b_sandbox_catches_a_real_forward_pass_shape_mismatch():
    """
    Regression test: the mocked tests above only verify report-parsing --
    they never exercise the real sandbox call. The real integration used to
    always fail with ModuleNotFoundError (torch not in the default sandbox
    template, no install step), and even once that was fixed, the original
    harness only checked that the class could be instantiated -- never
    actually ran forward() -- yet unconditionally reported
    checks["forward"] = True.

    This hits the real E2B API; the sandbox template doesn't have torch
    preinstalled, so this includes a ~2-4 minute cold-start pip install.
    """
    generator = PaperToCodeGenerator()
    spec = {
        "model_family": "gan",
        "input": {"channels": 3, "spatial_dims": [224, 224]},
        "output": {"num_classes": 10},
    }

    report = generator.validate_generated_code(_LIVE_BROKEN_CODE, "llm", spec)

    assert report["checks"]["exec"] is True  # torch installed, class instantiated
    assert report["checks"]["forward"] is False  # real forward pass genuinely failed
    assert report["passed"] is False
    assert "shapes cannot be multiplied" in report["error"]


@pytest.mark.live
@pytest.mark.skipif(
    not (_LIVE_PHASE2_ENABLED and os.getenv("E2B_API_KEY")),
    reason="requires RUN_LIVE_PHASE2=1 and a real E2B_API_KEY",
)
def test_real_e2b_sandbox_passes_genuinely_valid_code():
    generator = PaperToCodeGenerator()
    spec = {
        "model_family": "gan",
        "input": {"channels": 3, "spatial_dims": [224, 224]},
        "output": {"num_classes": 10},
    }

    report = generator.validate_generated_code(_LIVE_VALID_CODE, "llm", spec)

    assert report["passed"] is True
    assert report["checks"] == {"syntax": True, "exec": True, "forward": True}
    assert report["output_shape"] == [1, 10]
