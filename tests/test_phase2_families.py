"""
Phase 2.5: Tests for all 4 supported known families and unsupported family fallback.
"""

import pytest
from core.paper_to_code_generator import PaperToCodeGenerator


@pytest.mark.parametrize(
    ("family", "expected_shape"),
    [
        ("resnet", [1, 1000]),
        ("unet", [1, 2, 256, 256]),
        ("vit", [1, 1000]),
        ("transformer", [1, 1000]),
    ],
)
def test_all_four_supported_families_generate_self_contained_executable_code(family, expected_shape):
    """Verify that all 4 supported families produce runnable, self-contained PyTorch code."""
    generator = PaperToCodeGenerator()
    spec = {"model_family": family}

    source = generator._builder_code(family, spec)
    report = generator.validate_generated_code(source, "builder", spec)

    assert "from core." not in source
    assert "import torch" in source
    assert report["passed"] is True, report
    assert report["checks"] == {"syntax": True, "exec": True, "forward": True}
    assert report["output_shape"] == expected_shape


def test_unsupported_family_produces_honest_needs_review_verdict():
    """Verify unsupported family falls back cleanly to skeleton/LLM with needs_review report."""
    generator = PaperToCodeGenerator()
    generator.groq_available = False  # force skeleton fallback

    unsupported_spec = {"model_family": "gan", "output": {"num_classes": 10}}
    code, code_source = generator._generate_code(unsupported_spec, generator.pipeline.run_single({"name": "GAN", "layers": [{"type": "conv2d", "params": {}}, {"type": "linear", "params": {}}], "connections": [["layer_0", "layer_1"]]})["graph"])

    assert code_source == "skeleton"
    assert "class" in code

    report = generator.validate_generated_code(code, code_source, unsupported_spec)
    assert report["status"] == "needs_review"
    assert report["passed"] is False
