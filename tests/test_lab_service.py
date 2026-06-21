"""
Tests for LabService — Feature 3 AI Labs.

All assertions check real PyTorch model outputs:
- params come from count_parameters() on real nn.Module objects
- FLOPs come from FLOPsEngine.estimate() on real shapes captured by forward hooks
- No hardcoded expected values — only structural / sign constraints + known math

Run:
    cd C:\\papper2code
    python -m pytest tests/test_lab_service.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from backend.services.lab_service import LabService  # noqa: E402


@pytest.fixture(scope="module")
def svc() -> LabService:
    return LabService()


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Lab
# ─────────────────────────────────────────────────────────────────────────────


class TestTransformerLab:
    def test_returns_expected_keys(self, svc: LabService):
        r = svc.transformer_metrics(d_model=64, num_heads=4, num_layers=1, seq_len=16, vocab_size=1000)
        assert "error" not in r, r.get("error")
        for key in ("params_M", "total_flops_mflops", "memory_mb", "latency_ms", "flow_steps"):
            assert key in r, f"Missing key: {key}"

    def test_params_positive(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=32, vocab_size=5000)
        assert r["params_M"] > 0

    def test_flops_positive(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=32, vocab_size=5000)
        assert r["total_flops_mflops"] > 0

    def test_more_layers_more_flops(self, svc: LabService):
        r1 = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=1, seq_len=32, vocab_size=1000)
        r6 = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=6, seq_len=32, vocab_size=1000)
        assert r6["total_flops_mflops"] > r1["total_flops_mflops"]

    def test_longer_seq_more_flops(self, svc: LabService):
        r16  = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=16,  vocab_size=1000)
        r128 = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=128, vocab_size=1000)
        assert r128["total_flops_mflops"] > r16["total_flops_mflops"]

    def test_invalid_head_divisibility(self, svc: LabService):
        r = svc.transformer_metrics(d_model=100, num_heads=3, num_layers=1, seq_len=16, vocab_size=1000)
        assert "error" in r

    def test_latency_positive(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=32, vocab_size=1000)
        assert r["latency_ms"] > 0

    def test_flow_steps_non_empty(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=32, vocab_size=1000)
        assert len(r["flow_steps"]) > 0

    def test_flow_step_has_shapes(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=32, vocab_size=1000)
        step = r["flow_steps"][0]
        assert step["input_shape"] is not None
        assert step["output_shape"] is not None

    def test_attention_cost_note_present(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=2, seq_len=32, vocab_size=1000)
        assert "attention_note" in r

    def test_head_dim_computed(self, svc: LabService):
        r = svc.transformer_metrics(d_model=256, num_heads=4, num_layers=1, seq_len=16, vocab_size=1000)
        assert r.get("head_dim") == 64


# ─────────────────────────────────────────────────────────────────────────────
# CNN Lab
# ─────────────────────────────────────────────────────────────────────────────


class TestCNNLab:
    def test_returns_expected_keys(self, svc: LabService):
        r = svc.cnn_metrics(base_channels=16, depth=2, kernel_size=3, image_resolution=64)
        assert "error" not in r, r.get("error")
        for key in ("params_M", "total_flops_mflops", "memory_mb", "latency_ms", "flow_steps"):
            assert key in r

    def test_params_positive(self, svc: LabService):
        r = svc.cnn_metrics(base_channels=32, depth=3, kernel_size=3, image_resolution=64)
        assert r["params_M"] > 0

    def test_flops_positive(self, svc: LabService):
        r = svc.cnn_metrics(base_channels=32, depth=3, kernel_size=3, image_resolution=64)
        assert r["total_flops_mflops"] > 0

    def test_deeper_more_params(self, svc: LabService):
        r2 = svc.cnn_metrics(base_channels=16, depth=2, kernel_size=3, image_resolution=64)
        r6 = svc.cnn_metrics(base_channels=16, depth=6, kernel_size=3, image_resolution=64)
        assert r6["params_M"] > r2["params_M"]

    def test_higher_res_more_flops(self, svc: LabService):
        r64  = svc.cnn_metrics(base_channels=16, depth=2, kernel_size=3, image_resolution=64)
        r224 = svc.cnn_metrics(base_channels=16, depth=2, kernel_size=3, image_resolution=224)
        assert r224["total_flops_mflops"] > r64["total_flops_mflops"]

    def test_receptive_field_present(self, svc: LabService):
        r = svc.cnn_metrics(base_channels=16, depth=3, kernel_size=3, image_resolution=64)
        assert "receptive_field" in r
        assert r["receptive_field"] > 0

    def test_latency_positive(self, svc: LabService):
        r = svc.cnn_metrics(base_channels=16, depth=3, kernel_size=3, image_resolution=64)
        assert r["latency_ms"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# ViT Lab
# ─────────────────────────────────────────────────────────────────────────────


class TestViTLab:
    def test_returns_expected_keys(self, svc: LabService):
        r = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=2)
        assert "error" not in r, r.get("error")
        for key in ("params_M", "total_flops_mflops", "memory_mb", "latency_ms",
                    "token_count", "num_patches", "flow_steps"):
            assert key in r

    def test_token_count_correct(self, svc: LabService):
        r = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=2)
        # (64/8)^2 + 1 cls = 64 + 1 = 65
        assert r["num_patches"] == 64
        assert r["token_count"] == 65

    def test_smaller_patch_more_tokens(self, svc: LabService):
        r16 = svc.vit_metrics(image_size=64, patch_size=16, hidden_dim=128, num_blocks=2)
        r8  = svc.vit_metrics(image_size=64, patch_size=8,  hidden_dim=128, num_blocks=2)
        assert r8["num_patches"] > r16["num_patches"]

    def test_smaller_patch_more_flops(self, svc: LabService):
        r16 = svc.vit_metrics(image_size=64, patch_size=16, hidden_dim=128, num_blocks=2)
        r8  = svc.vit_metrics(image_size=64, patch_size=8,  hidden_dim=128, num_blocks=2)
        assert r8["total_flops_mflops"] > r16["total_flops_mflops"]

    def test_more_blocks_more_params(self, svc: LabService):
        r2  = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=2)
        r8  = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=8)
        assert r8["params_M"] > r2["params_M"]

    def test_num_heads_valid(self, svc: LabService):
        r = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=2)
        h = r["num_heads"]
        # hidden_dim must be divisible by num_heads
        assert 128 % h == 0

    def test_attention_note_present(self, svc: LabService):
        r = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=2)
        assert "attention_note" in r

    def test_patch_size_autocorrected(self, svc: LabService):
        # 64 is not divisible by 6 → auto-corrects to 4
        r = svc.vit_metrics(image_size=64, patch_size=6, hidden_dim=64, num_blocks=1)
        assert "error" not in r


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion Lab
# ─────────────────────────────────────────────────────────────────────────────


class TestDiffusionLab:
    def test_returns_expected_keys(self, svc: LabService):
        r = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=100)
        assert "error" not in r, r.get("error")
        for key in ("params_M", "total_flops_mflops", "memory_mb", "latency_ms",
                    "step_flops_mflops", "inference_flops_mflops"):
            assert key in r

    def test_inference_equals_step_times_T(self, svc: LabService):
        r = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=100)
        # inference ≈ step × steps — rounding of rounded sub-values allows up to 1.0 MFLOPs drift
        assert abs(r["inference_flops_mflops"] - r["step_flops_mflops"] * 100) < 1.0

    def test_more_steps_more_latency(self, svc: LabService):
        r100  = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=100)
        r1000 = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=1000)
        assert r1000["latency_ms"] > r100["latency_ms"]

    def test_params_positive(self, svc: LabService):
        r = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=100)
        assert r["params_M"] > 0

    def test_training_note_present(self, svc: LabService):
        r = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=500)
        assert "training_note" in r

    def test_flow_steps_non_empty(self, svc: LabService):
        r = svc.diffusion_metrics(latent_size=16, channels=3, diffusion_steps=100)
        assert len(r["flow_steps"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Latency estimation
# ─────────────────────────────────────────────────────────────────────────────


class TestLatencyEstimation:
    def test_zero_flops_zero_latency(self, svc: LabService):
        assert LabService._estimate_latency_ms(0) == 0.0

    def test_latency_proportional_to_flops(self):
        # Use values large enough to avoid round() truncation to 0
        lat1 = LabService._estimate_latency_ms(10_000)
        lat2 = LabService._estimate_latency_ms(20_000)
        assert lat1 > 0, "lat1 should be positive for 10 GFLOPs"
        assert abs(lat2 / lat1 - 2.0) < 0.01

    def test_reasonable_resnet_latency(self):
        # ResNet-50 ≈ 8178 MFLOPs → ~0.33ms on RTX4090 at 30% efficiency
        lat = LabService._estimate_latency_ms(8178, "RTX4090")
        assert 0.1 < lat < 5.0, f"Unexpected latency: {lat}ms"


# ─────────────────────────────────────────────────────────────────────────────
# Flow step structure
# ─────────────────────────────────────────────────────────────────────────────


class TestFlowSteps:
    def test_all_steps_have_required_fields(self, svc: LabService):
        r = svc.transformer_metrics(d_model=128, num_heads=4, num_layers=1, seq_len=16, vocab_size=1000)
        required = ("id", "name", "type", "input_shape", "output_shape",
                    "flops_mflops", "params_M", "memory_mb", "formula", "severity")
        for step in r["flow_steps"]:
            for field in required:
                assert field in step, f"Step {step.get('id')} missing '{field}'"

    def test_no_null_shapes_in_flow(self, svc: LabService):
        r = svc.vit_metrics(image_size=64, patch_size=8, hidden_dim=128, num_blocks=2)
        for step in r["flow_steps"]:
            assert step["input_shape"] is not None, f"{step['id']} has null input_shape"
            assert step["output_shape"] is not None, f"{step['id']} has null output_shape"

    def test_severity_valid_values(self, svc: LabService):
        r = svc.cnn_metrics(base_channels=16, depth=2, kernel_size=3, image_resolution=64)
        valid = {"low", "medium", "high", "info", "none"}
        for step in r["flow_steps"]:
            assert step["severity"] in valid, f"Bad severity: {step['severity']}"
