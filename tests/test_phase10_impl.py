"""
tests/test_phase10_impl.py

Unit tests for Phase 10: Research Engineer Mode.
Tests cover code mapper, training configs, cost estimator, and reproduction cards.
"""

import pytest
from core.implementation.code_mapper import get_module_implementation, get_architecture_implementation
from core.implementation.training_config import get_training_config, get_hyperparameter_explanations
from core.implementation.cost_estimator import estimate_training_cost, GPU_SPECS, ARCH_PROFILES
from core.implementation.reproduction_cards import get_reproduction_card
from core.agents.tutor_agent import tutor_manager


# ── Test 1: Module Code Mapping ───────────────────────────────────────────────

class TestModuleCodeMapping:
    """All major module types must return valid, labeled implementation views."""

    MODULE_TYPES = [
        "conv2d", "residualblock", "bottleneckblock", "denseblock",
        "multiheadattention", "patchembedding", "transformerblock",
        "upsample", "linear", "batchnorm2d", "layernorm",
    ]

    def test_all_module_types_return_results(self):
        for mtype in self.MODULE_TYPES:
            result = get_module_implementation(mtype)
            assert result is not None, f"No result for module type: {mtype}"
            assert "pytorch_code" in result or "pytorch_template" in result or "label" in result, \
                f"Missing required fields for: {mtype}"

    def test_label_field_is_valid(self):
        valid_labels = {"Educational Implementation", "Reference Implementation", "Pseudo Implementation"}
        for mtype in self.MODULE_TYPES:
            result = get_module_implementation(mtype)
            assert result["label"] in valid_labels, \
                f"Invalid label '{result['label']}' for {mtype}"

    def test_pytorch_code_not_empty(self):
        result = get_module_implementation("residualblock")
        assert len(result["pytorch_code"]) > 100, "ResidualBlock pytorch code too short"
        assert "nn." in result["pytorch_code"], "Missing nn. reference in ResidualBlock code"

    def test_pseudocode_present(self):
        result = get_module_implementation("multiheadattention")
        assert "pseudocode" in result
        assert len(result["pseudocode"]) > 50, "MHSA pseudocode too short"

    def test_design_rationale_present(self):
        result = get_module_implementation("patchembedding")
        assert "design_rationale" in result
        assert len(result["design_rationale"]) > 50, "PatchEmbedding rationale too short"

    def test_unknown_module_type_returns_pseudo(self):
        result = get_module_implementation("totally_unknown_custom_block")
        assert result["label"] == "Pseudo Implementation"

    def test_params_are_interpolated(self):
        """Template parameters like in_channels should be filled from params dict."""
        result = get_module_implementation("conv2d", {"in_channels": 128, "out_channels": 256})
        assert "128" in result["pytorch_code"] or "256" in result["pytorch_code"], \
            "Params not interpolated into PyTorch code"

    def test_architecture_implementation_structure(self):
        """get_architecture_implementation should produce correct top-level structure."""
        modules = [
            {"id": 1, "layer_name": "Conv Stem", "module_type": "conv2d",
             "explanation": "Initial conv", "graph_nodes": []},
            {"id": 2, "layer_name": "Residual Stage 1", "module_type": "residualblock",
             "explanation": "Residual blocks", "graph_nodes": []},
        ]
        result = get_architecture_implementation("ResNet-50", "ResNet", modules)
        assert result["paper_title"] == "ResNet-50"
        assert result["classification"] == "ResNet"
        assert len(result["modules"]) == 2
        assert "safety_notice" in result
        assert "label" in result
        # Each module should have implementation sub-dict
        for m in result["modules"]:
            assert "implementation" in m
            assert "label" in m["implementation"]


# ── Test 2: Training Configs ──────────────────────────────────────────────────

class TestTrainingConfig:
    """ResNet and Transformer must return distinct, complete training configs."""

    REQUIRED_FIELDS = [
        "loss_function", "optimizer", "learning_rate", "batch_size",
        "epochs", "augmentations"
    ]

    def test_resnet_config_complete(self):
        config = get_training_config("ResNet")
        for field in self.REQUIRED_FIELDS:
            assert field in config, f"Missing field '{field}' in ResNet config"

    def test_transformer_config_complete(self):
        config = get_training_config("Transformer")
        for field in self.REQUIRED_FIELDS:
            assert field in config, f"Missing field '{field}' in Transformer config"

    def test_resnet_vs_transformer_distinct(self):
        resnet = get_training_config("ResNet")
        transformer = get_training_config("Transformer")
        # Different optimizers
        assert resnet["optimizer"]["name"] != transformer["optimizer"]["name"], \
            "ResNet and Transformer should use different optimizers"
        # Different batch sizes
        assert resnet["batch_size"] != transformer["batch_size"], \
            "ResNet and Transformer should have different batch sizes"

    def test_vit_has_warmup(self):
        config = get_training_config("ViT")
        assert config.get("warmup") is not None, "ViT config must include warmup"

    def test_unet_has_segmentation_loss(self):
        config = get_training_config("Encoder-Decoder")
        loss_name = config["loss_function"]["name"].lower()
        assert any(keyword in loss_name for keyword in ["bce", "dice", "cross"]), \
            "U-Net should have segmentation-appropriate loss"

    def test_unknown_classification_falls_back(self):
        config = get_training_config("SomeUnknownArch")
        # Should fall back to CNN config
        assert "loss_function" in config

    def test_hyperparameter_cards_complete(self):
        hp = get_hyperparameter_explanations()
        expected = ["Learning Rate", "Weight Decay", "Batch Size", "Dropout",
                    "Label Smoothing", "Attention Heads", "Hidden Dimension"]
        for name in expected:
            assert name in hp, f"Missing hyperparameter card: {name}"
            card = hp[name]
            assert "what_it_does" in card
            assert "increase_effect" in card
            assert "decrease_effect" in card


# ── Test 3: Cost Estimator ────────────────────────────────────────────────────

class TestCostEstimator:
    """Known inputs must produce deterministic, sensible outputs."""

    def test_resnet_a100_basic(self):
        result = estimate_training_cost(
            architecture="ResNet",
            dataset_size=1_000_000,
            batch_size=256,
            gpu_type="A100",
        )
        assert result["gpu_memory_gb"] > 0
        assert result["training_hours"] > 0
        assert result["compute_cost_usd"] > 0
        assert result["steps_total"] > 0

    def test_deterministic_output(self):
        """Same inputs must always produce same outputs."""
        r1 = estimate_training_cost("ViT", 1_000_000, 32, "V100")
        r2 = estimate_training_cost("ViT", 1_000_000, 32, "V100")
        assert r1["training_hours"] == r2["training_hours"]
        assert r1["compute_cost_usd"] == r2["compute_cost_usd"]

    def test_larger_batch_reduces_steps(self):
        small = estimate_training_cost("ResNet", 100_000, 32, "A100")
        large = estimate_training_cost("ResNet", 100_000, 256, "A100")
        assert large["steps_per_epoch"] < small["steps_per_epoch"]

    def test_assumptions_field_present(self):
        result = estimate_training_cost("DenseNet", 500_000, 64, "T4")
        assert "assumptions" in result
        assert len(result["assumptions"]) >= 4

    def test_label_field_present(self):
        result = estimate_training_cost("ResNet", 1_000_000, 256, "A100")
        assert "label" in result
        assert "Estimate" in result["label"] or "estimate" in result["label"]

    def test_vit_memory_higher_than_resnet(self):
        """ViT should require more VRAM than ResNet at same batch size."""
        resnet = estimate_training_cost("ResNet", 1_000_000, 32, "A100")
        vit = estimate_training_cost("ViT", 1_000_000, 32, "A100")
        assert vit["gpu_memory_gb"] > resnet["gpu_memory_gb"]

    def test_all_gpu_types_work(self):
        # Function normalises legacy aliases (e.g. "RTX 3090" → "RTX3090")
        # so we check that the returned gpu_type is a valid canonical key.
        _normalize = {"RTX 3090": "RTX3090", "RTX 4090": "RTX4090"}
        for gpu in GPU_SPECS.keys():
            result = estimate_training_cost("ResNet", 100_000, 32, gpu)
            expected = _normalize.get(gpu, gpu)
            assert result["gpu_type"] == expected
            assert result["training_hours"] > 0

    def test_all_arch_profiles_work(self):
        # Function normalises alias keys (e.g. "ResNet" → "ResNet50")
        # so we check the returned architecture is a valid canonical key.
        _normalize = {
            "ResNet": "ResNet50",
            "CNN": "VGG16",
            "DenseNet": "DenseNet121",
            "U-Net": "UNet",
            "Encoder-Decoder": "UNet",
            "EfficientNet-B0": "EfficientNetB0",
        }
        for arch in ARCH_PROFILES.keys():
            result = estimate_training_cost(arch, 100_000, 32, "A100")
            expected = _normalize.get(arch, arch)
            assert result["architecture"] == expected


# ── Test 4: Reproduction Cards ────────────────────────────────────────────────

class TestReproductionCards:
    """All 5 architectures must have complete reproduction cards."""

    ARCHITECTURES = ["ResNet", "Transformer", "ViT", "Encoder-Decoder", "DenseNet"]
    REQUIRED_CARD_FIELDS = [
        "paper", "authors", "paper_summary", "architecture",
        "training_config", "expected_results", "known_limitations",
        "common_failure_modes", "reproduction_difficulty"
    ]

    def test_all_architectures_have_cards(self):
        for arch in self.ARCHITECTURES:
            card = get_reproduction_card(arch)
            assert card is not None, f"No reproduction card for: {arch}"

    def test_all_required_fields_present(self):
        for arch in self.ARCHITECTURES:
            card = get_reproduction_card(arch)
            for field in self.REQUIRED_CARD_FIELDS:
                assert field in card, f"Missing field '{field}' in {arch} reproduction card"

    def test_failure_modes_have_fixes(self):
        """Each failure mode must have a 'fix' key."""
        for arch in self.ARCHITECTURES:
            card = get_reproduction_card(arch)
            for mode in card["common_failure_modes"]:
                assert "fix" in mode, f"Failure mode missing 'fix' in {arch}: {mode}"
                assert "symptom" in mode
                assert "cause" in mode

    def test_expected_results_nonempty(self):
        for arch in self.ARCHITECTURES:
            card = get_reproduction_card(arch)
            assert len(card["expected_results"]) > 0, f"No expected results for {arch}"

    def test_fuzzy_classification_match(self):
        """Partial classification strings should still resolve."""
        card = get_reproduction_card("CNN")  # Should match as fallback
        assert card is not None
        card2 = get_reproduction_card("Vision Transformer")  # Should match ViT
        assert card2 is not None

    def test_unknown_classification_returns_default(self):
        card = get_reproduction_card("SomeUnknownArchX99")
        assert card is not None  # Falls back to ResNet


# ── Test 5: Tutor Implementation Context ─────────────────────────────────────

class TestTutorImplementationContext:
    """Tutor must correctly handle 'implementation' context_type."""

    def test_build_context_summary_implementation(self):
        """_build_context_summary must produce a non-empty summary for implementation context."""
        context_data = {
            "paper_title": "ResNet-50",
            "layer_name": "Residual Block Stage 2",
            "module_type": "residualblock",
            "implementation": {
                "component": "Residual Block (He et al. 2016)",
                "concept": "Skip Connections & Gradient Flow",
                "design_rationale": "Skip connections allow gradients to flow directly.",
                "label": "Reference Implementation",
            }
        }
        summary = tutor_manager._build_context_summary("implementation", context_data)
        assert "residualblock" in summary.lower() or "residual" in summary.lower()
        assert "ResNet-50" in summary
        assert "Reference Implementation" in summary
        assert len(summary) > 100

    def test_implementation_context_distinct_from_module(self):
        """Implementation context summary should differ from module context summary."""
        module_ctx = {
            "paper_title": "ResNet-50",
            "layer_name": "Residual Block",
            "module_type": "residualblock",
            "explanation": "A residual block",
            "flops_context": {}
        }
        impl_ctx = {
            "paper_title": "ResNet-50",
            "layer_name": "Residual Block",
            "module_type": "residualblock",
            "implementation": {
                "component": "Residual Block",
                "concept": "Skip Connections",
                "design_rationale": "Enables deep networks.",
                "label": "Reference Implementation"
            }
        }
        summary_module = tutor_manager._build_context_summary("module", module_ctx)
        summary_impl = tutor_manager._build_context_summary("implementation", impl_ctx)
        assert summary_module != summary_impl, "Module and implementation contexts should produce different summaries"
        assert "PyTorch Component" in summary_impl
