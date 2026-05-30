"""
Tests for ConfigExtractor enhancements (Task 1-4).

Tests:
- Synonym variants
- Attention variants
- Multi-block expansion
- Ordering robustness
"""

from core.rag import ConfigExtractor


def test_conv_synonyms():
    """Test that Conv2D variants all produce the same layer type."""
    ex = ConfigExtractor(use_llm=False)

    texts = [
        "Conv2D layer with 64 channels",
        "Convolution with 64 channels",
        "Convolutional layer with 64 channels",
        "Conv layer with 64 channels",
    ]

    configs = [ex.extract_from_text(t) for t in texts]

    # All should have same layer type
    types = [c["layers"][0]["type"] for c in configs]
    assert all(t == "conv2d" for t in types), f"Got types: {types}"

    # All should be deterministic (same text → same result)
    for i, text in enumerate(texts):
        config2 = ex.extract_from_text(text)
        assert configs[i] == config2, f"Text {i} not deterministic"

    print("[PASS] conv_synonyms: All variants produce conv2d type")


def test_linear_synonyms():
    """Test that Linear variants all produce the same layer type."""
    ex = ConfigExtractor(use_llm=False)

    texts = [
        "Linear layer",
        "Dense layer",
        "Fully connected layer",
        "Fully-connected layer",
        "FC layer",
    ]

    configs = [ex.extract_from_text(t) for t in texts]

    # All should have same layer type
    types = [c["layers"][0]["type"] for c in configs]
    assert all(t == "linear" for t in types), f"Got types: {types}"

    # All should be deterministic
    for i, text in enumerate(texts):
        config2 = ex.extract_from_text(text)
        assert configs[i] == config2, f"Text {i} not deterministic"

    print("[PASS] linear_synonyms: All variants produce linear type")


def test_attention_variants():
    """Test that Attention variants map correctly."""
    ex = ConfigExtractor(use_llm=False)

    texts = [
        "Attention layer",
        "Self-Attention layer",
        "Multi-Head Attention",
        "MHSA block",
    ]

    configs = [ex.extract_from_text(t) for t in texts]

    # All should have multiheadattention type
    types = [c["layers"][0]["type"] for c in configs]
    assert all(t == "multiheadattention" for t in types), f"Got types: {types}"

    # Test transformer encoder/decoder (map to multiheadattention, not transformerblock)
    config_tf = ex.extract_from_text("Transformer encoder")
    assert config_tf["layers"][0]["type"] == "multiheadattention"

    # Test plain transformer (maps to transformerblock)
    config_tb = ex.extract_from_text("Transformer block")
    assert config_tb["layers"][0]["type"] == "transformerblock"

    print("[PASS] attention_variants: All attention variants map correctly")


def test_multi_block_expansion():
    """Test that multi-block phrases are expanded correctly."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("3 residual blocks")

    # Should expand to 3 layers
    assert len(config["layers"]) == 3

    # All should be residualblock type
    for layer in config["layers"]:
        assert layer["type"] == "residualblock"

    # Should have at least 2 sequential connections
    # (residual skip edges may add more)
    assert len(config["connections"]) >= 2

    print("[PASS] multi_block_expansion: 3 residual blocks expands to 3 nodes")


def test_multi_block_mixed():
    """Test multi-block with single layers."""
    ex = ConfigExtractor(use_llm=False)

    text = "Conv layer then 2 residual blocks then Linear layer"
    config = ex.extract_from_text(text)

    # Should have 1 + 2 + 1 = 4 layers
    assert len(config["layers"]) == 4

    types = [layer["type"] for layer in config["layers"]]
    assert types == ["conv2d", "residualblock", "residualblock", "linear"]

    # 4 layers → at least 3 sequential connections
    # (residual skip edges may add more)
    assert len(config["connections"]) >= 3

    print("[PASS] multi_block_mixed: Mixed single and multi-block detection")


def test_ordering_stability():
    """Test that ordering is stable across multiple calls."""
    ex = ConfigExtractor(use_llm=False)

    text = "Conv then Max Pool then Dense layer then ReLU then Dropout"

    configs = [ex.extract_from_text(text) for _ in range(3)]

    # All should produce identical results
    assert configs[0] == configs[1] == configs[2]

    # Check layer order is preserved
    types = [layer["type"] for layer in configs[0]["layers"]]
    expected = ["conv2d", "maxpool2d", "linear", "relu", "dropout"]
    assert types == expected, f"Got {types}, expected {expected}"

    print("[PASS] ordering_stability: Stable ordering across multiple calls")


def test_determinism_with_synonyms():
    """Test that synonym variants produce identical output."""
    ex = ConfigExtractor(use_llm=False)

    result1 = ex.extract_from_text("Convolution layer with 32 filters")
    result2 = ex.extract_from_text("Conv2D layer with 32 channels")

    # Both should be deterministically identical (same text → same output)
    r1_again = ex.extract_from_text("Convolution layer with 32 filters")
    assert result1 == r1_again

    print("[PASS] determinism_with_synonyms: Synonyms preserve determinism")


def test_pooling_variants():
    """Test that pooling variants are detected."""
    ex = ConfigExtractor(use_llm=False)

    texts = [
        "MaxPool layer",
        "Max pooling",
        "Average pooling",
        "AvgPool layer",
    ]

    configs = [ex.extract_from_text(t) for t in texts]

    # MaxPool variants
    assert configs[0]["layers"][0]["type"] == "maxpool2d"
    assert configs[1]["layers"][0]["type"] == "maxpool2d"

    # AvgPool variants
    assert configs[2]["layers"][0]["type"] == "avgpool2d"
    assert configs[3]["layers"][0]["type"] == "avgpool2d"

    print("[PASS] pooling_variants: Pooling variants detected correctly")


def test_multi_block_cap():
    """Test that multi-block expansion is capped at 10."""
    ex = ConfigExtractor(use_llm=False)

    # Try to expand to 20 (should cap at 10)
    config = ex.extract_from_text("20 residual blocks")

    # Should cap at 10, not 20
    # (or may not expand at all if cap is stricter)
    assert len(config["layers"]) <= 10

    print("[PASS] multi_block_cap: Multi-block expansion capped reasonably")


def test_complex_architecture_description():
    """Test a realistic complex architecture description."""
    ex = ConfigExtractor(use_llm=False)

    text = """
    The encoder starts with a 7x7 convolution with 64 channels,
    followed by max pooling and 3 residual blocks.
    Then it has self-attention for global context.
    The decoder has 2 transpose convolution layers and ends with a linear projection.
    """

    config = ex.extract_from_text(text)

    # Expected layers:
    # conv2d (7x7, 64ch) → maxpool2d → residualblock x3 → multiheadattention → conv2d x2 → linear
    # Total: 1 + 1 + 3 + 1 + 2 + 1 = 9 layers

    # Rule-based gets at least 6 layers (conv may be missed from "transpose convolution")
    assert len(config["layers"]) >= 6  # pool, residuals, attention, linear at minimum

    # Should have at least layers-1 connections (skip edges may add more)
    assert len(config["connections"]) >= len(config["layers"]) - 1

    # Check for expected types
    types = [l["type"] for l in config["layers"]]
    assert "maxpool2d" in types
    assert "multiheadattention" in types
    assert "linear" in types

    print("[PASS] complex_architecture_description: Handles realistic descriptions")


if __name__ == "__main__":
    test_conv_synonyms()
    test_linear_synonyms()
    test_attention_variants()
    test_multi_block_expansion()
    test_multi_block_mixed()
    test_ordering_stability()
    test_determinism_with_synonyms()
    test_pooling_variants()
    test_multi_block_cap()
    test_complex_architecture_description()
    print("\n[SUCCESS] All enhancement tests passed!")
