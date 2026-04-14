"""
Tests for ConfigExtractor hardening enhancements (Phase RAG Hardening).

Tests:
- Synonym collision validation
- Multi-block repeat metadata preservation
- Transformer encoder/decoder mapping
- Overlapping multi-block phrase prevention
- Determinism across hardening changes
"""

from src.rag import ConfigExtractor
import src.rag.normalizer  # Import to trigger synonym validation


def test_synonym_map_validation():
    """Test that synonym map is validated at module load time."""
    # If we get here, module load didn't raise AssertionError
    # This means _validate_synonym_map() passed
    print("[PASS] synonym_map_validation: _SYNONYM_MAP validated at module load")


def test_transformer_encoder_mapping():
    """Test that transformer encoder/decoder map to multiheadattention."""
    ex = ConfigExtractor(use_llm=False)

    texts = [
        "Transformer encoder",
        "Transformer decoder",
    ]

    configs = [ex.extract_from_text(t) for t in texts]

    # Both should map to multiheadattention, NOT transformerblock
    for i, config in enumerate(configs):
        layer_type = config["layers"][0]["type"]
        assert layer_type == "multiheadattention", \
            f"Text {i} should map to multiheadattention, got {layer_type}"

    print("[PASS] transformer_encoder_mapping: encoder/decoder -> multiheadattention")


def test_multi_block_repeat_metadata():
    """Test that multi-block expansions have repeat metadata."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("3 residual blocks")

    # Should have 3 layers
    assert len(config["layers"]) == 3

    # Check that repeat metadata is present
    for i, layer in enumerate(config["layers"]):
        assert "_repeat_group" in layer["params"], \
            f"Layer {i} missing _repeat_group"
        assert "_repeat_index" in layer["params"], \
            f"Layer {i} missing _repeat_index"
        assert "_repeat_total" in layer["params"], \
            f"Layer {i} missing _repeat_total"

        # Verify repeat metadata values
        assert layer["params"]["_repeat_total"] == 3, \
            f"Layer {i} has wrong repeat total"
        assert layer["params"]["_repeat_index"] == i, \
            f"Layer {i} has wrong repeat index (expected {i}, got {layer['params']['_repeat_index']})"
        assert layer["params"]["_repeat_group"] == "residualblock", \
            f"Layer {i} has wrong repeat group format (expected 'residualblock', no _block suffix)"

    print("[PASS] multi_block_repeat_metadata: Repeat metadata preserved")


def test_multi_block_repeat_metadata_mixed():
    """Test repeat metadata in mixed single/multi-block architectures."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("Conv layer then 2 residual blocks then Linear")

    # Should have 1 + 2 + 1 = 4 layers
    assert len(config["layers"]) == 4

    # First layer (Conv) should NOT have repeat metadata
    assert "__repeat_group" not in config["layers"][0]["params"]
    assert "__repeat_index" not in config["layers"][0]["params"]
    assert "__repeat_total" not in config["layers"][0]["params"]

    # Next 2 layers (residual blocks) SHOULD have repeat metadata
    for i in [1, 2]:
        assert "_repeat_group" in config["layers"][i]["params"]
        assert config["layers"][i]["params"]["_repeat_total"] == 2
        assert config["layers"][i]["params"]["_repeat_index"] == i - 1

    # Last layer (Linear) should NOT have repeat metadata
    assert "_repeat_group" not in config["layers"][3]["params"]

    print("[PASS] multi_block_repeat_metadata_mixed: Mixed architectures track metadata")


def test_overlapping_phrase_prevention():
    """Test that overlapping multi-block phrases don't double-match."""
    ex = ConfigExtractor(use_llm=False)

    # A phrase where "blocks" could overlap with adjacent words
    # "2 conv layers" followed immediately by text mentioning "blocks"
    text = "2 conv layers blocks are used"

    config = ex.extract_from_text(text)

    # Should handle without creating extra layers from overlapping matches
    # The "2 conv layers" should expand to 2 conv2d layers
    # The word "blocks" should not trigger a second expansion
    conv_layers = [l for l in config["layers"] if l["type"] == "conv2d"]
    assert len(conv_layers) >= 2, "Should have at least 2 conv layers"

    # Make sure we don't have duplicates from overlapping detection
    # (exact count depends on how "blocks" is interpreted, but no >5 is reasonable)
    assert len(config["layers"]) <= 5, "Overlapping phrases created too many layers"

    print("[PASS] overlapping_phrase_prevention: No duplicate expansion from overlaps")


def test_determinism_after_hardening():
    """Test determinism is preserved after all hardening changes."""
    ex = ConfigExtractor(use_llm=False)

    text = """
    ResNet architecture with:
    - 3 Conv2D layers with kernel 3
    - 2 residual blocks
    - Transformer encoder for attention
    - Linear projection layer
    """

    # Run extraction multiple times
    results = [ex.extract_from_text(text) for _ in range(3)]

    # All results should be identical (deterministic)
    assert results[0] == results[1] == results[2], \
        "Hardening changes broke determinism"

    # Verify structure is valid
    config = results[0]
    assert len(config["layers"]) > 0
    assert all("id" in l and "type" in l and "params" in l for l in config["layers"])
    assert len(config["connections"]) == len(config["layers"]) - 1

    print("[PASS] determinism_after_hardening: All hardening preserves determinism")


def test_transformer_variants_consistency():
    """Test that all transformer variants are consistent."""
    ex = ConfigExtractor(use_llm=False)

    # These should still map to transformerblock
    texts_transformer_block = [
        "Transformer block",
        "Transformer layer",
    ]

    # These should map to multiheadattention
    texts_attention = [
        "Transformer encoder",
        "Transformer decoder",
    ]

    configs_block = [ex.extract_from_text(t) for t in texts_transformer_block]
    configs_attention = [ex.extract_from_text(t) for t in texts_attention]

    # Verify transformer block mappings
    for config in configs_block:
        assert config["layers"][0]["type"] == "transformerblock", \
            "Transformer block should map to transformerblock"

    # Verify attention mappings
    for config in configs_attention:
        assert config["layers"][0]["type"] == "multiheadattention", \
            "Transformer encoder/decoder should map to multiheadattention"

    print("[PASS] transformer_variants_consistency: All variants map correctly")


def test_single_layer_no_repeat_metadata():
    """Test that single (non-expanded) layers don't have repeat metadata."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("Conv2D layer")

    assert len(config["layers"]) == 1
    layer = config["layers"][0]

    # Single layer should NOT have repeat metadata
    assert "_repeat_group" not in layer["params"]
    assert "_repeat_index" not in layer["params"]
    assert "_repeat_total" not in layer["params"]

    print("[PASS] single_layer_no_repeat_metadata: Single layers clean of metadata")


def test_param_normalization_preserves_repeat_metadata():
    """Test that parameter normalization preserves repeat metadata."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("2 conv layers with 64 channels")

    # Should have 2 conv layers
    assert len(config["layers"]) == 2

    # Both should have repeat metadata
    for layer in config["layers"]:
        assert "_repeat_group" in layer["params"]
        assert "_repeat_index" in layer["params"]
        assert "_repeat_total" in layer["params"]

    # And should also have the channels parameter (normalized)
    for layer in config["layers"]:
        assert "channels" in layer["params"], \
            "Channels parameter should be normalized and preserved"
        assert layer["params"]["channels"] == 64

    print("[PASS] param_normalization_preserves_repeat_metadata: Both params and metadata preserved")


def test_complex_architecture_with_hardening():
    """Test a realistic complex architecture with hardening features."""
    ex = ConfigExtractor(use_llm=False)

    text = """
    Vision Transformer with:
    - Input Conv2D layer (kernel 7, 64 channels)
    - 3 residual blocks for feature extraction
    - Transformer encoder for global context
    - 2 attention heads for refinement
    - Linear classification layer
    """

    config = ex.extract_from_text(text)

    # Should have reasonable number of layers
    assert 5 <= len(config["layers"]) <= 15, \
        f"Expected 5-15 layers, got {len(config['layers'])}"

    # Should have proper connections
    assert len(config["connections"]) == len(config["layers"]) - 1

    # Check for expected layer types
    types = [l["type"] for l in config["layers"]]
    assert "conv2d" in types, "Should have conv2d"
    assert "residualblock" in types, "Should have residualblock"
    assert "multiheadattention" in types, "Should have multiheadattention (from transformer encoder)"
    assert "linear" in types, "Should have linear"

    # Find residual blocks and verify repeat metadata
    residual_indices = [i for i, l in enumerate(config["layers"]) if l["type"] == "residualblock"]
    if len(residual_indices) >= 2:
        # If we have multiple residual blocks, they should have repeat metadata
        for idx in residual_indices:
            layer = config["layers"][idx]
            if "_repeat_total" in layer["params"]:
                assert layer["params"]["_repeat_total"] > 1, \
                    "Repeat metadata indicates multi-block"

    print("[PASS] complex_architecture_with_hardening: Complex architecture valid")


if __name__ == "__main__":
    test_synonym_map_validation()
    test_transformer_encoder_mapping()
    test_multi_block_repeat_metadata()
    test_multi_block_repeat_metadata_mixed()
    test_overlapping_phrase_prevention()
    test_determinism_after_hardening()
    test_transformer_variants_consistency()
    test_single_layer_no_repeat_metadata()
    test_param_normalization_preserves_repeat_metadata()
    test_complex_architecture_with_hardening()
    print("\n[SUCCESS] All hardening enhancement tests passed!")
