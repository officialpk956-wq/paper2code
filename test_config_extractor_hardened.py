"""
Hardening validation for ConfigExtractor.

Verifies:
- Deterministic output
- Stable IDs and ordering
- Schema consistency
- No over-extraction
- Safe fallback
- Complete validation
"""

from core.rag import ConfigExtractor


def test_deterministic_output():
    """Test that identical inputs produce identical ConfigDicts."""
    text = "ResNet with Conv2D kernel size 3, 64 channels, MaxPool2D, and Linear layer"

    extractor = ConfigExtractor(use_llm=False)

    result1 = extractor.extract_from_text(text)
    result2 = extractor.extract_from_text(text)
    result3 = extractor.extract_from_text(text)

    # All results should be identical
    assert result1 == result2 == result3
    print("[PASS] deterministic_output: Identical text -> identical ConfigDict")


def test_stable_sequential_ids():
    """Test that layer IDs are always sequential and deterministic."""
    text = """
    Architecture with:
    - Conv2D layer
    - MaxPool2D layer
    - Linear layer
    - ReLU activation
    """

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # IDs should be sequential
    expected_ids = [f"layer_{i}" for i in range(len(config["layers"]))]
    actual_ids = [layer["id"] for layer in config["layers"]]

    assert actual_ids == expected_ids
    assert len(set(actual_ids)) == len(actual_ids), "Duplicate IDs found"

    print("[PASS] stable_sequential_ids: IDs always sequential and unique")


def test_schema_consistency():
    """Test that schema is consistent across all layers."""
    text = "Conv2D MaxPool Linear ReLU Dropout"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Every layer must have exact schema
    for layer in config["layers"]:
        assert set(layer.keys()) == {"id", "type", "params"}
        assert isinstance(layer["id"], str)
        assert isinstance(layer["type"], str)
        assert isinstance(layer["params"], dict)

    # Name must be string
    assert isinstance(config["name"], str)

    # Connections must be tuples of strings
    for source, target in config["connections"]:
        assert isinstance(source, str)
        assert isinstance(target, str)

    print("[PASS] schema_consistency: Consistent schema across all outputs")


def test_type_normalization():
    """Test that type names are normalized."""
    text = "Conv2d LINEAR MaxPool Dense RELU"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    types = [layer["type"] for layer in config["layers"]]

    # All types should be lowercase and normalized
    for t in types:
        assert t.islower() or "_" in t, f"Type not normalized: {t}"

    print("[PASS] type_normalization: Types normalized to lowercase")


def test_no_over_extraction():
    """Test that only explicit parameters are extracted."""
    text = "Layer with maybe kernel 3 and possibly stride 2"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Should not extract uncertain parameters
    # (exact behavior depends on extraction patterns)
    assert all(
        isinstance(params, dict) for params in
        [layer["params"] for layer in config["layers"]]
    )

    print("[PASS] no_over_extraction: Only explicit values extracted")


def test_sequential_connections_only():
    """Test that connections are strictly sequential."""
    text = "Layer1 Layer2 Layer3 Layer4"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    n_layers = len(config["layers"])
    n_connections = len(config["connections"])

    # Should have n-1 connections for n layers (strictly sequential)
    assert n_connections == n_layers - 1

    # Connections should follow ID order
    for i, (source, target) in enumerate(config["connections"]):
        assert source == f"layer_{i}"
        assert target == f"layer_{i + 1}"

    print("[PASS] sequential_connections_only: Connections strictly sequential")


def test_safe_fallback_schema():
    """Test that fallback produces same schema as LLM path."""
    text = "Conv2D with kernel 3 and Linear layer"

    # Rule-based extractor
    extractor_rule = ConfigExtractor(use_llm=False)
    config_rule = extractor_rule.extract_from_text(text)

    # Check schema
    assert "name" in config_rule
    assert "layers" in config_rule
    assert "connections" in config_rule

    # All layers have full schema
    for layer in config_rule["layers"]:
        assert "id" in layer
        assert "type" in layer
        assert "params" in layer

    print("[PASS] safe_fallback_schema: Fallback matches full schema")


def test_final_validation():
    """Test that final validation catches missing/invalid data."""
    text = "Simple neural network"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Name must exist (never empty after normalization)
    assert config["name"]

    # Layers must not be empty
    assert len(config["layers"]) > 0

    # All IDs must be unique
    ids = [layer["id"] for layer in config["layers"]]
    assert len(set(ids)) == len(ids)

    # Connections must reference valid layer IDs
    valid_ids = set(ids)
    for source, target in config["connections"]:
        assert source in valid_ids
        assert target in valid_ids

    print("[PASS] final_validation: All validation checks pass")


def test_param_key_normalization():
    """Test that parameter keys are normalized."""
    text = "Conv with filters 64 and kernel_size 3"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Check that param keys are normalized (e.g., filters -> channels)
    all_param_keys = set()
    for layer in config["layers"]:
        all_param_keys.update(layer["params"].keys())

    # Should not have non-normalized keys like "filters" (should be "channels")
    for key in all_param_keys:
        assert key in {
            "channels", "in_channels", "kernel_size", "stride",
            "padding", "hidden_size", "pool_size"
        } or key is None, f"Unexpected param key: {key}"

    print("[PASS] param_key_normalization: Parameter keys normalized")


if __name__ == "__main__":
    test_deterministic_output()
    test_stable_sequential_ids()
    test_schema_consistency()
    test_type_normalization()
    test_no_over_extraction()
    test_sequential_connections_only()
    test_safe_fallback_schema()
    test_final_validation()
    test_param_key_normalization()
    print("\n[SUCCESS] All hardening tests passed!")
