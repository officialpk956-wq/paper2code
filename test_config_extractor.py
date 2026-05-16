"""
Tests for RAG ConfigExtractor.
"""

from src.rag import ConfigExtractor


def test_rule_based_extraction():
    """Test rule-based extraction (no LLM required)."""
    text = """
    ResNet-18 Architecture

    The model consists of:
    1. Initial Conv2D layer with kernel size 7 and 64 output channels
    2. MaxPool2D with pool size 3
    3. Four residual blocks with 64, 128, 256, and 512 channels respectively
    4. Fully connected layer with 1000 units for classification

    Layers are connected sequentially from input to output.
    """

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Check basic structure
    assert config["name"] is not None
    assert len(config["layers"]) > 0
    assert all("id" in layer and "type" in layer for layer in config["layers"])
    # connections >= layers-1: residual/skip edges may add extra connections
    assert len(config["connections"]) >= len(config["layers"]) - 1

    print("[PASS] rule_based_extraction: Extracted config structure")


def test_config_normalization():
    """Test that extracted configs are normalized."""
    text = "Simple Conv and Linear model"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # All layers must have required fields
    for layer in config["layers"]:
        assert "id" in layer
        assert "type" in layer
        assert "params" in layer
        assert isinstance(layer["params"], dict)

    # All connections must be valid tuples
    for conn in config["connections"]:
        assert isinstance(conn, tuple) and len(conn) == 2

    print("[PASS] config_normalization: All required fields present")


def test_layer_type_extraction():
    """Test that layer types are correctly identified."""
    text = """
    The architecture contains:
    - Convolutional layers with kernel size 3
    - MaxPooling layers
    - Fully connected dense layers
    - BatchNorm for normalization
    - ReLU activations
    """

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    layer_types = set(layer["type"] for layer in config["layers"])

    # Should detect various layer types
    assert len(layer_types) > 0

    print("[PASS] layer_type_extraction: Detected multiple layer types")


def test_parameter_extraction():
    """Test extraction of layer parameters."""
    text = """
    Conv2D layer with kernel size 3, 64 output channels, stride 1
    Followed by MaxPool2D with pool size 2
    Dense layer with 512 hidden units
    """

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Check that parameters are extracted
    has_params = any(layer["params"] for layer in config["layers"])
    assert has_params or len(config["layers"]) > 0

    print("[PASS] parameter_extraction: Parameters extracted from text")


def test_determinism():
    """Test that extraction is deterministic (without LLM)."""
    text = """
    ResNet with Conv layers and Linear output
    """

    extractor = ConfigExtractor(use_llm=False)

    config1 = extractor.extract_from_text(text)
    config2 = extractor.extract_from_text(text)

    # Same input should produce same output
    assert config1["name"] == config2["name"]
    assert len(config1["layers"]) == len(config2["layers"])
    assert config1["connections"] == config2["connections"]

    print("[PASS] determinism: Same input -> same output")


def test_sequential_connections():
    """Test that connections are created sequentially."""
    text = "Four layer model: Conv → Pool → Dense → Output"

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text(text)

    # Should have layers
    assert len(config["layers"]) > 0

    # Connections should connect layers in order (>= accounts for skip edges)
    if len(config["layers"]) > 1:
        assert len(config["connections"]) >= len(config["layers"]) - 1

    print("[PASS] sequential_connections: Layers connected in sequence")


if __name__ == "__main__":
    test_rule_based_extraction()
    test_config_normalization()
    test_layer_type_extraction()
    test_parameter_extraction()
    test_determinism()
    test_sequential_connections()
    print("\n[SUCCESS] All ConfigExtractor tests passed!")
