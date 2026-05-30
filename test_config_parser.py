"""
Quick test for ConfigParsingAgent.
"""

from core.agents.config_parser import ConfigParsingAgent


def test_simple_resnet():
    """Test parsing a simple ResNet-like config."""
    config = {
        "name": "SimpleResNet",
        "description": "A simple ResNet configuration",
        "layers": [
            {
                "id": "stem",
                "type": "Conv2D",
                "label": "Stem Conv",
                "params": {"out_channels": 64, "kernel_size": 7},
                "description": "Initial convolution layer"
            },
            {
                "id": "pool1",
                "type": "MaxPool2D",
                "label": "Max Pool 1",
                "params": {"kernel_size": 3},
                "description": "First pooling layer"
            },
            {
                "id": "residual_block",
                "type": "ResidualBlock",
                "label": "Residual Block",
                "params": {"channels": 64},
                "description": "Residual block with skip connection"
            },
            {
                "id": "linear",
                "type": "Linear",
                "label": "Classification Head",
                "params": {"out_features": 1000},
                "description": "Final linear layer for classification"
            }
        ],
        "connections": [
            ("stem", "pool1"),
            ("pool1", "residual_block"),
            ("residual_block", "linear")
        ]
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    # Verify graph properties
    assert graph.name == "SimpleResNet"
    assert len(graph.nodes) == 4, f"Expected 4 nodes, got {len(graph.nodes)}"
    assert len(graph.edges) == 3, f"Expected 3 edges, got {len(graph.edges)}"

    # Verify nodes have semantic_params
    stem_node = next(n for n in graph.nodes if n.id == "stem")
    assert stem_node.type == "Conv2D"
    assert stem_node.semantic_params.get("flops") == "high"
    assert stem_node.semantic_params.get("compute_role") == "feature_extraction"

    pool_node = next(n for n in graph.nodes if n.id == "pool1")
    assert pool_node.type == "MaxPool2D"
    assert pool_node.semantic_params.get("feature_map") == "downsampling"
    assert pool_node.semantic_params.get("compute_role") == "pooling"

    residual_node = next(n for n in graph.nodes if n.id == "residual_block")
    assert residual_node.semantic_params.get("skip_connection") == "yes"

    linear_node = next(n for n in graph.nodes if n.id == "linear")
    assert linear_node.semantic_params.get("flops") == "high"

    # Verify edges (all default to "flow" unless explicitly specified)
    assert len(graph.edges) == 3
    assert graph.edges[0].source == "stem" and graph.edges[0].target == "pool1"
    assert all(edge.edge_type == "flow" for edge in graph.edges)

    print("[PASS] test_simple_resnet passed")


def test_unet_config():
    """Test parsing a U-Net-like config with upsampling."""
    config = {
        "name": "SimpleUNet",
        "layers": [
            {"id": "conv1", "type": "Conv2D", "label": "Encoder Conv 1"},
            {"id": "pool", "type": "MaxPool2D", "label": "Downsample"},
            {"id": "conv2", "type": "Conv2D", "label": "Bottleneck Conv"},
            {"id": "upsample", "type": "UpSample", "label": "Upsample"},
            {"id": "conv3", "type": "Conv2D", "label": "Decoder Conv"},
        ],
        "connections": [
            ("conv1", "pool"),
            ("pool", "conv2"),
            ("conv2", "upsample"),
            ("upsample", "conv3"),
        ]
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    assert graph.name == "SimpleUNet"
    assert len(graph.nodes) == 5
    assert len(graph.edges) == 4

    # Verify upsampling semantic_params
    upsample_node = next(n for n in graph.nodes if n.id == "upsample")
    assert upsample_node.semantic_params.get("feature_map") == "upsampling"
    assert upsample_node.semantic_params.get("compute_role") == "reconstruction"

    print("[PASS] test_unet_config passed")


def test_transformer_config():
    """Test parsing a Transformer-like config with attention."""
    config = {
        "name": "SimpleTransformer",
        "layers": [
            {"id": "embedding", "type": "Linear", "label": "Embedding"},
            {"id": "attention", "type": "MultiHeadAttention", "label": "Multi-Head Attention"},
            {"id": "ffn", "type": "Linear", "label": "Feed Forward"},
        ],
        "connections": [
            ("embedding", "attention"),
            ("attention", "ffn"),
        ]
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    assert graph.name == "SimpleTransformer"
    assert len(graph.nodes) == 3

    # Verify attention semantic_params
    attention_node = next(n for n in graph.nodes if n.id == "attention")
    assert attention_node.semantic_params.get("flops") == "very high"
    assert attention_node.semantic_params.get("attention") == "quadratic"
    assert attention_node.semantic_params.get("compute_role") == "attention"

    print("[PASS] test_transformer_config passed")


def test_error_handling():
    """Test error handling for invalid configs."""
    parser = ConfigParsingAgent()

    # Missing name
    try:
        parser.parse({"layers": []})
        assert False, "Should raise ParsingError for missing name"
    except Exception as e:
        assert "name" in str(e).lower()
        print("[PASS] Correctly rejected config without name")

    # Invalid layer type
    try:
        parser.parse({
            "name": "test",
            "layers": [{"id": "x"}],  # Missing type
            "connections": []
        })
        assert False, "Should raise ParsingError for missing type"
    except Exception as e:
        assert "type" in str(e).lower()
        print("[PASS] Correctly rejected layer without type")

    # Invalid connection (references non-existent node)
    try:
        parser.parse({
            "name": "test",
            "layers": [{"id": "a", "type": "Conv2D"}],
            "connections": [("a", "nonexistent")]
        })
        assert False, "Should raise ParsingError for invalid connection"
    except Exception as e:
        assert "not found" in str(e).lower()
        print("[PASS] Correctly rejected connection to non-existent node")


if __name__ == "__main__":
    test_simple_resnet()
    test_unet_config()
    test_transformer_config()
    test_error_handling()
    print("\n[SUCCESS] All tests passed!")
