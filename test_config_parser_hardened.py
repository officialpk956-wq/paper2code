"""
Verification tests for ConfigParsingAgent hardening.
Tests the 5 corrections applied.
"""

from src.agents.config_parser import ConfigParsingAgent


def test_edge_type_defaults_to_flow():
    """Correction 1: Edge type defaults to 'flow', not inferred from semantic_params."""
    config = {
        "name": "EdgeTypeTest",
        "layers": [
            {"id": "a", "type": "Residual", "label": "Has skip_connection=yes"},
            {"id": "b", "type": "Linear", "label": "Target"},
        ],
        "connections": [("a", "b")]
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    # Even though "a" is Residual with skip_connection="yes",
    # the edge should default to "flow" (not inferred as "residual")
    assert graph.edges[0].edge_type == "flow"
    print("[PASS] Edge types default to 'flow', not inferred from semantic_params")


def test_explicit_edge_type_in_config():
    """Correction 1: Edge type can be explicitly set in config."""
    config = {
        "name": "ExplicitEdgeType",
        "layers": [
            {"id": "a", "type": "Conv2D"},
            {"id": "b", "type": "Conv2D"},
        ],
        "connections": [
            ("a", "b", "residual"),  # Explicit edge type in tuple
        ]
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    assert graph.edges[0].edge_type == "residual"
    print("[PASS] Explicit edge_type in tuple format works")

    # Also test dict format
    config2 = {
        "name": "ExplicitEdgeType2",
        "layers": [
            {"id": "x", "type": "Conv2D"},
            {"id": "y", "type": "Conv2D"},
        ],
        "connections": [
            {"source": "x", "target": "y", "edge_type": "skip"}
        ]
    }

    graph2 = parser.parse(config2)
    assert graph2.edges[0].edge_type == "skip"
    print("[PASS] Explicit edge_type in dict format works")


def test_complete_semantic_params():
    """Correction 2: Every node has complete semantic_params with defaults."""
    config = {
        "name": "SemanticParamsTest",
        "layers": [
            {"id": "a", "type": "Conv2D"},
            {"id": "b", "type": "UnknownLayerType"},
            {"id": "c", "type": "MultiHeadAttention"},
        ],
        "connections": []
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    # Conv2D: overrides defaults
    conv_node = next(n for n in graph.nodes if n.id == "a")
    assert conv_node.semantic_params["flops"] == "high"
    assert conv_node.semantic_params["compute_role"] == "feature_extraction"

    # Unknown: keeps defaults
    unknown_node = next(n for n in graph.nodes if n.id == "b")
    assert unknown_node.semantic_params["flops"] == "low"
    assert unknown_node.semantic_params["compute_role"] == "unknown"

    # MultiHeadAttention: overrides defaults
    attention_node = next(n for n in graph.nodes if n.id == "c")
    assert attention_node.semantic_params["flops"] == "very high"
    assert attention_node.semantic_params["attention"] == "quadratic"
    assert attention_node.semantic_params["compute_role"] == "attention"

    print("[PASS] All nodes have complete semantic_params with defaults")


def test_explicit_semantic_params_override():
    """Correction 2: Explicit semantic_params in config override type-based rules."""
    config = {
        "name": "ExplicitOverride",
        "layers": [
            {
                "id": "custom",
                "type": "Conv2D",
                "semantic_params": {"flops": "low", "custom_field": "value"}
            }
        ],
        "connections": []
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    node = graph.nodes[0]
    # Override: flops="low" instead of default "high" for Conv2D
    assert node.semantic_params["flops"] == "low"
    # New field added
    assert node.semantic_params["custom_field"] == "value"
    # Other rules still applied
    assert node.semantic_params["compute_role"] == "feature_extraction"

    print("[PASS] Explicit semantic_params override type-based rules")


def test_case_insensitive_type_matching():
    """Correction 3: Type matching is case-insensitive and robust."""
    config = {
        "name": "CaseInsensitive",
        "layers": [
            {"id": "a", "type": "CONV2D"},
            {"id": "b", "type": "Conv2d"},
            {"id": "c", "type": "conv2d"},
            {"id": "d", "type": "ConV2D"},
            {"id": "e", "type": "Linear"},
            {"id": "f", "type": "linear"},
        ],
        "connections": []
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    # All conv nodes should get same semantic_params
    conv_nodes = [n for n in graph.nodes if "conv" in n.type.lower()]
    for node in conv_nodes:
        assert node.semantic_params["flops"] == "high"
        assert node.semantic_params["compute_role"] == "feature_extraction"

    # All linear nodes should get same semantic_params
    linear_nodes = [n for n in graph.nodes if "linear" in n.type.lower()]
    for node in linear_nodes:
        assert node.semantic_params["flops"] == "high"
        assert node.semantic_params["compute_role"] == "projection"

    print("[PASS] Case-insensitive type matching works")


def test_no_composite_block_handling():
    """Correction 4: Composite blocks are treated as flat nodes."""
    config = {
        "name": "NoComposite",
        "layers": [
            {"id": "rb1", "type": "ResidualBlock"},
            {"id": "tb1", "type": "TransformerBlock"},
        ],
        "connections": []
    }

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    # Both should be flat nodes, no internal_graph
    for node in graph.nodes:
        assert node.internal_graph is None
        # But they should have semantic_params
        assert isinstance(node.semantic_params, dict)

    # ResidualBlock gets specific semantic_params
    rb = next(n for n in graph.nodes if n.id == "rb1")
    assert rb.semantic_params["skip_connection"] == "yes"

    # TransformerBlock is unknown (falls back to defaults)
    tb = next(n for n in graph.nodes if n.id == "tb1")
    assert tb.semantic_params["flops"] == "low"
    assert tb.semantic_params["compute_role"] == "unknown"

    print("[PASS] Composite blocks treated as flat nodes")


def test_no_input_mutation():
    """Correction 5: Input ConfigDict is not mutated."""
    config = {
        "name": "NoMutation",
        "description": "Original description",
        "layers": [
            {"id": "x", "type": "Conv2D", "label": "Original label"}
        ],
        "connections": []
    }

    # Make a copy to compare
    import copy
    config_copy = copy.deepcopy(config)

    parser = ConfigParsingAgent()
    graph = parser.parse(config)

    # Verify config is unchanged
    assert config == config_copy
    assert config["name"] == "NoMutation"
    assert config["layers"][0]["type"] == "Conv2D"

    print("[PASS] Input ConfigDict is not mutated")


if __name__ == "__main__":
    test_edge_type_defaults_to_flow()
    test_explicit_edge_type_in_config()
    test_complete_semantic_params()
    test_explicit_semantic_params_override()
    test_case_insensitive_type_matching()
    test_no_composite_block_handling()
    test_no_input_mutation()
    print("\n[SUCCESS] All hardening tests passed!")
