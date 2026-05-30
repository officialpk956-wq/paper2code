"""
Tests for Paper2CodePipeline enhancements (Tasks 1-5).

Tests:
1. Stable cache keys (normalized text)
2. Large input safety guard (layer capping)
3. Repeat metadata in explanations
4. Streamlit RAG mode compatibility
5. Backward compatibility (no breaking changes)
"""

from core.orchestrator.pipeline import Paper2CodePipeline
from core.rag import ConfigExtractor, preprocess_text


def test_cache_key_normalization():
    """Task 1: Cache keys should be normalized (handle whitespace/casing)."""
    pipeline = Paper2CodePipeline()

    # Different formatting of the same architecture
    text1 = "Conv2D then Linear"
    text2 = "conv2d  then  linear"  # Extra spaces
    text3 = "CONV2D THEN LINEAR"  # Different case

    # All should use same normalized key
    result1 = pipeline.run_from_text(text1)
    result2 = pipeline.run_from_text(text2)
    result3 = pipeline.run_from_text(text3)

    # Should be identical (same cached result)
    assert result1 == result2, "Different whitespace should use same cache key"
    assert result1 == result3, "Different casing should use same cache key"

    # All should come from cache (same object)
    assert result1 is result2 is result3, "All variants should return same cached object"

    print("[PASS] cache_key_normalization: whitespace/casing differences handled")


def test_layer_cap_at_max():
    """Task 2: Layers should be capped at MAX_LAYERS (75)."""
    pipeline = Paper2CodePipeline()

    # Create a config with many layers
    config = {
        "name": "LargeModel",
        "layers": [
            {"type": "conv2d", "params": {"channels": 64}} for _ in range(100)
        ]
    }

    result = pipeline.run_single(config)

    # Should be capped at MAX_LAYERS
    assert len(result["graph"].nodes) <= pipeline.MAX_LAYERS, \
        f"Graph has {len(result['graph'].nodes)} nodes, expected <= {pipeline.MAX_LAYERS}"

    # Should have truncation metadata
    assert "metadata" in result
    assert result["metadata"]["truncated"] is True
    assert result["metadata"]["original_layer_count"] == 100
    assert result["metadata"]["layer_count"] == pipeline.MAX_LAYERS

    print("[PASS] layer_cap_at_max: large inputs capped at 75 layers")


def test_layer_cap_below_max():
    """Task 2: Inputs below MAX_LAYERS should NOT be truncated."""
    pipeline = Paper2CodePipeline()

    config = {
        "name": "SmallModel",
        "layers": [
            {"type": "conv2d", "params": {"channels": 64}} for _ in range(20)
        ]
    }

    result = pipeline.run_single(config)

    # Should not be truncated
    assert len(result["graph"].nodes) == 20
    assert "metadata" not in result or not result["metadata"].get("truncated")

    print("[PASS] layer_cap_below_max: inputs below 75 layers not truncated")


def test_repeat_metadata_in_explanation():
    """Task 3: Repeat patterns should appear in explanation."""
    from core.agents import ParsingAgentImpl
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))

    # Extract from text that contains multi-block patterns
    text = "3 residual blocks then 2 conv layers"

    result = pipeline.run_from_text(text)
    explanation = result["explanation"]

    # Should mention repeated patterns
    assert "Repeated Patterns" in explanation or "repeated" in explanation.lower(), \
        "Explanation should mention repeated structures"

    # Should mention residualblock repetition
    assert "residual" in explanation.lower() and "3" in explanation, \
        "Explanation should mention residual repeated 3 times"

    # Should mention conv2d repetition
    assert "conv2d" in explanation.lower() and "2" in explanation, \
        "Explanation should mention conv2d repeated 2 times"

    print("[PASS] repeat_metadata_in_explanation: repeated patterns exposed in text")


def test_no_repeat_metadata_when_not_repeated():
    """Task 3: Repeat patterns section should NOT appear for non-repeated blocks."""
    pipeline = Paper2CodePipeline()

    config = {
        "name": "SimpleModel",
        "layers": [
            {"type": "conv2d", "params": {"channels": 64}},
            {"type": "linear", "params": {}}
        ]
    }

    result = pipeline.run_single(config)
    explanation = result["explanation"]

    # Should not have "Repeated Patterns" section
    assert "Repeated Patterns" not in explanation, \
        "Repeated Patterns section should not appear for non-repeated architecture"

    print("[PASS] no_repeat_metadata_when_not_repeated: no false positives")


def test_pipeline_determinism_with_safety_cap():
    """Task 2: Safety cap should be deterministic."""
    pipeline = Paper2CodePipeline()

    config = {
        "name": "TestModel",
        "layers": [{"type": "conv2d", "params": {"channels": 64}} for _ in range(100)]
    }

    result1 = pipeline.run_single(config)
    result2 = pipeline.run_single(config)

    # Both should be identical
    assert result1["graph"] == result2["graph"]
    assert result1["explanation"] == result2["explanation"]

    print("[PASS] pipeline_determinism_with_safety_cap: determinism preserved")


def test_backward_compatibility_run_single():
    """Task 5: run_single() should still work with existing code."""
    pipeline = Paper2CodePipeline()

    extractor = ConfigExtractor(use_llm=False)
    config = extractor.extract_from_text("Conv2D then Linear")

    result = pipeline.run_single(config)

    # Should have all expected fields
    assert "graph" in result
    assert "visual" in result
    assert "explanation" in result

    # metadata is optional (only if truncated)
    if "metadata" in result:
        assert result["metadata"]["truncated"] is False

    print("[PASS] backward_compatibility_run_single: existing API unchanged")


def test_backward_compatibility_run_comparison():
    """Task 5: run_comparison() should still work with existing code."""
    pipeline = Paper2CodePipeline()

    extractor = ConfigExtractor(use_llm=False)
    config_a = extractor.extract_from_text("Conv2D then Linear")
    config_b = extractor.extract_from_text("Conv2D then MaxPool then Linear")

    result = pipeline.run_comparison(config_a, config_b)

    # Should have all expected fields
    assert "graph_a" in result
    assert "graph_b" in result
    assert "comparison_result" in result
    assert "visual_a" in result
    assert "visual_b" in result
    assert "explanation" in result

    print("[PASS] backward_compatibility_run_comparison: comparison API unchanged")


def test_run_from_text_with_large_input():
    """Task 2 + Task 4: RAG mode should handle realistic inputs gracefully."""
    pipeline = Paper2CodePipeline()

    # Realistic text (multi-block expansion caps at 10)
    text = "Conv2D then 10 residual blocks then Linear"

    result = pipeline.run_from_text(text)

    # Should NOT be truncated (only 12 layers total)
    assert "metadata" not in result or not result["metadata"].get("truncated")

    print("[PASS] run_from_text_with_large_input: RAG mode handles realistic inputs")


def test_cache_stability_across_runs():
    """Task 1: Cache should be stable and reused across calls."""
    pipeline = Paper2CodePipeline()

    text = "Conv2D then Linear"

    # First run (cache miss)
    result1 = pipeline.run_from_text(text)

    # Second run (cache hit)
    result2 = pipeline.run_from_text(text)

    # Third run (cache hit)
    result3 = pipeline.run_from_text(text)

    # Should all be same object (cached)
    assert result1 is result2 is result3, "Cache should return same object"

    print("[PASS] cache_stability_across_runs: cache is reused")


def test_comparison_from_text_integration():
    """Task 4: run_comparison_from_text should work end-to-end."""
    from core.agents import ParsingAgentImpl
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))

    text_a = "Conv2D then Linear"
    text_b = "Conv2D then MaxPool then Linear"

    result = pipeline.run_comparison_from_text(text_a, text_b)

    # Should have all expected fields
    assert "graph_a" in result
    assert "graph_b" in result
    assert "explanation" in result

    # Should be deterministic (same comparison gives same result)
    result2 = pipeline.run_comparison_from_text(text_a, text_b)
    assert result == result2

    print("[PASS] comparison_from_text_integration: RAG comparison mode works")


def test_no_logic_duplication():
    """Task 5: Verify no logic duplication in enhancements."""
    from core.agents import ParsingAgentImpl
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))
    extractor = ConfigExtractor(use_llm=False)

    text = "Conv2D with 64 channels then Linear"

    # Path 1: Text-based
    result_text = pipeline.run_from_text(text)

    # Path 2: Config-based
    config = extractor.extract_from_text(text)
    result_config = pipeline.run_single(config)

    # Should produce identical graphs and explanations
    assert result_text["graph"] == result_config["graph"]
    assert result_text["explanation"] == result_config["explanation"]

    print("[PASS] no_logic_duplication: both paths produce identical results")


if __name__ == "__main__":
    test_cache_key_normalization()
    test_layer_cap_at_max()
    test_layer_cap_below_max()
    test_repeat_metadata_in_explanation()
    test_no_repeat_metadata_when_not_repeated()
    test_pipeline_determinism_with_safety_cap()
    test_backward_compatibility_run_single()
    test_backward_compatibility_run_comparison()
    test_run_from_text_with_large_input()
    test_cache_stability_across_runs()
    test_comparison_from_text_integration()
    test_no_logic_duplication()

    print("\n[SUCCESS] All pipeline enhancement tests passed!")
