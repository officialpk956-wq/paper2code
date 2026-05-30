"""
Tests for RAG.4 pipeline integration.

Tests:
- run_from_text produces identical output as run_single(extracted_config)
- run_comparison_from_text matches run_comparison(extracted configs)
- Cache returns identical results
"""

from core.orchestrator.pipeline import Paper2CodePipeline
from core.rag import ConfigExtractor
from core.agents.parsing_agent_impl import ParsingAgentImpl


def test_run_from_text_basic():
    """run_from_text should extract and process text."""
    pipeline = Paper2CodePipeline()

    text = "Conv2D layer with 64 channels then MaxPool then Linear"

    result = pipeline.run_from_text(text)

    # Should have all expected fields
    assert "graph" in result
    assert "visual" in result
    assert "explanation" in result

    # Graph should have layers
    assert len(result["graph"].nodes) > 0

    print("[PASS] run_from_text_basic: extracts and processes text")


def test_run_from_text_matches_direct():
    """run_from_text should produce identical output as run_single(extracted_config)."""
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))
    extractor = ConfigExtractor(use_llm=False)

    text = "Conv2D with 64 channels then 3 residual blocks then Linear"

    # Path 1: use run_from_text
    result_from_text = pipeline.run_from_text(text)

    # Path 2: extract config separately and use run_single
    config = extractor.extract_from_text(text)
    result_from_config = pipeline.run_single(config)

    # Both paths should produce identical results
    assert result_from_text["graph"] == result_from_config["graph"], \
        "Graphs differ between run_from_text and run_single"
    assert result_from_text["explanation"] == result_from_config["explanation"], \
        "Explanations differ"
    # Visual should have same structure
    assert result_from_text["visual"]["graphviz_dot"] == result_from_config["visual"]["graphviz_dot"], \
        "Graphviz visuals differ"

    print("[PASS] run_from_text_matches_direct: identical to run_single(extracted_config)")


def test_run_comparison_from_text_basic():
    """run_comparison_from_text should extract and compare both texts."""
    pipeline = Paper2CodePipeline()

    text_a = "Conv2D then MaxPool then Linear"
    text_b = "Conv2D then 2 residual blocks then Linear"

    result = pipeline.run_comparison_from_text(text_a, text_b)

    # Should have all expected fields
    assert "graph_a" in result
    assert "graph_b" in result
    assert "comparison_result" in result
    assert "visual_a" in result
    assert "visual_b" in result
    assert "explanation" in result

    # Both graphs should exist
    assert len(result["graph_a"].nodes) > 0
    assert len(result["graph_b"].nodes) > 0

    print("[PASS] run_comparison_from_text_basic: extracts and compares both texts")


def test_run_comparison_from_text_matches_direct():
    """run_comparison_from_text should produce identical output as run_comparison(extracted configs)."""
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))
    extractor = ConfigExtractor(use_llm=False)

    text_a = "Conv2D with 32 channels then Linear"
    text_b = "Conv2D with 64 channels then Linear"

    # Path 1: use run_comparison_from_text
    result_from_text = pipeline.run_comparison_from_text(text_a, text_b)

    # Path 2: extract configs separately and use run_comparison
    config_a = extractor.extract_from_text(text_a)
    config_b = extractor.extract_from_text(text_b)
    result_from_configs = pipeline.run_comparison(config_a, config_b)

    # Both paths should produce identical results
    assert result_from_text["graph_a"] == result_from_configs["graph_a"]
    assert result_from_text["graph_b"] == result_from_configs["graph_b"]
    assert result_from_text["explanation"] == result_from_configs["explanation"]

    print("[PASS] run_comparison_from_text_matches_direct: identical to run_comparison(extracted_configs)")


def test_cache_returns_identical_results():
    """Cache should return identical results for same text."""
    pipeline = Paper2CodePipeline()

    text = "Conv2D then MaxPool then Linear"

    # First call - should compute and cache
    result1 = pipeline.run_from_text(text)

    # Second call - should return from cache
    result2 = pipeline.run_from_text(text)

    # Both results should be identical
    assert result1 == result2, "Cache did not return identical results"
    assert result1 is result2, "Cache should return same object (not just equal)"

    print("[PASS] cache_returns_identical_results: same text returns cached result")


def test_cache_different_texts():
    """Different texts should not share cache entries."""
    pipeline = Paper2CodePipeline()

    text_a = "Conv2D then Linear"
    text_b = "Conv2D then MaxPool then Linear"

    result_a = pipeline.run_from_text(text_a)
    result_b = pipeline.run_from_text(text_b)

    # Should have different number of layers
    assert len(result_a["graph"].nodes) != len(result_b["graph"].nodes), \
        "Different texts should produce different graphs"

    # Retrieve again - should come from cache
    result_a2 = pipeline.run_from_text(text_a)
    result_b2 = pipeline.run_from_text(text_b)

    # Cached results should match original results
    assert result_a == result_a2
    assert result_b == result_b2

    print("[PASS] cache_different_texts: different texts have separate cache entries")


def test_determinism_from_text():
    """run_from_text should be deterministic (same text → same output multiple times)."""
    pipeline = Paper2CodePipeline()

    text = "3 Conv2D layers with 64 channels then Linear"

    # Run multiple times
    results = [pipeline.run_from_text(text) for _ in range(3)]

    # All results should be identical
    for i in range(1, 3):
        assert results[0] == results[i], \
            f"Run {i} differs from run 0"

    # Cache should be used for runs 2 and 3
    # (they should be the same object due to caching)
    assert results[1] is results[2], \
        "Cached results should be same object"

    print("[PASS] determinism_from_text: identical text always produces same output")


def test_pipeline_agent_separation():
    """Pipeline should maintain agent separation (no duplication of logic)."""
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))
    text = "Conv2D then Linear"
    config = ConfigExtractor(use_llm=False).extract_from_text(text)

    # Both approaches should use same underlying agents
    result_text = pipeline.run_from_text(text)
    result_config = pipeline.run_single(config)

    # Should be identical (same agents processing same config)
    assert result_text == result_config, \
        "Both approaches should use identical agents"

    print("[PASS] pipeline_agent_separation: no logic duplication between paths")


def test_cache_only_affects_text_input():
    """Cache should not affect run_single or run_comparison (they take configs, not text)."""
    pipeline = Paper2CodePipeline(parsing_agent=ParsingAgentImpl(use_llm=False))
    extractor = ConfigExtractor(use_llm=False)

    text = "Conv2D then Linear"
    config = extractor.extract_from_text(text)

    # Use run_from_text (uses cache)
    result1 = pipeline.run_from_text(text)

    # Use run_single with same config (does not use cache)
    result2 = pipeline.run_single(config)

    # Results should be identical
    assert result1 == result2, \
        "Cache should not affect correctness"

    # But objects should be different (no caching in run_single)
    assert result1 is not result2, \
        "run_single should not return cached object"

    print("[PASS] cache_only_affects_text_input: cache transparent to run_single")


if __name__ == "__main__":
    test_run_from_text_basic()
    test_run_from_text_matches_direct()
    test_run_comparison_from_text_basic()
    test_run_comparison_from_text_matches_direct()
    test_cache_returns_identical_results()
    test_cache_different_texts()
    test_determinism_from_text()
    test_pipeline_agent_separation()
    test_cache_only_affects_text_input()

    print("\n[SUCCESS] All RAG.4 pipeline integration tests passed!")
