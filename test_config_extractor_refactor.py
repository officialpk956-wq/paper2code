"""
Targeted tests for ConfigExtractor refactoring (5 improvements).

Tests:
1. Metadata naming uses single underscore (_repeat_*)
2. Repeat group uses "_block" suffix
3. Truncation flag appears only when needed
4. Invalid canonical types raise AssertionError
5. Determinism: identical input -> identical output
"""

import pytest
from src.rag import ConfigExtractor
from src.rag.normalizer import normalize_config, CANONICAL_TYPES


# ---------------------------------------------------------------------------
# Test 1: Metadata naming uses single underscore
# ---------------------------------------------------------------------------

def test_metadata_uses_single_underscore():
    """_repeat_* fields must use single underscore, not double."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("3 residual blocks")
    assert len(config["layers"]) == 3

    for i, layer in enumerate(config["layers"]):
        params = layer["params"]

        # Must have single-underscore keys
        assert "_repeat_group" in params, f"Layer {i}: missing _repeat_group"
        assert "_repeat_index" in params, f"Layer {i}: missing _repeat_index"
        assert "_repeat_total" in params, f"Layer {i}: missing _repeat_total"

        # Must NOT have double-underscore keys
        assert "__repeat_group" not in params, f"Layer {i}: double-underscore found"
        assert "__repeat_index" not in params, f"Layer {i}: double-underscore found"
        assert "__repeat_total" not in params, f"Layer {i}: double-underscore found"

    print("[PASS] metadata_uses_single_underscore: _repeat_* keys correct")


def test_metadata_preserved_through_normalize():
    """Internal metadata must survive the normalize_config() pipeline."""
    raw = {
        "name": "TestModel",
        "layers": [
            {
                "type": "residualblock",
                "params": {
                    "_repeat_group": "residualblock_block",
                    "_repeat_index": 0,
                    "_repeat_total": 3,
                    "channels": 64,
                },
            },
            {
                "type": "residualblock",
                "params": {
                    "_repeat_group": "residualblock_block",
                    "_repeat_index": 1,
                    "_repeat_total": 3,
                    "channels": 64,
                },
            },
        ],
    }

    config = normalize_config(raw)

    # Metadata must be preserved
    for layer in config["layers"]:
        assert "_repeat_group" in layer["params"]
        assert "_repeat_index" in layer["params"]
        assert "_repeat_total" in layer["params"]
        assert layer["params"]["_repeat_total"] == 3
        # Regular params also preserved
        assert layer["params"]["channels"] == 64

    print("[PASS] metadata_preserved_through_normalize: survives normalize_config()")


# ---------------------------------------------------------------------------
# Test 2: Repeat group uses "_block" suffix
# ---------------------------------------------------------------------------

def test_repeat_group_has_block_suffix():
    """_repeat_group is '{type}' or '{type}_block', never a raw position."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("3 residual blocks")
    assert len(config["layers"]) == 3

    for layer in config["layers"]:
        group = layer["params"]["_repeat_group"]
        # residualblock already ends with "block", so no suffix needed
        assert group == "residualblock", \
            f"Expected 'residualblock', got '{group}'"
        assert not group.isdigit(), \
            f"_repeat_group must not be a raw position number"

    print("[PASS] repeat_group_has_block_suffix: smart naming (no redundant _block)")


def test_repeat_group_suffix_for_different_types():
    """_repeat_group naming is smart: adds _block only if needed."""
    ex = ConfigExtractor(use_llm=False)

    # conv2d doesn't end with "block", so gets "_block" suffix
    config_conv = ex.extract_from_text("2 conv layers")
    for layer in config_conv["layers"]:
        if "_repeat_group" in layer["params"]:
            assert layer["params"]["_repeat_group"] == "conv2d_block"

    # residualblock already ends with "block", so no suffix
    config_residual = ex.extract_from_text("3 residual blocks")
    for layer in config_residual["layers"]:
        if "_repeat_group" in layer["params"]:
            assert layer["params"]["_repeat_group"] == "residualblock"

    print("[PASS] repeat_group_suffix_for_different_types: smart naming consistency")


def test_repeat_index_is_sequential():
    """_repeat_index must be 0-based and sequential within each group."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("4 residual blocks")
    assert len(config["layers"]) == 4

    indices = [l["params"]["_repeat_index"] for l in config["layers"]]
    assert indices == [0, 1, 2, 3], f"Expected [0,1,2,3], got {indices}"

    totals = [l["params"]["_repeat_total"] for l in config["layers"]]
    assert all(t == 4 for t in totals), f"All totals should be 4, got {totals}"

    print("[PASS] repeat_index_is_sequential: 0-based sequential indices")


# ---------------------------------------------------------------------------
# Test 3: Truncation flag appears only when needed
# ---------------------------------------------------------------------------

def test_truncation_flag_not_set_for_small_counts():
    """_repeat_truncated must NOT appear when count is within cap."""
    ex = ConfigExtractor(use_llm=False)

    for count in [1, 5, 10]:
        config = ex.extract_from_text(f"{count} residual blocks")
        for layer in config["layers"]:
            assert "_repeat_truncated" not in layer["params"], \
                f"count={count}: _repeat_truncated should not be set"

    print("[PASS] truncation_flag_not_set_for_small_counts: no false positives")


def test_truncation_flag_set_for_large_counts():
    """_repeat_truncated must be True when original count > 10 (cap)."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("20 residual blocks")

    # Should be capped at 10
    assert len(config["layers"]) == 10, \
        f"Expected 10 layers (capped), got {len(config['layers'])}"

    # Every capped layer must have _repeat_truncated = True
    for i, layer in enumerate(config["layers"]):
        assert "_repeat_truncated" in layer["params"], \
            f"Layer {i}: missing _repeat_truncated flag"
        assert layer["params"]["_repeat_truncated"] is True, \
            f"Layer {i}: _repeat_truncated should be True"

    print("[PASS] truncation_flag_set_for_large_counts: True when count > 10")


def test_truncation_flag_exact_boundary():
    """Boundary: count=10 should NOT set _repeat_truncated (at cap, not over)."""
    ex = ConfigExtractor(use_llm=False)

    config = ex.extract_from_text("10 residual blocks")
    assert len(config["layers"]) == 10

    for i, layer in enumerate(config["layers"]):
        assert "_repeat_truncated" not in layer["params"], \
            f"Layer {i}: count=10 is at cap, should not be marked truncated"

    print("[PASS] truncation_flag_exact_boundary: count=10 not truncated")


def test_truncation_flag_preserved_through_normalize():
    """_repeat_truncated must survive normalize_config()."""
    raw = {
        "name": "TestModel",
        "layers": [
            {
                "type": "residualblock",
                "params": {
                    "_repeat_group": "residualblock_block",
                    "_repeat_index": 0,
                    "_repeat_total": 10,
                    "_repeat_truncated": True,
                },
            },
        ],
    }

    config = normalize_config(raw)
    layer = config["layers"][0]

    assert "_repeat_truncated" in layer["params"]
    assert layer["params"]["_repeat_truncated"] is True

    print("[PASS] truncation_flag_preserved_through_normalize: survives pipeline")


# ---------------------------------------------------------------------------
# Test 4: Invalid canonical types raise AssertionError
# ---------------------------------------------------------------------------

def test_invalid_type_raises_assertion():
    """normalize_config() must raise AssertionError for invalid layer types."""
    raw = {
        "name": "TestModel",
        "layers": [
            {"type": "not_a_real_layer", "params": {}},
        ],
    }

    with pytest.raises(AssertionError) as exc_info:
        normalize_config(raw)

    assert "Invalid canonical type" in str(exc_info.value), \
        f"Expected 'Invalid canonical type' in error, got: {exc_info.value}"

    print("[PASS] invalid_type_raises_assertion: AssertionError on bad type")


def test_known_synonym_does_not_raise():
    """Known synonym variants must NOT raise — they normalize cleanly."""
    synonyms = [
        "conv", "convolution", "convolutional", "convlayer",
        "fc", "dense", "fullyconnected",
        "maxpool", "maxpooling", "avgpool", "averagepooling",
        "attention", "selfattention", "mha", "mhsa",
        "transformer", "transformerblock",
        "batchnorm", "layernorm",
        "relu", "dropout", "upsample",
        "residual", "residualblock",
    ]

    for synonym in synonyms:
        raw = {
            "name": "TestModel",
            "layers": [{"type": synonym, "params": {}}],
        }
        try:
            config = normalize_config(raw)
            assert config["layers"][0]["type"] in CANONICAL_TYPES, \
                f"'{synonym}' normalized to non-canonical type"
        except AssertionError as e:
            raise AssertionError(f"Known synonym '{synonym}' raised: {e}") from e

    print("[PASS] known_synonym_does_not_raise: all synonyms normalize correctly")


def test_empty_type_defaults_to_conv2d():
    """Missing/empty type must default to conv2d without raising."""
    raw = {
        "name": "TestModel",
        "layers": [
            {"type": "", "params": {}},
            {"type": None, "params": {}},
        ],
    }

    config = normalize_config(raw)
    for layer in config["layers"]:
        assert layer["type"] == "conv2d", \
            f"Empty type should default to conv2d, got '{layer['type']}'"

    print("[PASS] empty_type_defaults_to_conv2d: no error on missing type")


def test_canonical_types_constant_is_complete():
    """CANONICAL_TYPES must contain all types used in _SYNONYM_MAP values."""
    from src.rag.normalizer import _SYNONYM_MAP

    all_mapped_values = set(_SYNONYM_MAP.values())
    missing = all_mapped_values - CANONICAL_TYPES

    assert not missing, \
        f"Types in _SYNONYM_MAP not in CANONICAL_TYPES: {missing}"

    print("[PASS] canonical_types_constant_is_complete: all synonym targets are canonical")


# ---------------------------------------------------------------------------
# Test 5: Determinism — identical input produces identical output
# ---------------------------------------------------------------------------

def test_determinism_basic():
    """Same text must produce exactly equal ConfigDicts."""
    ex = ConfigExtractor(use_llm=False)

    text = "Conv2D with 64 channels then MaxPool then 3 residual blocks then Linear"

    results = [ex.extract_from_text(text) for _ in range(5)]
    for i in range(1, 5):
        assert results[0] == results[i], \
            f"Run {i} differs from run 0"

    print("[PASS] determinism_basic: 5 identical runs produce same ConfigDict")


def test_determinism_includes_metadata():
    """Determinism must hold for all fields including repeat metadata."""
    ex = ConfigExtractor(use_llm=False)

    text = "3 residual blocks with 64 channels"

    r1 = ex.extract_from_text(text)
    r2 = ex.extract_from_text(text)

    assert r1 == r2, "Repeat metadata broke determinism"

    # Explicitly check metadata values are stable
    for i, (l1, l2) in enumerate(zip(r1["layers"], r2["layers"])):
        assert l1["params"].get("_repeat_group") == l2["params"].get("_repeat_group")
        assert l1["params"].get("_repeat_index") == l2["params"].get("_repeat_index")
        assert l1["params"].get("_repeat_total") == l2["params"].get("_repeat_total")

    print("[PASS] determinism_includes_metadata: metadata fields are stable")


def test_determinism_across_synonym_variants():
    """Same architecture described with synonyms must be individually deterministic."""
    ex = ConfigExtractor(use_llm=False)

    # Each synonym form is deterministic within itself
    variants = [
        "Conv2D layer",
        "Convolutional layer",
        "Conv layer",
    ]

    for text in variants:
        r1 = ex.extract_from_text(text)
        r2 = ex.extract_from_text(text)
        assert r1 == r2, f"Not deterministic for: '{text}'"
        assert r1["layers"][0]["type"] == "conv2d"

    print("[PASS] determinism_across_synonym_variants: each synonym form is stable")


def test_determinism_after_truncation():
    """Truncated expansion must also be deterministic."""
    ex = ConfigExtractor(use_llm=False)

    text = "15 residual blocks"

    r1 = ex.extract_from_text(text)
    r2 = ex.extract_from_text(text)

    assert r1 == r2
    assert len(r1["layers"]) == 10  # capped

    for layer in r1["layers"]:
        assert layer["params"]["_repeat_truncated"] is True

    print("[PASS] determinism_after_truncation: truncated output is stable")


if __name__ == "__main__":
    # Test 1: Metadata naming
    test_metadata_uses_single_underscore()
    test_metadata_preserved_through_normalize()

    # Test 2: Repeat group suffix
    test_repeat_group_has_block_suffix()
    test_repeat_group_suffix_for_different_types()
    test_repeat_index_is_sequential()

    # Test 3: Truncation flag
    test_truncation_flag_not_set_for_small_counts()
    test_truncation_flag_set_for_large_counts()
    test_truncation_flag_exact_boundary()
    test_truncation_flag_preserved_through_normalize()

    # Test 4: Canonical type enforcement
    test_invalid_type_raises_assertion()
    test_known_synonym_does_not_raise()
    test_empty_type_defaults_to_conv2d()
    test_canonical_types_constant_is_complete()

    # Test 5: Determinism
    test_determinism_basic()
    test_determinism_includes_metadata()
    test_determinism_across_synonym_variants()
    test_determinism_after_truncation()

    print("\n[SUCCESS] All refactor tests passed!")
