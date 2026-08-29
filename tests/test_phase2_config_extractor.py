"""
Phase 2.1: Tests for ConfigExtractor wiring and extraction consistency.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.config_extractor import ConfigExtractor
from backend.schemas.architecture_spec import ArchitectureSpec
from backend.services.paper_ingestion_service import _architecture_spec_payload


REAL_RESNET_EXCERPT = """
The ResNet-50 architecture consists of a 7x7 convolutional stem with 64 channels and stride 2,
followed by a 3x3 max pooling layer with stride 2. The residual network backbone comprises
four stages with bottleneck residual blocks. Stage 1 has 3 bottleneck blocks with 64 channels.
Stage 2 has 4 bottleneck blocks with 128 channels and downsampling with stride 2.
Stage 3 has 6 bottleneck blocks with 256 channels and stride 2.
Stage 4 has 3 bottleneck blocks with 512 channels and stride 2.
Finally, a global average pooling layer and a linear classification head output 1000 classes.
"""

REAL_TRANSFORMER_EXCERPT = """
The Transformer model uses a standard encoder architecture with 6 encoder layers.
The model dimension d_model is 512, with 8 attention heads and a feed-forward network
hidden dimension of 2048. Dropout rate is 0.1. A vocabulary size of 10000 tokens is used,
and a linear projection head produces classification over 1000 classes.
"""

_LIVE_PHASE2_ENABLED = os.getenv("RUN_LIVE_PHASE2") == "1"


def test_config_extractor_returns_populated_config_dict():
    """Verify ConfigExtractor extracts structured layers and connections."""
    extractor = ConfigExtractor(use_llm=False)  # rule-based test for deterministic CI
    config = extractor.extract_from_text(REAL_RESNET_EXCERPT)

    assert isinstance(config, dict)
    assert "name" in config
    assert "layers" in config
    assert "connections" in config
    assert len(config["layers"]) >= 2
    assert len(config["connections"]) >= 1


def test_config_extractor_repeated_consistency():
    """Verify ConfigExtractor produces consistent output across 3 repeated runs."""
    extractor = ConfigExtractor(use_llm=False)
    results = [extractor.extract_from_text(REAL_RESNET_EXCERPT) for _ in range(3)]

    # Confirm all 3 runs produced identical layer counts and types
    layer_counts = [len(r["layers"]) for r in results]
    assert len(set(layer_counts)) == 1, f"Inconsistent layer counts across runs: {layer_counts}"

    layer_types_0 = [l["type"] for l in results[0]["layers"]]
    for i in range(1, 3):
        layer_types_i = [l["type"] for l in results[i]["layers"]]
        assert layer_types_0 == layer_types_i


def test_generator_run_pipeline_uses_config_extractor_and_derives_family():
    """Verify _run_pipeline utilizes ConfigExtractor to build graph and generate code."""
    generator = PaperToCodeGenerator()
    generator.config_extractor = ConfigExtractor(use_llm=False)

    # Test with rule-based extraction
    result = generator._run_pipeline(REAL_RESNET_EXCERPT, "resnet50_paper")

    assert result["family"] == "resnet"
    assert result["generation_status"] == "success"
    assert result["verification_report"]["passed"] is True
    assert "ResNetBuilder" in result["code"]
    assert result["verification_report"]["output_shape"] == [1, 1000]


@pytest.mark.parametrize(
    ("layers", "expected_family"),
    [
        (
            [
                {"type": "conv2d", "params": {}},
                {"type": "residualblock", "params": {}},
            ],
            "resnet",
        ),
        (
            [
                {"type": "conv2d", "params": {}},
                {"type": "upsample", "params": {}},
            ],
            "unet",
        ),
        (
            [
                {"type": "patchembedding", "params": {"patch_size": 16}},
                {"type": "transformerblock", "params": {}},
            ],
            "vit",
        ),
        (
            [
                {"type": "multiheadattention", "params": {}},
                {"type": "linear", "params": {}},
            ],
            "transformer",
        ),
    ],
)
def test_anonymous_config_uses_graph_family_for_builder(layers, expected_family):
    """An `unknown` placeholder must not mask deterministic graph classification."""
    generator = PaperToCodeGenerator()
    generator.groq_available = False
    config = {
        "name": "UnknownModel",
        "layers": layers,
        "connections": [
            [f"layer_{index}", f"layer_{index + 1}"]
            for index in range(len(layers) - 1)
        ],
    }

    with patch.object(generator.config_extractor, "extract_from_text", return_value=config):
        result = generator._run_pipeline("realistic methods text", "uploaded-paper")

    assert result["family"] == expected_family
    assert result["code_source"] == "builder"
    assert result["generation_status"] == "success"


def test_config_graph_adapts_to_learning_module_schema_without_data_loss():
    generator = PaperToCodeGenerator()
    config = {
        "name": "Anonymous Transformer",
        "layers": [
            {"type": "multiheadattention", "params": {"num_heads": 8, "d_model": 512}},
            {"type": "linear", "params": {"channels": 1000}},
        ],
        "connections": [["layer_0", "layer_1"]],
    }
    graph = generator.pipeline.run_single(config)["graph"]

    payload = _architecture_spec_payload(
        {"model_family": "transformer", "layers": config["layers"]},
        {"family": "transformer", "graph": graph},
    )
    validated = ArchitectureSpec(**payload)

    assert validated.family == "transformer"
    assert validated.input_shape == [64]
    assert [layer.name for layer in validated.layers] == [
        "multiheadattention",
        "linear",
    ]
    assert validated.layers[0].heads == 8
    assert validated.layers[0].hidden_size == 512


@pytest.mark.live
@pytest.mark.skipif(
    not (_LIVE_PHASE2_ENABLED and os.getenv("GROQ_API_KEY")),
    reason="requires RUN_LIVE_PHASE2=1 and a real GROQ_API_KEY",
)
def test_config_extractor_real_llm_path_is_consistent():
    """
    Regression test: the tests above all use ConfigExtractor(use_llm=False)
    -- the deterministic rule-based fallback -- which is trivially
    consistent by construction and proves nothing about the actual
    production path (ConfigExtractor() defaults to use_llm=True).

    A live run of the real LLM path previously produced 20/8/20 layers for
    3 identical calls with the same input text -- non-determinism caused by
    Groq's rate limit on the new model exhausting mid-extraction and
    silently falling back to Gemini for some calls but not others. See
    core/llm_client.py's rate-limit retry fix and
    tests/test_llm_client_retry.py.

    This hits real Groq/Gemini APIs and costs real tokens -- skipped
    unless GROQ_API_KEY is actually configured.
    """
    extractor = ConfigExtractor()
    assert extractor.use_llm is True

    results = [extractor.extract_from_text(REAL_RESNET_EXCERPT) for _ in range(3)]

    layer_counts = [len(r.get("layers", [])) for r in results]
    assert len(set(layer_counts)) == 1, (
        f"Inconsistent layer counts across 3 real runs: {layer_counts}"
    )

    # Layer *count* consistency is the structurally significant signal --
    # that's what was actually broken (20/8/20) before the rate-limit retry
    # fix, since a mid-pipeline fallback to a different model produced a
    # visibly shorter/different extraction. Individual layer *names* can
    # still vary by harmless synonym (e.g. "avgpool2d" vs
    # "globalavgpool2d" for the same semantic layer) since the LLM isn't
    # forced to pick from a fixed vocabulary -- that's wording variance,
    # not the non-determinism bug this test guards against.
    _POOL_SYNONYMS = {"avgpool2d", "globalavgpool2d", "adaptiveavgpool2d"}

    def _canonical(layer_type: str) -> str:
        return "avgpool2d" if layer_type in _POOL_SYNONYMS else layer_type

    layer_types = [
        [_canonical(l["type"]) for l in r.get("layers", [])] for r in results
    ]
    assert all(t == layer_types[0] for t in layer_types), (
        f"Layer types differ beyond known synonyms across 3 real runs: {layer_types}"
    )
