"""
Phase 2.1: Tests for ConfigExtractor wiring and extraction consistency.
"""

import json
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.paper_to_code_generator import PaperToCodeGenerator
from core.rag.config_extractor import (
    _ARCHITECTURE_QUERY,
    _LLM_EXTRACTION_PROMPT,
    _PARAM_PATTERNS,
    _operation_context,
    ConfigExtractor,
)
from core.rag.normalizer import CANONICAL_TYPES, _PARAM_MAP, _normalize_params
from core.knowledge.operations import OPERATIONS
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


def _prompt_allowed_types() -> set[str]:
    match = re.search(
        r'- "type" must be one of:\s*(.*?)(?=\n- "params")',
        _LLM_EXTRACTION_PROMPT,
        re.DOTALL,
    )
    assert match is not None
    return {
        entry.strip()
        for entry in match.group(1).replace("\n", " ").split(",")
        if entry.strip()
    }


def test_hybrid_architecture_query_includes_common_dimension_signals():
    for value in ("64", "128", "256", "512", "768", "1024", "2048"):
        assert value in _ARCHITECTURE_QUERY.split()


@pytest.mark.parametrize("layer_type", sorted(_prompt_allowed_types()))
def test_prompt_allowed_types_are_canonical(layer_type):
    assert layer_type in CANONICAL_TYPES


@pytest.mark.parametrize(
    "internal_type",
    ["query_projection", "key_projection", "value_projection", "attention_merge"],
)
def test_prompt_excludes_attention_decomposition_types(internal_type):
    assert internal_type not in _prompt_allowed_types()


@pytest.mark.parametrize(
    "label_path",
    sorted((Path(__file__).resolve().parents[1] / "benchmarks" / "labels").glob("*.json")),
)
def test_benchmark_label_types_are_canonical(label_path):
    label = json.loads(label_path.read_text(encoding="utf-8"))
    layer_types = label.get("expected", {}).get("layer_types", [])
    assert set(layer_types).issubset(CANONICAL_TYPES)


def _assert_label_hyperparameter_keys_are_supported(label: dict) -> None:
    produced_keys = {key for key, _patterns in _PARAM_PATTERNS}
    produced_keys.update(_PARAM_MAP.values())
    unsupported = set(label.get("expected", {}).get("key_hyperparams", {})) - produced_keys
    assert not unsupported, f"Unsupported benchmark hyperparameter keys: {sorted(unsupported)}"


@pytest.mark.parametrize(
    "label_path",
    sorted((Path(__file__).resolve().parents[1] / "benchmarks" / "labels").glob("*.json")),
)
def test_benchmark_label_hyperparameter_keys_are_producible(label_path):
    _assert_label_hyperparameter_keys_are_supported(
        json.loads(label_path.read_text(encoding="utf-8"))
    )


def test_benchmark_label_hyperparameter_key_guard_rejects_unknown_key():
    with pytest.raises(AssertionError, match="wibble_size"):
        _assert_label_hyperparameter_keys_are_supported(
            {"expected": {"key_hyperparams": {"wibble_size": 1}}}
        )


def test_parameter_key_normalization_preserves_existing_and_disambiguates_dimensions():
    assert _normalize_params({"out_features": 1000}) == {"hidden_size": 1000}
    assert _normalize_params({"num_heads": 12}) == {"num_heads": 12}


def test_operation_context_renders_grounded_definitions():
    context = _operation_context("We use layer normalization and a GELU activation.")

    assert "layernorm" in context
    assert "gelu" in context
    assert "eps" in context
    assert "nn.LayerNorm" in context


def test_operation_context_is_empty_when_no_operation_is_mentioned():
    assert _operation_context("This paper is about datasets.") == ""


def test_operation_context_limits_mentioned_operations():
    context = _operation_context(" ".join(OPERATIONS), limit=8)

    assert sum(line.startswith("- ") for line in context.splitlines()) <= 8


def test_extraction_prompt_has_a_format_safe_operation_context_slot():
    assert "{operation_context}" in _LLM_EXTRACTION_PROMPT
    rendered = _LLM_EXTRACTION_PROMPT.format(
        few_shot="few shot",
        graph_rules="graph rules",
        operation_context="operation context",
        text="paper text",
    )
    assert "operation context" in rendered


def test_extraction_prompt_names_supported_numeric_parameter_keys():
    params_rule = next(
        line for line in _LLM_EXTRACTION_PROMPT.splitlines() if line.startswith('- "params"')
    )
    for key in (
        "channels",
        "kernel_size",
        "stride",
        "padding",
        "hidden_size",
        "num_heads",
        "num_layers",
        "num_classes",
    ):
        assert key in params_rule

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


def test_config_extractor_records_rule_based_fallback_reason(caplog, monkeypatch):
    extractor = ConfigExtractor(use_llm=False)
    extractor.use_llm = True
    extractor.verify = False
    monkeypatch.setattr(extractor, "_extract_with_llm", lambda _text: (_ for _ in ()).throw(ValueError("quota")))

    config = extractor.extract_from_text("The network uses a convolutional layer.")

    assert config["extraction_method"] == "rule_based_fallback"
    assert config["extraction_reason"] == "ValueError: quota"
    assert "LLM extraction failed; using rule-based fallback" in caplog.text


def test_rule_based_extraction_collects_explicit_convolution_parameters():
    config = ConfigExtractor(use_llm=False).extract_from_text(
        "The model uses a 7x7 convolution with 64 filters, stride 2."
    )

    params = next(layer["params"] for layer in config["layers"] if layer["type"] == "conv2d")
    assert params["kernel_size"] == 7
    assert params["channels"] == 64
    assert params["stride"] == 2


def test_rule_based_extraction_collects_explicit_num_classes_only():
    config = ConfigExtractor(use_llm=False).extract_from_text("The model performs 1000-way classification.")

    params = config["layers"][0]["params"]
    assert params["num_classes"] == 1000


def test_rule_based_extraction_omits_num_classes_when_not_stated():
    config = ConfigExtractor(use_llm=False).extract_from_text("The network uses a convolutional layer.")

    params = config["layers"][0]["params"]
    assert "num_classes" not in params


def test_rule_based_extraction_does_not_fabricate_layer_parameters():
    config = ConfigExtractor(use_llm=False).extract_from_text("The network uses a convolutional layer.")

    assert config["layers"][0]["params"] == {}


@pytest.mark.parametrize(
    "phrase",
    [
        "transposed convolution",
        "transposed convolutions",
        "transpose convolution",
        "transpose convolutions",
        "deconvolution",
        "deconvolutions",
        "fractional-strided convolution",
        "fractional-strided convolutions",
        "fractionally strided convolution",
        "fractionally strided convolutions",
    ],
)
def test_rule_based_extraction_recognizes_transposed_convolution_terminology(phrase):
    config = ConfigExtractor(use_llm=False).extract_from_text(f"The generator uses {phrase}.")

    assert "convtranspose2d" in [layer["type"] for layer in config["layers"]]


@pytest.mark.parametrize("phrase", ["strided convolution", "strided convolutions", "convolution", "convolutions"])
def test_rule_based_extraction_does_not_treat_strided_convolution_as_transposed(phrase):
    config = ConfigExtractor(use_llm=False).extract_from_text(
        f"The discriminator uses a {phrase}."
    )

    assert "convtranspose2d" not in [layer["type"] for layer in config["layers"]]


def test_config_extractor_uses_injected_chunk_retriever_over_real_chunks():
    """
    Phase 3 Half B follow-up: ConfigExtractor._focus_text must actually call
    an injected chunk_retriever (dense/hybrid retrieval) when source_chunks
    are provided, instead of silently ignoring them and always falling back
    to plain BM25 over a re-chunked flat string.
    """
    filler = "This section discusses unrelated background literature and citations. " * 30
    target = (
        "The residual block adds the input tensor back to the output via a "
        "shortcut path around two convolutional layers, the skip connection."
    )
    source_chunks = [{"text": f"{filler} filler chunk {i}."} for i in range(8)]
    source_chunks.insert(4, {"text": target})
    full_text = "\n\n".join(c["text"] for c in source_chunks)
    assert len(full_text) > 10_000  # must exceed max_context_chars to trigger retrieval

    calls = []

    def stub_retriever(query, texts, top_k):
        calls.append((query, len(texts), top_k))
        return [t for t in texts if "shortcut path" in t][:top_k] or texts[:top_k]

    extractor = ConfigExtractor(use_llm=False, verify=False, chunk_retriever=stub_retriever)
    focused = extractor._focus_text(full_text, source_chunks=source_chunks)

    assert calls, "chunk_retriever was never called -- source_chunks wiring is dead"
    assert calls[0][1] == len(source_chunks)
    assert "shortcut path" in focused


def test_config_extractor_falls_back_to_bm25_when_retriever_raises():
    """A broken/misbehaving chunk_retriever must not take down extraction --
    _focus_text should fall back to plain BM25 over the real chunks."""
    filler = "This section discusses unrelated background literature and citations. " * 30
    source_chunks = [{"text": f"{filler} filler chunk {i}."} for i in range(8)]
    full_text = "\n\n".join(c["text"] for c in source_chunks)
    assert len(full_text) > 10_000

    def broken_retriever(query, texts, top_k):
        raise RuntimeError("embedder exploded")

    extractor = ConfigExtractor(use_llm=False, verify=False, chunk_retriever=broken_retriever)
    focused = extractor._focus_text(full_text, source_chunks=source_chunks)
    assert focused  # didn't crash, still returned something


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


# ── JSON parse robustness ─────────────────────────────────────────────────────
# Observed in production: densenet121 and unet both fell back to rule-based
# extraction with "LLM did not return valid JSON", because the model emitted
# `// bottleneck 1x1` inside the JSON. The broad except in extract_from_text
# turned that into a silently wrong spec.

def test_parse_json_response_tolerates_line_comments():
    parsed = ConfigExtractor._parse_json_response(
        '{"name": "DenseNet", "layers": [\n'
        '  {"type": "conv2d", "params": {"kernel_size": 1}},   // bottleneck 1x1\n'
        '  {"type": "concat", "params": {}}\n]}'
    )
    assert parsed["name"] == "DenseNet"
    assert len(parsed["layers"]) == 2


def test_parse_json_response_tolerates_block_comments_and_trailing_commas():
    parsed = ConfigExtractor._parse_json_response(
        '{"layers": [{"type": "relu", "params": {}},], /* note */ "name": "X",}'
    )
    assert parsed["name"] == "X"


def test_parse_json_response_preserves_slashes_inside_strings():
    """A URL must not be mistaken for a comment."""
    parsed = ConfigExtractor._parse_json_response('{"src": "http://arxiv.org//abs//1", "n": 1}')
    assert parsed["src"] == "http://arxiv.org//abs//1"


def test_parse_json_response_handles_preamble_and_truncated_fence():
    assert ConfigExtractor._parse_json_response('Here you go:\n{"a": 1}\nDone.') == {"a": 1}
    assert ConfigExtractor._parse_json_response('```json\n{"a": 1}') == {"a": 1}


def test_parse_json_response_leaves_wellformed_json_untouched():
    assert ConfigExtractor._parse_json_response('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_parse_json_response_still_raises_on_garbage():
    with pytest.raises(ValueError, match="did not return valid JSON"):
        ConfigExtractor._parse_json_response("there is no json here at all")


# ── Focused-context composition: reserved table/caption slots ─────────────────
# Measured cause of the dimension gap: EfficientNet parameterizes 18 of 18
# convolutions because its architecture table was retrieved; ViT's dimension
# table and U-Net's 64->128->256->512 progression never reached the model.

def _chunks_with_tables(n_prose=20, n_tables=3):
    prose = [{"text": f"Background discussion of prior work number {i}. " * 12,
              "chunk_type": "text"} for i in range(n_prose)]
    tables = [{"text": f"Table {i}: stage {i} conv3x3 channels 64 128 256 512",
               "chunk_type": "table"} for i in range(n_tables)]
    return prose + tables


def test_structured_chunks_are_reserved_against_prose_ranking():
    """A retriever that only ever ranks prose must not evict every table."""
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:k])
    chunks = _chunks_with_tables()
    selected = ex._select_focus_chunks(chunks)
    assert any("Table" in s for s in selected), "no table survived selection"
    assert len(selected) <= 6


def test_reservation_is_bounded_and_leaves_room_for_prose():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:k])
    selected = ex._select_focus_chunks(_chunks_with_tables(n_tables=10))
    tables = [s for s in selected if s.startswith("Table")]
    assert len(tables) <= 2, f"reservation unbounded: {len(tables)} tables"
    assert len(selected) - len(tables) >= 1, "tables crowded out all prose"


def test_numeric_structured_chunks_are_reserved():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:k])
    chunks = _chunks_with_tables(n_prose=10, n_tables=1)
    selected = ex._select_focus_chunks(chunks)
    assert any(text.startswith("Table") for text in selected)


@pytest.mark.parametrize(
    "text",
    [
        "Table 2: SQuAD 1.1 results. The BERT ensemble answer when s > s +tau, i,j null",
        "Table 7: CoNLL-2003 Named Entity Recognition re- sults. Hyperparameters were se-",
        "Figure 1: The U-Net architecture with contracting and expanding paths",
        "Table 3: Results on ImageNet. 76.3 77.1 78.8",
    ],
)
def test_results_and_nonnumeric_structured_chunks_do_not_qualify_for_reservation(text):
    from core.rag.config_extractor import _has_architectural_structured_content

    assert not _has_architectural_structured_content(text)


@pytest.mark.parametrize(
    "text",
    [
        "Conv3x3 32 MBConv1 16 MBConv6 24 MBConv6 40 MBConv6 80 MBConv6 112 MBConv6 192 MBConv6 320 Conv1x1&Pooling&FC 1280",
        "Table 1: EfficientNet-B0 baseline network. Stage 1 Conv3x3 resolution 224x224 channels 32 layers 1",
    ],
)
def test_architectural_numeric_structured_chunks_qualify_for_reservation(text):
    from core.rag.config_extractor import _has_architectural_structured_content

    assert _has_architectural_structured_content(text)


def test_non_numeric_structured_chunks_do_not_consume_reserved_slots():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:k])
    prose = [{"text": f"prose {i} " * 30, "chunk_type": "text"} for i in range(10)]
    chunks = prose + [{"text": "Table 2: ablation study of components", "chunk_type": "table"}]
    selected = ex._select_focus_chunks(chunks)
    assert len(selected) == 6
    assert all(not text.startswith("Table") for text in selected)


def test_focus_expansion_adds_forward_neighbors_when_budget_allows():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:1])
    chunks = [{"text": f"chunk {index} " * 20, "chunk_type": "text"} for index in range(4)]

    selected = ex._select_focus_chunks(
        chunks, total=1, expand_neighbors=True, max_context_chars=10_000
    )

    assert selected == [chunk["text"] for chunk in chunks[:3]]


def test_focus_expansion_prioritizes_retrieval_rank_over_document_position():
    texts = [f"chunk-{index}" for index in range(5)]
    ex = ConfigExtractor(
        use_llm=False,
        verify=False,
        chunk_retriever=lambda q, ranked_texts, k: [ranked_texts[3], ranked_texts[0]],
    )
    chunks = [{"text": text, "chunk_type": "text"} for text in texts]
    one_expansion_budget = sum(len(text) for text in (texts[0], texts[3], texts[4])) + 2 * len("\n\n---\n\n")

    selected = ex._select_focus_chunks(
        chunks,
        total=2,
        expand_neighbors=True,
        max_context_chars=one_expansion_budget,
    )

    assert selected == [texts[0], texts[3], texts[4]]
    assert texts[1] not in selected


def test_focus_expansion_stops_when_no_budget_remains():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:1])
    chunks = [{"text": f"chunk {index} " * 20, "chunk_type": "text"} for index in range(3)]
    ranked = ex._select_focus_chunks(chunks, total=1)

    selected = ex._select_focus_chunks(
        chunks,
        total=1,
        expand_neighbors=True,
        max_context_chars=len(ranked[0]),
    )

    assert selected == ranked


def test_focus_expansion_keeps_ranked_chunks_and_deduplicates_neighbors():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:k])
    chunks = [{"text": f"chunk {index} " * 20, "chunk_type": "text"} for index in range(4)]
    ranked = ex._select_focus_chunks(chunks, total=2)

    selected = ex._select_focus_chunks(
        chunks, total=2, expand_neighbors=True, max_context_chars=10_000
    )

    assert set(ranked).issubset(selected)
    assert len(selected) == len(set(selected))
    assert selected == [chunk["text"] for chunk in chunks]


def test_focus_expansion_handles_empty_and_single_chunk_inputs():
    ex = ConfigExtractor(use_llm=False, verify=False)
    assert ex._select_focus_chunks([], expand_neighbors=True, max_context_chars=100) == []
    assert ex._select_focus_chunks(
        [{"text": "only chunk", "chunk_type": "text"}],
        expand_neighbors=True,
        max_context_chars=100,
    ) == ["only chunk"]


def test_no_structured_chunks_behaves_as_before():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: texts[:k])
    prose = [{"text": f"prose {i} " * 30, "chunk_type": "text"} for i in range(10)]
    assert len(ex._select_focus_chunks(prose)) == 6


def test_selection_preserves_reading_order():
    ex = ConfigExtractor(use_llm=False, verify=False,
                         chunk_retriever=lambda q, texts, k: list(reversed(texts))[:k])
    chunks = [{"text": f"chunk {i} " * 20, "chunk_type": "text"} for i in range(6)]
    selected = ex._select_focus_chunks(chunks)
    order = [chunks.index(next(c for c in chunks if c["text"] == s)) for s in selected]
    assert order == sorted(order), "reading order not preserved"


def test_selection_survives_a_broken_retriever():
    def boom(q, texts, k):
        raise RuntimeError("embedder down")
    ex = ConfigExtractor(use_llm=False, verify=False, chunk_retriever=boom)
    assert ex._select_focus_chunks(_chunks_with_tables())


def test_selection_handles_empty_and_blank_chunks():
    ex = ConfigExtractor(use_llm=False, verify=False)
    assert ex._select_focus_chunks([]) == []
    assert ex._select_focus_chunks([{"text": "   ", "chunk_type": "text"}]) == []


def test_provider_fallback_ignores_litellm_routing_prefix():
    """litellm reports 'openai/gpt-oss-120b' for a call to 'groq/openai/...'.
    A bare != comparison flagged every call as a cross-provider fallback,
    making the alarm useless. Same model must not be flagged."""
    from core.rag.config_extractor import _bare_model_id
    assert _bare_model_id("groq/openai/gpt-oss-120b") == _bare_model_id("openai/gpt-oss-120b")
    assert _bare_model_id("gemini/gemini-3.6-flash") != _bare_model_id("groq/openai/gpt-oss-120b")
    assert _bare_model_id(None) == ""
