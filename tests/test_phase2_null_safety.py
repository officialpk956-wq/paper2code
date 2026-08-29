"""
Phase 2.2: Tests for systematic null-safety across normalizers, classifiers, and builders.
"""

import pytest
from core.normalizer import normalize_model_spec
from core.classification import classify_architecture, infer_family_from_name
from core.paper_to_code_generator import PaperToCodeGenerator
from core.architecture_graph import ArchitectureGraph, GraphNode
from core.rag.normalizer import _normalize_type, normalize_config


def test_normalizer_handles_explicit_none_in_nested_fields():
    """Verify normalizer handles explicit None in stem, block, stages, params."""
    raw_spec = {
        "stem": {"type": None, "params": {"out_channels": None, "stride": None, "kernel_size": None}},
        "block": {"type": None, "params": None},
        "stages": [
            {"name": None, "repeats": None, "out_channels": None, "params": None},
            {"params": {"num_filters": None}},
        ],
    }

    normalized = normalize_model_spec(raw_spec)

    assert normalized["stem"]["params"]["out_channels"] == 64
    assert normalized["stem"]["params"]["stride"] == 2
    assert normalized["stem"]["params"]["kernel"] == 7
    assert normalized["block"]["params"] == {}
    assert len(normalized["stages"]) == 2
    assert normalized["stages"][0]["repeats"] == 1


def test_classification_handles_none_node_types_and_labels():
    """Verify classify_architecture does not crash when nodes have None type or label."""
    graph = ArchitectureGraph(name="test_graph")
    graph.add_node(GraphNode(id="node_1", type=None, label=None, params=None))
    graph.add_node(GraphNode(id="node_2", type="conv2d", label=None, params={}))

    family = classify_architecture(graph)
    assert isinstance(family, str)
    assert family == "cnn"


def test_infer_family_from_name_handles_none():
    """Verify infer_family_from_name handles None and empty string."""
    assert infer_family_from_name(None) is None
    assert infer_family_from_name("") is None
    assert infer_family_from_name("resnet50") == "resnet"
    assert infer_family_from_name("u-net-segmentation") == "unet"
    assert infer_family_from_name("vision-transformer-paper") == "vit"


def test_builder_schema_handles_none_values():
    """Verify _prepare_builder_schema handles explicit None in all fields."""
    generator = PaperToCodeGenerator()

    none_spec = {
        "model_family": "resnet",
        "input": None,
        "output": None,
        "stem": None,
        "stages": None,
    }

    prepared = generator._prepare_builder_schema("resnet", none_spec)
    assert prepared["input"]["channels"] == 3
    assert prepared["output"]["num_classes"] == 1000
    assert len(prepared["stages"]) == 4

    prepared_vit = generator._prepare_builder_schema("vit", {"model_family": "vit", "stages": [None]})
    assert prepared_vit["output"]["num_classes"] == 1000

    prepared_trans = generator._prepare_builder_schema("transformer", {"model_family": "transformer", "input": {"vocab_size": None}})
    assert prepared_trans["output"]["num_classes"] == 1000
    assert prepared_trans["input"]["vocab_size"] == 10000


def test_rag_normalizer_handles_none_in_config():
    """Verify core/rag/normalizer.py handles explicit None in layers and params."""
    raw_config = {
        "name": None,
        "layers": [
            {"id": None, "type": None, "params": None},
            {"id": "layer_1", "type": "conv2d", "params": {"channels": None, "stride": None}},
        ],
    }

    normalized = normalize_config(raw_config)
    assert normalized["name"] == "UnknownModel"
    assert len(normalized["layers"]) >= 1


def test_rag_normalizer_accepts_explicit_concat_merge_nodes():
    normalized = normalize_config(
        {
            "name": "U-Net",
            "layers": [
                {"type": "conv2d", "params": {"channels": 64}},
                {"type": "upsample", "params": {}},
                {"type": "concat", "params": {}},
                {"type": "conv2d", "params": {"channels": 64}},
            ],
        }
    )

    assert [layer["type"] for layer in normalized["layers"]] == [
        "conv2d",
        "upsample",
        "concat",
        "conv2d",
    ]


def test_builder_spec_handles_explicit_none_patch_parameters():
    generator = PaperToCodeGenerator()
    spec = generator._config_dict_to_builder_spec(
        {
            "name": "ViT",
            "layers": [
                {
                    "type": "patchembedding",
                    "params": {
                        "patch_size": None,
                        "embed_dim": None,
                        "in_channels": None,
                        "num_patches": None,
                    },
                }
            ],
            "connections": [],
        },
        "vit",
        "paper",
    )

    assert spec["stem"]["params"] == {
        "patch_size": 16,
        "embed_dim": 192,
        "in_channels": 3,
        "num_patches": 196,
    }


def test_normalize_type_degrades_gracefully_for_unrecognized_types():
    """
    Regression test: _normalize_type used a hard `assert` for any layer type
    outside CANONICAL_TYPES, crashing the entire extraction call for one
    unfamiliar layer name -- e.g. a diffusion model's legitimate
    "timestep_embedding" layer, which has no reason to be in a vocabulary
    built around conv/attention architectures. Downstream consumers
    (ConfigParsingAgent._compute_semantic_params) already default gracefully
    for unrecognized types, so there was never a need to hard-fail here.
    """
    assert _normalize_type("timestep_embedding") == "timestep_embedding"
    assert _normalize_type("conv2d") == "conv2d"  # still normalizes known types
    assert _normalize_type(None) == "conv2d"


def test_e2b_input_candidates_handles_scalar_spatial_dims():
    """
    Regression test: _e2b_test_input_candidates did `dims = spatial or
    [224, 224]` then `dims[0]` -- if spatial_dims came back as a bare
    int/float (a paper described as "224x224" can plausibly extract as a
    single 224 rather than a [h, w] list), `dims` stayed a scalar and
    `dims[0]` raised "'int' object is not subscriptable". This crashed
    E2B validation before any generated code even ran, identically on
    every repair attempt (since this executes regardless of what code is
    being validated), silently wasting the entire repair budget on a bug
    that had nothing to do with the code being repaired.
    """
    generator = PaperToCodeGenerator()

    candidates = generator._e2b_test_input_candidates(
        {"input": {"channels": 3, "spatial_dims": 224}}
    )
    assert candidates == ["torch.randn(1, 3, 224, 224)"]

    candidates = generator._e2b_test_input_candidates(
        {"input": {"channels": 3, "spatial_dims": [256, 256]}}
    )
    assert candidates == ["torch.randn(1, 3, 256, 256)"]

    candidates = generator._e2b_test_input_candidates({"input": {}})
    assert len(candidates) == 2
