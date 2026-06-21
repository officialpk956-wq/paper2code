"""
Tests for backend.services.architecture_reconstruction_service (Phase 14A).

All tests are deterministic — no external calls, no LLMs, no file I/O.
"""

from __future__ import annotations

import pytest
from backend.services.architecture_reconstruction_service import (
    reconstruct_architecture,
    _extract_signals,
    _detect_primary_arch,
    _compute_confidence,
    _simulate_tensor_flow,
    _resnet_template,
    _vit_template,
    _unet_template,
    _transformer_template,
    _gan_template,
    _vae_template,
    _diffusion_template,
    Component,
    Connection,
    ArchitectureBlueprint,
)
from core.rag.flops_engine import FLOPsEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kg(arch_name: str, relation: str = "introduces", href: str | None = None) -> dict:
    paper_id = "paper-test"
    arch_id  = f"arch-{arch_name.lower().replace(' ', '-')}"
    return {
        "nodes": [
            {"id": paper_id, "name": "Test Paper", "type": "paper", "href": None},
            {"id": arch_id,  "name": arch_name,    "type": "architecture", "href": href},
        ],
        "edges": [
            {"source": paper_id, "target": arch_id, "relation": relation},
        ],
    }


def _empty_kg() -> dict:
    return {"nodes": [{"id": "paper-x", "name": "X", "type": "paper", "href": None}], "edges": []}


def _section(content: str, title: str = "Methods") -> dict:
    return {"id": "sec-1", "title": title, "content": content, "level": 1}


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

class TestSignalExtraction:
    def test_extracts_depth_from_layer_count(self):
        sigs = _extract_signals("A 50-layer residual network", "", [])
        assert sigs["depth"] == 50

    def test_extracts_depth_from_L_notation(self):
        sigs = _extract_signals("Our model", "We use L = 12 encoder layers", [])
        assert sigs["depth"] == 12

    def test_extracts_num_heads(self):
        sigs = _extract_signals("", "We use 8 attention heads in each layer.", [])
        assert sigs["num_heads"] == 8

    def test_extracts_d_model(self):
        sigs = _extract_signals("", "The model has d_model = 512 hidden units.", [])
        assert sigs["d_model"] == 512

    def test_extracts_d_model_hidden_dim(self):
        sigs = _extract_signals("", "hidden_dim = 768 is used throughout.", [])
        assert sigs["d_model"] == 768

    def test_extracts_ffn_dim(self):
        sigs = _extract_signals("", "", [_section("ffn_dim = 2048 for the feed-forward layers.")])
        assert sigs["ffn_dim"] == 2048

    def test_extracts_patch_size(self):
        sigs = _extract_signals("", "Images are divided using patch_size = 16.", [])
        assert sigs["patch_size"] == 16

    def test_extracts_num_classes(self):
        sigs = _extract_signals("", "evaluated on 1000 class ImageNet benchmark.", [])
        assert sigs["num_classes"] == 1000

    def test_extracts_vocab_size(self):
        sigs = _extract_signals("", "vocab = 30000 subword tokens are used.", [])
        assert sigs["vocab_size"] == 30000

    def test_extracts_seq_len(self):
        sigs = _extract_signals("", "max sequence length = 512 tokens.", [])
        assert sigs["seq_len"] == 512

    def test_returns_none_for_missing_signals(self):
        sigs = _extract_signals("", "", [])
        assert sigs["depth"] is None
        assert sigs["num_heads"] is None

    def test_does_not_extract_zero(self):
        sigs = _extract_signals("", "Using 0 extra layers (baseline).", [])
        assert sigs["depth"] is None


# ---------------------------------------------------------------------------
# Architecture detection from knowledge graph
# ---------------------------------------------------------------------------

class TestDetectPrimaryArch:
    def test_detects_introduces_relation(self):
        kg = _make_kg("ResNet", relation="introduces", href="/architectures/resnet")
        name, href = _detect_primary_arch(kg)
        assert name == "ResNet"
        assert href == "/architectures/resnet"

    def test_detects_uses_relation_as_fallback(self):
        kg = _make_kg("ViT", relation="uses", href="/architectures/vit")
        name, href = _detect_primary_arch(kg)
        assert name == "ViT"

    def test_prefers_introduces_over_uses(self):
        kg = {
            "nodes": [
                {"id": "p1", "name": "Paper", "type": "paper", "href": None},
                {"id": "a1", "name": "ResNet", "type": "architecture", "href": None},
                {"id": "a2", "name": "ViT",    "type": "architecture", "href": None},
            ],
            "edges": [
                {"source": "p1", "target": "a1", "relation": "introduces"},
                {"source": "p1", "target": "a2", "relation": "uses"},
            ],
        }
        name, _ = _detect_primary_arch(kg)
        assert name == "ResNet"

    def test_returns_unknown_for_empty_kg(self):
        name, href = _detect_primary_arch(_empty_kg())
        assert name == "unknown"
        assert href is None

    def test_returns_unknown_if_no_paper_node(self):
        kg = {"nodes": [], "edges": []}
        name, href = _detect_primary_arch(kg)
        assert name == "unknown"

    def test_ignores_non_architecture_targets(self):
        kg = {
            "nodes": [
                {"id": "p1", "name": "Paper", "type": "paper", "href": None},
                {"id": "d1", "name": "ImageNet", "type": "dataset", "href": None},
            ],
            "edges": [
                {"source": "p1", "target": "d1", "relation": "evaluates_on"},
            ],
        }
        name, _ = _detect_primary_arch(kg)
        assert name == "unknown"


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    def test_introduces_gives_high_confidence(self):
        score = _compute_confidence(
            arch_name="ResNet",
            kg_edges=[{"source": "p", "target": "a", "relation": "introduces"}],
            sections=[],
            equations=[],
            signals={k: None for k in ["depth","num_heads","d_model","ffn_dim","patch_size",
                                        "image_size","channels","num_classes","vocab_size","seq_len"]},
        )
        assert score >= 0.35

    def test_uses_gives_lower_confidence(self):
        score_introduces = _compute_confidence(
            "ResNet",
            [{"relation": "introduces"}], [], [], {}
        )
        score_uses = _compute_confidence(
            "ViT",
            [{"relation": "uses"}], [], [], {}
        )
        assert score_introduces > score_uses

    def test_signals_boost_confidence(self):
        base = _compute_confidence("ResNet", [{"relation": "introduces"}], [], [], {})
        with_signals = _compute_confidence(
            "ResNet",
            [{"relation": "introduces"}], [], [],
            {"depth": 50, "d_model": 768, "num_heads": 12, "ffn_dim": 3072},
        )
        assert with_signals > base

    def test_method_section_boosts_confidence(self):
        base = _compute_confidence("ResNet", [{"relation": "introduces"}], [], [], {})
        with_section = _compute_confidence(
            "ResNet",
            [{"relation": "introduces"}],
            [_section("We propose ResNet.", title="Methods")],
            [], {},
        )
        assert with_section > base

    def test_confidence_capped_at_1(self):
        score = _compute_confidence(
            "ResNet",
            [{"relation": "introduces"}],
            [_section("method", title="Methods")],
            [{"text": "eq1"}, {"text": "eq2"}, {"text": "eq3"}],
            {"depth": 50, "num_heads": 12, "d_model": 768, "ffn_dim": 3072, "channels": 64},
        )
        assert score <= 1.0

    def test_unknown_arch_gives_low_confidence(self):
        score = _compute_confidence("unknown", [], [], [], {})
        assert score == 0.0


# ---------------------------------------------------------------------------
# ResNet template
# ---------------------------------------------------------------------------

class TestResNetTemplate:
    def test_default_depth_50(self):
        comps, conns, in_s, out_s = _resnet_template({}, 1000)
        ids = {c.id for c in comps}
        assert "stem" in ids
        assert "stage1" in ids
        assert "stage4" in ids
        assert "head" in ids

    def test_stage_count_is_4(self):
        comps, _, _, _ = _resnet_template({}, 1000)
        stage_ids = [c.id for c in comps if c.id.startswith("stage")]
        assert len(stage_ids) == 4

    def test_sequential_connections_form_chain(self):
        _, conns, _, _ = _resnet_template({}, 1000)
        seq_pairs = [(c.source, c.target) for c in conns if c.connection_type == "sequential"]
        assert len(seq_pairs) >= 6

    def test_input_shape_is_imagenet(self):
        _, _, in_s, _ = _resnet_template({}, 1000)
        assert in_s == [1, 3, 224, 224]

    def test_depth_18_uses_basic_blocks(self):
        comps, _, _, _ = _resnet_template({"depth": 18}, 1000)
        stage1 = next(c for c in comps if c.id == "stage1")
        assert "basic" in stage1.name.lower()

    def test_depth_50_uses_bottleneck(self):
        comps, _, _, _ = _resnet_template({"depth": 50}, 1000)
        stage1 = next(c for c in comps if c.id == "stage1")
        assert "bottleneck" in stage1.name.lower()

    def test_stage_has_repeat_count(self):
        comps, _, _, _ = _resnet_template({"depth": 50}, 1000)
        stage1 = next(c for c in comps if c.id == "stage1")
        assert stage1.metadata.get("repeat_count", 0) >= 1

    def test_custom_n_classes(self):
        comps, _, _, out_s = _resnet_template({"num_classes": 200}, 200)
        head = next(c for c in comps if c.id == "head")
        assert head.shape[-1] == 200
        assert out_s[-1] == 200


# ---------------------------------------------------------------------------
# ViT template
# ---------------------------------------------------------------------------

class TestViTTemplate:
    def test_has_patch_embed_and_encoder_and_head(self):
        comps, _, _, _ = _vit_template({}, 1000)
        ids = {c.id for c in comps}
        assert "patch_embed" in ids
        assert "encoder" in ids
        assert "head" in ids

    def test_encoder_has_repeat_count(self):
        comps, _, _, _ = _vit_template({"depth": 12}, 1000)
        enc = next(c for c in comps if c.id == "encoder")
        assert enc.metadata.get("repeat_count") == 12

    def test_embed_dim_from_signal(self):
        comps, _, _, _ = _vit_template({"d_model": 1024}, 1000)
        enc = next(c for c in comps if c.id == "encoder")
        assert enc.metadata.get("embed_dim") == 1024

    def test_patch_size_from_signal(self):
        comps, _, _, _ = _vit_template({"patch_size": 32}, 1000)
        pe = next(c for c in comps if c.id == "patch_embed")
        assert pe.metadata.get("patch_size") == 32

    def test_all_connections_sequential(self):
        _, conns, _, _ = _vit_template({}, 1000)
        assert all(c.connection_type == "sequential" for c in conns)


# ---------------------------------------------------------------------------
# U-Net template
# ---------------------------------------------------------------------------

class TestUNetTemplate:
    def test_has_4_encoder_stages(self):
        comps, _, _, _ = _unet_template({}, 2)
        enc_ids = [c.id for c in comps if c.id.startswith("enc")]
        assert len(enc_ids) == 4

    def test_has_4_decoder_stages(self):
        comps, _, _, _ = _unet_template({}, 2)
        dec_ids = [c.id for c in comps if c.id.startswith("dec")]
        assert len(dec_ids) == 4

    def test_has_concat_skip_connections(self):
        _, conns, _, _ = _unet_template({}, 2)
        skips = [c for c in conns if c.connection_type == "concat"]
        assert len(skips) == 4

    def test_skip_enc1_to_dec1(self):
        _, conns, _, _ = _unet_template({}, 2)
        skip_pairs = {(c.source, c.target) for c in conns if c.connection_type == "concat"}
        assert ("enc1", "dec1") in skip_pairs

    def test_input_shape_is_grayscale(self):
        _, _, in_s, _ = _unet_template({}, 2)
        assert in_s[1] == 1  # 1 channel

    def test_channel_scaling(self):
        comps, _, _, _ = _unet_template({"channels": 32}, 2)
        enc1 = next(c for c in comps if c.id == "enc1")
        assert enc1.shape[1] == 32


# ---------------------------------------------------------------------------
# Transformer template
# ---------------------------------------------------------------------------

class TestTransformerTemplate:
    def test_has_encoder_decoder_and_head(self):
        comps, _, _, _ = _transformer_template({}, 1000)
        ids = {c.id for c in comps}
        assert "enc_stack" in ids
        assert "dec_stack" in ids
        assert "head" in ids

    def test_cross_attention_connection(self):
        _, conns, _, _ = _transformer_template({}, 1000)
        cross = [c for c in conns if c.connection_type == "cross_attention"]
        assert len(cross) >= 1

    def test_num_layers_from_signal(self):
        comps, _, _, _ = _transformer_template({"depth": 6}, 1000)
        enc = next(c for c in comps if c.id == "enc_stack")
        assert enc.metadata.get("repeat_count") == 6


# ---------------------------------------------------------------------------
# Tensor flow simulation & FLOPs
# ---------------------------------------------------------------------------

class TestTensorFlowSimulation:
    def test_resnet_tensor_flow_has_flops(self):
        comps, conns, in_s, _ = _resnet_template({}, 1000)
        fe = FLOPsEngine()
        flow = _simulate_tensor_flow(comps, conns, in_s, fe)
        # Stem should have non-zero FLOPs
        stem_step = next(s for s in flow if s["component_id"] == "stem")
        assert stem_step["flops_mflops"] > 0

    def test_vit_encoder_has_flops(self):
        comps, conns, in_s, _ = _vit_template({"depth": 12}, 1000)
        fe = FLOPsEngine()
        flow = _simulate_tensor_flow(comps, conns, in_s, fe)
        enc_step = next(s for s in flow if s["component_id"] == "encoder")
        assert enc_step["flops_mflops"] > 0

    def test_pooling_has_zero_flops(self):
        comps, conns, in_s, _ = _resnet_template({}, 1000)
        fe = FLOPsEngine()
        flow = _simulate_tensor_flow(comps, conns, in_s, fe)
        pool_step = next(s for s in flow if s["component_id"] == "pool1")
        assert pool_step["flops_mflops"] == 0.0

    def test_repeat_count_scales_flops(self):
        comps, conns, in_s, _ = _resnet_template({"depth": 50}, 1000)
        fe = FLOPsEngine()
        flow = _simulate_tensor_flow(comps, conns, in_s, fe)
        stage1 = next(c for c in comps if c.id == "stage1")
        stage1_step = next(s for s in flow if s["component_id"] == "stage1")
        repeat = stage1.metadata.get("repeat_count", 1)
        assert stage1_step["flops_mflops"] == pytest.approx(
            stage1_step["flops_mflops"], rel=1e-3
        )
        assert repeat >= 1

    def test_first_component_input_is_blueprint_input(self):
        comps, conns, in_s, _ = _resnet_template({}, 1000)
        fe = FLOPsEngine()
        flow = _simulate_tensor_flow(comps, conns, in_s, fe)
        assert flow[0]["input_shape"] == in_s

    def test_flow_length_equals_component_count(self):
        comps, conns, in_s, _ = _vit_template({}, 1000)
        fe = FLOPsEngine()
        flow = _simulate_tensor_flow(comps, conns, in_s, fe)
        assert len(flow) == len(comps)


# ---------------------------------------------------------------------------
# Blueprint generation (end-to-end)
# ---------------------------------------------------------------------------

class TestBlueprintGeneration:
    def test_reconstruct_returns_dict(self):
        result = reconstruct_architecture(
            paper_title="ResNet: Deep Residual Learning",
            abstract="We propose a 50-layer ResNet.",
            sections=[],
            equations=[],
            knowledge_graph=_make_kg("ResNet", "introduces", "/architectures/resnet"),
        )
        assert isinstance(result, dict)

    def test_blueprint_has_required_keys(self):
        result = reconstruct_architecture("T", "T", [], [], _empty_kg())
        for key in ("id", "name", "components", "connections",
                    "input_shape", "output_shape", "estimated_parameters",
                    "estimated_flops", "confidence_score", "tensor_flow",
                    "architecture_href"):
            assert key in result, f"Missing key: {key}"

    def test_components_are_dicts(self):
        result = reconstruct_architecture("T", "T", [], [], _empty_kg())
        assert all(isinstance(c, dict) for c in result["components"])

    def test_connections_are_dicts(self):
        result = reconstruct_architecture("T", "T", [], [], _empty_kg())
        assert all(isinstance(c, dict) for c in result["connections"])

    def test_resnet_blueprint_estimated_params_nonzero(self):
        result = reconstruct_architecture(
            "ResNet: Deep Residual Learning",
            "We propose a 50-layer ResNet.",
            [],
            [],
            _make_kg("ResNet", "introduces"),
        )
        assert result["estimated_parameters"] > 0

    def test_vit_blueprint_flops_nonzero(self):
        result = reconstruct_architecture(
            "ViT: An Image is Worth 16x16 Words",
            "patch_size = 16, d_model = 768, 12 attention heads.",
            [],
            [],
            _make_kg("ViT", "introduces", "/architectures/vit"),
        )
        assert result["estimated_flops"] > 0

    def test_unet_blueprint_has_concat_connections(self):
        result = reconstruct_architecture(
            "U-Net Segmentation",
            "We propose a U-Net with skip connections.",
            [],
            [],
            _make_kg("U-Net", "introduces"),
        )
        concat_conns = [c for c in result["connections"]
                        if c["connection_type"] == "concat"]
        assert len(concat_conns) >= 1

    def test_architecture_href_set_when_known(self):
        result = reconstruct_architecture(
            "ResNet study",
            "ResNet trained on ImageNet.",
            [], [],
            _make_kg("ResNet", "introduces", "/architectures/resnet"),
        )
        assert result["architecture_href"] == "/architectures/resnet"

    def test_architecture_href_none_when_unknown(self):
        result = reconstruct_architecture(
            "Novel Architecture",
            "A new architecture for image recognition.",
            [], [],
            _empty_kg(),
        )
        assert result["architecture_href"] is None

    def test_confidence_score_between_0_and_1(self):
        result = reconstruct_architecture("T", "T", [], [], _empty_kg())
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_introduces_relation_gives_high_confidence(self):
        result = reconstruct_architecture(
            "ResNet study",
            "A 50-layer ResNet with 12 heads.",
            [_section("methods")], [{"text": "eq1"}],
            _make_kg("ResNet", "introduces"),
        )
        assert result["confidence_score"] > 0.30

    def test_tensor_flow_len_equals_components(self):
        result = reconstruct_architecture("T", "T", [], [], _make_kg("ResNet", "uses"))
        assert len(result["tensor_flow"]) == len(result["components"])

    def test_paper_id_stored_in_id_field(self):
        result = reconstruct_architecture("T", "T", [], [], _empty_kg(), paper_id=42)
        assert "42" in result["id"]

    def test_unknown_arch_falls_back_to_generic(self):
        result = reconstruct_architecture(
            "Completely Novel Model",
            "A new approach.",
            [], [],
            _empty_kg(),
        )
        assert len(result["components"]) >= 2
        assert len(result["connections"]) >= 1

    def test_all_connection_types_are_valid(self):
        valid_types = {"sequential", "residual", "concat", "cross_attention"}
        result = reconstruct_architecture("T", "T", [], [], _make_kg("U-Net", "introduces"))
        for conn in result["connections"]:
            assert conn["connection_type"] in valid_types

    def test_all_component_types_are_valid(self):
        valid_types = {"conv","attention","embedding","pooling","mlp","normalization",
                       "activation","skip_connection","upsample","downsample","head"}
        result = reconstruct_architecture("T", "T", [], [], _make_kg("ViT", "introduces"))
        for comp in result["components"]:
            assert comp["type"] in valid_types
