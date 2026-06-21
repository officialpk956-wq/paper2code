"""
Tests for backend.services.knowledge_extraction_service (Phase 13B).

All tests are deterministic — no external calls, no LLMs, no file I/O.
"""

from __future__ import annotations

import pytest
from backend.services.knowledge_extraction_service import (
    build_knowledge_graph,
    extract_architectures,
    extract_concepts,
    extract_datasets,
    extract_metrics,
    _safe_id,
    _search_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(content: str, title: str = "Methods") -> dict:
    return {"id": "sec-1", "title": title, "content": content, "level": 1}


def _node_names(nodes: list) -> set:
    return {n["name"] for n in nodes}


def _edge_pairs(edges: list) -> list:
    return [(e["source"], e["target"], e["relation"]) for e in edges]


# ---------------------------------------------------------------------------
# Unit: _safe_id
# ---------------------------------------------------------------------------

class TestSafeId:
    def test_lowercases(self):
        assert _safe_id("ResNet") == "resnet"

    def test_replaces_spaces(self):
        assert _safe_id("Multi-Head Attention") == "multi-head-attention"

    def test_strips_leading_trailing_dash(self):
        result = _safe_id("  U-Net  ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_removes_special_chars(self):
        # + is not alphanumeric so it's replaced by -, then trailing - is stripped
        result = _safe_id("DeepLabV3+")
        assert "deeplabv3" in result


class TestSearchText:
    def test_matches_case_insensitive(self):
        assert _search_text(r"\btransformer\b", "The TRANSFORMER architecture") is True

    def test_does_not_match_absent(self):
        assert _search_text(r"\btransformer\b", "convolutional neural network") is False

    def test_respects_word_boundary(self):
        # "transformers" should match \btransformer\b? No — \b requires end of word
        assert _search_text(r"\btransformer\b", "transformers") is False


# ---------------------------------------------------------------------------
# extract_architectures
# ---------------------------------------------------------------------------

class TestExtractArchitectures:
    def test_detects_transformer_in_abstract(self):
        nodes = extract_architectures(
            title="A New Vision Model",
            abstract="We propose a Transformer-based approach to image classification.",
            sections=[],
        )
        assert any(n["name"] == "Transformer" for n in nodes)

    def test_sets_in_title_flag_for_title_match(self):
        nodes = extract_architectures(
            title="ResNet: Deep Residual Learning",
            abstract="",
            sections=[],
        )
        resnet_node = next(n for n in nodes if n["name"] == "ResNet")
        assert resnet_node["_in_title"] is True

    def test_sets_in_title_false_for_abstract_only(self):
        nodes = extract_architectures(
            title="An Efficient Model",
            abstract="We compare against ResNet baseline.",
            sections=[],
        )
        resnet_node = next(n for n in nodes if n["name"] == "ResNet")
        assert resnet_node["_in_title"] is False

    def test_deduplicates_same_architecture(self):
        nodes = extract_architectures(
            title="Transformer study",
            abstract="Transformer Transformer Transformer",
            sections=[_section("attention is all you need")],
        )
        names = [n["name"] for n in nodes]
        assert names.count("Transformer") == 1

    def test_assigns_href_for_known_slug(self):
        nodes = extract_architectures(
            title="BERT model",
            abstract="We fine-tune BERT on downstream tasks.",
            sections=[],
        )
        bert = next(n for n in nodes if n["name"] == "BERT")
        assert bert["href"] == "/architectures/bert"

    def test_href_none_for_unknown_slug(self):
        nodes = extract_architectures(
            title="YOLO detector",
            abstract="We use YOLO for real-time detection.",
            sections=[],
        )
        yolo = next(n for n in nodes if n["name"] == "YOLO")
        assert yolo["href"] is None

    def test_detects_architecture_in_sections(self):
        nodes = extract_architectures(
            title="Paper",
            abstract="",
            sections=[_section("We train a U-Net for segmentation.")],
        )
        assert any(n["name"] == "U-Net" for n in nodes)

    def test_empty_inputs_return_empty(self):
        nodes = extract_architectures(title="", abstract="", sections=[])
        assert nodes == []

    def test_node_type_is_architecture(self):
        nodes = extract_architectures(
            title="GPT study",
            abstract="We analyse GPT models.",
            sections=[],
        )
        assert all(n["type"] == "architecture" for n in nodes)


# ---------------------------------------------------------------------------
# extract_concepts
# ---------------------------------------------------------------------------

class TestExtractConcepts:
    def test_detects_self_attention(self):
        nodes = extract_concepts([_section("Using self-attention layers.")])
        assert any(n["name"] == "Self-Attention" for n in nodes)

    def test_detects_batch_normalization(self):
        nodes = extract_concepts([_section("Batch normalization is applied after each conv.")])
        assert any(n["name"] == "Batch Normalization" for n in nodes)

    def test_deduplicates(self):
        nodes = extract_concepts([_section("dropout dropout dropout")])
        names = [n["name"] for n in nodes]
        assert names.count("Dropout") == 1

    def test_empty_sections_return_empty(self):
        nodes = extract_concepts([])
        assert nodes == []

    def test_node_type_is_concept(self):
        nodes = extract_concepts([_section("layer normalization is used")])
        assert all(n["type"] == "concept" for n in nodes)

    def test_does_not_search_title_or_abstract(self):
        # Title and abstract are NOT passed to extract_concepts — only sections
        nodes = extract_concepts([_section("nothing related here")])
        concept_names = _node_names(nodes)
        assert "Self-Attention" not in concept_names


# ---------------------------------------------------------------------------
# extract_datasets
# ---------------------------------------------------------------------------

class TestExtractDatasets:
    def test_detects_imagenet(self):
        nodes = extract_datasets([_section("We evaluate on ImageNet classification.")])
        assert any(n["name"] == "ImageNet" for n in nodes)

    def test_detects_coco(self):
        nodes = extract_datasets([_section("Evaluation on COCO benchmark.")])
        assert any(n["name"] == "COCO" for n in nodes)

    def test_detects_squad(self):
        nodes = extract_datasets([_section("We test on SQuAD question answering.")])
        assert any(n["name"] == "SQuAD" for n in nodes)

    def test_deduplicates(self):
        nodes = extract_datasets([_section("ImageNet ImageNet ImageNet")])
        names = [n["name"] for n in nodes]
        assert names.count("ImageNet") == 1

    def test_empty_sections_return_empty(self):
        nodes = extract_datasets([])
        assert nodes == []

    def test_node_type_is_dataset(self):
        nodes = extract_datasets([_section("trained on MNIST")])
        assert all(n["type"] == "dataset" for n in nodes)


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_detects_accuracy(self):
        nodes = extract_metrics([_section("We achieve 95% accuracy on the test set.")])
        assert any(n["name"] == "Accuracy" for n in nodes)

    def test_detects_bleu(self):
        nodes = extract_metrics([_section("BLEU score is computed for translation.")])
        assert any(n["name"] == "BLEU" for n in nodes)

    def test_detects_fid(self):
        nodes = extract_metrics([_section("Images are evaluated using FID.")])
        assert any(n["name"] == "FID" for n in nodes)

    def test_deduplicates(self):
        nodes = extract_metrics([_section("accuracy accuracy accuracy")])
        names = [n["name"] for n in nodes]
        assert names.count("Accuracy") == 1

    def test_node_type_is_metric(self):
        nodes = extract_metrics([_section("we report precision and recall")])
        assert all(n["type"] == "metric" for n in nodes)


# ---------------------------------------------------------------------------
# build_knowledge_graph — structure
# ---------------------------------------------------------------------------

class TestBuildKnowledgeGraph:
    def test_always_includes_paper_node(self):
        kg = build_knowledge_graph(
            paper_title="My Paper",
            abstract="",
            sections=[],
            equations=[],
        )
        paper_nodes = [n for n in kg["nodes"] if n["type"] == "paper"]
        assert len(paper_nodes) == 1

    def test_paper_node_name_matches_title(self):
        kg = build_knowledge_graph(
            paper_title="Attention Is All You Need",
            abstract="",
            sections=[],
            equations=[],
        )
        paper_node = next(n for n in kg["nodes"] if n["type"] == "paper")
        assert paper_node["name"] == "Attention Is All You Need"

    def test_returns_nodes_and_edges_keys(self):
        kg = build_knowledge_graph("P", "", [], [])
        assert "nodes" in kg
        assert "edges" in kg

    def test_no_private_metadata_in_output_nodes(self):
        kg = build_knowledge_graph(
            paper_title="ResNet paper",
            abstract="We use ResNet.",
            sections=[],
            equations=[],
        )
        for node in kg["nodes"]:
            assert "_in_title" not in node

    def test_architecture_node_detected(self):
        kg = build_knowledge_graph(
            paper_title="ResNet study",
            abstract="ResNet is a deep residual network.",
            sections=[],
            equations=[],
        )
        arch_names = {n["name"] for n in kg["nodes"] if n["type"] == "architecture"}
        assert "ResNet" in arch_names

    def test_introduces_edge_for_title_architecture(self):
        kg = build_knowledge_graph(
            paper_title="ResNet: Deep Residual Learning",
            abstract="We propose ResNet.",
            sections=[],
            equations=[],
        )
        edges = _edge_pairs(kg["edges"])
        paper_id = next(n["id"] for n in kg["nodes"] if n["type"] == "paper")
        arch_id = next(n["id"] for n in kg["nodes"] if n["name"] == "ResNet")
        assert (paper_id, arch_id, "introduces") in edges

    def test_uses_edge_for_non_title_architecture(self):
        kg = build_knowledge_graph(
            paper_title="Efficient Image Classification",
            abstract="We compare against ResNet.",
            sections=[],
            equations=[],
        )
        edges = _edge_pairs(kg["edges"])
        paper_id = next(n["id"] for n in kg["nodes"] if n["type"] == "paper")
        arch_id = next(n["id"] for n in kg["nodes"] if n["name"] == "ResNet")
        assert (paper_id, arch_id, "uses") in edges

    def test_evaluates_on_edge_for_dataset(self):
        kg = build_knowledge_graph(
            paper_title="My Paper",
            abstract="",
            sections=[_section("We evaluate on ImageNet.")],
            equations=[],
        )
        edges = _edge_pairs(kg["edges"])
        paper_id = next(n["id"] for n in kg["nodes"] if n["type"] == "paper")
        ds_id = next(n["id"] for n in kg["nodes"] if n["name"] == "ImageNet")
        assert (paper_id, ds_id, "evaluates_on") in edges

    def test_reports_edge_for_metric(self):
        kg = build_knowledge_graph(
            paper_title="My Paper",
            abstract="",
            sections=[_section("We report top-1 accuracy.")],
            equations=[],
        )
        edges = _edge_pairs(kg["edges"])
        paper_id = next(n["id"] for n in kg["nodes"] if n["type"] == "paper")
        metric_id = next(n["id"] for n in kg["nodes"] if n["name"] == "Top-1 Accuracy")
        assert (paper_id, metric_id, "reports") in edges

    def test_derives_from_edge_for_equations(self):
        eqs = [{"id": "p1-eq1", "page": 1, "text": "y = Wx + b"}]
        kg = build_knowledge_graph(
            paper_title="My Paper",
            abstract="",
            sections=[],
            equations=eqs,
        )
        edges = _edge_pairs(kg["edges"])
        paper_id = next(n["id"] for n in kg["nodes"] if n["type"] == "paper")
        eq_id = next(n["id"] for n in kg["nodes"] if n["type"] == "equation")
        assert (paper_id, eq_id, "derives_from") in edges

    def test_arch_concept_edge_from_static_map(self):
        kg = build_knowledge_graph(
            paper_title="Transformer paper",
            abstract="We use the Transformer architecture.",
            sections=[_section("We apply multi-head attention and positional encoding.")],
            equations=[],
        )
        edges = _edge_pairs(kg["edges"])
        arch_id = next(n["id"] for n in kg["nodes"] if n["name"] == "Transformer")
        concept_id = next(
            (n["id"] for n in kg["nodes"] if n["name"] == "Multi-Head Attention"),
            None,
        )
        if concept_id:
            assert (arch_id, concept_id, "uses") in edges

    def test_arch_lineage_edge(self):
        # BERT derives_from Transformer
        kg = build_knowledge_graph(
            paper_title="BERT pretraining",
            abstract="BERT is a transformer-based model.",
            sections=[],
            equations=[],
        )
        edges = _edge_pairs(kg["edges"])
        bert_id = next((n["id"] for n in kg["nodes"] if n["name"] == "BERT"), None)
        transformer_id = next((n["id"] for n in kg["nodes"] if n["name"] == "Transformer"), None)
        if bert_id and transformer_id:
            assert (bert_id, transformer_id, "derives_from") in edges

    def test_equations_capped_at_10(self):
        eqs = [{"id": f"p1-eq{i}", "page": 1, "text": f"eq {i}"} for i in range(20)]
        kg = build_knowledge_graph(
            paper_title="Math paper",
            abstract="",
            sections=[],
            equations=eqs,
        )
        eq_nodes = [n for n in kg["nodes"] if n["type"] == "equation"]
        assert len(eq_nodes) <= 10

    def test_empty_inputs_graph(self):
        kg = build_knowledge_graph("", "", [], [])
        # Should have exactly one node (the paper node) and no edges
        assert len(kg["nodes"]) == 1
        assert kg["nodes"][0]["type"] == "paper"
        assert kg["edges"] == []

    def test_node_ids_are_unique(self):
        kg = build_knowledge_graph(
            paper_title="ResNet on ImageNet with Accuracy",
            abstract="ResNet evaluated on ImageNet. We report accuracy.",
            sections=[_section("Batch normalization. Residual connection. Accuracy on ImageNet.")],
            equations=[{"id": "p1-eq1", "page": 1, "text": "L = sum"}],
        )
        ids = [n["id"] for n in kg["nodes"]]
        assert len(ids) == len(set(ids))
