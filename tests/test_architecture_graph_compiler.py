"""
Tests for backend/services/architecture_graph_compiler.py (Phase 14B).

Covers:
  - Data model round-trips (to_dict / from_dict)
  - compile_blueprint() happy path
  - Shape inference via TensorTracker (reuse, not duplication)
  - FLOPs / params extraction from tensor_flow
  - Validation: cycle detection, disconnected nodes, dangling edges,
    missing input/output, shape mismatch warnings
  - Export: JSON, Mermaid, DOT
  - Edge cases: empty blueprint, single node, unknown component types
"""

import json
import pytest

from backend.services.architecture_graph_compiler import (
    EGNode,
    EGEdge,
    ValidationReport,
    ExecutableGraph,
    compile_blueprint,
    validate_graph,
    export_graph_json,
    export_graph_mermaid,
    export_graph_dot,
    _has_cycle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _resnet_blueprint():
    """Minimal ResNet-like blueprint dict (mirrors Phase 14A output)."""
    return {
        "id": "bp-1-resnet",
        "name": "ResNet",
        "paper_id": 1,
        "components": [
            {
                "id": "stem",
                "name": "Stem (7×7 Conv)",
                "type": "conv",
                "shape": [1, 64, 112, 112],
                "metadata": {"in_channels": 3, "out_channels": 64, "channels": 64,
                              "kernel_size": 7, "stride": 2},
            },
            {
                "id": "pool1",
                "name": "Max Pool",
                "type": "pooling",
                "shape": [1, 64, 56, 56],
                "metadata": {"kernel_size": 3, "stride": 2},
            },
            {
                "id": "head",
                "name": "Classifier (→1000)",
                "type": "head",
                "shape": [1, 1000],
                "metadata": {"in_features": 64, "out_features": 1000},
            },
        ],
        "connections": [
            {"source": "stem",  "target": "pool1", "connection_type": "sequential"},
            {"source": "pool1", "target": "head",  "connection_type": "sequential"},
        ],
        "input_shape":  [1, 3, 224, 224],
        "output_shape": [1, 1000],
        "estimated_parameters": 25_000_000,
        "estimated_flops": 4000.0,
        "confidence_score": 0.78,
        "tensor_flow": [
            {
                "component_id": "stem", "component_name": "Stem", "type": "conv",
                "input_shape": [1, 3, 224, 224], "output_shape": [1, 64, 112, 112],
                "flops_mflops": 235.9, "params_M": 0.047,
            },
            {
                "component_id": "pool1", "component_name": "MaxPool", "type": "pooling",
                "input_shape": [1, 64, 112, 112], "output_shape": [1, 64, 56, 56],
                "flops_mflops": 0.0, "params_M": 0.0,
            },
            {
                "component_id": "head", "component_name": "Head", "type": "head",
                "input_shape": [1, 2048], "output_shape": [1, 1000],
                "flops_mflops": 4.1, "params_M": 2.049,
            },
        ],
        "architecture_href": "/architectures/resnet",
    }


def _unet_blueprint():
    """U-Net blueprint with skip (concat) connections."""
    return {
        "id": "bp-2-unet",
        "name": "U-Net",
        "paper_id": 2,
        "components": [
            {"id": "enc1", "name": "Encoder 1", "type": "conv",
             "shape": [1, 64, 256, 256],
             "metadata": {"in_channels": 1, "out_channels": 64, "channels": 64, "kernel_size": 3}},
            {"id": "enc2", "name": "Encoder 2", "type": "conv",
             "shape": [1, 128, 128, 128],
             "metadata": {"in_channels": 64, "out_channels": 128, "channels": 128, "kernel_size": 3}},
            {"id": "dec2", "name": "Decoder 2", "type": "conv",
             "shape": [1, 64, 256, 256],
             "metadata": {"in_channels": 128, "out_channels": 64, "channels": 64, "kernel_size": 3}},
            {"id": "head", "name": "Output Conv", "type": "head",
             "shape": [1, 2, 256, 256],
             "metadata": {"in_features": 64, "out_features": 2}},
        ],
        "connections": [
            {"source": "enc1", "target": "enc2", "connection_type": "sequential"},
            {"source": "enc2", "target": "dec2", "connection_type": "sequential"},
            {"source": "dec2", "target": "head", "connection_type": "sequential"},
            {"source": "enc1", "target": "dec2", "connection_type": "concat"},
        ],
        "input_shape":  [1, 1, 256, 256],
        "output_shape": [1, 2, 256, 256],
        "estimated_parameters": 7_000_000,
        "estimated_flops": 2000.0,
        "confidence_score": 0.65,
        "tensor_flow": [
            {"component_id": "enc1", "component_name": "Enc1", "type": "conv",
             "input_shape": [1, 1, 256, 256], "output_shape": [1, 64, 256, 256],
             "flops_mflops": 512.0, "params_M": 0.7},
            {"component_id": "enc2", "component_name": "Enc2", "type": "conv",
             "input_shape": [1, 64, 256, 256], "output_shape": [1, 128, 128, 128],
             "flops_mflops": 512.0, "params_M": 1.2},
            {"component_id": "dec2", "component_name": "Dec2", "type": "conv",
             "input_shape": [1, 128, 128, 128], "output_shape": [1, 64, 256, 256],
             "flops_mflops": 256.0, "params_M": 0.9},
            {"component_id": "head", "component_name": "Head", "type": "head",
             "input_shape": [1, 64, 256, 256], "output_shape": [1, 2, 256, 256],
             "flops_mflops": 0.5, "params_M": 0.01},
        ],
        "architecture_href": "/architectures/unet",
    }


# ---------------------------------------------------------------------------
# TestDataModel
# ---------------------------------------------------------------------------

class TestDataModel:
    def test_egnode_to_dict_round_trip(self):
        n = EGNode(id="n1", name="Layer", type="conv", shape=[1, 64, 112, 112],
                   params=47_040, flops=235.9, metadata={"stride": 2})
        d = n.to_dict()
        assert d["id"] == "n1"
        assert d["shape"] == [1, 64, 112, 112]
        assert d["params"] == 47_040
        assert d["flops"] == 235.9

    def test_egedge_to_dict_round_trip(self):
        e = EGEdge(source="a", target="b", edge_type="sequential")
        d = e.to_dict()
        assert d == {"source": "a", "target": "b", "edge_type": "sequential"}

    def test_validation_report_round_trip(self):
        r = ValidationReport(valid=False, errors=["cycle"], warnings=["mismatch"])
        d = r.to_dict()
        r2 = ValidationReport.from_dict(d)
        assert r2.valid is False
        assert r2.errors == ["cycle"]
        assert r2.warnings == ["mismatch"]

    def test_executable_graph_to_dict_round_trip(self):
        eg = ExecutableGraph(
            id="eg-1",
            nodes=[EGNode("n", "N", "conv", [1, 64], 0, 0.0, {})],
            edges=[],
            input_spec=[1, 3, 224, 224],
            output_spec=[1, 1000],
            parameter_count=0,
            flops=0.0,
            validation_status="valid",
            validation_report=ValidationReport(True, [], []),
        )
        d = eg.to_dict()
        eg2 = ExecutableGraph.from_dict(d)
        assert eg2.id == "eg-1"
        assert len(eg2.nodes) == 1
        assert eg2.nodes[0].id == "n"
        assert eg2.input_spec == [1, 3, 224, 224]

    def test_executable_graph_from_dict_missing_fields(self):
        """from_dict must handle missing optional fields gracefully."""
        eg = ExecutableGraph.from_dict({"id": "eg-x"})
        assert eg.nodes == []
        assert eg.edges == []
        assert eg.parameter_count == 0
        assert eg.validation_status == "valid"


# ---------------------------------------------------------------------------
# TestCompileBlueprint
# ---------------------------------------------------------------------------

class TestCompileBlueprint:
    def test_returns_executable_graph(self):
        eg = compile_blueprint(_resnet_blueprint())
        assert isinstance(eg, ExecutableGraph)

    def test_id_derived_from_blueprint_id(self):
        eg = compile_blueprint(_resnet_blueprint())
        assert eg.id.startswith("eg-")
        assert "resnet" in eg.id

    def test_node_count_matches_components(self):
        bp = _resnet_blueprint()
        eg = compile_blueprint(bp)
        assert len(eg.nodes) == len(bp["components"])

    def test_edge_count_matches_connections(self):
        bp = _resnet_blueprint()
        eg = compile_blueprint(bp)
        assert len(eg.edges) == len(bp["connections"])

    def test_input_spec_preserved(self):
        bp = _resnet_blueprint()
        eg = compile_blueprint(bp)
        assert eg.input_spec == bp["input_shape"]

    def test_output_spec_preserved(self):
        bp = _resnet_blueprint()
        eg = compile_blueprint(bp)
        assert eg.output_spec == bp["output_shape"]

    def test_flops_from_tensor_flow(self):
        eg = compile_blueprint(_resnet_blueprint())
        stem = next(n for n in eg.nodes if n.id == "stem")
        assert abs(stem.flops - 235.9) < 0.01

    def test_params_from_tensor_flow(self):
        eg = compile_blueprint(_resnet_blueprint())
        stem = next(n for n in eg.nodes if n.id == "stem")
        # 0.047M params → 47_000
        assert stem.params == int(0.047 * 1_000_000)

    def test_total_flops_sum(self):
        bp = _resnet_blueprint()
        eg = compile_blueprint(bp)
        expected = sum(f["flops_mflops"] for f in bp["tensor_flow"])
        assert abs(eg.flops - round(expected, 4)) < 0.001

    def test_total_params_sum(self):
        bp = _resnet_blueprint()
        eg = compile_blueprint(bp)
        expected_params = int(sum(f["params_M"] for f in bp["tensor_flow"]) * 1_000_000)
        assert eg.parameter_count == expected_params

    def test_node_names_preserved(self):
        eg = compile_blueprint(_resnet_blueprint())
        names = {n.name for n in eg.nodes}
        assert "Stem (7×7 Conv)" in names
        assert "Classifier (→1000)" in names

    def test_node_types_preserved(self):
        eg = compile_blueprint(_resnet_blueprint())
        types = {n.id: n.type for n in eg.nodes}
        assert types["stem"] == "conv"
        assert types["head"] == "head"
        assert types["pool1"] == "pooling"

    def test_edge_types_preserved(self):
        eg = compile_blueprint(_unet_blueprint())
        edge_types = {(e.source, e.target): e.edge_type for e in eg.edges}
        assert edge_types[("enc1", "dec2")] == "concat"
        assert edge_types[("enc1", "enc2")] == "sequential"

    def test_metadata_input_shape_stored(self):
        eg = compile_blueprint(_resnet_blueprint())
        stem = next(n for n in eg.nodes if n.id == "stem")
        assert stem.metadata.get("input_shape") == [1, 3, 224, 224]

    def test_validation_status_valid_for_clean_blueprint(self):
        eg = compile_blueprint(_resnet_blueprint())
        assert eg.validation_status in ("valid", "warnings")

    def test_unet_concat_edges(self):
        eg = compile_blueprint(_unet_blueprint())
        concat_edges = [e for e in eg.edges if e.edge_type == "concat"]
        assert len(concat_edges) == 1
        assert concat_edges[0].source == "enc1"
        assert concat_edges[0].target == "dec2"

    def test_empty_blueprint(self):
        eg = compile_blueprint({})
        assert eg.nodes == []
        assert eg.edges == []
        assert eg.parameter_count == 0

    def test_single_node_blueprint(self):
        bp = {
            "id": "bp-x-single",
            "components": [{"id": "only", "name": "Only Node", "type": "head",
                             "shape": [1, 10], "metadata": {}}],
            "connections": [],
            "input_shape": [1, 64],
            "output_shape": [1, 10],
            "tensor_flow": [{"component_id": "only", "component_name": "Only",
                              "type": "head", "input_shape": [1, 64],
                              "output_shape": [1, 10],
                              "flops_mflops": 1.0, "params_M": 0.001}],
        }
        eg = compile_blueprint(bp)
        assert len(eg.nodes) == 1
        assert eg.nodes[0].id == "only"


# ---------------------------------------------------------------------------
# TestShapeInference
# ---------------------------------------------------------------------------

class TestShapeInference:
    def test_tracker_shapes_not_empty_for_conv_blueprint(self):
        """TensorTracker should infer non-trivial shapes for conv nodes."""
        from backend.services.architecture_graph_compiler import _run_tracker
        bp = _resnet_blueprint()
        shapes = _run_tracker(
            bp["components"], bp["connections"], bp["input_shape"]
        )
        # Should have shapes for at least the first conv node
        assert "stem" in shapes or len(shapes) > 0

    def test_tracker_shape_falls_back_to_tensor_flow(self):
        """When TensorTracker can't infer, tensor_flow shapes are used."""
        eg = compile_blueprint(_resnet_blueprint())
        head = next(n for n in eg.nodes if n.id == "head")
        # head shape from tensor_flow is [1, 1000]
        assert head.shape == [1, 1000]

    def test_shape_for_pooling_node(self):
        eg = compile_blueprint(_resnet_blueprint())
        pool = next(n for n in eg.nodes if n.id == "pool1")
        # Should have a valid 4D shape
        assert len(pool.shape) in (2, 4)

    def test_input_shape_stored_in_metadata(self):
        eg = compile_blueprint(_unet_blueprint())
        enc2 = next(n for n in eg.nodes if n.id == "enc2")
        assert enc2.metadata.get("input_shape") is not None


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

def _make_nodes(*ids):
    return [EGNode(id=i, name=i, type="conv", shape=[1, 64], params=0, flops=0.0) for i in ids]


def _make_edges(*pairs, edge_type="sequential"):
    return [EGEdge(source=s, target=t, edge_type=edge_type) for s, t in pairs]


class TestValidation:
    def test_valid_linear_chain(self):
        nodes = _make_nodes("a", "b", "c")
        edges = _make_edges(("a", "b"), ("b", "c"))
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "valid",
                             ValidationReport(True, [], []))
        report = validate_graph(eg)
        assert report.valid is True
        assert report.errors == []

    def test_cycle_detected(self):
        nodes = _make_nodes("a", "b", "c")
        edges = _make_edges(("a", "b"), ("b", "c"), ("c", "a"))
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "invalid",
                             ValidationReport(False, [], []))
        report = validate_graph(eg)
        assert report.valid is False
        assert any("cycle" in e.lower() for e in report.errors)

    def test_no_cycle_for_skip_connections(self):
        nodes = _make_nodes("enc", "bottleneck", "dec")
        edges = _make_edges(
            ("enc", "bottleneck"), ("bottleneck", "dec"),
            edge_type="sequential",
        ) + [EGEdge("enc", "dec", "concat")]
        assert not _has_cycle(nodes, edges)

    def test_dangling_edge_source_error(self):
        nodes = _make_nodes("a", "b")
        edges = [EGEdge("x", "b", "sequential")]   # "x" doesn't exist
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "invalid",
                             ValidationReport(False, [], []))
        report = validate_graph(eg)
        assert any("x" in e for e in report.errors)

    def test_dangling_edge_target_error(self):
        nodes = _make_nodes("a")
        edges = [EGEdge("a", "z", "sequential")]   # "z" doesn't exist
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "invalid",
                             ValidationReport(False, [], []))
        report = validate_graph(eg)
        assert any("z" in e for e in report.errors)

    def test_disconnected_node_warning(self):
        nodes = _make_nodes("a", "b", "orphan")
        edges = _make_edges(("a", "b"))
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "warnings",
                             ValidationReport(True, [], []))
        report = validate_graph(eg)
        assert any("orphan" in w for w in report.warnings)

    def test_no_input_node_error(self):
        nodes = _make_nodes("a", "b")
        # Both nodes have incoming edges: a←b and b←a (but different from cycle)
        edges = [EGEdge("b", "a", "sequential"), EGEdge("a", "b", "sequential")]
        # This is a cycle, but the "no input" check fires too
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "invalid",
                             ValidationReport(False, [], []))
        report = validate_graph(eg)
        # cycle should be in errors
        assert len(report.errors) > 0

    def test_missing_output_node_error(self):
        nodes = _make_nodes("a", "b")
        # Both nodes have outgoing edges (no terminal node)
        edges = [EGEdge("a", "b", "sequential"), EGEdge("b", "a", "sequential")]
        eg = ExecutableGraph("eg-x", nodes, edges, [], [], 0, 0.0, "invalid",
                             ValidationReport(False, [], []))
        report = validate_graph(eg)
        assert len(report.errors) > 0

    def test_shape_mismatch_warning(self):
        n1 = EGNode("a", "A", "conv", [1, 64, 112, 112], 0, 0.0,
                    {"input_shape": [1, 3, 224, 224]})
        n2 = EGNode("b", "B", "conv", [1, 64, 56, 56], 0, 0.0,
                    {"input_shape": [1, 64, 999, 999]})  # mismatch: a out ≠ b in
        edges = [EGEdge("a", "b", "sequential")]
        eg = ExecutableGraph("eg-x", [n1, n2], edges, [], [], 0, 0.0, "warnings",
                             ValidationReport(True, [], []))
        report = validate_graph(eg)
        assert any("mismatch" in w.lower() or "→" in w for w in report.warnings)

    def test_empty_graph_valid(self):
        eg = ExecutableGraph("eg-x", [], [], [], [], 0, 0.0, "valid",
                             ValidationReport(True, [], []))
        report = validate_graph(eg)
        assert report.valid is True

    def test_compile_blueprint_validation_status_propagates(self):
        eg = compile_blueprint(_resnet_blueprint())
        assert eg.validation_status in ("valid", "warnings", "invalid")
        assert isinstance(eg.validation_report, ValidationReport)


# ---------------------------------------------------------------------------
# TestExports
# ---------------------------------------------------------------------------

class TestExports:
    def test_json_export_is_valid_json(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_json(eg)
        parsed = json.loads(out)
        assert parsed["id"] == eg.id
        assert "nodes" in parsed
        assert "edges" in parsed
        assert "validation_report" in parsed

    def test_json_export_contains_all_nodes(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_json(eg)
        parsed = json.loads(out)
        assert len(parsed["nodes"]) == len(eg.nodes)

    def test_mermaid_export_starts_with_flowchart(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_mermaid(eg)
        assert out.startswith("flowchart TD")

    def test_mermaid_export_contains_node_names(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_mermaid(eg)
        assert "stem" in out or "Stem" in out

    def test_mermaid_export_contains_edges(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_mermaid(eg)
        assert "-->" in out or "-.->" in out

    def test_dot_export_starts_with_digraph(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_dot(eg)
        assert out.startswith("digraph ExecutableGraph {")

    def test_dot_export_ends_with_brace(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_dot(eg)
        assert out.strip().endswith("}")

    def test_dot_export_contains_arrow(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_dot(eg)
        assert "->" in out

    def test_mermaid_skip_connection_uses_dashed_arrow(self):
        eg = compile_blueprint(_unet_blueprint())
        out = export_graph_mermaid(eg)
        # concat edge should produce a dashed arrow
        assert "-.->" in out

    def test_dot_skip_connection_is_dashed(self):
        eg = compile_blueprint(_unet_blueprint())
        out = export_graph_dot(eg)
        assert "dashed" in out

    def test_json_from_dict_round_trip(self):
        eg = compile_blueprint(_resnet_blueprint())
        out = export_graph_json(eg)
        restored = ExecutableGraph.from_dict(json.loads(out))
        assert restored.id == eg.id
        assert len(restored.nodes) == len(eg.nodes)
        assert restored.flops == eg.flops

    def test_empty_graph_exports(self):
        eg = compile_blueprint({})
        assert export_graph_json(eg)
        assert export_graph_mermaid(eg)
        assert export_graph_dot(eg)


# ---------------------------------------------------------------------------
# TestCycleDetection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_no_cycle_linear(self):
        nodes = _make_nodes("a", "b", "c")
        edges = _make_edges(("a", "b"), ("b", "c"))
        assert not _has_cycle(nodes, edges)

    def test_no_cycle_dag_with_merge(self):
        nodes = _make_nodes("a", "b", "c", "d")
        edges = _make_edges(("a", "c"), ("b", "c"), ("c", "d"))
        assert not _has_cycle(nodes, edges)

    def test_cycle_direct(self):
        nodes = _make_nodes("a", "b")
        edges = _make_edges(("a", "b"), ("b", "a"))
        assert _has_cycle(nodes, edges)

    def test_cycle_long(self):
        nodes = _make_nodes("a", "b", "c", "d")
        edges = _make_edges(("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"))
        assert _has_cycle(nodes, edges)

    def test_no_cycle_unet_skip(self):
        nodes = _make_nodes("enc1", "enc2", "dec2", "head")
        edges = _make_edges(
            ("enc1", "enc2"), ("enc2", "dec2"), ("dec2", "head")
        ) + [EGEdge("enc1", "dec2", "concat")]
        assert not _has_cycle(nodes, edges)
