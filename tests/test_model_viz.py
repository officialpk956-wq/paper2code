"""Tests for the ONNX model-viz cost estimator, health endpoint, and parse
integration. The estimator + health tests need no `onnx`; the parse test skips
where the (heavy) onnx package isn't installed.
"""

import pytest

from backend.services.onnx_flops import estimate_node_cost

_MB = 1024 * 1024


# ── pure FLOPs estimator ────────────────────────────────────────────────────
def test_conv_flops_from_weight_dims():
    c = estimate_node_cost("Conv", [[1, 3, 224, 224]], [1, 64, 112, 112], [[64, 3, 7, 7]], {})
    assert c["flops"] == 2 * (64 * 112 * 112) * (3 * 7 * 7)
    assert c["severity"] in ("medium", "high", "critical")


def test_conv_falls_back_to_kernel_attr_without_weight():
    c = estimate_node_cost("Conv", [[1, 8, 32, 32]], [1, 16, 32, 32], [], {"kernel_shape": [3, 3]})
    assert c["flops"] == 2 * (16 * 32 * 32) * (8 * 3 * 3)


def test_gemm_and_matmul_contract_dims():
    g = estimate_node_cost("Gemm", [[1, 2048]], [1, 1000], [[2048, 1000]], {})
    assert g["flops"] == 2 * 1000 * 2048
    m = estimate_node_cost("MatMul", [[1, 512, 768]], [1, 512, 768], [[768, 768]], {})
    assert m["flops"] == 2 * (512 * 768) * 768


def test_elementwise_and_memory():
    r = estimate_node_cost("Relu", [[1, 64, 112, 112]], [1, 64, 112, 112], [], {})
    assert r["flops"] == 64 * 112 * 112
    assert abs(r["memory_mb"] - (64 * 112 * 112 * 4 / _MB)) < 1e-6


def test_pool_norm_and_unknown_ops():
    p = estimate_node_cost(
        "MaxPool", [[1, 64, 112, 112]], [1, 64, 56, 56], [], {"kernel_shape": [2, 2]}
    )
    assert p["flops"] == (64 * 56 * 56) * 4
    n = estimate_node_cost("BatchNormalization", [[1, 64, 56, 56]], [1, 64, 56, 56], [], {})
    assert n["flops"] == (64 * 56 * 56) * 5
    # a shape/reshape op has no FLOPs but still reports activation memory
    u = estimate_node_cost("Reshape", [[1, 1000]], [1, 1000], [], {})
    assert u["flops"] == 0 and u["memory_mb"] > 0


def test_severity_bands():
    low = estimate_node_cost("Relu", [[1, 10]], [1, 10], [], {})
    assert low["severity"] == "low"
    big = estimate_node_cost("Gemm", [[1, 40000]], [1, 40000], [[40000, 40000]], {})
    assert big["severity"] in ("high", "critical")


# ── health endpoint ─────────────────────────────────────────────────────────
def test_model_viz_health_reports_dependency_availability(client):
    r = client.get("/api/model/health")
    assert r.status_code == 200
    body = r.json()
    assert set(("onnx", "e2b", "onnx_parse_ready", "pytorch_parse_ready")) <= body.keys()
    assert isinstance(body["onnx"]["available"], bool)
    # the *_ready flags must mirror the dependency check
    assert body["onnx_parse_ready"] == body["onnx"]["available"]
    assert body["pytorch_parse_ready"] == body["e2b"]["available"]


# ── real parse integration (only where onnx is installed) ───────────────────
def test_parse_onnx_attaches_cost_to_nodes():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    W = helper.make_tensor("W", TensorProto.FLOAT, [4, 3], [0.0] * 12)
    B = helper.make_tensor("B", TensorProto.FLOAT, [3], [0.0] * 3)
    node = helper.make_node("Gemm", ["X", "W", "B"], ["Y"])
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    from backend.services.onnx_parser import parse_onnx

    res = parse_onnx(model.SerializeToString())
    n = res["nodes"][0]
    assert n["op_type"] == "Gemm"
    assert n["params"] == 4 * 3 + 3
    assert n["flops"] == 2 * 3 * 4  # 2 * N_out * K
    assert n["severity"] == "low"
    assert "memory_mb" in n
    assert res["meta"]["total_flops"] == n["flops"]
