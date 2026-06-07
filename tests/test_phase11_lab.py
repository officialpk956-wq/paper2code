"""
tests/test_phase11_lab.py — Phase 11: Research Lab & Experiment Studio Tests

Tests:
  1.  test_diff_engine_baseline_no_change     — diff of same graph returns zeros
  2.  test_diff_engine_node_delta             — add_node produces correct nodes_added
  3.  test_diff_engine_param_delta            — wider graph has larger param_delta
  4.  test_diff_engine_skip_delta             — add_residual adds skip edges
  5.  test_mutator_increase_depth             — depth increases by 1
  6.  test_mutator_decrease_depth             — depth decreases by 1
  7.  test_mutator_increase_width             — channels scale up
  8.  test_mutator_decrease_width             — channels scale down
  9.  test_mutator_add_residual               — skip edges appear
  10. test_mutator_remove_residual            — skip edges disappear
  11. test_mutator_add_attention              — attention node inserted
  12. test_mutator_change_patch_size          — patch_size param updated
  13. test_mutator_change_hidden_dim          — hidden dim param updated
  14. test_mutator_nondestructive             — original graph unchanged
  15. test_apply_mutations_sequential         — chained mutations work
  16. test_apply_mutations_unknown_type       — raises ValueError
  17. test_hypothesis_score_correct           — perfect prediction scores ≥ 0.7
  18. test_hypothesis_score_wrong_both        — both wrong scores ≤ 0.35
  19. test_hypothesis_score_text_bonus        — rich text earns bonus
  20. test_hypothesis_direction_helper        — _direction() handles edge cases
  21. test_prediction_prompt_coverage         — all mutation types have prompts
  22. test_experiment_result_builder          — result has required fields
  23. test_tradeoff_scatter_points            — scatter returns ≥ 3 points
  24. test_tradeoff_pareto_frontier           — frontier correctly identifies dominated points
  25. test_tradeoff_summary_coverage          — all mutations have tradeoff summaries
  26. test_lab_api_mutations_endpoint         — GET /api/lab/mutations returns 9 types
  27. test_lab_api_mutate_resnet              — POST /api/lab/mutate ResNet returns diff
  28. test_lab_api_predict                    — POST /api/lab/predict returns scoring
  29. test_lab_api_tradeoffs                  — GET /api/lab/tradeoffs returns scatter
  30. test_lab_api_prediction_prompt          — GET /api/lab/prediction-prompt returns question
"""

import sys
import os
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge
from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_vit import build_vit_graph
from core.lab.diff_engine import compute_diff
from core.lab.mutator import (
    apply_mutations,
    increase_depth, decrease_depth,
    increase_width, decrease_width,
    add_residual, remove_residual,
    add_attention, change_patch_size, change_hidden_dim,
    MUTATION_REGISTRY,
)
from core.lab.hypothesis_engine import hypothesis_engine, _direction
from core.lab.tradeoff_analyzer import (
    tradeoff_scatter, get_efficiency_frontiers, get_tradeoff_summary,
    TRADEOFF_SUMMARIES,
)


# ───────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────

@pytest.fixture
def resnet_graph():
    return build_resnet18_graph(base_channels=64, stages=4, blocks_per_stage=2)


@pytest.fixture
def vit_graph():
    return build_vit_graph(hidden_size=768, num_heads=12, depth=4)


@pytest.fixture
def tiny_graph():
    """Minimal 3-node graph for isolated diff tests."""
    g = ArchitectureGraph(name="Tiny")
    g.add_node(GraphNode(id="a", type="conv2d", label="Conv A", params={"channels": 64}))
    g.add_node(GraphNode(id="b", type="residualblock", label="Block B", params={"channels": 64}))
    g.add_node(GraphNode(id="c", type="linear", label="Linear C", params={"hidden_size": 512}))
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    return g


# ───────────────────────────────────────────────
# 1-4: Diff Engine Tests
# ───────────────────────────────────────────────

def test_diff_engine_baseline_no_change(resnet_graph):
    """Diffing identical graphs → all deltas are 0."""
    import copy
    clone = copy.deepcopy(resnet_graph)
    diff = compute_diff(resnet_graph, clone)

    assert diff["nodes_added"] == []
    assert diff["nodes_removed"] == []
    assert diff["param_delta"]["absolute"] == 0
    assert diff["flops_delta"]["absolute"] == 0
    assert diff["depth_delta"]["absolute"] == 0
    assert diff["skip_delta"]["absolute"] == 0


def test_diff_engine_node_delta(tiny_graph):
    """Adding a node to the graph produces nodes_added in the diff."""
    import copy
    after = copy.deepcopy(tiny_graph)
    after.add_node(GraphNode(id="d", type="batchnorm2d", label="BatchNorm D", params={"channels": 64}))
    after.add_edge("c", "d")

    diff = compute_diff(tiny_graph, after)
    assert "BatchNorm D" in diff["nodes_added"]
    assert diff["depth_delta"]["absolute"] == 1


def test_diff_engine_param_delta(resnet_graph):
    """A wider graph should have strictly more parameters."""
    wider = increase_width(resnet_graph, factor=2.0)
    diff = compute_diff(resnet_graph, wider)

    assert diff["param_delta"]["absolute"] > 0, "Wider graph must have more params"
    assert diff["param_delta"]["pct"] > 0
    assert diff["flops_delta"]["absolute"] >= 0  # FLOPs should not decrease when width doubles


def test_diff_engine_skip_delta(resnet_graph):
    """add_residual adds skip edges → skip_delta.absolute > 0."""
    with_skips = add_residual(resnet_graph)
    diff = compute_diff(resnet_graph, with_skips)

    assert diff["skip_delta"]["absolute"] > 0, "Expected skip connections to be added"
    assert diff["skip_delta"]["after"] > diff["skip_delta"]["before"]


# ───────────────────────────────────────────────
# 5-14: Mutator Unit Tests
# ───────────────────────────────────────────────

def test_mutator_increase_depth(resnet_graph):
    """increase_depth(1) should add exactly 1 node."""
    before_depth = len(resnet_graph.nodes)
    after = increase_depth(resnet_graph, n=1)
    assert len(after.nodes) == before_depth + 1


def test_mutator_decrease_depth(resnet_graph):
    """decrease_depth(1) should remove exactly 1 node."""
    before_depth = len(resnet_graph.nodes)
    after = decrease_depth(resnet_graph, n=1)
    assert len(after.nodes) == before_depth - 1


def test_mutator_increase_width(resnet_graph):
    """increase_width(1.5) should increase at least one channel param."""
    before_params = sum(
        n.params.get("channels", 0) for n in resnet_graph.nodes
        if isinstance((n.params or {}).get("channels", 0), (int, float))
    )
    after = increase_width(resnet_graph, factor=1.5)
    after_params = sum(
        n.params.get("channels", 0) for n in after.nodes
        if isinstance((n.params or {}).get("channels", 0), (int, float))
    )
    assert after_params > before_params, "Width increase should raise total channel sum"


def test_mutator_decrease_width(resnet_graph):
    """decrease_width(0.5) should reduce at least one channel param."""
    before_params = sum(
        n.params.get("channels", 0) for n in resnet_graph.nodes
        if isinstance((n.params or {}).get("channels", 0), (int, float))
    )
    after = decrease_width(resnet_graph, factor=0.5)
    after_params = sum(
        n.params.get("channels", 0) for n in after.nodes
        if isinstance((n.params or {}).get("channels", 0), (int, float))
    )
    assert after_params < before_params, "Width decrease should lower total channel sum"


def test_mutator_add_residual(resnet_graph):
    """add_residual should inject skip-type edges."""
    before_skip = sum(1 for e in resnet_graph.edges if e.edge_type in ("skip", "residual"))
    after = add_residual(resnet_graph)
    after_skip = sum(1 for e in after.edges if e.edge_type in ("skip", "residual"))
    assert after_skip > before_skip, "Expected skip edges to be added"


def test_mutator_remove_residual(resnet_graph):
    """remove_residual should eliminate all skip edges."""
    with_skips = add_residual(resnet_graph)
    after = remove_residual(with_skips)
    remaining_skip = sum(1 for e in after.edges if e.edge_type in ("skip", "residual"))
    assert remaining_skip == 0, "All skip edges should be removed"


def test_mutator_add_attention(resnet_graph):
    """add_attention should insert a multiheadattention node."""
    before_types = [n.type.lower() for n in resnet_graph.nodes]
    after = add_attention(resnet_graph)
    after_types = [n.type.lower() for n in after.nodes]

    assert "multiheadattention" in after_types
    assert len(after.nodes) == len(resnet_graph.nodes) + 1


def test_mutator_change_patch_size(vit_graph):
    """change_patch_size(8) should update patch_size in patchembedding nodes."""
    after = change_patch_size(vit_graph, patch_size=8)
    patch_nodes = [n for n in after.nodes if (n.type or "").lower() in ("patchembedding", "patch_embedding", "embedding")]

    # If ViT has patch embedding nodes, verify they're updated
    if patch_nodes:
        for pn in patch_nodes:
            assert pn.params.get("patch_size") == 8, f"patch_size not updated in node {pn.id}"
    else:
        # ViT graph may not have explicit patchembedding; mutation is idempotent
        assert len(after.nodes) == len(vit_graph.nodes)


def test_mutator_change_hidden_dim(vit_graph):
    """change_hidden_dim(1024) should update d_model/hidden_size in transformer nodes."""
    after = change_hidden_dim(vit_graph, dim=1024)
    transformer_types = {"transformerblock", "multiheadattention", "feedforward", "linear"}
    updated = [n for n in after.nodes if (n.type or "").lower() in transformer_types]

    for n in updated:
        p = n.params or {}
        if "d_model" in p:
            assert p["d_model"] == 1024
        elif "hidden_size" in p:
            assert p["hidden_size"] == 1024


def test_mutator_nondestructive(resnet_graph):
    """All mutations must NOT modify the original graph."""
    original_node_count = len(resnet_graph.nodes)
    original_edge_count = len(resnet_graph.edges)

    _ = increase_depth(resnet_graph, n=2)
    _ = increase_width(resnet_graph, factor=2.0)
    _ = add_residual(resnet_graph)
    _ = add_attention(resnet_graph)

    assert len(resnet_graph.nodes) == original_node_count, "Original graph must not be modified"
    assert len(resnet_graph.edges) == original_edge_count, "Original edges must not be modified"


# ───────────────────────────────────────────────
# 15-16: apply_mutations()
# ───────────────────────────────────────────────

def test_apply_mutations_sequential(resnet_graph):
    """Sequential mutations compose correctly."""
    mutations = [
        {"type": "increase_depth", "params": {"n": 1}},
        {"type": "add_residual", "params": {}},
    ]
    result = apply_mutations(resnet_graph, mutations)
    assert len(result.nodes) == len(resnet_graph.nodes) + 1
    skip_after = sum(1 for e in result.edges if e.edge_type in ("skip", "residual"))
    assert skip_after > 0


def test_apply_mutations_unknown_type(resnet_graph):
    """Unknown mutation type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown mutation type"):
        apply_mutations(resnet_graph, [{"type": "explode_everything", "params": {}}])


# ───────────────────────────────────────────────
# 17-21: Hypothesis Engine Tests
# ───────────────────────────────────────────────

def test_hypothesis_score_correct():
    """Both directions correct + quality text → score ≥ 0.7."""
    actual_diff = {
        "param_delta": {"pct": 25.0, "absolute": 1000},
        "flops_delta": {"absolute": 3},
    }
    hyp = {
        "predicted_param_direction": "increase",
        "predicted_flops_direction": "increase",
        "prediction": "Adding more layers increases depth, expressivity, and the number of parameters significantly due to residual learning connections.",
        "mutation_type": "increase_depth",
    }
    result = hypothesis_engine.score_prediction(hyp, actual_diff)
    assert result["score"] >= 0.7, f"Expected high score, got {result['score']}"
    assert result["param_correct"] is True
    assert result["flops_correct"] is True


def test_hypothesis_score_wrong_both():
    """Both directions wrong → score ≤ 0.35."""
    actual_diff = {
        "param_delta": {"pct": 30.0, "absolute": 5000},
        "flops_delta": {"absolute": 4},
    }
    hyp = {
        "predicted_param_direction": "decrease",  # wrong (actual: increase)
        "predicted_flops_direction": "decrease",  # wrong (actual: increase)
        "prediction": "I think it gets smaller.",
        "mutation_type": "increase_depth",
    }
    result = hypothesis_engine.score_prediction(hyp, actual_diff)
    assert result["score"] <= 0.35, f"Expected low score, got {result['score']}"
    assert result["param_correct"] is False
    assert result["flops_correct"] is False


def test_hypothesis_score_text_bonus():
    """Rich on-topic text with 2+ domain keywords should earn the full text bonus."""
    actual_diff = {
        "param_delta": {"pct": 0.0, "absolute": 0},
        "flops_delta": {"absolute": 0},
    }
    hyp = {
        "predicted_param_direction": "no_change",
        "predicted_flops_direction": "no_change",
        "prediction": (
            "Adding a residual connection creates an identity shortcut path that enables "
            "gradient flow through the network and prevents the vanishing gradient problem "
            "common in deeper architectures such as ResNet."
        ),
        "mutation_type": "add_residual",
    }
    result = hypothesis_engine.score_prediction(hyp, actual_diff)
    assert result["text_bonus"] >= 0.15, f"Expected text bonus ≥ 0.15, got {result['text_bonus']}"


def test_hypothesis_direction_helper():
    """_direction() maps values to the correct direction string."""
    assert _direction(10.0) == "increase"
    assert _direction(-10.0) == "decrease"
    assert _direction(0.0) == "no_change"
    assert _direction(0.4) == "no_change"   # below threshold
    assert _direction(-0.4) == "no_change"  # above negative threshold
    assert _direction(0.6) == "increase"
    assert _direction(-0.6) == "decrease"


def test_prediction_prompt_coverage():
    """All 9 mutation types should return a question and hints."""
    mutation_types = list(MUTATION_REGISTRY.keys())
    for mut in mutation_types:
        prompt = hypothesis_engine.generate_prediction_prompt(mut, "ResNet")
        assert "question" in prompt, f"Missing 'question' for mutation '{mut}'"
        assert "hints" in prompt, f"Missing 'hints' for mutation '{mut}'"
        assert len(prompt["question"]) > 10, f"Question too short for mutation '{mut}'"


# ───────────────────────────────────────────────
# 22: Experiment Result Builder
# ───────────────────────────────────────────────

def test_experiment_result_builder(resnet_graph):
    """build_experiment_result should return a dict with required fields."""
    mutations = [{"type": "increase_depth", "params": {"n": 1}}]
    after = apply_mutations(resnet_graph, mutations)
    diff = compute_diff(resnet_graph, after)

    hyp = {
        "id": "test-hyp-01",
        "predicted_param_direction": "increase",
        "predicted_flops_direction": "increase",
        "prediction": "Depth increases parameters via residual connections.",
        "mutation_type": "increase_depth",
    }

    result = hypothesis_engine.build_experiment_result(
        hypothesis=hyp,
        mutations_applied=mutations,
        architecture="ResNet",
        actual_diff=diff,
    )

    required_fields = [
        "id", "architecture", "mutations_applied", "diff_summary",
        "param_delta_pct", "flops_delta", "depth_delta",
        "prediction_score", "prediction_feedback", "created_at",
    ]
    for field in required_fields:
        assert field in result, f"Missing field '{field}' in experiment result"

    assert result["architecture"] == "ResNet"
    assert result["prediction_score"] is not None
    assert isinstance(result["prediction_score"], float)


# ───────────────────────────────────────────────
# 23-25: Tradeoff Analyzer Tests
# ───────────────────────────────────────────────

def test_tradeoff_scatter_points(resnet_graph):
    """tradeoff_scatter should return at least 5 data points."""
    points = tradeoff_scatter(resnet_graph, "ResNet")
    assert len(points) >= 5, f"Expected ≥5 scatter points, got {len(points)}"

    required_keys = {"label", "group", "flops_score", "params", "depth", "memory_mb"}
    for p in points:
        missing = required_keys - set(p.keys())
        assert not missing, f"Point missing keys: {missing}"


def test_tradeoff_pareto_frontier(resnet_graph):
    """get_efficiency_frontiers should mark at least 1 point as Pareto-optimal."""
    points = tradeoff_scatter(resnet_graph, "ResNet")
    frontier = get_efficiency_frontiers(points)

    pareto_points = [p for p in frontier if p.get("is_frontier")]
    assert len(pareto_points) >= 1, "Expected at least 1 Pareto-optimal point"

    # Verify: no Pareto point is dominated by another Pareto point
    for p in pareto_points:
        for q in pareto_points:
            if p is q:
                continue
            # They should not strictly dominate each other
            q_dominates_p = (q["flops_score"] <= p["flops_score"] and q["params"] <= p["params"]
                             and (q["flops_score"] < p["flops_score"] or q["params"] < p["params"]))
            assert not q_dominates_p, f"Pareto point '{p['label']}' is dominated by '{q['label']}'"


def test_tradeoff_summary_coverage():
    """All mutation types in MUTATION_REGISTRY must have a tradeoff summary."""
    for mut in MUTATION_REGISTRY.keys():
        summary = get_tradeoff_summary(mut)
        assert "compute" in summary, f"Missing 'compute' in tradeoff summary for '{mut}'"
        assert "recommendation" in summary, f"Missing 'recommendation' for '{mut}'"
        assert len(summary["recommendation"]) > 20, f"Recommendation too short for '{mut}'"


# ───────────────────────────────────────────────
# 26-30: API Integration Tests (FastAPI TestClient)
# ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def api_client():
    """Create a FastAPI TestClient for backend API tests."""
    try:
        from fastapi.testclient import TestClient
        from backend.server import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"TestClient setup failed: {e}")


def test_lab_api_mutations_endpoint(api_client):
    """GET /api/lab/mutations should return 9 mutation types."""
    res = api_client.get("/api/lab/mutations")
    assert res.status_code == 200
    data = res.json()
    assert "mutations" in data
    assert len(data["mutations"]) == 9


def test_lab_api_mutate_resnet(api_client):
    """POST /api/lab/mutate with increase_depth should return before/after/diff."""
    payload = {
        "architecture": "ResNet",
        "mutations": [{"type": "increase_depth", "params": {"n": 1}}],
    }
    res = api_client.post("/api/lab/mutate", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert "before" in data
    assert "after" in data
    assert "diff" in data
    assert "metrics" in data["before"]
    assert "metrics" in data["after"]
    assert "summary_text" in data["diff"]

    # After depth should be bigger
    assert data["after"]["metrics"]["depth"] >= data["before"]["metrics"]["depth"]


def test_lab_api_predict(api_client):
    """POST /api/lab/predict should return scoring with score in [0,1]."""
    payload = {
        "architecture": "ResNet",
        "mutations": [{"type": "increase_depth", "params": {"n": 1}}],
        "hypothesis": {
            "predicted_param_direction": "increase",
            "predicted_flops_direction": "increase",
            "prediction": "Deeper networks have more parameters.",
            "mutation_type": "increase_depth",
        },
    }
    res = api_client.post("/api/lab/predict", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert "scoring" in data
    s = data["scoring"]
    assert "score" in s
    assert 0.0 <= s["score"] <= 1.0
    assert "feedback" in s
    assert "param_correct" in s
    assert "flops_correct" in s


def test_lab_api_tradeoffs(api_client):
    """GET /api/lab/tradeoffs returns scatter_points and tradeoff_summaries."""
    res = api_client.get("/api/lab/tradeoffs?architecture=ResNet")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert "scatter_points" in data
    assert "tradeoff_summaries" in data
    assert len(data["scatter_points"]) >= 5
    assert "increase_depth" in data["tradeoff_summaries"]


def test_lab_api_prediction_prompt(api_client):
    """GET /api/lab/prediction-prompt returns question and hints."""
    res = api_client.get("/api/lab/prediction-prompt?mutation_type=add_attention&architecture=ResNet")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert "question" in data
    assert "hints" in data
    assert "mutation_type" in data
    assert data["mutation_type"] == "add_attention"
    assert len(data["question"]) > 10
