"""Labs metrics are pinned to published model sizes, not to whatever the heuristic returns."""

import pytest

from core.lab.diff_engine import compute_diff
from core.lab.lab_metrics import estimate_params_and_memory
from core.lab.mutator import apply_mutations
from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_unet import build_unet_graph
from core.visualizer_vit import build_transformer_graph, build_vit_graph


@pytest.mark.parametrize(
    ("graph", "published_millions"),
    [
        (build_resnet18_graph(), 11.69),  # ResNet-18
        (build_unet_graph(64, 4), 31.03),  # original U-Net
        (build_vit_graph(768, 12, 12), 86.6),  # ViT-B/16
    ],
    ids=["resnet18", "unet", "vit-b16"],
)
def test_params_match_published_sizes(graph, published_millions):
    params, _ = estimate_params_and_memory(graph)
    assert params / 1e6 == pytest.approx(published_millions, rel=0.01)


@pytest.mark.parametrize(
    "graph",
    [build_resnet18_graph(), build_unet_graph(), build_vit_graph(), build_transformer_graph()],
    ids=["resnet", "unet", "vit", "transformer"],
)
def test_every_lab_architecture_has_params_and_activation_memory(graph):
    params, mem_mb = estimate_params_and_memory(graph)
    assert params > 1_000_000
    assert mem_mb > 1


def test_transformer_lab_is_token_based_not_a_copy_of_the_vit():
    transformer, vit = build_transformer_graph(), build_vit_graph()
    assert transformer.nodes[0].label == "Token Embedding"
    assert vit.nodes[0].label == "Patch Embedding"
    # every edge still points at a node that exists after the embedding swap
    ids = {n.id for n in transformer.nodes}
    assert all(e.source in ids and e.target in ids for e in transformer.edges)
    assert estimate_params_and_memory(transformer)[0] != estimate_params_and_memory(vit)[0]


def test_mutations_move_metrics_in_the_expected_direction():
    base = build_resnet18_graph()
    deeper = apply_mutations(base, [{"type": "increase_depth"}])
    wider = apply_mutations(base, [{"type": "increase_width"}])
    p0, m0 = estimate_params_and_memory(base)
    pd, md = estimate_params_and_memory(deeper)
    pw, mw = estimate_params_and_memory(wider)
    assert pd > p0 and md > m0
    assert pw > p0 and mw > m0


def test_diff_uses_the_supplied_estimator():
    before = build_resnet18_graph()
    after = apply_mutations(before, [{"type": "increase_depth"}])
    diff = compute_diff(before, after, params_and_memory=estimate_params_and_memory)
    assert diff["param_delta"]["before"] == estimate_params_and_memory(before)[0]
    assert diff["param_delta"]["after"] == estimate_params_and_memory(after)[0]
    assert diff["memory_delta"]["before_mb"] > 0


def test_unet_increase_depth_adds_a_real_block_not_the_output_head():
    base = build_unet_graph()
    deeper = apply_mutations(base, [{"type": "increase_depth"}])
    added = [n for n in deeper.nodes if n.id not in {b.id for b in base.nodes}]
    assert len(added) == 1 and added[0].block != "Head"
    assert estimate_params_and_memory(deeper)[0] > estimate_params_and_memory(base)[0]
