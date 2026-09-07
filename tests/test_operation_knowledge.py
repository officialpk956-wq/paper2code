from core.knowledge.operations import OPERATIONS, _ALIAS_INDEX, find_mentioned, lookup
from core.rag.normalizer import CANONICAL_TYPES
from core.architecture_graph import ArchitectureGraph, GraphNode
from core.codegen import _generate_skeleton, _node_to_layer


def test_operations_have_required_well_formed_fields():
    required = {"formula", "latex", "syntax", "functional", "aliases", "notes"}
    assert set(OPERATIONS) == {
        "sigmoid", "relu", "leakyrelu", "gelu", "silu", "swish", "tanh", "softmax",
        "batchnorm2d", "layernorm", "groupnorm", "rmsnorm",
        "scaled_dot_product_attention", "multiheadattention",
        "linear", "conv2d", "depthwise_conv2d", "dropout", "residual_add", "concat",
        "flatten", "globalavgpool2d", "patchembedding",
    }
    for entry in OPERATIONS.values():
        assert required <= entry.keys()
        assert entry["aliases"]
        assert all(alias == alias.lower() for alias in entry["aliases"])


def test_alias_index_has_no_collisions():
    total_aliases = sum(len(entry["aliases"]) for entry in OPERATIONS.values())
    assert len(_ALIAS_INDEX) == total_aliases


def test_syntax_and_functional_prefixes():
    for entry in OPERATIONS.values():
        if entry["syntax"] is not None:
            assert entry["syntax"].startswith("nn.")
        if entry["functional"] is not None:
            assert entry["functional"].startswith(("torch.", "F."))


def test_canonical_intersection_uses_identical_spelling():
    for canonical in set(OPERATIONS) & CANONICAL_TYPES:
        assert canonical in CANONICAL_TYPES


def test_lookup_normalises_logistic_variants():
    expected = OPERATIONS["sigmoid"]
    assert lookup("Logistic Function") is expected
    assert lookup("logistic_function") is expected
    assert lookup("sigmoid") is expected


def test_lookup_returns_none_for_unknown_operation():
    assert lookup("some_novel_op_xyz") is None


def test_find_mentioned_returns_first_appearance_order():
    assert find_mentioned("We apply layer normalization then a GELU activation.") == ["layernorm", "gelu"]


def test_find_mentioned_empty_text_returns_empty_list():
    assert find_mentioned("") == []


def test_mathematical_correctness_details_are_present():
    assert "eps" in OPERATIONS["layernorm"]["formula"]
    assert "sqrt" in OPERATIONS["scaled_dot_product_attention"]["formula"]
    assert "groups" in OPERATIONS["depthwise_conv2d"]["syntax"]


def test_codegen_uses_operation_fallback_after_parameterized_map():
    tanh = GraphNode(id="tanh", type="tanh", label="Tanh")
    conv2d = GraphNode(
        id="conv", type="conv2d", label="Conv", params={"channels": 32}
    )

    assert _node_to_layer(tanh) == "nn.Tanh()"
    assert "32" in _node_to_layer(conv2d)


def test_codegen_keeps_unknown_operations_as_valid_passthrough():
    node = GraphNode(
        id="unknown",
        type="timestep_embedding_xyz",
        label="Timestep embedding",
    )
    graph = ArchitectureGraph(name="UnknownOps", nodes=[node])

    assert _node_to_layer(node) is None
    code = _generate_skeleton(graph)
    compile(code, "<gen>", "exec")
