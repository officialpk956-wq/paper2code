from core.fidelity import score_fidelity


MATCHING_SPEC = {
    "layers": [
        {"type": "conv2d", "params": {"channels": 16, "kernel_size": 3}},
        {"type": "relu", "params": {}},
        {"type": "linear", "params": {"hidden_size": 8}},
    ],
    "connection_types": ["residual"],
}

MATCHING_CODE = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        residual = x
        x = self.relu(self.conv(x))
        x = x + residual
        return self.fc(x.mean(dim=(2, 3)))
'''


def _check(result, name):
    return next(check for check in result["checks"] if check["name"] == name)


def test_matching_spec_and_code_score_one():
    result = score_fidelity(MATCHING_SPEC, MATCHING_CODE)
    assert result["score"] == 1.0
    assert result["mismatches"] == []


def test_half_the_layers_fails_count_with_real_counts():
    spec = {**MATCHING_SPEC, "connection_types": []}
    code = MATCHING_CODE.replace("        self.relu = nn.ReLU()\n", "").replace("self.relu(self.conv(x))", "self.conv(x)")
    result = score_fidelity(spec, code)
    check = _check(result, "layer_count")
    assert check["passed"] is False
    # Detail now reports DISTINCT constructors; raw call-site counts are not
    # comparable to logical layer counts (loops, helper classes).
    assert "3" in check["detail"] and "2" in check["detail"]
    assert result["score"] < 1.0


def test_forward_use_without_assignment_fails_declared_vs_used():
    code = MATCHING_CODE.replace("self.relu(self.conv(x))", "self.conv2(self.conv(x))")
    result = score_fidelity(MATCHING_SPEC, code)
    check = _check(result, "declared_vs_used")
    assert check["passed"] is False
    assert "conv2" in check["detail"]


def test_stated_hyperparameter_mismatch_names_expected_and_found_values():
    spec = {"layers": [{"type": "multiheadattention", "params": {"num_heads": 8}}]}
    code = '''
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
    def forward(self, x):
        return self.attn(x, x, x)[0]
'''
    result = score_fidelity(spec, code)
    check = _check(result, "key_hyperparams.num_heads")
    assert check["passed"] is False
    assert "8" in check["detail"] and "4" in check["detail"]


def test_not_stated_hyperparameter_is_informational_and_not_scored():
    spec = {"layers": [{"type": "relu", "params": {}}]}
    code = '''
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x)
'''
    result = score_fidelity(spec, code)
    assert _check(result, "key_hyperparams.num_heads")["detail"] == "not_stated"
    assert result["score"] == 1.0


def test_plain_spec_omits_residual_check():
    spec = {"layers": [{"type": "relu", "params": {}}], "connection_types": []}
    result = score_fidelity(spec, MATCHING_CODE)
    assert "residual_present" not in {check["name"] for check in result["checks"]}


def test_empty_inputs_return_zero_without_raising():
    assert score_fidelity({}, "")["score"] == 0.0


def test_syntax_error_returns_zero_with_explanatory_check():
    result = score_fidelity({"layers": [{"type": "relu", "params": {}}]}, "def broken(:")
    assert result["score"] == 0.0
    assert result["checks"][0]["name"] == "syntax"
    assert "could not be parsed" in result["checks"][0]["detail"]


# ---------------------------------------------------------------------------
# status: a 0.0 that means "could not score" must not read as a measured zero.
# Four return paths used to give a bare 0.0 -- bad spec, empty code, syntax
# error, scorer crash -- and the benchmark averaged them with real scores.
# ---------------------------------------------------------------------------


def test_a_real_zero_is_scored_not_unscorable():
    """Analysable code that fails every check is a measurement of 0.0."""
    spec = {"layers": [{"type": "conv2d", "params": {"channels": 64}}]}
    result = score_fidelity(spec, "import torch\n\n\nclass M:\n    pass\n")
    assert result["status"] == "scored"
    assert result["score"] == 0.0


def test_missing_spec_is_unscorable_not_zero():
    result = score_fidelity({}, "class M:\n    pass\n")
    assert result["status"] == "unscorable"
    assert result["score"] == 0.0


def test_empty_code_is_unscorable_not_zero():
    result = score_fidelity({"layers": [{"type": "relu", "params": {}}]}, "   ")
    assert result["status"] == "unscorable"


def test_unparseable_code_is_unscorable_not_zero():
    result = score_fidelity({"layers": [{"type": "relu", "params": {}}]}, "def broken(:")
    assert result["status"] == "unscorable"


def test_a_broken_scorer_reports_error_not_zero(monkeypatch):
    """If the instrument breaks, say so. 0.0 here is not a fact about the code."""
    import core.fidelity as fid

    monkeypatch.setattr(fid, "_nn_constructor_calls", lambda tree: 1 / 0)
    result = fid.score_fidelity({"layers": [{"type": "relu", "params": {}}]}, "x = 1\n")
    assert result["status"] == "error"
    assert result["score"] == 0.0
    assert "fidelity analysis failed" in result["mismatches"][0]


def test_score_fidelity_no_longer_accepts_a_graph_argument():
    """It was accepted and never read -- callers passing it got a silent no-op."""
    import inspect

    assert "graph" not in inspect.signature(score_fidelity).parameters


def test_benchmark_aggregates_only_scored_results(tmp_path, monkeypatch):
    """An unscorable paper must leave fidelity_score None, not contribute 0.0."""
    from benchmarks import harness

    label = {"schema_version": harness.LABEL_SCHEMA_VERSION, "paper_id": "p", "source": "arxiv:1",
             "family": "resnet", "notes": "", "expected": {"layer_types": ["conv2d"]}}
    extraction = {"spec": {"name": "M", "layers": [{"type": "conv2d", "params": {}}]},
                  "family": "resnet", "code": "def broken(:"}
    row = harness._score_label(label, extraction, fidelity=True)
    assert row["fidelity_status"] == "unscorable"
    assert row["fidelity_score"] is None, "an unscorable run must not average in as 0.0"


def test_declared_vs_used_still_counts_when_there_are_modules():
    """The applicability guard must not disable the check for real code."""
    spec = {"layers": [{"type": "conv2d", "params": {}}]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.a = nn.Conv2d(3, 64, 3)\n"
        "        self.unused = nn.Linear(4, 4)\n\n"
        "    def forward(self, x):\n"
        "        return self.a(x)\n"
    )
    result = score_fidelity(spec, code)
    check = next(c for c in result["checks"] if c["name"] == "declared_vs_used")
    assert check["passed"] is False, "an unused declared module must still fail the check"
    assert "unused" in check["detail"]


# ---------------------------------------------------------------------------
# layer_types_present must measure the CODE, not this module's mapping table.
# 18 of 45 canonical types had no nn.* constructor, and each was counted as
# "missing" regardless of the code. ViT scored 0.00 on a file that contained
# every constructor its spec named.
# ---------------------------------------------------------------------------


def test_unmapped_type_does_not_fail_code_that_contains_everything_else():
    spec = {"layers": [
        {"type": "layernorm", "params": {}},
        {"type": "clstoken", "params": {}},          # nn.Parameter
        {"type": "positionalembedding", "params": {}},  # nn.Parameter
    ]}
    code = (
        "import torch\nimport torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.norm = nn.LayerNorm(768)\n"
        "        self.cls = nn.Parameter(torch.zeros(1, 1, 768))\n\n"
        "    def forward(self, x):\n"
        "        return self.norm(x) + self.cls\n"
    )
    check = next(c for c in score_fidelity(spec, code)["checks"] if c["name"] == "layer_types_present")
    assert check["passed"] is True, check["detail"]


def test_structural_types_are_reported_but_not_scored():
    """residualblock has no single nn.* form; residual_present covers it."""
    spec = {"layers": [{"type": "conv2d", "params": {}}, {"type": "residualblock", "params": {}}]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.c = nn.Conv2d(3, 64, 3)\n\n"
        "    def forward(self, x):\n"
        "        return self.c(x) + x\n"
    )
    check = next(c for c in score_fidelity(spec, code)["checks"] if c["name"] == "layer_types_present")
    assert check["passed"] is True
    assert "structural" in check["detail"]


def test_a_genuinely_absent_constructor_still_fails():
    """The fix must not make the check unfailable."""
    spec = {"layers": [{"type": "conv2d", "params": {}}, {"type": "layernorm", "params": {}}]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.c = nn.Conv2d(3, 64, 3)\n\n"
        "    def forward(self, x):\n"
        "        return self.c(x)\n"
    )
    check = next(c for c in score_fidelity(spec, code)["checks"] if c["name"] == "layer_types_present")
    assert check["passed"] is False
    assert "layernorm" in check["detail"]


def test_every_canonical_type_is_either_mappable_or_declared_structural():
    """A new canonical type must not silently become an unscorable failure."""
    from core.fidelity import _STRUCTURAL_TYPES, _expected_constructor
    from core.rag.normalizer import CANONICAL_TYPES

    orphans = sorted(
        t for t in CANONICAL_TYPES
        if _expected_constructor(t, {}) is None and t not in _STRUCTURAL_TYPES
    )
    assert not orphans, f"canonical types with no constructor and not declared structural: {orphans}"


# ---------------------------------------------------------------------------
# declared_vs_used and layer_count both compared incomparable things.
# ---------------------------------------------------------------------------


def test_custom_submodule_is_not_reported_as_unassigned():
    """Real builders write `self.patch_embed = PatchEmbedding(...)`. Only
    nn.* assignments were recorded, so every custom submodule was flagged."""
    spec = {"layers": [{"type": "conv2d", "params": {}}]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class PatchEmbedding(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.proj = nn.Conv2d(3, 768, 16)\n\n"
        "    def forward(self, x):\n"
        "        return self.proj(x)\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.patch_embed = PatchEmbedding()\n"
        "        self.d_model = int(768)\n\n"
        "    def forward(self, x):\n"
        "        return self.patch_embed(x) * self.d_model\n"
    )
    check = _check(score_fidelity(spec, code), "declared_vs_used")
    assert check["passed"] is True, check["detail"]


def test_genuinely_unassigned_attribute_still_fails():
    """The fix must not make the check unfailable."""
    spec = {"layers": [{"type": "conv2d", "params": {}}]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.conv = nn.Conv2d(3, 64, 3)\n\n"
        "    def forward(self, x):\n"
        "        return self.never_set(self.conv(x))\n"
    )
    check = _check(score_fidelity(spec, code), "declared_vs_used")
    assert check["passed"] is False and "never_set" in check["detail"]


def test_a_loop_built_model_is_not_penalised_for_call_site_count():
    """12 blocks built in a loop is one call site; the spec lists 12 layers."""
    spec = {"layers": [{"type": "multiheadattention", "params": {}} for _ in range(12)]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.blocks = nn.ModuleList(\n"
        "            [nn.MultiheadAttention(768, 12) for _ in range(12)]\n"
        "        )\n\n"
        "    def forward(self, x):\n"
        "        for b in self.blocks:\n"
        "            x = b(x)\n"
        "        return x\n"
    )
    check = _check(score_fidelity(spec, code), "layer_count")
    assert check["passed"] is True, check["detail"]


def test_under_built_code_still_fails_the_breadth_check():
    spec = {"layers": [
        {"type": "conv2d", "params": {}},
        {"type": "layernorm", "params": {}},
        {"type": "multiheadattention", "params": {}},
    ]}
    code = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.c = nn.Conv2d(3, 64, 3)\n\n"
        "    def forward(self, x):\n"
        "        return self.c(x)\n"
    )
    check = _check(score_fidelity(spec, code), "layer_count")
    assert check["passed"] is False, check["detail"]


# ---------------------------------------------------------------------------
# Hyperparameters reach builder-path models through a DEFAULT_SCHEMA dict, not
# as call arguments. transformer_base had d_model 512, num_heads 8, ffn 2048
# and 6 repeats all correct, and the scorer reported every one as missing.
# ---------------------------------------------------------------------------


def test_hyperparameters_in_a_schema_dict_are_found():
    spec = {"layers": [{"type": "multiheadattention",
                        "params": {"hidden_size": 512, "num_heads": 8}}]}
    code = (
        "import torch.nn as nn\n\n"
        "DEFAULT_SCHEMA = {'block': {'params': {'d_model': 512, 'num_heads': 8}}}\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self, schema):\n"
        "        self.attn = nn.MultiheadAttention(\n"
        "            schema['block']['params']['d_model'],\n"
        "            schema['block']['params']['num_heads'],\n"
        "        )\n\n"
        "    def forward(self, x):\n"
        "        return self.attn(x, x, x)\n"
    )
    result = score_fidelity(spec, code)
    for name in ("hidden_size", "num_heads"):
        check = _check(result, f"key_hyperparams.{name}")
        assert check["passed"] is True, check["detail"]


def test_hyperparameter_in_an_assignment_or_default_is_found():
    spec = {"layers": [{"type": "linear", "params": {"hidden_size": 768}}]}
    assigned = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self):\n"
        "        self.d_model = 768\n"
        "        self.fc = nn.Linear(self.d_model, 10)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    default = (
        "import torch.nn as nn\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self, dim=768):\n"
        "        self.fc = nn.Linear(dim, 10)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    for code in (assigned, default):
        check = _check(score_fidelity(spec, code), "key_hyperparams.hidden_size")
        assert check["passed"] is True, check["detail"]


def test_a_hyperparameter_genuinely_absent_still_fails():
    """Broadening must not make the check unfailable."""
    spec = {"layers": [{"type": "linear", "params": {"hidden_size": 768}}]}
    code = (
        "import torch.nn as nn\n\n"
        "DEFAULT_SCHEMA = {'block': {'params': {'d_model': 512}}}\n\n\n"
        "class M(nn.Module):\n"
        "    def __init__(self, schema):\n"
        "        self.fc = nn.Linear(512, 10)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    check = _check(score_fidelity(spec, code), "key_hyperparams.hidden_size")
    assert check["passed"] is False
    assert "768" in check["detail"]
