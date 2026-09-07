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
    assert "spec=3" in check["detail"] and "code=2" in check["detail"]
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
