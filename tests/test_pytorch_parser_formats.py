"""Model-viz PyTorch path: which file formats it accepts and what it tells the user.

Runs the embedded sandbox script LOCALLY (no E2B) against real files, and
exercises the server-side ONNX decode with the sandbox mocked out.

Before 2026-09-19 the only formats a user could produce -- a pickled
nn.Module and a state_dict -- both failed, and the state_dict error told
them to switch to the pickled format. TorchScript is self-contained; it now
loads first and goes through the ONNX parser.
"""
import base64
import io
import json
import re
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

import torch.nn as nn  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).mean(dim=(2, 3)))


def _sandbox_body() -> str:
    src = open("backend/services/pytorch_parser.py", encoding="utf-8").read()
    return re.search(r'_SANDBOX_SCRIPT = r"""(.*?)"""', src, re.S).group(1).replace("__INPUT_SHAPE__", "[3, 8, 8]")


def _run_sandbox_locally(model_path) -> dict:
    """Execute the sandbox script in a fresh interpreter -- one with NO _Tiny
    class defined, exactly like the E2B sandbox."""
    script = _sandbox_body().replace("/home/user/model.pt", str(model_path).replace("\\", "/"))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    assert lines, f"no JSON from sandbox script\nstdout={proc.stdout[-500:]}\nstderr={proc.stderr[-500:]}"
    return json.loads(lines[-1])


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    m = _Tiny().eval()
    torch.jit.trace(m, torch.zeros(1, 3, 8, 8)).save(d / "scripted.pt")
    torch.save(m.state_dict(), d / "weights.pt")
    # The pickled module must reference a class in __main__, as a user's
    # `torch.save(model, ...)` from a script does. Defined here, it would live
    # in an importable test module and the sandbox script could load it.
    script = f"""
import torch, torch.nn as nn
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1); self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).mean(dim=(2, 3)))
torch.save(_Tiny().eval(), r"{d / 'pickled.pt'}")
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, timeout=300)
    return d


def test_torchscript_is_loaded_without_the_class_and_exported_to_onnx(files):
    out = _run_sandbox_locally(files / "scripted.pt")
    assert "onnx_b64" in out, out
    assert out["method"] == "torchscript_onnx"
    from backend.services.onnx_parser import parse_onnx

    graph = parse_onnx(base64.b64decode(out["onnx_b64"]))
    assert {n["op_type"] for n in graph["nodes"]} >= {"Conv", "Relu", "Gemm"}


def test_pickled_module_without_its_class_gets_an_actionable_error(files):
    out = _run_sandbox_locally(files / "pickled.pt")
    assert out["error"] == "class_unavailable"
    assert "torch.jit.trace" in out["message"] and "torch.onnx.export" in out["message"]
    assert "torch.save(model" not in out["message"], "must not recommend the format that just failed"


def test_state_dict_error_no_longer_recommends_the_other_dead_end(files):
    out = _run_sandbox_locally(files / "weights.pt")
    assert out["error"] == "state_dict_only"
    assert "torch.jit.trace" in out["message"]
    assert "torch.save(model, 'model.pt')" not in out["message"]


def test_server_decodes_onnx_bytes_returned_by_the_sandbox(monkeypatch, files):
    """The server side of the TorchScript route, with E2B mocked out."""
    from backend.services import pytorch_parser as pp

    m = torch.jit.load(files / "scripted.pt")
    buf = io.BytesIO()
    torch.onnx.export(m, torch.zeros(1, 3, 8, 8), buf, opset_version=17, dynamo=False)
    payload = json.dumps({"onnx_b64": base64.b64encode(buf.getvalue()).decode(), "method": "torchscript_onnx"})

    class _Result:
        stdout = payload + "\n"
        stderr = ""
        exit_code = 0

    class _Files:
        def write(self, *a, **k):
            return None

    class _Commands:
        def run(self, *a, **k):
            return _Result()

    class _Sandbox:
        files = _Files()
        commands = _Commands()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        @classmethod
        def create(cls, *a, **k):
            return cls()

    import e2b_code_interpreter

    monkeypatch.setattr(e2b_code_interpreter, "Sandbox", _Sandbox)
    graph = pp.parse_pytorch(b"irrelevant", [3, 8, 8])
    assert graph["meta"]["method"] == "torchscript_onnx"
    assert graph["meta"]["source_format"] == "torchscript"
    assert len(graph["nodes"]) >= 3
