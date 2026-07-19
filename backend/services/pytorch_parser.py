"""
backend/services/pytorch_parser.py

Parse a PyTorch model (.pt / .pth) file in an E2B sandbox and return the same
graph JSON format that onnx_parser.parse_onnx() produces:

    {
        nodes: [...],
        edges: [...],
        meta:  { total_nodes, total_params, total_edges,
                 graph_inputs, graph_outputs, ir_version, opset_version }
    }

WHY E2B?
  torch is not installed on the main backend (it's a 2 GB dependency).
  Instead we spin up an ephemeral sandbox that already has (or can install)
  the PyTorch CPU wheel, run the trace inside it, and stream JSON back.

FIRST-RUN LATENCY:
  If the sandbox template doesn't have torch pre-installed, pip install runs
  on the first call and takes 2-4 minutes.  Subsequent calls reuse the cached
  template state.  Set E2B_SANDBOX_TEMPLATE to a custom template ID that has
  torch pre-installed to avoid this cold start.

TWO-STRATEGY TRACE:
  1. torch.fx.symbolic_trace — exact computation graph; fails for models with
     dynamic control flow (if/else on tensor values, dynamic loops).
  2. named_modules() leaf-walk — always works; gives a sequential layout of
     the model's submodule tree.  Less accurate but still useful.
"""

import json
import logging

logger = logging.getLogger(__name__)

# How long to allow the sandbox to run (seconds).
# 300 s gives enough time to pip install torch on a cold template start.
_PYTORCH_SANDBOX_TIMEOUT = 300

# ── sandbox Python code ────────────────────────────────────────────────────────
# INPUT_SHAPE placeholder is replaced with the actual list before execution.

_SANDBOX_SCRIPT = r"""
import sys, json, subprocess

# ── ensure torch is available ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
except ImportError:
    print(json.dumps({"__status__": "installing_torch"}), flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet"],
        check=True,
    )
    import torch
    import torch.nn as nn

INPUT_SHAPE = __INPUT_SHAPE__   # replaced by pytorch_parser.py

# ── load model ────────────────────────────────────────────────────────────────
try:
    obj = torch.load("/home/user/model.pt", map_location="cpu", weights_only=False)
except Exception as exc:
    print(json.dumps({"error": "load_failed", "message": str(exc)}))
    sys.exit(1)

# state_dict only — no architecture information
if isinstance(obj, dict):
    print(json.dumps({
        "error": "state_dict_only",
        "message": (
            "This file contains only weights (state_dict), not a full model. "
            "Save the whole model with torch.save(model, 'model.pt') rather "
            "than torch.save(model.state_dict(), 'weights.pt')."
        ),
    }))
    sys.exit(1)

if not isinstance(obj, nn.Module):
    print(json.dumps({
        "error": "not_a_module",
        "message": f"Expected nn.Module, got {type(obj).__name__}.",
    }))
    sys.exit(1)

model = obj
model.eval()
total_params = sum(p.numel() for p in model.parameters())

# ── helper: shape-inference hooks ─────────────────────────────────────────────
def _run_forward_hooks(model, input_shape):
    # Returns {module_name: output_shape} via forward-pass hooks.
    shape_map = {}
    hooks = []

    def _hook(name):
        def fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                shape_map[name] = list(out.shape)
        return fn

    for name, mod in model.named_modules():
        if name:
            hooks.append(mod.register_forward_hook(_hook(name)))

    final_out = None
    try:
        dummy = torch.zeros([1] + input_shape)
        with torch.no_grad():
            final_out = model(dummy)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    return shape_map, final_out


def _attr_val(v):
    # Normalise a module attribute for JSON serialisation.
    if v is None:
        return None
    if isinstance(v, (int, float, str, bool)):
        return round(v, 6) if isinstance(v, float) else v
    if hasattr(v, "__iter__"):
        return list(v)
    return None


_MODULE_ATTRS = (
    "kernel_size", "stride", "padding", "dilation", "groups",
    "in_channels", "out_channels", "in_features", "out_features",
    "num_heads", "eps", "normalized_shape", "p",
)

# ── strategy 1: torch.fx symbolic trace ───────────────────────────────────────
nodes_list, edges_list, edge_id = [], [], 0
method = "symbolic_trace"

try:
    import torch.fx

    traced = torch.fx.symbolic_trace(model)
    g = traced.graph

    shape_map, final_out = _run_forward_hooks(model, INPUT_SHAPE)

    node_id_map = {}
    for i, node in enumerate(g.nodes):
        node_id_map[node.name] = f"node_{i}"

    for i, node in enumerate(g.nodes):
        nid = f"node_{i}"

        if node.op == "placeholder":
            op_type, params, attrs = "Input", 0, {}
        elif node.op == "output":
            op_type, params, attrs = "Output", 0, {}
        elif node.op == "call_module":
            submod = traced.get_submodule(node.target)
            op_type = type(submod).__name__
            params = sum(p.numel() for p in submod.parameters())
            attrs = {k: _attr_val(getattr(submod, k, None))
                     for k in _MODULE_ATTRS
                     if getattr(submod, k, None) is not None}
        elif node.op == "call_function":
            op_type = getattr(node.target, "__name__", str(node.target))
            params, attrs = 0, {}
        elif node.op == "call_method":
            op_type = str(node.target)
            params, attrs = 0, {}
        else:
            op_type = node.op
            params, attrs = 0, {}

        out_shape = []
        if node.op == "call_module" and node.target in shape_map:
            out_shape = shape_map[node.target]

        nodes_list.append({
            "id": nid,
            "op_type": op_type,
            "label": node.name,
            "inputs": [],
            "outputs": [node.name],
            "input_shapes": {},
            "output_shapes": {node.name: out_shape} if out_shape else {},
            "primary_out_shape": out_shape,
            "params": params,
            "attrs": {k: v for k, v in attrs.items() if v is not None},
        })

    seen: set = set()
    for i, node in enumerate(g.nodes):
        nid = f"node_{i}"
        for arg in node.args:
            if isinstance(arg, torch.fx.node.Node):
                src = node_id_map.get(arg.name)
                if src and src != nid and (src, nid) not in seen:
                    seen.add((src, nid))
                    edges_list.append({
                        "id": f"e_{edge_id}",
                        "source": src,
                        "target": nid,
                        "shape": [],
                        "tensor": arg.name,
                    })
                    edge_id += 1

    out_shape_meta = list(final_out.shape) if isinstance(final_out, torch.Tensor) else []

except Exception as fx_err:
    # ── strategy 2: named_modules() leaf walk ─────────────────────────────────
    nodes_list, edges_list, edge_id = [], [], 0
    method = "named_modules"

    leaf_mods = [
        (name, mod)
        for name, mod in model.named_modules()
        if name and len(list(mod.children())) == 0
    ]

    shape_map, final_out = _run_forward_hooks(model, INPUT_SHAPE)

    node_id_map = {}
    for i, (name, mod) in enumerate(leaf_mods):
        nid = f"node_{i}"
        node_id_map[name] = nid
        op_type = type(mod).__name__
        params = sum(p.numel() for p in mod.parameters())
        out_shape = shape_map.get(name, [])
        attrs = {k: _attr_val(getattr(mod, k, None))
                 for k in _MODULE_ATTRS
                 if getattr(mod, k, None) is not None}

        nodes_list.append({
            "id": nid,
            "op_type": op_type,
            "label": name,
            "inputs": [],
            "outputs": [name],
            "input_shapes": {},
            "output_shapes": {name: out_shape} if out_shape else {},
            "primary_out_shape": out_shape,
            "params": params,
            "attrs": {k: v for k, v in attrs.items() if v is not None},
        })

    # Sequential edges between adjacent leaf nodes
    for i in range(len(nodes_list) - 1):
        src = nodes_list[i]["id"]
        tgt = nodes_list[i + 1]["id"]
        edges_list.append({
            "id": f"e_{edge_id}",
            "source": src,
            "target": tgt,
            "shape": nodes_list[i]["primary_out_shape"],
            "tensor": "",
        })
        edge_id += 1

    out_shape_meta = list(final_out.shape) if isinstance(final_out, torch.Tensor) else []

# ── emit result ───────────────────────────────────────────────────────────────
print(json.dumps({
    "nodes": nodes_list,
    "edges": edges_list,
    "meta": {
        "total_nodes": len(nodes_list),
        "total_params": total_params,
        "total_edges": len(edges_list),
        "graph_inputs": {"input": [1] + INPUT_SHAPE},
        "graph_outputs": {"output": out_shape_meta},
        "ir_version": 0,
        "opset_version": 0,
        "method": method,
    },
}))
"""


def parse_pytorch(file_bytes: bytes, input_shape: list[int]) -> dict:
    """
    Run the PyTorch parsing script in an E2B sandbox.

    Args:
        file_bytes:  Raw bytes of the .pt / .pth file.
        input_shape: Spatial dimensions WITHOUT batch, e.g. [3, 224, 224].
                     Batch dim (1) is prepended automatically.

    Returns:
        Graph dict matching parse_onnx() output format.

    Raises:
        ImportError  — e2b not installed
        RuntimeError — sandbox error, torch load failure, etc.
    """
    import os
    import time

    from e2b import CommandExitException  # type: ignore
    from e2b_code_interpreter import Sandbox  # type: ignore

    api_key = os.getenv("E2B_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "E2B_API_KEY is not configured. Set it in your environment to enable "
            "PyTorch model parsing."
        )

    template = os.getenv("E2B_SANDBOX_TEMPLATE") or None
    script = _SANDBOX_SCRIPT.replace("__INPUT_SHAPE__", repr(input_shape))

    logger.info("Starting E2B sandbox for PyTorch parse (input_shape=%s)", input_shape)
    t0 = time.monotonic()

    with Sandbox.create(
        template=template,
        api_key=api_key,
        timeout=_PYTORCH_SANDBOX_TIMEOUT,
    ) as sb:
        # Write the .pt file bytes into the sandbox
        sb.files.write("/home/user/model.pt", file_bytes)
        sb.files.write("/home/user/parse.py", script)

        try:
            result = sb.commands.run(
                f"python3 /home/user/parse.py",
                timeout=_PYTORCH_SANDBOX_TIMEOUT - 10,
            )
            stdout: str = result.stdout or ""
            stderr: str = result.stderr or ""
            exit_code: int = result.exit_code
        except CommandExitException as exc:
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or ""
            exit_code = getattr(exc, "exit_code", 1) or 1

    elapsed = time.monotonic() - t0
    logger.info("E2B sandbox completed in %.1f s (exit_code=%d)", elapsed, exit_code)

    # Parse the JSON output (last non-empty line that looks like JSON)
    json_line = ""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            json_line = line
            break

    if not json_line:
        raise RuntimeError(
            f"PyTorch sandbox produced no JSON output.\nstdout: {stdout[:500]}\nstderr: {stderr[:500]}"
        )

    try:
        data = json.loads(json_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse sandbox output as JSON: {exc}") from exc

    # Surface user-facing errors from inside the sandbox
    if "error" in data:
        raise RuntimeError(data.get("message", data["error"]))

    return data
