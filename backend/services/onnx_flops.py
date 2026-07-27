"""Per-node FLOPs + activation-memory estimator for ONNX graphs.

`parse_onnx` gives us op_type, data-input shapes, the output shape, weight
initializer dims, and attributes — enough to estimate compute cost per node so
the viewer can colour nodes by where the FLOPs/memory actually go (audit #3).

Estimates are per forward pass at the shapes in the model (batch dim as-is;
dynamic dims counted as 1). A pure function — no `onnx` import — so it is unit
testable on hand-built node data even where the onnx package isn't installed.
"""

from __future__ import annotations

from typing import Any

_BYTES_FP32 = 4
_MB = 1024 * 1024

_POOL = {"MaxPool", "AveragePool", "LpPool"}
_GLOBAL_POOL = {"GlobalAveragePool", "GlobalMaxPool", "GlobalLpPool"}
_NORM = {
    "BatchNormalization",
    "LayerNormalization",
    "InstanceNormalization",
    "GroupNormalization",
}
_SOFTMAX = {"Softmax", "LogSoftmax"}
_ELEMENTWISE = {
    "Relu",
    "LeakyRelu",
    "PRelu",
    "Sigmoid",
    "Tanh",
    "Elu",
    "Selu",
    "Celu",
    "Gelu",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sqrt",
    "Exp",
    "Log",
    "Clip",
    "Abs",
    "Neg",
    "HardSigmoid",
    "HardSwish",
    "Erf",
    "Softplus",
    "Mish",
    "Swish",
    "SiLU",
    "Sin",
    "Cos",
    "Reciprocal",
    "Sign",
    "Round",
    "Floor",
    "Ceil",
}

# severity bands, in MFLOPs and MB (mirrors core/rag FLOPsEngine)
_MF = {"medium": 10, "high": 200, "critical": 2000}
_MEM = {"medium": 50, "high": 500, "critical": 4000}


def _prod(dims) -> int:
    p = 1
    for d in dims or []:
        if isinstance(d, int) and d > 0:
            p *= d
    return p


def _last_pos(shape) -> int:
    for d in reversed(shape or []):
        if isinstance(d, int) and d > 0:
            return d
    return 1


def _pick(weight_dims, ndim):
    for w in weight_dims or []:
        if len(w) == ndim:
            return w
    return None


def _gemm_contract(weight_dims, n_out, input_shapes) -> int:
    """The contracted (K) dimension of a Gemm/linear layer."""
    w = _pick(weight_dims, 2) or (weight_dims[0] if weight_dims else None)
    if w and len(w) == 2:
        a, b = w[0], w[1]
        if b == n_out:
            return a
        if a == n_out:
            return b
        return max(a, b)
    return _last_pos(input_shapes[0]) if input_shapes else 1


def _severity(flops: int, mem_mb: float) -> str:
    mf = flops / 1e6
    if mf >= _MF["critical"] or mem_mb >= _MEM["critical"]:
        return "critical"
    if mf >= _MF["high"] or mem_mb >= _MEM["high"]:
        return "high"
    if mf >= _MF["medium"] or mem_mb >= _MEM["medium"]:
        return "medium"
    return "low"


def estimate_node_cost(
    op_type: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    weight_dims: list[list[int]],
    attrs: dict[str, Any] | None = None,
) -> dict:
    """Return {flops, memory_mb, severity} for one ONNX node."""
    attrs = attrs or {}
    out_elems = _prod(output_shape)
    mem_mb = out_elems * _BYTES_FP32 / _MB
    flops = 0

    if op_type in ("Conv", "ConvTranspose"):
        w = (
            _pick(weight_dims, 4)
            or _pick(weight_dims, 3)
            or (weight_dims[0] if weight_dims else None)
        )
        if w and len(w) >= 2:
            macs_per_out = _prod(w[1:])  # (C_in / groups) * kH * kW
        else:
            k = _prod(attrs.get("kernel_shape") or [1])
            c_in = input_shapes[0][1] if input_shapes and len(input_shapes[0]) > 1 else 1
            macs_per_out = max(1, (c_in if isinstance(c_in, int) and c_in > 0 else 1) * k)
        flops = 2 * out_elems * macs_per_out

    elif op_type == "Gemm":
        n_out = _last_pos(output_shape)
        flops = 2 * out_elems * _gemm_contract(weight_dims, n_out, input_shapes)

    elif op_type == "MatMul":
        k = _last_pos(input_shapes[0]) if input_shapes else _last_pos(output_shape)
        flops = 2 * out_elems * k

    elif op_type in _POOL:
        flops = out_elems * max(1, _prod(attrs.get("kernel_shape") or [1, 1]))

    elif op_type in _GLOBAL_POOL:
        flops = _prod(input_shapes[0]) if input_shapes else out_elems

    elif op_type in _NORM or op_type in _SOFTMAX:
        flops = out_elems * 5

    elif op_type in _ELEMENTWISE:
        flops = out_elems

    return {
        "flops": int(flops),
        "memory_mb": round(mem_mb, 4),
        "severity": _severity(flops, mem_mb),
    }


def _demo():
    # Conv 3->64, 7x7, output 112x112: 2 * (64*112*112) * (3*7*7)
    c = estimate_node_cost("Conv", [[1, 3, 224, 224]], [1, 64, 112, 112], [[64, 3, 7, 7]], {})
    assert c["flops"] == 2 * (64 * 112 * 112) * (3 * 7 * 7), c
    assert c["severity"] in ("medium", "high"), c
    # Gemm 2048 -> 1000: 2 * 1000 * 2048
    g = estimate_node_cost("Gemm", [[1, 2048]], [1, 1000], [[2048, 1000]], {})
    assert g["flops"] == 2 * 1000 * 2048, g
    # MatMul contract on last input dim
    m = estimate_node_cost("MatMul", [[1, 512, 768]], [1, 512, 768], [[768, 768]], {})
    assert m["flops"] == 2 * (512 * 768) * 768, m
    # Relu elementwise = out elems; memory = elems*4/MB
    r = estimate_node_cost("Relu", [[1, 64, 112, 112]], [1, 64, 112, 112], [], {})
    assert r["flops"] == 64 * 112 * 112, r
    assert abs(r["memory_mb"] - (64 * 112 * 112 * 4 / _MB)) < 1e-6, r
    # unknown op -> 0 flops, still reports memory
    u = estimate_node_cost("Reshape", [[1, 1000]], [1, 1000], [], {})
    assert u["flops"] == 0 and u["memory_mb"] > 0, u
    print("onnx_flops demo OK")


if __name__ == "__main__":
    _demo()
