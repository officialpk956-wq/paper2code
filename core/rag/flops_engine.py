"""
FLOPs and Activation-Memory Propagation Engine.

Computes per-layer computational complexity and memory footprint
with exact symbolic formulas during tensor propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BYTES_PER_FLOAT32 = 4  # fp32
MB = 1024 * 1024

# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class FLOPsResult:
    node_id: str
    node_type: str
    node_label: str = ""

    flops_mflops: float = 0.0  # Mega-FLOPs (multiply-adds × 2)
    params_M: float = 0.0  # Parameters in millions
    memory_mb: float = 0.0  # Activation memory in MB

    formula: str = "—"  # Symbolic expression
    complexity: str = "O(1)"  # Big-O class
    severity: str = "low"  # "low" | "medium" | "high" | "critical"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "node_label": self.node_label,
            "flops_mflops": round(self.flops_mflops, 4),
            "params_M": round(self.params_M, 4),
            "memory_mb": round(self.memory_mb, 3),
            "formula": self.formula,
            "complexity": self.complexity,
            "severity": self.severity,
            "warnings": self.warnings,
        }


# ──────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────


class FLOPsEngine:
    """
    Computes FLOPs, parameter counts, and activation memory for each
    layer type.  All estimates are per-sample (B=1); multiply by batch
    for real VRAM usage.
    """

    # Thresholds (MFLOPs / MB) for severity bands
    _FLOPS_THRESH = {"medium": 10, "high": 200, "critical": 2000}
    _MEM_THRESH = {"medium": 50, "high": 500, "critical": 4000}

    # Dispatch table
    _DISPATCH: dict[str, str] = {
        "conv2d": "_conv2d",
        "conv1d": "_conv1d",
        "conv": "_conv2d",
        "linear": "_linear",
        "dense": "_linear",
        "residualblock": "_residual_block",
        "block": "_residual_block",
        "stage": "_residual_block",
        "upsample": "_upsample",
        "convtranspose2d": "_conv2d",
        "upconvolution": "_conv2d",
        "multiheadattention": "_mha",
        "mhsa": "_mha",
        "self_attention": "_mha",
        "causal_attention": "_mha",
        "cross_attention": "_cross_attention",
        "feedforward": "_feedforward",
        "patchembedding": "_patch_embedding",
        "token_embedding": "_embedding",
        "positionalembedding": "_embedding",
        "segment_embedding": "_embedding",
        "layernorm": "_layernorm",
        "rmsnorm": "_layernorm",
        "batchnorm2d": "_batchnorm",
        "invertedresidual": "_residual_block",
        "mbconvblock": "_residual_block",
    }

    # ── public entry point ─────────────────────────────────

    def estimate(
        self,
        node_id: str,
        node_type: str,
        in_shape: tuple,
        out_shape: tuple,
        params: dict[str, Any],
        label: str = "",
    ) -> FLOPsResult:
        r = FLOPsResult(node_id=node_id, node_type=node_type, node_label=label or node_id)

        method_name = self._DISPATCH.get(node_type.lower())
        if method_name:
            getattr(self, method_name)(r, in_shape, out_shape, params)

        # Activation memory from output shape (always computed)
        r.memory_mb = self._activation_memory(out_shape)

        # Severity & warnings
        r.severity = self._severity(r.flops_mflops, r.memory_mb)
        r.warnings = self._generate_warnings(r, node_type, in_shape, params)
        return r

    # ── per-type estimators ────────────────────────────────

    def _conv2d(self, r: FLOPsResult, in_s, out_s, p):
        C_in = _int(in_s, 1, 3)
        C_out = _int(out_s, 1, _int(p, "channels", _int(p, "filters", 64)))
        H_out = _int(out_s, 2, 1)
        W_out = _int(out_s, 3, 1)
        K = _int(p, "kernel_size", 3)

        flops = 2 * C_in * C_out * K * K * H_out * W_out
        params = C_out * (C_in * K * K + 1)

        r.flops_mflops = flops / 1e6
        r.params_M = params / 1e6
        r.formula = f"2 × C_in({C_in}) × C_out({C_out}) × K²({K}²) × H({H_out}) × W({W_out})"
        r.complexity = "O(C_in × C_out × K² × H × W)"

    def _conv1d(self, r: FLOPsResult, in_s, out_s, p):
        C_in = _int(in_s, 1, 1)
        C_out = _int(out_s, 1, _int(p, "channels", 64))
        L = _int(out_s, 2, 1)
        K = _int(p, "kernel_size", 3)

        r.flops_mflops = 2 * C_in * C_out * K * L / 1e6
        r.params_M = (C_out * (C_in * K + 1)) / 1e6
        r.formula = f"2 × C_in({C_in}) × C_out({C_out}) × K({K}) × L({L})"
        r.complexity = "O(C_in × C_out × K × L)"

    def _linear(self, r: FLOPsResult, in_s, out_s, p):
        D_in = _last_int(in_s, _int(p, "hidden_size", 768))
        D_out = _last_int(out_s, D_in)
        N = _int(in_s, 1, 1) if len(in_s) == 3 else 1

        r.flops_mflops = 2 * N * D_in * D_out / 1e6
        r.params_M = (D_in * D_out + D_out) / 1e6
        r.formula = f"2 × N({N}) × D_in({D_in}) × D_out({D_out})"
        r.complexity = "O(N × D_in × D_out)"

    def _mha(self, r: FLOPsResult, in_s, out_s, p):
        D = _last_int(in_s, _int(p, "embed_dim", 768))
        N = _int(in_s, 1, _int(p, "seq_len", 512))
        H = _int(p, "num_heads", 12)

        # QKV projections: 3 × 2 × N × D × D
        qkv = 3 * 2 * N * D * D
        # Score computation: 2 × N² × D  (quadratic bottleneck)
        scores = 2 * N * N * D
        # AV product: 2 × N² × D
        av = 2 * N * N * D
        # Output projection: 2 × N × D × D
        out_p = 2 * N * D * D

        total = qkv + scores + av + out_p

        r.flops_mflops = total / 1e6
        r.params_M = (4 * D * D) / 1e6  # Q, K, V, O projections
        r.formula = (
            f"QKV: 3·2·N·D² + Scores: 2·N²·D + AV: 2·N²·D + Out: 2·N·D²\n"
            f"= N={N}, D={D}, H={H}  →  {total / 1e6:.1f} MFLOPs"
        )
        r.complexity = "O(N²·D + N·D²)"

    def _feedforward(self, r: FLOPsResult, in_s, out_s, p):
        D = _last_int(in_s, _int(p, "embed_dim", 768))
        N = _int(in_s, 1, 1)
        D_ff = _int(p, "ff_dim", _int(p, "intermediate_size", D * 4))

        flops = 2 * N * D * D_ff + 2 * N * D_ff * D
        params = (D * D_ff + D_ff) + (D_ff * D + D)

        r.flops_mflops = flops / 1e6
        r.params_M = params / 1e6
        r.formula = f"2·N·D·D_ff + 2·N·D_ff·D  |  N={N}, D={D}, D_ff={D_ff}"
        r.complexity = "O(N × D × D_ff)"

    def _patch_embedding(self, r: FLOPsResult, in_s, out_s, p):
        C_in = _int(in_s, 1, 3)
        P = _int(p, "patch_size", 16)
        E = _int(p, "embed_dim", 768)
        N = _int(out_s, 1, 196)  # num_patches

        flops = 2 * C_in * P * P * E * N
        params = C_in * P * P * E + E

        r.flops_mflops = flops / 1e6
        r.params_M = params / 1e6
        r.formula = f"2 × C_in({C_in}) × P²({P}²) × D({E}) × N_patches({N})"
        r.complexity = "O(C × P² × D × N)"

    def _embedding(self, r: FLOPsResult, in_s, out_s, p):
        D = _last_int(out_s, _int(p, "embed_dim", 768))
        vocab = _int(p, "vocab_size", 30522)

        r.flops_mflops = 0.001  # lookup — negligible
        r.params_M = (vocab * D) / 1e6
        r.formula = f"Table lookup — vocab={vocab:,}, D={D}"
        r.complexity = "O(N) — table lookup"

    def _layernorm(self, r: FLOPsResult, in_s, out_s, p):
        D = _last_int(in_s, 768)
        N = _int(in_s, 1, 1)

        r.flops_mflops = 5 * N * D / 1e6  # mean, var, normalize, scale, shift
        r.params_M = (2 * D) / 1e6  # gamma, beta
        r.formula = f"5 × N({N}) × D({D})  [mean+var+norm+scale+shift]"
        r.complexity = "O(N × D)"

    def _batchnorm(self, r: FLOPsResult, in_s, out_s, p):
        C = _int(in_s, 1, 64)
        r.flops_mflops = 0.001
        r.params_M = (2 * C) / 1e6
        r.formula = f"Running stats — C={C} channels"
        r.complexity = "O(C)"

    def _residual_block(self, r: FLOPsResult, in_s, out_s, p):
        """Approximates a residual block as two conv2d layers + optional projection."""
        C_in = _int(in_s, 1, 64)
        C_out = _int(out_s, 1, p.get("out_channels", p.get("channels", p.get("filters", C_in))))
        if not isinstance(C_out, int):
            C_out = C_in
        H = _int(out_s, 2, 56)
        W = _int(out_s, 3, 56)
        K = _int(p, "kernel_size", 3)
        stride = _int(p, "stride", 1)
        # Two conv layers in the block
        flops_conv1 = 2 * C_in * C_out * K * K * H * W
        flops_conv2 = 2 * C_out * C_out * K * K * H * W
        # Projection shortcut if downsampling
        flops_proj = 2 * C_in * C_out * 1 * 1 * H * W if stride > 1 else 0
        total = flops_conv1 + flops_conv2 + flops_proj
        params = (C_in * C_out * K * K) + (C_out * C_out * K * K)
        r.flops_mflops = total / 1e6
        r.params_M = params / 1e6
        r.formula = f"2·C_in({C_in})·C_out({C_out})·K²({K}²)·H·W + 2·C_out²·K²·H·W"
        r.complexity = "O(C_in × C_out × K² × H × W)"

    def _upsample(self, r: FLOPsResult, in_s, out_s, p):
        """Bilinear upsample: 4 multiplications per element."""
        C = _int(in_s, 1, 64)
        H = _int(out_s, 2, 112)
        W = _int(out_s, 3, 112)
        r.flops_mflops = 4 * C * H * W / 1e6
        r.params_M = 0.0
        r.formula = f"4 × C({C}) × H({H}) × W({W})  [bilinear]"
        r.complexity = "O(C × H × W)"

    def _cross_attention(self, r: FLOPsResult, in_s, out_s, p):
        D = _last_int(in_s, _int(p, "embed_dim", 768))
        N_q = _int(in_s, 1, _int(p, "seq_len_q", 128))
        N_k = _int(p, "seq_len_k", N_q * 4)
        H = _int(p, "num_heads", 8)

        # Q proj: 2·N_q·D²; K proj: 2·N_k·D²; V proj: 2·N_k·D²
        qkv = 2 * N_q * D * D + 2 * N_k * D * D + 2 * N_k * D * D
        # Q·K^T scores: 2·N_q·N_k·D  (rectangle, not square)
        scores = 2 * N_q * N_k * D
        # A·V:          2·N_q·N_k·D
        av = 2 * N_q * N_k * D

        total = qkv + scores + av

        r.flops_mflops = total / 1e6
        r.params_M = (4 * D * D) / 1e6
        r.formula = (
            f"Q·K^T: N_q({N_q}) × N_k({N_k}) × D({D})\n= {total / 1e6:.1f} MFLOPs across {H} heads"
        )
        r.complexity = "O(N_q × N_k × D)"

    # ── helpers ────────────────────────────────────────────

    def _activation_memory(self, shape: tuple) -> float:
        try:
            elems = 1
            for d in shape:
                if isinstance(d, int):
                    elems *= d
            return (elems * BYTES_PER_FLOAT32) / MB
        except Exception:
            return 0.0

    def _severity(self, flops: float, mem: float) -> str:
        ft, mt = self._FLOPS_THRESH, self._MEM_THRESH
        if flops >= ft["critical"] or mem >= mt["critical"]:
            return "critical"
        if flops >= ft["high"] or mem >= mt["high"]:
            return "high"
        if flops >= ft["medium"] or mem >= mt["medium"]:
            return "medium"
        return "low"

    def _generate_warnings(self, r: FLOPsResult, ntype: str, in_s: tuple, p: dict) -> list[str]:
        w = []
        if r.memory_mb > 500:
            w.append(f"High VRAM: {r.memory_mb:.1f} MB activation memory")
        if ntype.lower() in (
            "multiheadattention",
            "mhsa",
            "causal_attention",
            "cross_attention",
            "self_attention",
        ):
            N = _int(in_s, 1, 0)
            if isinstance(N, int) and N > 1024:
                w.append(
                    f"Quadratic attention explosion: N={N} → {N * N:,} score elements per head"
                )
        if r.flops_mflops > 2000:
            w.append(
                f"Compute-heavy: {r.flops_mflops:.0f} MFLOPs — "
                "consider linear attention or chunking"
            )
        return w


# ──────────────────────────────────────────────────────────
# Utility accessors
# ──────────────────────────────────────────────────────────


def _int(src, key_or_idx, default=0):
    """Safely extract an int from either a tuple (by index) or a dict (by key)."""
    try:
        if isinstance(src, (tuple, list)):
            v = src[key_or_idx]
        elif isinstance(src, dict):
            v = src.get(key_or_idx, default)
        else:
            return default
        return int(v) if isinstance(v, (int, float)) else default
    except (IndexError, TypeError, KeyError):
        return default


def _last_int(shape: tuple, default: int = 0) -> int:
    """Return the last numeric dimension of a shape tuple."""
    for d in reversed(shape):
        if isinstance(d, int):
            return d
    return default
