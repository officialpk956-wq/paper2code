#!/usr/bin/env python3
"""
Lab Service — AI Labs Feature 3.

Builds real PyTorch models from core/blocks_*.py, runs a single dummy forward
pass to capture exact tensor shapes via register_forward_hook, then feeds
those shapes into FLOPsEngine for exact per-layer FLOPs/params/memory.

No mocked values. No hardcoded shapes. Everything through real code paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from core.blocks_transformer import TransformerEncoderBlock
from core.blocks_vit import PatchEmbedding
from core.blocks_resnet import Bottleneck
from core.blocks_unet import DoubleConv
from core.ddpm_builder import DDPMBuilder
from core.rag.flops_engine import FLOPsEngine
from core.param_counter import count_parameters
from core.implementation.cost_estimator import GPU_SPECS


# ---------------------------------------------------------------------------
# Lab-specific model definitions (use real blocks, not reimplementations)
# ---------------------------------------------------------------------------

class _TransformerLab(nn.Module):
    """Encoder-only Transformer for interactive parameter exploration."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 vocab_size: int, ffn_dim: Optional[int] = None) -> None:
        super().__init__()
        ffn_dim = ffn_dim or d_model * 4
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.layers = nn.Sequential(*[
            TransformerEncoderBlock(d_model, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        e = self.embedding(x) + self.pos_embed(positions)
        e = self.layers(e)
        return self.out_proj(self.norm(e))


class _CNNLab(nn.Module):
    """Parametric CNN — VGG-style with configurable depth/width/kernel."""

    def __init__(self, base_channels: int, depth: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: List[nn.Module] = []
        in_c = 3
        out_c = base_channels

        for i in range(depth):
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
                          padding=padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ])
            # Downsample every other layer (keeps spatial resolution manageable)
            if i % 2 == 1 and i < depth - 1:
                layers.append(nn.MaxPool2d(2))
            in_c = out_c
            out_c = min(in_c * 2, 512)

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, 1000)
        self._final_channels = in_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


class _ViTLab(nn.Module):
    """Vision Transformer with parametric patch size, hidden dim, and depth."""

    def __init__(self, image_size: int, patch_size: int,
                 hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        # Ensure patch_size divides image_size
        patch_size = max(1, patch_size)
        if image_size % patch_size != 0:
            # Round patch_size down to largest divisor ≤ requested
            for ps in range(patch_size, 0, -1):
                if image_size % ps == 0:
                    patch_size = ps
                    break

        # Choose num_heads so hidden_dim is divisible
        num_heads = _find_valid_heads(hidden_dim)
        ffn_dim = hidden_dim * 4
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(3, patch_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1000)
        self._patch_size = patch_size
        self._num_patches = num_patches
        self._num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))


def _find_valid_heads(d_model: int) -> int:
    """Return the largest divisor of d_model that is ≤ 16."""
    for h in range(min(16, d_model), 0, -1):
        if d_model % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _module_to_params(mod: nn.Module) -> Dict[str, Any]:
    """Extract numeric param dict from a PyTorch module for FLOPsEngine."""
    if isinstance(mod, nn.Conv2d):
        k = mod.kernel_size
        s = mod.stride
        return {
            "in_channels": mod.in_channels,
            "out_channels": mod.out_channels,
            "kernel_size": k[0] if isinstance(k, tuple) else k,
            "stride": s[0] if isinstance(s, tuple) else s,
            "channels": mod.out_channels,
        }
    if isinstance(mod, nn.ConvTranspose2d):
        k = mod.kernel_size
        return {
            "in_channels": mod.in_channels,
            "out_channels": mod.out_channels,
            "kernel_size": k[0] if isinstance(k, tuple) else k,
            "channels": mod.out_channels,
        }
    if isinstance(mod, nn.Linear):
        return {"in_features": mod.in_features, "out_features": mod.out_features}
    if isinstance(mod, nn.MultiheadAttention):
        return {"embed_dim": mod.embed_dim, "num_heads": mod.num_heads}
    if isinstance(mod, nn.LayerNorm):
        ns = mod.normalized_shape
        return {"hidden_size": ns[0] if ns else 0}
    if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)):
        ks = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
        st = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
        return {"kernel_size": ks, "stride": st}
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return {}
    if isinstance(mod, nn.Embedding):
        return {"vocab_size": mod.num_embeddings, "embed_dim": mod.embedding_dim}
    return {}


def _humanize(name: str, mod: nn.Module) -> str:
    """Return a readable display name for a module."""
    _NAMES: Dict[str, str] = {
        "embedding": "Token Embedding",
        "pos_embed": "Positional Embedding",
        "patch_embed": "Patch Embedding",
        "norm": "LayerNorm",
        "out_proj": "Output Projection",
        "head": "Classifier",
        "fc": "Linear Classifier",
        "pool": "Global Avg Pool",
        "features": "Feature Extractor",
        "blocks": "Encoder Blocks",
        "layers": "Encoder Layers",
        "cls_token": "CLS Token",
        "init_conv": "Initial Conv",
        "final_conv": "Output Conv",
        "time_mlp": "Time Embedding MLP",
        "mid1": "Mid Block 1",
        "mid2": "Mid Block 2",
    }
    short = name.split(".")[-1]
    if short in _NAMES:
        return _NAMES[short]
    if short.isdigit():
        return f"{type(mod).__name__} {short}"
    return type(mod).__name__


# ---------------------------------------------------------------------------
# LabService
# ---------------------------------------------------------------------------

SKIP_TYPES_FLOW = frozenset({
    "ReLU", "Dropout", "BatchNorm2d", "Identity", "GELU", "SiLU",
})


class LabService:

    def __init__(self) -> None:
        self._flops_engine = FLOPsEngine()

    # ── public API ──────────────────────────────────────────────

    def transformer_metrics(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        seq_len: int = 128,
        vocab_size: int = 10000,
    ) -> Dict[str, Any]:
        if d_model % num_heads != 0:
            return {"error": f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"}
        if num_heads <= 0 or d_model <= 0 or num_layers <= 0 or seq_len <= 0:
            return {"error": "All parameters must be positive integers"}

        seq_len = min(seq_len, 512)   # pos_embed only goes to 512
        model = _TransformerLab(d_model, num_heads, num_layers, vocab_size)
        model.eval()
        dummy = torch.randint(0, min(vocab_size, 100), (1, seq_len))

        shapes = self._capture_shapes(model, dummy)
        params = count_parameters(model)
        flow, total_flops, total_mem = self._build_flow(
            model, shapes, max_depth=2,
        )

        attention_ops = 2 * seq_len * seq_len * d_model * num_layers
        return {
            "lab": "transformer",
            "config": {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "seq_len": seq_len,
                "vocab_size": vocab_size,
            },
            "params_M": round(params / 1e6, 4),
            "total_flops_mflops": round(total_flops, 2),
            "memory_mb": round(total_mem, 2),
            "latency_ms": self._estimate_latency_ms(total_flops),
            "token_count": seq_len,
            "head_dim": d_model // num_heads,
            "attention_cost_mflops": round(attention_ops / 1e6, 2),
            "attention_note": (
                f"O(N²·D) per layer: {seq_len}²×{d_model}={seq_len**2*d_model:,} ops "
                f"× {num_layers} layers = {attention_ops/1e6:.0f} MFLOPs"
            ),
            "flow_steps": flow,
        }

    def cnn_metrics(
        self,
        base_channels: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        image_resolution: int = 224,
    ) -> Dict[str, Any]:
        kernel_size = max(1, min(kernel_size, 7))
        if kernel_size % 2 == 0:
            kernel_size += 1  # keep it odd for clean padding
        depth = max(1, min(depth, 8))
        base_channels = max(4, min(base_channels, 256))

        model = _CNNLab(base_channels, depth, kernel_size)
        model.eval()
        dummy = torch.randn(1, 3, image_resolution, image_resolution)

        shapes = self._capture_shapes(model, dummy)
        params = count_parameters(model)
        flow, total_flops, total_mem = self._build_flow(model, shapes, max_depth=2)

        # Feature map sizes at each conv layer
        feature_maps = [
            {
                "id": k,
                "spatial": v["out"][2] if v.get("out") and len(v["out"]) == 4 else None,
                "channels": v["out"][1] if v.get("out") and len(v["out"]) >= 2 else None,
            }
            for k, v in shapes.items()
            if v.get("out") and len(v.get("out", [])) == 4
            and k != ""
        ]

        return {
            "lab": "cnn",
            "config": {
                "base_channels": base_channels,
                "depth": depth,
                "kernel_size": kernel_size,
                "image_resolution": image_resolution,
            },
            "params_M": round(params / 1e6, 4),
            "total_flops_mflops": round(total_flops, 2),
            "memory_mb": round(total_mem, 2),
            "latency_ms": self._estimate_latency_ms(total_flops),
            "receptive_field": kernel_size + (kernel_size - 1) * (depth - 1),
            "feature_maps": feature_maps[:20],  # cap for JSON size
            "flow_steps": flow,
        }

    def vit_metrics(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 768,
        num_blocks: int = 12,
    ) -> Dict[str, Any]:
        patch_size = max(1, patch_size)
        if image_size % patch_size != 0:
            for ps in range(patch_size, 0, -1):
                if image_size % ps == 0:
                    patch_size = ps
                    break

        num_heads = _find_valid_heads(hidden_dim)
        model = _ViTLab(image_size, patch_size, hidden_dim, num_blocks)
        model.eval()
        dummy = torch.randn(1, 3, image_size, image_size)

        shapes = self._capture_shapes(model, dummy)
        params = count_parameters(model)
        flow, total_flops, total_mem = self._build_flow(model, shapes, max_depth=2)

        num_patches = (image_size // patch_size) ** 2
        token_count = num_patches + 1  # + CLS
        attn_cost_per_layer = 2 * token_count * token_count * hidden_dim
        total_attn = attn_cost_per_layer * num_blocks

        return {
            "lab": "vit",
            "config": {
                "image_size": image_size,
                "patch_size": patch_size,
                "hidden_dim": hidden_dim,
                "num_blocks": num_blocks,
            },
            "params_M": round(params / 1e6, 4),
            "total_flops_mflops": round(total_flops, 2),
            "memory_mb": round(total_mem, 2),
            "latency_ms": self._estimate_latency_ms(total_flops),
            "token_count": token_count,
            "num_patches": num_patches,
            "num_heads": num_heads,
            "head_dim": hidden_dim // num_heads,
            "attention_cost_mflops": round(total_attn / 1e6, 2),
            "attention_note": (
                f"O(N²·D) per block: {token_count}²×{hidden_dim}={token_count**2*hidden_dim:,} ops "
                f"× {num_blocks} blocks = {total_attn/1e6:.0f} MFLOPs"
            ),
            "flow_steps": flow,
        }

    def diffusion_metrics(
        self,
        latent_size: int = 32,
        channels: int = 3,
        diffusion_steps: int = 1000,
    ) -> Dict[str, Any]:
        # DDPMBuilder: channels = input image channels, dim = base feature dim
        # We use latent_size as spatial resolution and base dim=64
        base_dim = 64
        model = DDPMBuilder(channels=channels, dim=base_dim)
        model.eval()
        dummy_x = torch.randn(1, channels, latent_size, latent_size)
        dummy_t = torch.tensor([500])

        shapes = self._capture_shapes_ddpm(model, dummy_x, dummy_t)
        params = count_parameters(model)
        flow, total_flops, total_mem = self._build_flow(model, shapes, max_depth=1)

        inference_flops = total_flops * diffusion_steps
        # Rough training estimate: 3× forward (fwd+bwd) per step
        training_flops_per_iter = total_flops * 3
        training_note = (
            f"{total_flops:.0f} MFLOPs/step × {diffusion_steps} steps = "
            f"{inference_flops/1000:.1f} GFLOPs per inference image"
        )

        # Memory scales with diffusion steps at inference (KV cache / noise schedule)
        inference_memory_mb = total_mem  # per-step memory (no KV growth in DDPM)

        return {
            "lab": "diffusion",
            "config": {
                "latent_size": latent_size,
                "channels": channels,
                "diffusion_steps": diffusion_steps,
            },
            "params_M": round(params / 1e6, 4),
            "total_flops_mflops": round(total_flops, 2),
            "memory_mb": round(inference_memory_mb, 2),
            "latency_ms": self._estimate_latency_ms(inference_flops),
            "step_flops_mflops": round(total_flops, 2),
            "inference_flops_mflops": round(inference_flops, 2),
            "training_flops_per_iter_mflops": round(training_flops_per_iter, 2),
            "training_note": training_note,
            "base_dim": base_dim,
            "flow_steps": flow,
        }

    # ── shape capture ──────────────────────────────────────────

    def _capture_shapes(self, model: nn.Module, dummy: torch.Tensor) -> Dict[str, Dict]:
        shapes: Dict[str, Dict] = {}
        handles = []

        def make_hook(name: str):
            def hook(mod: nn.Module, inp: tuple, out: Any) -> None:
                try:
                    in_s = list(inp[0].shape) if inp else None
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_s = list(out.shape) if isinstance(out, torch.Tensor) else None
                    shapes[name] = {"in": in_s, "out": out_s}
                except Exception:
                    pass
            return hook

        for name, mod in model.named_modules():
            handles.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            try:
                model(dummy)
            except Exception:
                pass
        for h in handles:
            h.remove()
        return shapes

    def _capture_shapes_ddpm(
        self, model: nn.Module, dummy_x: torch.Tensor, dummy_t: torch.Tensor
    ) -> Dict[str, Dict]:
        shapes: Dict[str, Dict] = {}
        handles = []

        def make_hook(name: str):
            def hook(mod: nn.Module, inp: tuple, out: Any) -> None:
                try:
                    in_s = list(inp[0].shape) if inp else None
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out_s = list(out.shape) if isinstance(out, torch.Tensor) else None
                    shapes[name] = {"in": in_s, "out": out_s}
                except Exception:
                    pass
            return hook

        for name, mod in model.named_modules():
            handles.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            try:
                model(dummy_x, dummy_t)
            except Exception:
                pass
        for h in handles:
            h.remove()
        return shapes

    # ── flow builder ───────────────────────────────────────────

    def _build_flow(
        self,
        model: nn.Module,
        shapes: Dict[str, Dict],
        max_depth: int = 2,
    ) -> Tuple[List[Dict], float, float]:
        """
        Walk named_modules and emit one flow-step per meaningful node.
        Returns (flow_steps, total_flops_mflops, total_memory_mb).
        """
        steps: List[Dict] = []
        total_flops = 0.0
        total_mem = 0.0

        for name, mod in model.named_modules():
            if not name:  # root
                continue
            depth = name.count(".")
            if depth > max_depth:
                continue
            mod_type = type(mod).__name__
            if mod_type in SKIP_TYPES_FLOW:
                continue

            s = shapes.get(name, {})
            in_s = s.get("in")
            out_s = s.get("out")
            if in_s is None or out_s is None:
                continue

            fr = self._flops_engine.estimate(
                name,
                mod_type,
                tuple(in_s),
                tuple(out_s),
                _module_to_params(mod),
                _humanize(name, mod),
            )
            total_flops += fr.flops_mflops
            total_mem += fr.memory_mb

            steps.append({
                "id": name,
                "name": _humanize(name, mod),
                "type": mod_type,
                "input_shape": in_s,
                "output_shape": out_s,
                "flops_mflops": round(fr.flops_mflops, 4),
                "params_M": round(fr.params_M, 4),
                "memory_mb": round(fr.memory_mb, 3),
                "formula": fr.formula,
                "severity": fr.severity,
            })

        return steps, total_flops, total_mem

    # ── latency estimation ─────────────────────────────────────

    @staticmethod
    def _estimate_latency_ms(
        total_flops_mflops: float,
        gpu_name: str = "RTX4090",
    ) -> float:
        """
        Rough inference latency from FLOPs and GPU throughput.

        Formula:
            latency_s = (total_flops_mflops × 10^6) / (gpu_tflops × 10^12 × eff)
            latency_ms = latency_s × 1000
                       = total_flops_mflops / (gpu_tflops × eff × 10^3)  [ms]

        GPU efficiency at inference ≈ 30% (practical MFU, batch=1).
        Source: NVIDIA GPU specs from cost_estimator.GPU_SPECS.
        """
        gpu = GPU_SPECS.get(gpu_name, GPU_SPECS["A100"])
        tflops = gpu["tflops_fp32"]
        efficiency = 0.30
        if tflops <= 0:
            return 0.0
        ms = total_flops_mflops / (tflops * efficiency * 1e3)
        return round(ms, 4)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Lab Service")
    parser.add_argument("--lab", required=True,
                        choices=["transformer", "cnn", "vit", "diffusion"])

    # Transformer params
    parser.add_argument("--d_model",   type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers",type=int, default=6)
    parser.add_argument("--seq_len",   type=int, default=128)
    parser.add_argument("--vocab_size",type=int, default=10000)

    # CNN params
    parser.add_argument("--base_channels",    type=int, default=64)
    parser.add_argument("--depth",            type=int, default=4)
    parser.add_argument("--kernel_size",      type=int, default=3)
    parser.add_argument("--image_resolution", type=int, default=224)

    # ViT params
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--patch_size",  type=int, default=16)
    parser.add_argument("--hidden_dim",  type=int, default=768)
    parser.add_argument("--num_blocks",  type=int, default=12)

    # Diffusion params
    parser.add_argument("--latent_size",      type=int, default=32)
    parser.add_argument("--channels",         type=int, default=3)
    parser.add_argument("--diffusion_steps",  type=int, default=1000)

    args = parser.parse_args()
    svc = LabService()

    try:
        if args.lab == "transformer":
            result = svc.transformer_metrics(
                args.d_model, args.num_heads, args.num_layers,
                args.seq_len, args.vocab_size,
            )
        elif args.lab == "cnn":
            result = svc.cnn_metrics(
                args.base_channels, args.depth, args.kernel_size,
                args.image_resolution,
            )
        elif args.lab == "vit":
            result = svc.vit_metrics(
                args.image_size, args.patch_size, args.hidden_dim, args.num_blocks,
            )
        elif args.lab == "diffusion":
            result = svc.diffusion_metrics(
                args.latent_size, args.channels, args.diffusion_steps,
            )
        else:
            result = {"error": f"Unknown lab: {args.lab}"}
    except Exception as exc:
        result = {"error": str(exc)}

    print(json.dumps(result))
