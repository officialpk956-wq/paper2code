#!/usr/bin/env python3
"""
Block Visualization Service.

Builds a real 3-level expandable hierarchy from actual PyTorch block definitions.

Data sources (real, no mocks):
  - core/blocks_resnet.py        (Bottleneck)
  - core/blocks_transformer.py   (TransformerEncoderBlock)
  - core/blocks_unet.py          (DoubleConv)
  - core/blocks_vit.py           (PatchEmbedding)
  - core/rag/flops_engine.py     (FLOPs computation)

Shape inference: PyTorch forward hooks during a single forward pass with a
dummy input tensor — no hard-coded shape tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from core.blocks_resnet import Bottleneck
from core.blocks_transformer import TransformerEncoderBlock
from core.blocks_unet import DoubleConv
from core.blocks_vit import PatchEmbedding
from core.rag.flops_engine import FLOPsEngine

# ---------------------------------------------------------------------------
# Architecture model definitions (using real blocks from core/blocks_*.py)
# ---------------------------------------------------------------------------


class _ResNet50(nn.Module):
    """ResNet-50 built from core/blocks_resnet.py Bottleneck blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(in_c=64, mid_c=64, n=3, stride=1)
        self.stage2 = self._make_stage(in_c=256, mid_c=128, n=4, stride=2)
        self.stage3 = self._make_stage(in_c=512, mid_c=256, n=6, stride=2)
        self.stage4 = self._make_stage(in_c=1024, mid_c=512, n=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 1000)

    def _make_stage(self, in_c: int, mid_c: int, n: int, stride: int) -> nn.Sequential:
        blocks = [Bottleneck(in_c, mid_c, stride=stride, downsample=True)]
        for _ in range(1, n):
            blocks.append(Bottleneck(mid_c * 4, mid_c))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class _ViTB16(nn.Module):
    """ViT-B/16 built from core/blocks_vit.py + core/blocks_transformer.py."""

    def __init__(self) -> None:
        super().__init__()
        d_model = 768
        self.patch_embed = PatchEmbedding(in_channels=3, patch_size=16, embed_dim=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, d_model))  # 196 patches + 1 CLS
        self.pos_drop = nn.Dropout(0.0)
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    d_model=d_model, num_heads=12, ffn_dim=d_model * 4, dropout=0.0
                )
                for _ in range(12)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)  # (B, 196, 768)
        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)  # (B, 197, 768)
        x = self.pos_drop(x + self.pos_embed)
        x = self.encoder(x)  # (B, 197, 768)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token → (B, 1000)


class _UNet(nn.Module):
    """U-Net built from core/blocks_unet.py DoubleConv blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1_conv = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2_conv = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3_conv = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_conv = DoubleConv(512, 256)  # 256 from up3 + 256 skip
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv = DoubleConv(256, 128)  # 128 from up2 + 128 skip
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv = DoubleConv(128, 64)  # 64 from up1 + 64 skip
        self.out_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1_conv(x)
        e2 = self.enc2_conv(self.pool1(e1))
        e3 = self.enc3_conv(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3_conv(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2_conv(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1_conv(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class _TransformerBase(nn.Module):
    """Transformer-Base (Attention Is All You Need) built from core/blocks_transformer.py."""

    def __init__(self) -> None:
        super().__init__()
        d_model, num_heads, ffn_dim, num_layers = 512, 8, 2048, 6
        self.embedding = nn.Embedding(10000, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        self.encoder_layers = nn.Sequential(
            *[TransformerEncoderBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 10000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) — integer token IDs
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        e = self.embedding(x) + self.pos_embed(positions)
        e = self.encoder_layers(e)
        e = self.norm(e)
        return self.out_proj(e)


# ---------------------------------------------------------------------------
# Types that are "atomic blocks" — shown as Level 2 nodes, not expanded
# ---------------------------------------------------------------------------

ATOMIC_BLOCK_TYPES = (Bottleneck, DoubleConv, TransformerEncoderBlock, PatchEmbedding)


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_ARCH_MAP: dict[str, dict[str, Any]] = {
    "resnet50": {
        "cls": _ResNet50,
        "name": "ResNet-50",
        "description": "Deep Residual Learning for Image Recognition (He et al., 2015). "
        "50-layer network using bottleneck residual blocks.",
        "input_shape": [1, 3, 224, 224],
        "dummy_fn": lambda: torch.randn(1, 3, 224, 224),
        "stage_groups": [
            {
                "id": "stem",
                "name": "Stem",
                "type": "stem",
                "modules": ["stem"],
                "description": "7×7 conv + BN + ReLU + MaxPool. Projects 3-channel "
                "image to 64 feature maps, halving spatial dims twice (224→56).",
            },
            {
                "id": "stage1",
                "name": "Stage 1 (×3)",
                "type": "backbone",
                "modules": ["stage1"],
                "description": "3 Bottleneck blocks. Channels: 64→256 (expansion 4×). "
                "Spatial: 56×56. First block adds projection shortcut.",
            },
            {
                "id": "stage2",
                "name": "Stage 2 (×4)",
                "type": "backbone",
                "modules": ["stage2"],
                "description": "4 Bottleneck blocks. Channels: 128→512. "
                "Spatial: 28×28 (stride-2 at entry).",
            },
            {
                "id": "stage3",
                "name": "Stage 3 (×6)",
                "type": "backbone",
                "modules": ["stage3"],
                "description": "6 Bottleneck blocks. Channels: 256→1024. Spatial: 14×14.",
            },
            {
                "id": "stage4",
                "name": "Stage 4 (×3)",
                "type": "backbone",
                "modules": ["stage4"],
                "description": "3 Bottleneck blocks. Channels: 512→2048. Spatial: 7×7.",
            },
            {
                "id": "head",
                "name": "Head",
                "type": "head",
                "modules": ["avgpool", "fc"],
                "description": "Global average pooling (7×7→1×1) → 2048-d vector → "
                "linear classifier → 1000-class softmax.",
            },
        ],
    },
    "vit": {
        "cls": _ViTB16,
        "name": "ViT-B/16",
        "description": "An Image is Worth 16×16 Words (Dosovitskiy et al., 2020). "
        "Vision Transformer with 12 encoder layers, d=768.",
        "input_shape": [1, 3, 224, 224],
        "dummy_fn": lambda: torch.randn(1, 3, 224, 224),
        "stage_groups": [
            {
                "id": "embedding",
                "name": "Patch Embedding",
                "type": "stem",
                "modules": ["patch_embed", "pos_drop"],
                "description": "16×16 patch projection (Conv2d stride=16) → 196 patches × 768-d. "
                "CLS token prepended; positional embedding added.",
            },
            {
                "id": "encoder",
                "name": "Transformer Encoder (×12)",
                "type": "backbone",
                "modules": ["encoder"],
                "description": "12 identical TransformerEncoderBlock layers. Each: "
                "MHSA (12 heads) + FFN (3072-d hidden) + LayerNorm residuals.",
            },
            {
                "id": "head",
                "name": "Head",
                "type": "head",
                "modules": ["norm", "head"],
                "description": "LayerNorm → CLS token → linear classifier → 1000 classes.",
            },
        ],
    },
    "unet": {
        "cls": _UNet,
        "name": "U-Net",
        "description": "U-Net: Convolutional Networks for Biomedical Image Segmentation "
        "(Ronneberger et al., 2015). Encoder-decoder with skip connections.",
        "input_shape": [1, 1, 256, 256],
        "dummy_fn": lambda: torch.randn(1, 1, 256, 256),
        "stage_groups": [
            {
                "id": "enc1",
                "name": "Encoder 1",
                "type": "encoder",
                "modules": ["enc1_conv", "pool1"],
                "description": "DoubleConv(1→64) + MaxPool×2. Output: 64×128×128.",
            },
            {
                "id": "enc2",
                "name": "Encoder 2",
                "type": "encoder",
                "modules": ["enc2_conv", "pool2"],
                "description": "DoubleConv(64→128) + MaxPool×2. Output: 128×64×64.",
            },
            {
                "id": "enc3",
                "name": "Encoder 3",
                "type": "encoder",
                "modules": ["enc3_conv", "pool3"],
                "description": "DoubleConv(128→256) + MaxPool×2. Output: 256×32×32.",
            },
            {
                "id": "bottleneck",
                "name": "Bottleneck",
                "type": "bottleneck",
                "modules": ["bottleneck"],
                "description": "DoubleConv(256→512) at lowest resolution. "
                "Captures global context. Output: 512×32×32.",
            },
            {
                "id": "dec3",
                "name": "Decoder 3",
                "type": "decoder",
                "modules": ["up3", "dec3_conv"],
                "description": "ConvTranspose×2 + concat skip from Enc3 → DoubleConv(512→256).",
            },
            {
                "id": "dec2",
                "name": "Decoder 2",
                "type": "decoder",
                "modules": ["up2", "dec2_conv"],
                "description": "ConvTranspose×2 + concat skip from Enc2 → DoubleConv(256→128).",
            },
            {
                "id": "dec1",
                "name": "Decoder 1",
                "type": "decoder",
                "modules": ["up1", "dec1_conv"],
                "description": "ConvTranspose×2 + concat skip from Enc1 → DoubleConv(128→64).",
            },
            {
                "id": "output",
                "name": "Output",
                "type": "head",
                "modules": ["out_conv"],
                "description": "1×1 conv → 2 segmentation channels (background / foreground).",
            },
        ],
    },
    "transformer": {
        "cls": _TransformerBase,
        "name": "Transformer-Base",
        "description": "Attention Is All You Need (Vaswani et al., 2017). "
        "6-layer encoder, d=512, 8 heads, FFN=2048.",
        "input_shape": [1, 128],
        "dummy_fn": lambda: torch.randint(0, 10000, (1, 128)),
        "stage_groups": [
            {
                "id": "embedding",
                "name": "Embedding",
                "type": "stem",
                "modules": ["embedding", "pos_embed"],
                "description": "Token embedding (vocab=10K, d=512) + positional embedding. "
                "Sum gives contextualized token representation.",
            },
            {
                "id": "encoder",
                "name": "Encoder Stack (×6)",
                "type": "backbone",
                "modules": ["encoder_layers"],
                "description": "6 TransformerEncoderBlock layers. Each: "
                "MHSA (8 heads, d_k=64) + FFN (2048-d) + LayerNorm + residuals.",
            },
            {
                "id": "head",
                "name": "Head",
                "type": "head",
                "modules": ["norm", "out_proj"],
                "description": "LayerNorm + linear projection → vocabulary logits (10K classes).",
            },
        ],
    },
}

_ARCH_ALIASES = {
    "resnet": "resnet50",
    "resnet-50": "resnet50",
    "deep-residual-learning": "resnet50",
    "he2015": "resnet50",
    "vit-b16": "vit",
    "vit-b/16": "vit",
    "an-image-is-worth-16x16-words": "vit",
    "dosovitskiy2020": "vit",
    "unet-biomedical": "unet",
    "ronneberger2015": "unet",
    "attention-is-all-you-need": "transformer",
    "vaswani2017": "transformer",
}


# ---------------------------------------------------------------------------
# BlockVizService
# ---------------------------------------------------------------------------


class BlockVizService:
    def __init__(self) -> None:
        self._flops_engine = FLOPsEngine()

    # ── public API ─────────────────────────────────────────

    def get_block_hierarchy(self, paper_id: str) -> dict[str, Any]:
        arch_id, cfg = self._resolve(paper_id)
        model = cfg["cls"]()
        model.eval()
        shapes = self._capture_shapes(model, cfg["dummy_fn"]())
        return self._build_hierarchy(arch_id, model, shapes, cfg)

    def get_block_detail(self, paper_id: str, block_id: str) -> dict[str, Any]:
        """Return detailed FLOPs breakdown for a single block (Level 2 node)."""
        arch_id, cfg = self._resolve(paper_id)
        model = cfg["cls"]()
        model.eval()
        shapes = self._capture_shapes(model, cfg["dummy_fn"]())

        # Find the module by block_id path
        mod = _get_module_by_path(model, block_id)
        if mod is None:
            raise ValueError(f"Block '{block_id}' not found in {arch_id}")

        s = shapes.get(block_id, {})
        layers = self._get_layers(mod, shapes, block_id)

        # Use layer-sum FLOPs (consistent with _build_hierarchy); engine as fallback
        block_flops = sum(l["flops_mflops"] for l in layers)
        block_params = sum(l["params_M"] for l in layers)

        fr = self._flops_engine.estimate(
            block_id,
            type(mod).__name__,
            tuple(s.get("in") or ()),
            tuple(s.get("out") or ()),
            _module_to_params(mod),
            _humanize_name(block_id.split(".")[-1], mod),
        )
        if block_flops == 0.0:
            block_flops = fr.flops_mflops
            block_params = fr.params_M

        severity = self._flops_engine._severity(block_flops, fr.memory_mb)

        return {
            "id": block_id,
            "name": _humanize_name(block_id.split(".")[-1], mod),
            "type": type(mod).__name__,
            "input_shape": s.get("in"),
            "output_shape": s.get("out"),
            "flops_mflops": round(block_flops, 4),
            "params_M": round(block_params, 4),
            "formula": fr.formula,
            "complexity": fr.complexity,
            "severity": severity,
            "warnings": fr.warnings,
            "layers": layers,
        }

    def get_animated_forward_pass(self, paper_id: str) -> dict[str, Any]:
        arch_id, cfg = self._resolve(paper_id)
        model = cfg["cls"]()
        model.eval()
        shapes = self._capture_shapes(model, cfg["dummy_fn"]())
        steps = self._build_forward_pass_steps(model, shapes)
        return {
            "paper_id": arch_id,
            "name": cfg["name"],
            "total_steps": len(steps),
            "steps": steps,
        }

    # ── shape capture ──────────────────────────────────────

    def _capture_shapes(self, model: nn.Module, dummy: torch.Tensor) -> dict[str, dict]:
        shapes: dict[str, dict] = {}
        handles = []

        def make_hook(name: str):
            def hook(mod: nn.Module, inp: tuple, out: Any) -> None:
                try:
                    in_s = list(inp[0].shape) if inp else None
                    # MultiheadAttention returns (output, attn_weights) tuple
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

    # ── hierarchy builder ──────────────────────────────────

    def _build_hierarchy(
        self,
        arch_id: str,
        model: nn.Module,
        shapes: dict[str, dict],
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        children_dict = dict(model.named_children())
        stages = []

        for sg in cfg["stage_groups"]:
            mod_names: list[str] = sg["modules"]

            # Stage input/output: first mod's in → last mod's out
            first_in = shapes.get(mod_names[0], {}).get("in")
            last_out = shapes.get(mod_names[-1], {}).get("out")

            # Build Level-2 blocks from this stage's modules
            blocks: list[dict] = []
            for mod_name in mod_names:
                if mod_name not in children_dict:
                    continue
                top_mod = children_dict[mod_name]

                # Expand stage-level containers into individual blocks
                for block_prefix, block_mod in _expand_to_blocks(top_mod, mod_name):
                    b_s = shapes.get(block_prefix, {})
                    layers = self._get_layers(block_mod, shapes, block_prefix)

                    block_flops = sum(l["flops_mflops"] for l in layers)
                    block_params = sum(l["params_M"] for l in layers)

                    # Fallback: estimate directly if no sub-layers captured
                    if block_flops == 0.0:
                        fr = self._flops_engine.estimate(
                            block_prefix,
                            type(block_mod).__name__,
                            tuple(b_s.get("in") or ()),
                            tuple(b_s.get("out") or ()),
                            _module_to_params(block_mod),
                            _humanize_name(block_prefix.split(".")[-1], block_mod),
                        )
                        block_flops = fr.flops_mflops
                        block_params = fr.params_M

                    blocks.append(
                        {
                            "id": block_prefix,
                            "name": _humanize_name(block_prefix.split(".")[-1], block_mod),
                            "type": type(block_mod).__name__,
                            "input_shape": b_s.get("in"),
                            "output_shape": b_s.get("out"),
                            "flops_mflops": round(block_flops, 4),
                            "params_M": round(block_params, 4),
                            "layers": layers,
                        }
                    )

            stage_flops = sum(b["flops_mflops"] for b in blocks)
            stage_params = sum(b["params_M"] for b in blocks)

            stages.append(
                {
                    "id": sg["id"],
                    "name": sg["name"],
                    "type": sg.get("type", "stage"),
                    "description": sg.get("description", ""),
                    "input_shape": first_in,
                    "output_shape": last_out,
                    "flops_mflops": round(stage_flops, 4),
                    "params_M": round(stage_params, 4),
                    "blocks": blocks,
                }
            )

        total_flops = sum(s["flops_mflops"] for s in stages)
        total_params = sum(s["params_M"] for s in stages)

        return {
            "paper_id": arch_id,
            "name": cfg["name"],
            "description": cfg.get("description", ""),
            "input_shape": cfg["input_shape"],
            "total_flops_mflops": round(total_flops, 2),
            "total_params_M": round(total_params, 2),
            "stages": stages,
        }

    # ── layer extraction (Level 3) ─────────────────────────

    def _get_layers(
        self,
        block_mod: nn.Module,
        shapes: dict[str, dict],
        block_prefix: str,
    ) -> list[dict]:
        layers: list[dict] = []

        for child_name, child_mod in block_mod.named_children():
            full_id = f"{block_prefix}.{child_name}"

            if isinstance(child_mod, nn.Sequential):
                # Recurse one level into Sequential wrappers (e.g. DoubleConv.net, ffn)
                for sub_name, sub_mod in child_mod.named_children():
                    sub_id = f"{full_id}.{sub_name}"
                    layers.append(
                        self._make_layer_node(sub_id, f"{child_name}.{sub_name}", sub_mod, shapes)
                    )
            else:
                layers.append(self._make_layer_node(full_id, child_name, child_mod, shapes))

        return layers

    def _make_layer_node(
        self,
        full_id: str,
        display_name: str,
        mod: nn.Module,
        shapes: dict[str, dict],
    ) -> dict:
        s = shapes.get(full_id, {})
        in_s = s.get("in")
        out_s = s.get("out")

        fr = self._flops_engine.estimate(
            full_id,
            type(mod).__name__,
            tuple(in_s) if in_s else (),
            tuple(out_s) if out_s else (),
            _module_to_params(mod),
            display_name,
        )

        return {
            "id": full_id,
            "name": _humanize_name(display_name, mod),
            "type": type(mod).__name__,
            "input_shape": in_s,
            "output_shape": out_s,
            "flops_mflops": round(fr.flops_mflops, 4),
            "params_M": round(fr.params_M, 4),
            "formula": fr.formula,
            "complexity": fr.complexity,
            "severity": fr.severity,
        }

    # ── forward-pass animation steps ──────────────────────

    def _build_forward_pass_steps(
        self,
        model: nn.Module,
        shapes: dict[str, dict],
    ) -> list[dict]:
        SKIP_TYPES = frozenset({"ReLU", "Dropout", "BatchNorm2d", "MaxPool2d"})
        steps = []

        for name, mod in model.named_modules():
            if not name:  # skip root
                continue
            depth = name.count(".")
            if depth > 2:  # skip deeply nested implementation details
                continue
            if type(mod).__name__ in SKIP_TYPES:
                continue

            s = shapes.get(name, {})
            in_s = s.get("in")
            out_s = s.get("out")
            if in_s is None or out_s is None:
                continue

            fr = self._flops_engine.estimate(
                name,
                type(mod).__name__,
                tuple(in_s),
                tuple(out_s),
                _module_to_params(mod),
                name,
            )

            steps.append(
                {
                    "step": len(steps),
                    "node_id": name,
                    "node_name": _humanize_name(name.split(".")[-1], mod),
                    "type": type(mod).__name__,
                    "input_shape": in_s,
                    "output_shape": out_s,
                    "flops_mflops": round(fr.flops_mflops, 4),
                    "params_M": round(fr.params_M, 4),
                    "severity": fr.severity,
                    "description": _describe_layer(mod, in_s, out_s),
                }
            )

        return steps

    # ── private helpers ────────────────────────────────────

    def _resolve(self, paper_id: str) -> tuple[str, dict]:
        pid = paper_id.lower().strip()
        arch_id = _ARCH_ALIASES.get(pid, pid)
        if arch_id not in _ARCH_MAP:
            arch_id = "resnet50"  # safe fallback
        return arch_id, _ARCH_MAP[arch_id]


# ---------------------------------------------------------------------------
# Module utilities
# ---------------------------------------------------------------------------


def _expand_to_blocks(mod: nn.Module, prefix: str) -> list[tuple[str, nn.Module]]:
    """
    Returns (prefix, module) pairs for Level-2 blocks.

    - Atomic block types (Bottleneck, DoubleConv, etc.) → returned as-is
    - Sequential containers → expanded into their direct children
    - Leaf modules (no children) → returned as-is
    """
    if isinstance(mod, ATOMIC_BLOCK_TYPES):
        return [(prefix, mod)]
    children = list(mod.named_children())
    if not children:
        return [(prefix, mod)]  # leaf
    if isinstance(mod, nn.Sequential):
        return [(f"{prefix}.{k}", v) for k, v in children]
    return [(f"{prefix}.{k}", v) for k, v in children]


def _get_module_by_path(model: nn.Module, path: str) -> nn.Module | None:
    """Navigate to a sub-module by dot-separated path."""
    parts = path.split(".")
    cur = model
    for p in parts:
        if hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            # Try integer index for Sequential
            try:
                cur = cur[int(p)]
            except (TypeError, IndexError, ValueError):
                return None
    return cur


def _module_to_params(mod: nn.Module) -> dict[str, Any]:
    """Extract numeric params dict from a PyTorch module for FLOPsEngine."""
    if isinstance(mod, nn.Conv2d):
        k = mod.kernel_size
        s = mod.stride
        p = mod.padding
        return {
            "in_channels": mod.in_channels,
            "out_channels": mod.out_channels,
            "kernel_size": k[0] if isinstance(k, tuple) else k,
            "stride": s[0] if isinstance(s, tuple) else s,
            "padding": p[0] if isinstance(p, tuple) else p,
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
    if isinstance(mod, nn.Embedding):
        return {"vocab_size": mod.num_embeddings, "embed_dim": mod.embedding_dim}
    return {}


_DISPLAY_NAMES: dict[str, str] = {
    "conv1": "Conv 1×1",
    "conv2": "Conv 3×3",
    "conv3": "Conv 1×1 (expand)",
    "bn1": "BatchNorm",
    "bn2": "BatchNorm",
    "bn3": "BatchNorm",
    "relu": "ReLU",
    "norm1": "LayerNorm",
    "norm2": "LayerNorm",
    "norm": "LayerNorm",
    "attn": "Multi-Head Attention",
    "dropout": "Dropout",
    "head": "Classifier",
    "patch_embed": "Patch Embedding",
    "pos_drop": "Pos Dropout",
    "enc1_conv": "Encoder Block 1",
    "enc2_conv": "Encoder Block 2",
    "enc3_conv": "Encoder Block 3",
    "bottleneck": "Bottleneck Block",
    "dec1_conv": "Decoder Block 1",
    "dec2_conv": "Decoder Block 2",
    "dec3_conv": "Decoder Block 3",
    "pool1": "MaxPool ×2",
    "pool2": "MaxPool ×2",
    "pool3": "MaxPool ×2",
    "up1": "Upsample ×2",
    "up2": "Upsample ×2",
    "up3": "Upsample ×2",
    "out_conv": "Output Conv 1×1",
    "fc": "Linear Classifier",
    "avgpool": "Global Avg Pool",
    "stem": "Stem",
    "stage1": "Stage 1",
    "stage2": "Stage 2",
    "stage3": "Stage 3",
    "stage4": "Stage 4",
    "embedding": "Token Embedding",
    "pos_embed": "Positional Embedding",
    "encoder_layers": "Encoder Stack",
    "encoder": "Encoder Stack",
    "out_proj": "Output Projection",
    "ffn": "FFN",
    "net.0": "Conv 3×3",
    "net.1": "ReLU",
    "net.2": "Conv 3×3",
    "net.3": "ReLU",
    "ffn.0": "FFN Linear (up)",
    "ffn.1": "ReLU",
    "ffn.2": "FFN Linear (down)",
    "downsample.0": "Shortcut Conv 1×1",
    "downsample.1": "Shortcut BN",
}


def _humanize_name(raw_name: str, mod: nn.Module) -> str:
    if raw_name in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[raw_name]
    # Numeric sequential index → "TypeName N"
    if raw_name.isdigit():
        return f"{type(mod).__name__} {raw_name}"
    return type(mod).__name__


def _describe_layer(mod: nn.Module, in_s: list, out_s: list) -> str:
    if isinstance(mod, nn.Conv2d):
        k = mod.kernel_size
        k_str = f"{k[0]}×{k[1]}" if isinstance(k, tuple) else f"{k}×{k}"
        return (
            f"{k_str} convolution — {mod.in_channels}→{mod.out_channels} channels, output: {out_s}"
        )
    if isinstance(mod, nn.ConvTranspose2d):
        return f"Transposed conv — doubles spatial dims: {in_s} → {out_s}"
    if isinstance(mod, nn.Linear):
        return f"Linear: {mod.in_features}→{mod.out_features} features"
    if isinstance(mod, nn.MultiheadAttention):
        return f"MHSA: {mod.num_heads} heads, d_model={mod.embed_dim} — {in_s} → {out_s}"
    if isinstance(mod, nn.MaxPool2d):
        return "Max pooling — halves spatial dimensions"
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return "Global average pooling — compresses spatial dims to 1×1"
    if isinstance(mod, nn.Embedding):
        return f"Token lookup: vocab={mod.num_embeddings}, d={mod.embedding_dim}"
    if isinstance(mod, nn.LayerNorm):
        return f"Layer normalization — stabilizes training: {in_s} → {out_s}"
    return f"{type(mod).__name__}: {in_s} → {out_s}"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block Visualization Service")
    parser.add_argument(
        "--architecture",
        required=True,
        help="Architecture ID: resnet50 | vit | unet | transformer (or alias)",
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["hierarchy", "detail", "forward-pass"],
        help="Action to perform",
    )
    parser.add_argument(
        "--block-id",
        default="",
        help="Block ID for --action detail",
    )
    args = parser.parse_args()

    svc = BlockVizService()
    try:
        if args.action == "hierarchy":
            result = svc.get_block_hierarchy(args.architecture)
        elif args.action == "forward-pass":
            result = svc.get_animated_forward_pass(args.architecture)
        elif args.action == "detail":
            if not args.block_id:
                print(json.dumps({"error": "--block-id required for detail action"}))
                sys.exit(1)
            result = svc.get_block_detail(args.architecture, args.block_id)
        else:
            result = {"error": f"Unknown action: {args.action}"}
    except Exception as e:
        result = {"error": str(e)}

    print(json.dumps(result))
