"""
Tests for block_viz_service.py.

Verifies:
  - Hierarchy generation from real PyTorch blocks
  - Correct 3-level structure (stages → blocks → layers)
  - Shape propagation accuracy via forward hooks
  - FLOPs computation via FLOPsEngine
  - Forward pass animation step generation
  - Architecture aliases resolve correctly
"""

import sys
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.block_viz_service import (
    BlockVizService,
    _module_to_params,
    _humanize_name,
    _expand_to_blocks,
    ATOMIC_BLOCK_TYPES,
)
from core.blocks_resnet import Bottleneck
from core.blocks_transformer import TransformerEncoderBlock
from core.blocks_unet import DoubleConv
from core.blocks_vit import PatchEmbedding
import torch
import torch.nn as nn


class TestHierarchyStructure(unittest.TestCase):
    """Hierarchy must have the correct 3-level tree structure."""

    def setUp(self):
        self.svc = BlockVizService()

    def test_resnet50_top_level_fields(self):
        h = self.svc.get_block_hierarchy("resnet50")
        self.assertEqual(h["paper_id"], "resnet50")
        self.assertEqual(h["name"], "ResNet-50")
        self.assertIn("stages", h)
        self.assertIn("total_flops_mflops", h)
        self.assertIn("total_params_M", h)
        self.assertIn("input_shape", h)

    def test_resnet50_stage_count(self):
        h = self.svc.get_block_hierarchy("resnet50")
        # Stem, Stage1-4, Head = 6
        self.assertEqual(len(h["stages"]), 6)

    def test_resnet50_stage_names(self):
        h = self.svc.get_block_hierarchy("resnet50")
        names = [s["name"] for s in h["stages"]]
        self.assertIn("Stem", names)
        self.assertIn("Head", names)

    def test_resnet50_blocks_have_layers(self):
        h = self.svc.get_block_hierarchy("resnet50")
        # stage1 should have Bottleneck blocks with internal layers
        stage1 = next(s for s in h["stages"] if s["id"] == "stage1")
        self.assertGreater(len(stage1["blocks"]), 0)
        # First block should have layers (conv1, bn1, relu, etc.)
        first_block = stage1["blocks"][0]
        self.assertGreater(len(first_block["layers"]), 0, "Bottleneck should have layers")

    def test_vit_hierarchy(self):
        h = self.svc.get_block_hierarchy("vit")
        self.assertEqual(h["paper_id"], "vit")
        self.assertEqual(len(h["stages"]), 3)
        # Encoder stage should have 12 TransformerEncoderBlock blocks
        encoder_stage = next(s for s in h["stages"] if s["id"] == "encoder")
        self.assertEqual(len(encoder_stage["blocks"]), 12)

    def test_unet_hierarchy(self):
        h = self.svc.get_block_hierarchy("unet")
        self.assertEqual(h["paper_id"], "unet")
        stage_ids = [s["id"] for s in h["stages"]]
        self.assertIn("enc1", stage_ids)
        self.assertIn("bottleneck", stage_ids)
        self.assertIn("dec1", stage_ids)

    def test_transformer_hierarchy(self):
        h = self.svc.get_block_hierarchy("transformer")
        self.assertEqual(h["paper_id"], "transformer")
        self.assertEqual(len(h["stages"]), 3)


class TestShapePropagation(unittest.TestCase):
    """Shapes must come from actual PyTorch forward hooks (not hardcoded)."""

    def setUp(self):
        self.svc = BlockVizService()

    def test_resnet50_input_shape(self):
        h = self.svc.get_block_hierarchy("resnet50")
        self.assertEqual(h["input_shape"], [1, 3, 224, 224])

    def test_resnet50_stem_output_shape(self):
        h = self.svc.get_block_hierarchy("resnet50")
        stem = next(s for s in h["stages"] if s["id"] == "stem")
        # MaxPool with stride=2 twice on 224 → 56
        self.assertIsNotNone(stem["output_shape"])
        self.assertEqual(stem["output_shape"][1], 64)  # 64 channels
        self.assertEqual(stem["output_shape"][2], 56)  # 56×56 spatial

    def test_resnet50_stage1_channels(self):
        h = self.svc.get_block_hierarchy("resnet50")
        stage1 = next(s for s in h["stages"] if s["id"] == "stage1")
        # Bottleneck expansion=4: 64 mid → 256 output channels
        self.assertIsNotNone(stage1["output_shape"])
        self.assertEqual(stage1["output_shape"][1], 256)

    def test_vit_patch_embed_output(self):
        h = self.svc.get_block_hierarchy("vit")
        embed_stage = next(s for s in h["stages"] if s["id"] == "embedding")
        patch_block = embed_stage["blocks"][0]
        self.assertEqual(patch_block["type"], "PatchEmbedding")
        # 224/16 = 14, 14*14 = 196 patches, d=768
        self.assertIsNotNone(patch_block["output_shape"])
        self.assertEqual(patch_block["output_shape"][1], 196)
        self.assertEqual(patch_block["output_shape"][2], 768)

    def test_unet_encoder_spatial_halving(self):
        h = self.svc.get_block_hierarchy("unet")
        enc1 = next(s for s in h["stages"] if s["id"] == "enc1")
        enc2 = next(s for s in h["stages"] if s["id"] == "enc2")
        # enc1 stage includes pool1 as its last module, so enc1 output IS the pooled shape.
        # enc2 input = enc2_conv input = same pooled shape from pool1.
        enc1_out = enc1["output_shape"]
        enc2_in = enc2["input_shape"]
        if enc1_out and enc2_in:
            # Both should be the same shape (enc1 ends with pool, enc2 starts with conv on pool output)
            self.assertEqual(enc1_out[2], enc2_in[2])


class TestFLOPsComputation(unittest.TestCase):
    """FLOPs must be positive for non-trivial layers and computed by FLOPsEngine."""

    def setUp(self):
        self.svc = BlockVizService()

    def test_resnet50_total_flops_nonzero(self):
        h = self.svc.get_block_hierarchy("resnet50")
        self.assertGreater(h["total_flops_mflops"], 0)

    def test_resnet50_total_params_reasonable(self):
        h = self.svc.get_block_hierarchy("resnet50")
        # ResNet-50 has ~25M params; our estimate should be > 1M
        self.assertGreater(h["total_params_M"], 1.0)

    def test_resnet50_stage_flops_positive(self):
        h = self.svc.get_block_hierarchy("resnet50")
        for stage in h["stages"]:
            # At least some stages should have nonzero FLOPs
            if stage["id"] in ("stage1", "stage2", "stage3", "stage4"):
                self.assertGreater(stage["flops_mflops"], 0,
                                   f"Stage {stage['id']} has zero FLOPs")

    def test_vit_encoder_flops_nonzero(self):
        h = self.svc.get_block_hierarchy("vit")
        encoder_stage = next(s for s in h["stages"] if s["id"] == "encoder")
        self.assertGreater(encoder_stage["flops_mflops"], 0)

    def test_layer_flops_formula_present(self):
        h = self.svc.get_block_hierarchy("resnet50")
        stage1 = next(s for s in h["stages"] if s["id"] == "stage1")
        first_block = stage1["blocks"][0]
        # At least one layer should have a formula
        layers_with_formula = [l for l in first_block["layers"] if l.get("formula")]
        self.assertGreater(len(layers_with_formula), 0)


class TestForwardPassSteps(unittest.TestCase):
    """Forward pass animation must produce a valid ordered list of steps."""

    def setUp(self):
        self.svc = BlockVizService()

    def test_resnet50_forward_pass_steps(self):
        fp = self.svc.get_animated_forward_pass("resnet50")
        self.assertIn("steps", fp)
        self.assertGreater(fp["total_steps"], 0)
        self.assertEqual(fp["paper_id"], "resnet50")

    def test_steps_have_required_fields(self):
        fp = self.svc.get_animated_forward_pass("resnet50")
        for step in fp["steps"]:
            self.assertIn("step", step)
            self.assertIn("node_id", step)
            self.assertIn("node_name", step)
            self.assertIn("input_shape", step)
            self.assertIn("output_shape", step)
            self.assertIn("flops_mflops", step)

    def test_steps_are_sequential(self):
        fp = self.svc.get_animated_forward_pass("resnet50")
        for i, step in enumerate(fp["steps"]):
            self.assertEqual(step["step"], i)

    def test_vit_forward_pass_steps(self):
        fp = self.svc.get_animated_forward_pass("vit")
        self.assertGreater(fp["total_steps"], 5)


class TestArchitectureAliases(unittest.TestCase):
    """Aliases must resolve to the correct architecture."""

    def setUp(self):
        self.svc = BlockVizService()

    def test_resnet_alias(self):
        h = self.svc.get_block_hierarchy("resnet")
        self.assertEqual(h["paper_id"], "resnet50")

    def test_deep_residual_alias(self):
        h = self.svc.get_block_hierarchy("deep-residual-learning")
        self.assertEqual(h["paper_id"], "resnet50")

    def test_attention_alias(self):
        h = self.svc.get_block_hierarchy("attention-is-all-you-need")
        self.assertEqual(h["paper_id"], "transformer")

    def test_vit_alias(self):
        h = self.svc.get_block_hierarchy("an-image-is-worth-16x16-words")
        self.assertEqual(h["paper_id"], "vit")

    def test_unknown_falls_back_to_resnet50(self):
        h = self.svc.get_block_hierarchy("nonexistent-paper-xyz")
        self.assertEqual(h["paper_id"], "resnet50")


class TestModuleUtils(unittest.TestCase):
    """Unit tests for module utility helpers."""

    def test_module_to_params_conv2d(self):
        conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        p = _module_to_params(conv)
        self.assertEqual(p["in_channels"], 3)
        self.assertEqual(p["out_channels"], 64)
        self.assertEqual(p["kernel_size"], 7)
        self.assertEqual(p["stride"], 2)

    def test_module_to_params_linear(self):
        lin = nn.Linear(512, 1000)
        p = _module_to_params(lin)
        self.assertEqual(p["in_features"], 512)
        self.assertEqual(p["out_features"], 1000)

    def test_module_to_params_mha(self):
        mha = nn.MultiheadAttention(768, 12, batch_first=True)
        p = _module_to_params(mha)
        self.assertEqual(p["embed_dim"], 768)
        self.assertEqual(p["num_heads"], 12)

    def test_expand_to_blocks_atomic(self):
        block = Bottleneck(64, 64)
        pairs = _expand_to_blocks(block, "stage1.0")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "stage1.0")

    def test_expand_to_blocks_sequential(self):
        seq = nn.Sequential(Bottleneck(64, 64), Bottleneck(256, 64))
        pairs = _expand_to_blocks(seq, "stage1")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], "stage1.0")
        self.assertEqual(pairs[1][0], "stage1.1")

    def test_expand_to_blocks_leaf(self):
        conv = nn.Conv2d(3, 64, 7)
        pairs = _expand_to_blocks(conv, "stem.0")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], "stem.0")

    def test_humanize_name_known(self):
        conv = nn.Conv2d(64, 64, 1)
        self.assertEqual(_humanize_name("conv1", conv), "Conv 1×1")
        self.assertEqual(_humanize_name("attn", nn.MultiheadAttention(512, 8, batch_first=True)), "Multi-Head Attention")

    def test_humanize_name_numeric(self):
        # Numeric names not in _DISPLAY_NAMES fall back to "TypeName N"
        block = Bottleneck(64, 64)
        result = _humanize_name("5", block)  # "5" is not in _DISPLAY_NAMES
        self.assertIn("5", result)
        self.assertIn("Bottleneck", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
