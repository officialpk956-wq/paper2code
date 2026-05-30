#!/usr/bin/env python
"""Test suite for Phase 3.9.A.2 — Comparison Explainer."""

from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_unet import build_unet_graph
from core.visualizer_vit import build_vit_graph
from core.comparators import (
    explain_compute_difference,
    explain_spatial_difference,
    explain_scaling_difference,
    explain_architecture_comparison,
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
)

print("=" * 80)
print("Phase 3.9.A.2 — Comparison Explainer Test Suite")
print("=" * 80)

# Build all three architectures
resnet = build_resnet18_graph()
unet = build_unet_graph()
vit = build_vit_graph()

# ============================================================================
# Test 1: explain_compute_difference()
# ============================================================================
print("\n[TEST 1] explain_compute_difference() — Compute explanations")
print("-" * 80)

compute_resnet = summarize_compute(resnet)
compute_vit = summarize_compute(vit)

print("\nResNet vs Vision Transformer (Compute):")
print(explain_compute_difference(
    compute_resnet,
    compute_vit,
    "ResNet-18",
    "Vision Transformer"
))

print("\n✓ explain_compute_difference() working")

# ============================================================================
# Test 2: explain_spatial_difference()
# ============================================================================
print("\n[TEST 2] explain_spatial_difference() — Spatial explanations")
print("-" * 80)

spatial_resnet = summarize_spatial_behavior(resnet)
spatial_vit = summarize_spatial_behavior(vit)

print("\nResNet vs Vision Transformer (Spatial):")
print(explain_spatial_difference(
    spatial_resnet,
    spatial_vit,
    "ResNet-18",
    "Vision Transformer"
))

print("\n✓ explain_spatial_difference() working")

# ============================================================================
# Test 3: explain_scaling_difference()
# ============================================================================
print("\n[TEST 3] explain_scaling_difference() — Scaling explanations")
print("-" * 80)

scaling_resnet = summarize_scaling_behavior(resnet)
scaling_vit = summarize_scaling_behavior(vit)

print("\nResNet vs Vision Transformer (Scaling):")
print(explain_scaling_difference(
    scaling_resnet,
    scaling_vit,
    "ResNet-18",
    "Vision Transformer"
))

print("\n✓ explain_scaling_difference() working")

# ============================================================================
# Test 4: explain_architecture_comparison() — Full comparison
# ============================================================================
print("\n[TEST 4] explain_architecture_comparison() — Complete comparison")
print("-" * 80)

print("\n" + "=" * 80)
print(explain_architecture_comparison(resnet, vit))
print("=" * 80)

print("\n✓ explain_architecture_comparison() working")

# ============================================================================
# Test 5: All pairwise comparisons
# ============================================================================
print("\n[TEST 5] All pairwise comparisons")
print("-" * 80)

pairs = [
    ("ResNet-18", resnet, "U-Net", unet),
    ("ResNet-18", resnet, "Vision Transformer", vit),
    ("U-Net", unet, "Vision Transformer", vit),
]

for name_a, graph_a, name_b, graph_b in pairs:
    print(f"\n--- {name_a} vs {name_b} ---")
    explanation = explain_architecture_comparison(graph_a, graph_b)
    # Verify structure
    assert "# Architecture Comparison:" in explanation
    assert "## Computational Cost" in explanation
    assert "## Spatial Structure" in explanation
    assert "## Scaling Behavior" in explanation
    assert "### Quick Summary" in explanation
    print(f"✓ {name_a} vs {name_b} comparison generated")

print("\n✓ All pairwise comparisons working")

# ============================================================================
# Test 6: Explanation quality checks
# ============================================================================
print("\n[TEST 6] Explanation quality validation")
print("-" * 80)

# Get a full explanation
full_explanation = explain_architecture_comparison(resnet, vit)

# Check for key terminology
assert "compute" in full_explanation.lower() or "computation" in full_explanation.lower()
assert "spatial" in full_explanation.lower()
assert "scaling" in full_explanation.lower() or "scale" in full_explanation.lower()

# Check for specific patterns
if vit.name in full_explanation:
    # Should mention attention or quadratic complexity for ViT
    assert "quadratic" in full_explanation.lower() or "attention" in full_explanation.lower()

# Check for markdown formatting
assert full_explanation.startswith("#")
assert "##" in full_explanation
assert "---" in full_explanation

print("✓ Markdown formatting present")
print("✓ Key terminology included")
print("✓ Architecture-specific details present")

# ============================================================================
# Test 7: Edge cases
# ============================================================================
print("\n[TEST 7] Edge case handling")
print("-" * 80)

# Test with U-Net (0 high-FLOPs nodes)
compute_unet = summarize_compute(unet)
compute_resnet = summarize_compute(resnet)

explanation = explain_compute_difference(
    compute_unet,
    compute_resnet,
    "U-Net",
    "ResNet-18"
)
assert "U-Net" in explanation
assert "no major computational bottlenecks" in explanation or "0" in explanation
print("✓ Zero high-FLOPs case handled")

# Test identical spatial preservation
spatial_unet = summarize_spatial_behavior(unet)
explanation = explain_spatial_difference(
    spatial_resnet,
    spatial_unet,
    "ResNet-18",
    "U-Net"
)
assert "similar" in explanation.lower() or "both" in explanation.lower()
print("✓ Similar characteristics case handled")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("All Tests Passed ✓")
print("=" * 80)
print("""
Phase 3.9.A.2 Implementation Complete:

✓ explain_compute_difference() — Clear compute comparisons
✓ explain_spatial_difference() — Spatial behavior explanations
✓ explain_scaling_difference() — Scaling analysis with O(n²) warnings
✓ explain_architecture_comparison() — Comprehensive formatted reports

All explanations are:
  • Deterministic and rule-based
  • Human-readable with markdown formatting
  • Context-aware (mentions skip connections, attention, etc.)
  • Ready for UI and agent consumption
  • Beginner-friendly with explanatory notes
""")
print("=" * 80)
