#!/usr/bin/env python
"""Edge case and robustness tests for architecture comparator."""

from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_unet import build_unet_graph
from core.visualizer_vit import build_vit_graph
from core.comparators import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
)

print("Edge Case & Robustness Tests")
print("=" * 70)

# Test all architectures
architectures = [
    ("ResNet-18", build_resnet18_graph()),
    ("U-Net", build_unet_graph()),
    ("Vision Transformer", build_vit_graph()),
]

print("\n[Test 1] All functions return proper dict structure")
print("-" * 70)
for name, graph in architectures:
    compute = summarize_compute(graph)
    spatial = summarize_spatial_behavior(graph)
    scaling = summarize_scaling_behavior(graph)
    
    # Validate compute dict
    assert "high_flops_nodes" in compute
    assert "total_high_flops" in compute
    assert "primary_bottleneck" in compute
    assert isinstance(compute["high_flops_nodes"], list)
    assert isinstance(compute["total_high_flops"], int)
    
    # Validate spatial dict
    assert "spatial_preservation" in spatial
    assert "reason" in spatial
    assert spatial["spatial_preservation"] in ["high", "medium", "low"]
    
    # Validate scaling dict
    assert "scaling" in scaling
    assert "reason" in scaling
    assert scaling["scaling"] in ["good", "moderate", "poor"]
    
    print(f"  ✓ {name}: All dicts properly structured")

print("\n[Test 2] Handle architectures with no high-FLOPs nodes")
print("-" * 70)
unet = build_unet_graph()
compute_unet = summarize_compute(unet)
assert compute_unet["total_high_flops"] == 0
assert compute_unet["primary_bottleneck"] is None
print("  ✓ U-Net handled correctly (0 high-FLOPs nodes)")

print("\n[Test 3] Handle architectures with multiple very-high nodes")
print("-" * 70)
vit = build_vit_graph()
compute_vit = summarize_compute(vit)
assert compute_vit["total_high_flops"] > 0
assert compute_vit["primary_bottleneck"] is not None
print(f"  ✓ ViT handled correctly ({compute_vit['total_high_flops']} high-FLOPs nodes)")

print("\n[Test 4] Spatial behavior detection works for all patterns")
print("-" * 70)
# ResNet should be high (CNN-based)
resnet = build_resnet18_graph()
spatial_resnet = summarize_spatial_behavior(resnet)
assert spatial_resnet["spatial_preservation"] == "high"
print("  ✓ ResNet correctly identified as high spatial preservation")

# U-Net should be high (skip connections)
spatial_unet = summarize_spatial_behavior(unet)
assert spatial_unet["spatial_preservation"] == "high"
print("  ✓ U-Net correctly identified as high spatial preservation")

# ViT should be low (token-based)
spatial_vit = summarize_spatial_behavior(vit)
assert spatial_vit["spatial_preservation"] == "low"
print("  ✓ ViT correctly identified as low spatial preservation")

print("\n[Test 5] Scaling behavior detection for quadratic attention")
print("-" * 70)
scaling_vit = summarize_scaling_behavior(vit)
assert scaling_vit["scaling"] == "poor"
assert "quadratic" in scaling_vit["reason"].lower()
print("  ✓ ViT correctly identified as poor scaling (quadratic attention)")

print("\n[Test 6] All summary reasons are non-empty strings")
print("-" * 70)
for name, graph in architectures:
    spatial = summarize_spatial_behavior(graph)
    scaling = summarize_scaling_behavior(graph)
    
    assert isinstance(spatial["reason"], str)
    assert len(spatial["reason"]) > 0
    assert isinstance(scaling["reason"], str)
    assert len(scaling["reason"]) > 0
    
    print(f"  ✓ {name}: All reasons are valid strings")

print("\n[Test 7] Case-insensitive behavior for node types and params")
print("-" * 70)
from core.architecture_graph import ArchitectureGraph, GraphNode

# Create a graph with mixed case properties
g_case = ArchitectureGraph("MixedCase")
g_case.add_node(GraphNode(id="n1", type="CONV2d", label="C1", semantic_params={"flops": "VERY High"}))
g_case.add_node(GraphNode(id="n2", type="conv2D", label="C2", semantic_params={"skip_connection": "YeS"}))
g_case.add_node(GraphNode(id="n3", type="Linear", label="L1", semantic_params={"attention": "QUADRATIC"}))

compute_case = summarize_compute(g_case)
assert compute_case["total_high_flops"] == 1
assert compute_case["high_flops_nodes"] == ["C1"]

spatial_case = summarize_spatial_behavior(g_case)
assert spatial_case["spatial_preservation"] == "high"
assert "skip connection" in spatial_case["reason"].lower()

scaling_case = summarize_scaling_behavior(g_case)
assert scaling_case["scaling"] == "poor"
assert "quadratic" in scaling_case["reason"].lower()

print("  ✓ Mixed case node types and semantic parameters are correctly parsed")

print("\n" + "=" * 70)
print("All Edge Cases Passed ✓")
print("=" * 70)
print("""
The comparator module is:
  • Robust to edge cases
  • Properly typed and structured
  • Deterministic and reliable
  • Ready for production use
""")
