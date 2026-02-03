"""
Test ResNet-18 vs Vision Transformer comparison.

Validates that:
- ViT is more compute-intensive than ResNet
- ResNet preserves spatial structure better than ViT
- ViT scales worse than ResNet
"""

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_vit import build_vit_graph
from src.comparators.architecture_comparator import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
    compare_graphs
)
from src.comparators.comparison_explainer import explain_architecture_comparison


def test_resnet_vs_vit_comparison():
    """Test that ResNet vs ViT comparison produces expected results."""
    
    # Build graphs
    print("Building ResNet-18 graph...")
    resnet_graph = build_resnet18_graph()
    
    print("Building Vision Transformer graph...")
    vit_graph = build_vit_graph()
    
    # Get compute summaries
    print("\nComputing summaries...")
    resnet_compute = summarize_compute(resnet_graph)
    vit_compute = summarize_compute(vit_graph)
    
    print(f"ResNet high-FLOPs operations: {resnet_compute['total_high_flops']}")
    print(f"ViT high-FLOPs operations: {vit_compute['total_high_flops']}")
    
    # Assertion 1: ViT is more compute-intensive
    assert vit_compute["total_high_flops"] > resnet_compute["total_high_flops"], \
        "ViT should have more high-FLOPs operations than ResNet"
    print("✓ ViT is more compute-intensive than ResNet")
    
    # Get spatial summaries
    resnet_spatial = summarize_spatial_behavior(resnet_graph)
    vit_spatial = summarize_spatial_behavior(vit_graph)
    
    print(f"\nResNet spatial preservation: {resnet_spatial['spatial_preservation']}")
    print(f"ViT spatial preservation: {vit_spatial['spatial_preservation']}")
    
    # Assertion 2: ResNet preserves spatial structure better
    spatial_levels = {"high": 3, "medium": 2, "low": 1}
    resnet_spatial_level = spatial_levels.get(resnet_spatial["spatial_preservation"], 0)
    vit_spatial_level = spatial_levels.get(vit_spatial["spatial_preservation"], 0)
    
    assert resnet_spatial_level > vit_spatial_level, \
        "ResNet should preserve spatial structure better than ViT"
    print("✓ ResNet preserves spatial structure better than ViT")
    
    # Get scaling summaries
    resnet_scaling = summarize_scaling_behavior(resnet_graph)
    vit_scaling = summarize_scaling_behavior(vit_graph)
    
    print(f"\nResNet scaling: {resnet_scaling['scaling']}")
    print(f"ViT scaling: {vit_scaling['scaling']}")
    
    # Assertion 3: ViT scales worse
    scaling_levels = {"good": 3, "moderate": 2, "acceptable": 2, "poor": 1}
    resnet_scaling_level = scaling_levels.get(resnet_scaling["scaling"], 0)
    vit_scaling_level = scaling_levels.get(vit_scaling["scaling"], 0)
    
    assert vit_scaling_level < resnet_scaling_level, \
        "ViT should have worse scaling than ResNet"
    print("✓ ViT scales worse than ResNet")
    
    # Generate full comparison
    print("\nGenerating comparison explanation...")
    comparison = compare_graphs(resnet_graph, vit_graph)
    explanation = explain_architecture_comparison(resnet_graph, vit_graph)
    
    print(f"\nComparison summary generated ({len(explanation)} characters)")
    assert len(explanation) > 0, "Explanation should not be empty"
    assert "compute" in explanation.lower(), "Explanation should mention compute"
    assert "spatial" in explanation.lower(), "Explanation should mention spatial"
    
    print("\n" + "="*60)
    print("All assertions passed! ✓")
    print("="*60)
    
    return True


if __name__ == "__main__":
    test_resnet_vs_vit_comparison()
