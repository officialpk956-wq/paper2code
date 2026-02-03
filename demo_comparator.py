#!/usr/bin/env python
"""Quick demo of architecture comparison engine."""

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_vit import build_vit_graph
from src.comparators import compare_graphs

# Compare ResNet vs Vision Transformer
resnet = build_resnet18_graph()
vit = build_vit_graph()

result = compare_graphs(resnet, vit)

print("Architecture Comparison: ResNet-18 vs Vision Transformer")
print("=" * 70)

print(f"\n{result['graph_a']['name']}:")
print(f"  Computational bottleneck: {result['graph_a']['compute']['primary_bottleneck']}")
print(f"  Spatial preservation: {result['graph_a']['spatial']['spatial_preservation']}")
print(f"  Scaling behavior: {result['graph_a']['scaling']['scaling']}")

print(f"\n{result['graph_b']['name']}:")
print(f"  Computational bottleneck: {result['graph_b']['compute']['primary_bottleneck']}")
print(f"  Spatial preservation: {result['graph_b']['spatial']['spatial_preservation']}")
print(f"  Scaling behavior: {result['graph_b']['scaling']['scaling']}")

print("\nKey Insights:")
for insight in result['summary']:
    print(f"  • {insight}")

print("\n" + "=" * 70)
print("This comparison is deterministic, rule-based, and ready for UI/agents!")
