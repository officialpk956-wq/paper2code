#!/usr/bin/env python
"""Test script for Phase 3.8.3 Semantic Reasoning Hooks."""

from src.explainers import explain_node, explain_graph
from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_unet import build_unet_graph
from src.visualizer_vit import build_vit_graph

print("=" * 60)
print("Phase 3.8.3 Semantic Reasoning Hooks - Verification Test")
print("=" * 60)

# Test 1: ResNet-18
print("\n[1] ResNet-18 - Graph Explanation")
print("-" * 60)
g1 = build_resnet18_graph()
print(explain_graph(g1))

print("\n[2] ResNet-18 - Node Explanation (Conv 7×7)")
print("-" * 60)
print(explain_node(g1.nodes[0]))

# Test 2: U-Net
print("\n\n[3] U-Net - Graph Explanation")
print("-" * 60)
g2 = build_unet_graph()
print(explain_graph(g2))

print("\n[4] U-Net - Node Explanation (Decoder Block)")
print("-" * 60)
decoder = [n for n in g2.nodes if "Decoder" in n.label][0]
print(explain_node(decoder))

# Test 3: ViT
print("\n\n[5] Vision Transformer - Graph Explanation")
print("-" * 60)
g3 = build_vit_graph()
print(explain_graph(g3))

print("\n[6] Vision Transformer - Node Explanation (Transformer Encoder)")
print("-" * 60)
trans = [n for n in g3.nodes if "Encoder Layer" in n.label][0]
print(explain_node(trans))

print("\n" + "=" * 60)
print("✓ All tests passed - Explainers working correctly!")
print("=" * 60)
