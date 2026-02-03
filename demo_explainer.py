#!/usr/bin/env python
"""Demo: Human-readable architecture comparisons."""

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_unet import build_unet_graph
from src.visualizer_vit import build_vit_graph
from src.comparators import explain_architecture_comparison

print("\n" + "=" * 80)
print("DEMO: Human-Readable Architecture Comparisons")
print("=" * 80)

# Build architectures
resnet = build_resnet18_graph()
unet = build_unet_graph()
vit = build_vit_graph()

# ============================================================================
# Comparison 1: ResNet vs U-Net
# ============================================================================
print("\n\n")
print(explain_architecture_comparison(resnet, unet))

# ============================================================================
# Comparison 2: U-Net vs Vision Transformer
# ============================================================================
print("\n\n")
print(explain_architecture_comparison(unet, vit))

# ============================================================================
# Comparison 3: ResNet vs Vision Transformer
# ============================================================================
print("\n\n")
print(explain_architecture_comparison(resnet, vit))

print("\n" + "=" * 80)
print("These explanations are:")
print("  • Fully deterministic (same architectures = same text)")
print("  • Beginner-friendly (explains O(n²), skip connections, etc.)")
print("  • Markdown-formatted (ready for Streamlit or docs)")
print("  • Context-aware (mentions specific architectural features)")
print("=" * 80)
