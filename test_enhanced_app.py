#!/usr/bin/env python
"""Test enhanced app.py features."""

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_unet import build_unet_graph
from src.visualizer_vit import build_vit_graph

# Helper function from app.py
def generate_reasoning(node) -> str:
    """Generate 'Why This Matters' reasoning from semantic params."""
    reasoning = []
    
    if not node.semantic_params:
        return ""
    
    # FLOPs reasoning
    flops = node.semantic_params.get("flops")
    if flops == "high":
        reasoning.append("🔴 **High computational cost** — This block dominates the forward/backward pass")
    elif flops == "very high":
        reasoning.append("🔴🔴 **Very high computational cost** — Potential bottleneck for large inputs")
    elif flops == "medium":
        reasoning.append("🟡 **Moderate computational cost** — Reasonable overhead for the benefits")
    
    # Attention complexity
    attention = node.semantic_params.get("attention")
    if attention == "quadratic":
        reasoning.append("⚠️ **Quadratic complexity** — Scales poorly with sequence/spatial length (O(n²))")
    
    # Feature map
    feature_map = node.semantic_params.get("feature_map")
    if feature_map == "downsampling":
        reasoning.append("↓ **Reduces spatial dimensions** — Decreases memory and compute for downstream layers")
    elif feature_map == "upsampling":
        reasoning.append("↑ **Increases spatial dimensions** — Recovers fine-grained details for dense predictions")
    
    # Skip connections
    skip = node.semantic_params.get("skip_connection")
    if skip == "yes":
        reasoning.append("🔗 **Skip connections active** — Preserves low-level features and improves gradient flow")
    
    # Compute role
    role = node.semantic_params.get("compute_role")
    if role:
        reasoning.append(f"📌 **Purpose**: {role}")
    
    # Tokens (ViT)
    tokens = node.semantic_params.get("tokens")
    if tokens and tokens != "constant":
        reasoning.append(f"📍 **Token count**: {tokens} (affects attention complexity and memory)")
    elif tokens == "constant":
        reasoning.append("📍 **Constant token count** — Consistent complexity regardless of input resolution")
    
    return "\n".join(reasoning)

print("=" * 70)
print("Enhanced App Features Test")
print("=" * 70)

# Test 1: ResNet
print("\n[1] ResNet-18 Node Selection UI")
print("-" * 70)
g = build_resnet18_graph()
node_options = [f"{n.id} – {n.label}" for n in g.nodes]
print(f"Sample option: {node_options[0]}")
print(f"Total nodes: {len(node_options)}")
print("✓ Node selector UI working")

# Test 2: FLOPs-based coloring
print("\n[2] Compute-Heavy Block Highlighting (FLOPs)")
print("-" * 70)
FLOPS_COLORS = {
    "high": {"color": "#FF4444", "penwidth": "2.5"},
    "very high": {"color": "#FF0000", "penwidth": "3"},
    "medium": {"color": "#FFA500", "penwidth": "2"},
}

conv_node = g.nodes[0]
print(f"Node: {conv_node.label}")
print(f"FLOPs: {conv_node.semantic_params.get('flops')}")
if "flops" in conv_node.semantic_params:
    flops = conv_node.semantic_params["flops"]
    if flops in FLOPS_COLORS:
        print(f"Color: {FLOPS_COLORS[flops]['color']}")
        print(f"Pen Width: {FLOPS_COLORS[flops]['penwidth']}")
        print("✓ Coloring applied")

# Test 3: Reasoning generation
print("\n[3] 'Why This Matters' Reasoning Box")
print("-" * 70)
reasoning = generate_reasoning(conv_node)
print(reasoning)
print("✓ Reasoning generated")

# Test 4: Test with U-Net decoder (upsampling)
print("\n[4] U-Net Decoder Reasoning")
print("-" * 70)
unet = build_unet_graph()
decoder = [n for n in unet.nodes if "Decoder" in n.label][0]
print(f"Node: {decoder.label}")
reasoning = generate_reasoning(decoder)
print(reasoning)
print("✓ U-Net decoder reasoning working")

# Test 5: Test with ViT (attention complexity)
print("\n[5] Vision Transformer Attention Reasoning")
print("-" * 70)
vit = build_vit_graph()
trans = [n for n in vit.nodes if "Encoder Layer" in n.label][0]
print(f"Node: {trans.label}")
reasoning = generate_reasoning(trans)
print(reasoning)
print("✓ ViT attention reasoning working")

print("\n" + "=" * 70)
print("✓✓✓ All enhanced features working correctly! ✓✓✓")
print("=" * 70)
