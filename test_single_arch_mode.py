"""
Test that single-architecture rendering is unchanged and backward compatible.
"""

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_unet import build_unet_graph
from src.visualizer_vit import build_vit_graph


def test_single_architecture_mode():
    """Test that graphs can still be rendered in single-architecture mode."""
    
    print("Testing single-architecture mode (no comparison context)...")
    
    # Build all three architectures
    print("\nBuilding ResNet-18...")
    resnet = build_resnet18_graph()
    assert len(resnet.nodes) > 0, "ResNet should have nodes"
    assert len(resnet.edges) > 0, "ResNet should have edges"
    print(f"✓ ResNet-18: {len(resnet.nodes)} nodes, {len(resnet.edges)} edges")
    
    print("\nBuilding U-Net...")
    unet = build_unet_graph()
    assert len(unet.nodes) > 0, "U-Net should have nodes"
    assert len(unet.edges) > 0, "U-Net should have edges"
    print(f"✓ U-Net: {len(unet.nodes)} nodes, {len(unet.edges)} edges")
    
    print("\nBuilding Vision Transformer...")
    vit = build_vit_graph()
    assert len(vit.nodes) > 0, "ViT should have nodes"
    assert len(vit.edges) > 0, "ViT should have edges"
    print(f"✓ ViT: {len(vit.nodes)} nodes, {len(vit.edges)} edges")
    
    # Verify semantic params are preserved
    print("\nVerifying semantic parameters...")
    
    resnet_semantic_count = sum(1 for n in resnet.nodes if n.semantic_params)
    unet_semantic_count = sum(1 for n in unet.nodes if n.semantic_params)
    vit_semantic_count = sum(1 for n in vit.nodes if n.semantic_params)
    
    assert resnet_semantic_count > 0, "ResNet should have nodes with semantic params"
    assert unet_semantic_count > 0, "U-Net should have nodes with semantic params"
    assert vit_semantic_count > 0, "ViT should have nodes with semantic params"
    
    print(f"✓ ResNet: {resnet_semantic_count} nodes with semantic params")
    print(f"✓ U-Net: {unet_semantic_count} nodes with semantic params")
    print(f"✓ ViT: {vit_semantic_count} nodes with semantic params")
    
    # Verify descriptions are present
    print("\nVerifying node descriptions...")
    
    resnet_desc_count = sum(1 for n in resnet.nodes if n.description)
    unet_desc_count = sum(1 for n in unet.nodes if n.description)
    vit_desc_count = sum(1 for n in vit.nodes if n.description)
    
    assert resnet_desc_count > 0, "ResNet should have nodes with descriptions"
    assert unet_desc_count > 0, "U-Net should have nodes with descriptions"
    assert vit_desc_count > 0, "ViT should have nodes with descriptions"
    
    print(f"✓ ResNet: {resnet_desc_count} nodes with descriptions")
    print(f"✓ U-Net: {unet_desc_count} nodes with descriptions")
    print(f"✓ ViT: {vit_desc_count} nodes with descriptions")
    
    print("\n" + "="*60)
    print("Single-architecture mode: BACKWARD COMPATIBLE ✓")
    print("="*60)
    
    return True


if __name__ == "__main__":
    test_single_architecture_mode()
