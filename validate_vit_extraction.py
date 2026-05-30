
import logging
import json
import torch
import torch.nn as nn
from core.orchestrator.pipeline import Paper2CodePipeline
from core.architecture_graph import ArchitectureGraph

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ViT-Validation")

def validate_extraction():
    pipeline = Paper2CodePipeline()
    
    # Standard ViT configuration for testing extraction
    config = {
        "name": "ViT-Validation-Test",
        "layers": [
            {
                "type": "patchembedding", 
                "params": {
                    "patch_size": 16, 
                    "embed_dim": 768,
                    "num_patches": 196
                }
            },
            {"type": "transformerblock", "params": {"num_heads": 12, "hidden_size": 768}},
            {"type": "sequence_pooling", "params": {}},
            {"type": "linear", "params": {"hidden_size": 1000}}
        ],
        "connections": [
            ["layer_0", "layer_1"], ["layer_1", "layer_2"], ["layer_2", "layer_3"]
        ]
    }

    logger.info("Step 1: Running Extraction Pipeline")
    result = pipeline.run_single(config)
    graph = result["graph"]
    
    logger.info("\nStep 2: ArchitectureGraph Nodes")
    for node in graph.nodes:
        logger.info(f"Node: ID={node.id}, Type={node.type}, Label={node.label}")
        logger.info(f"  - Input Shape: {getattr(node, 'input_shape', 'N/A')}")
        logger.info(f"  - Output Shape: {getattr(node, 'output_shape', 'N/A')}")
        logger.info(f"  - Semantic Params: {node.semantic_params}")

    # Step 3: Verify Metadata
    patch_node = graph.nodes[0]
    logger.info("\nStep 3: Verifying Patch Embedding Metadata")
    patch_info = patch_node.get_patch_info()
    for key, val in patch_info.items():
        logger.info(f"  - {key}: {val}")
        if val is None:
             logger.warning(f"  - Missing metadata: {key}")

    # Step 4: Verify Tensor Propagation
    logger.info("\nStep 4: Verifying Tensor Propagation")
    logger.info(f"  - Input: (B, 3, 224, 224)")
    logger.info(f"  - Patch Embedding Output: {patch_node.output_shape}")
    
    expected_output = (patch_node.input_shape[0], 196, 768)
    if patch_node.output_shape == expected_output:
        logger.info("  - PASS: Tensor propagation matches expected (B, 196, 768)")
    else:
        logger.error(f"  - FAIL: Expected {expected_output}, got {patch_node.output_shape}")

    # Step 5: Visualization Check
    from core.agents.types import VisualizationOptions
    vis_options = VisualizationOptions(highlight_compute=True, show_shapes=True)
    result_vis = pipeline.run_single(config, vis_options=vis_options)
    visual = result_vis["visual"]
    
    logger.info("\nStep 5: Visualization Check")
    logger.info(f"  - Visual Mode: {visual.get('mode')}")
    if "node_annotations" in visual:
        logger.info("  - Semantic labels found in visual annotations.")
        # Print some annotations for confirmation
        for node_id, anno in list(visual["node_annotations"].items())[:2]:
            logger.info(f"    - {node_id}: {anno.get('label')} ({anno.get('color')})")

    # Custom Mermaid Generator for terminal output
    logger.info("\nMermaid Representation:")
    mermaid_lines = ["graph TD"]
    for node in graph.nodes:
        shape_text = f"<br/>{node.output_shape}" if node.output_shape else ""
        label = f"{node.label}{shape_text}"
        mermaid_lines.append(f"    {node.id}[\"{label}\"]")
    for edge in graph.edges:
        mermaid_lines.append(f"    {edge.source} --> {edge.target}")
    
    print("\n".join(mermaid_lines))


if __name__ == "__main__":
    validate_extraction()
