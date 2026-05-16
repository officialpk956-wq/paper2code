
import logging
from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
from src.architecture_graph import ArchitectureGraph, GraphNode

# Setup logging to show the trace
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("ViT-Tensor-Evolution")

def run_simulation(name, nodes, initial_shape=(1, 3, 224, 224), expect_error=None):
    logger.info(f"\n--- Simulation: {name} ---")
    graph = ArchitectureGraph(name=name)
    for n in nodes:
        graph.add_node(n)
    
    # Add sequential edges
    for i in range(len(nodes) - 1):
        graph.add_edge(nodes[i].id, nodes[i+1].id)
    
    tracker = TensorTracker()
    try:
        tracker.propagate_shapes(graph, initial_shape=initial_shape)
        logger.info("Propagation Success!")
        for node in graph.nodes:
            logger.info(f"  {node.id} ({node.type}): {node.input_shape} -> {node.output_shape}")
        
        if expect_error:
            logger.error(f"FAIL: Expected error '{expect_error}' but propagation succeeded.")
        else:
            logger.info("PASS: Valid flow validated.")
            
    except TensorMismatchError as e:
        if expect_error and expect_error in str(e):
            logger.info(f"PASS: Caught expected error: {e}")
        else:
            logger.error(f"FAIL: Unexpected error: {e}")
            if not expect_error:
                raise e

def validate_evolution():
    # 1. Valid ViT Flow
    valid_nodes = [
        GraphNode(id="patch", type="patchembedding", label="Patch", params={"patch_size": 16, "embed_dim": 768}),
        GraphNode(id="cls", type="clstoken", label="CLS", params={}),
        GraphNode(id="pos", type="positionalembedding", label="Pos", params={"embed_dim": 768}),
        GraphNode(id="block", type="transformerblock", label="Encoder", params={"hidden_size": 768}),
    ]
    run_simulation("Valid ViT Flow", valid_nodes)

    # 2. Invalid Case: Mismatched Positional Embedding Dim
    invalid_pos_nodes = [
        GraphNode(id="patch", type="patchembedding", label="Patch", params={"patch_size": 16, "embed_dim": 768}),
        GraphNode(id="pos", type="positionalembedding", label="Pos", params={"embed_dim": 512}), # WRONG DIM
    ]
    run_simulation("Mismatched Positional Embedding", invalid_pos_nodes, expect_error="Dimension mismatch")

    # 3. Invalid Case: Incorrect CLS Token Context (Not 3D)
    invalid_cls_nodes = [
        GraphNode(id="conv", type="conv2d", label="Conv", params={"channels": 64}),
        GraphNode(id="cls", type="clstoken", label="CLS", params={}), # Fails because input is 4D (B, C, H, W)
    ]
    run_simulation("Incorrect CLS Context (4D input)", invalid_cls_nodes, expect_error="Expected 3D input")

    # 4. Invalid Case: Incompatible Token Count / Residual Add
    graph_res = ArchitectureGraph(name="Mismatched Sequence Length Residual")
    n1 = GraphNode(id="patch", type="patchembedding", label="Patch", params={"patch_size": 16, "embed_dim": 768})
    n2 = GraphNode(id="cls", type="clstoken", label="CLS", params={})
    n3 = GraphNode(id="add", type="residual_add", label="Add", params={})
    
    graph_res.add_node(n1)
    graph_res.add_node(n2)
    graph_res.add_node(n3)
    
    graph_res.add_edge("patch", "cls")
    graph_res.add_edge("cls", "add")
    graph_res.add_edge("patch", "add", edge_type="skip") # Skip connection bypasses CLS token insertion
    
    logger.info("\n--- Simulation: Mismatched Sequence Length Residual ---")
    tracker = TensorTracker()
    try:
        tracker.propagate_shapes(graph_res, initial_shape=(1, 3, 224, 224))
        logger.error("FAIL: Expected error but propagation succeeded.")
    except TensorMismatchError as e:
        logger.info(f"PASS: Caught expected error: {e}")


if __name__ == "__main__":
    validate_evolution()
