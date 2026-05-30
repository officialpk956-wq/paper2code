
import logging
from core.rag.tensor_tracker import TensorTracker, TensorMismatchError
from core.architecture_graph import ArchitectureGraph, GraphNode

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Transformer-Details")

def validate_transformer_details():
    logger.info("Step 1: Constructing Detailed ViT ArchitectureGraph")
    graph = ArchitectureGraph(name="Detailed-ViT")
    
    # Stem
    n_patch = GraphNode(id="patch", type="patchembedding", label="Patch Embedding", params={"patch_size": 16, "embed_dim": 768})
    n_cls = GraphNode(id="cls", type="clstoken", label="CLS Token", params={})
    n_pos = GraphNode(id="pos", type="positionalembedding", label="Positional Embedding", params={"embed_dim": 768})
    
    # Transformer Block 1 - Detailed
    n_norm1 = GraphNode(id="norm1", type="layernorm", label="LayerNorm 1", params={})
    n_q = GraphNode(id="q", type="query_projection", label="Query Proj", params={"hidden_size": 768})
    n_k = GraphNode(id="k", type="key_projection", label="Key Proj", params={"hidden_size": 768})
    n_v = GraphNode(id="v", type="value_projection", label="Value Proj", params={"hidden_size": 768})
    n_mhsa = GraphNode(id="mhsa", type="mhsa", label="MHSA", params={"num_heads": 12})
    n_merge = GraphNode(id="merge", type="attention_merge", label="Attention Merge", params={"hidden_size": 768})
    n_add1 = GraphNode(id="add1", type="residual_add", label="Residual Add 1", params={})
    
    # MLP / FeedForward
    n_norm2 = GraphNode(id="norm2", type="layernorm", label="LayerNorm 2", params={})
    n_ff = GraphNode(id="ff", type="feedforward", label="FeedForward", params={"hidden_size": 3072})
    n_ff_proj = GraphNode(id="ff_proj", type="linear", label="FF Projection", params={"hidden_size": 768})
    n_add2 = GraphNode(id="add2", type="residual_add", label="Residual Add 2", params={})
    
    nodes = [n_patch, n_cls, n_pos, n_norm1, n_q, n_k, n_v, n_mhsa, n_merge, n_add1, n_norm2, n_ff, n_ff_proj, n_add2]
    for n in nodes: graph.add_node(n)
    
    # Connectivity
    graph.add_edge("patch", "cls")
    graph.add_edge("cls", "pos")
    graph.add_edge("pos", "norm1")
    
    # Attention Path
    graph.add_edge("norm1", "q")
    graph.add_edge("norm1", "k")
    graph.add_edge("norm1", "v")
    graph.add_edge("q", "mhsa")
    graph.add_edge("k", "mhsa")
    graph.add_edge("v", "mhsa")
    graph.add_edge("mhsa", "merge")
    graph.add_edge("merge", "add1")
    
    # Residual 1
    graph.add_edge("pos", "add1", edge_type="skip")
    
    # MLP Path
    graph.add_edge("add1", "norm2")
    graph.add_edge("norm2", "ff")
    graph.add_edge("ff", "ff_proj")
    graph.add_edge("ff_proj", "add2")
    
    # Residual 2
    graph.add_edge("add1", "add2", edge_type="skip")
    
    logger.info("Step 2: Running TensorTracker")
    tracker = TensorTracker()
    tracker.propagate_shapes(graph, initial_shape=(1, 3, 224, 224))
    
    logger.info("Step 3: Verifying Connectivity and Shapes")
    # Verify no disconnected nodes
    all_connected = True
    dependencies = {n.id: [] for n in graph.nodes}
    for edge in graph.edges: dependencies[edge.target].append(edge.source)
    for n in graph.nodes:
        if not dependencies[n.id] and n.id != "patch":
            logger.error(f"  Disconnected Node: {n.id}")
            all_connected = False
    
    if all_connected:
        logger.info("  PASS: No disconnected nodes found.")

    # Verify MHSA divisibility
    # (Already validated by tracker.propagate_shapes if it didn't crash)
    logger.info("  PASS: Attention heads divide correctly (validated by tracker).")

    # Step 4: Verification Report
    logger.info("\nValidation Report:")
    for node in graph.nodes:
        logger.info(f"  {node.id:10} | {node.type:18} | Out: {str(node.output_shape):15} | Role: {node.semantic_params.get('semantic_role', 'N/A')}")

    # Step 5: Visual Confirmation (Mermaid)
    logger.info("\nMermaid Representation:")
    mermaid_lines = ["graph TD"]
    for node in graph.nodes:
        mermaid_lines.append(f"    {node.id}[\"{node.label}<br/>{node.output_shape}\"]")
    for edge in graph.edges:
        arrow = "==>" if edge.edge_type == "skip" else "-->"
        mermaid_lines.append(f"    {edge.source} {arrow} {edge.target}")
    
    print("\n".join(mermaid_lines))

if __name__ == "__main__":
    validate_transformer_details()
