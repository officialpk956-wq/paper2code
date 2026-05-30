import sys
import os

sys.path.append(os.getcwd())

from core.rag.tensor_tracker import TensorTracker, TensorMismatchError
from core.architecture_graph import ArchitectureGraph, GraphNode

def test_valid_flows():
    print("=== Testing Valid Transformer Flows ===")
    tracker = TensorTracker()
    
    graph = ArchitectureGraph("test")
    
    # (B, N, D)
    graph.add_node(GraphNode("input", "linear", "input", {"hidden_size": 768}))
    # Reshape testing
    graph.add_node(GraphNode("reshape1", "reshape", "reshape1", {"shape": ["B", -1, 384]}))
    # Transpose testing
    graph.add_node(GraphNode("transpose1", "transpose", "transpose1", {"dims": (1, 2)}))
    
    graph.add_edge("input", "reshape1")
    graph.add_edge("reshape1", "transpose1")
    
    try:
        tracker.propagate_shapes(graph, initial_shape=("B", 196, 768))
        print("[PASS] Reshape and transpose propagated successfully.")
    except Exception as e:
        print(f"[FAIL] {e}")
        
    # Split/Merge testing
    graph2 = ArchitectureGraph("test")
    graph2.add_node(GraphNode("input", "linear", "input", {"hidden_size": 768}))
    graph2.add_node(GraphNode("split", "split_heads", "split", {"num_heads": 12}))
    graph2.add_node(GraphNode("merge", "merge_heads", "merge", {}))
    graph2.add_edge("input", "split")
    graph2.add_edge("split", "merge")
    
    try:
        tracker.propagate_shapes(graph2, initial_shape=("B", 196, 768))
        print("[PASS] Split and merge heads propagated successfully.")
        for log in tracker.trace:
            print(f"  {log}")
    except Exception as e:
        print(f"[FAIL] {e}")

def test_invalid_flows():
    print("\n=== Testing Invalid Flows ===")
    
    # 1. Incompatible reshape
    try:
        tracker = TensorTracker()
        graph = ArchitectureGraph("test")
        graph.add_node(GraphNode("n1", "reshape", "n1", {"shape": ["B", 100, 100]}))
        tracker.propagate_shapes(graph, initial_shape=("B", 196, 768)) # 196*768 = 150528 != 10000
        print("[FAIL] Incompatible reshape should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught incompatible reshape: {e}")

    # 2. Invalid attention head count (divisibility)
    try:
        tracker = TensorTracker()
        graph = ArchitectureGraph("test")
        graph.add_node(GraphNode("n1", "split_heads", "n1", {"num_heads": 7})) # 768 % 7 != 0
        tracker.propagate_shapes(graph, initial_shape=("B", 196, 768))
        print("[FAIL] Invalid attention head count should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught invalid head count: {e}")

    # 3. Inconsistent embedding dimensions in Positional Embedding
    try:
        tracker = TensorTracker()
        graph = ArchitectureGraph("test")
        graph.add_node(GraphNode("n1", "positionalembedding", "n1", {"embed_dim": 512}))
        tracker.propagate_shapes(graph, initial_shape=("B", 196, 768))
        print("[FAIL] Inconsistent embedding dimensions should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught inconsistent embedding dimensions: {e}")
        
    # 4. Illegal token merge (Merge heads on wrong dims)
    try:
        tracker = TensorTracker()
        graph = ArchitectureGraph("test")
        graph.add_node(GraphNode("n1", "merge_heads", "n1", {}))
        tracker.propagate_shapes(graph, initial_shape=("B", 196, 768)) # Expected 4D
        print("[FAIL] Illegal token merge should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught illegal token merge: {e}")

def main():
    test_valid_flows()
    test_invalid_flows()

if __name__ == "__main__":
    main()
