import sys
import os

sys.path.append(os.getcwd())

from src.rag.tensor_tracker import TensorTracker, TensorMismatchError
from src.architecture_graph import ArchitectureGraph, GraphNode

def test_causal_mask_and_scores():
    print("=== Testing Causal Mask & Attention Scores ===")
    tracker = TensorTracker()
    graph = ArchitectureGraph("test_scores")
    
    # Q and K inputs (B, H, N, D_H) -> Output: (B, H, N, N)
    graph.add_node(GraphNode("q", "layernorm", "Q", {})) # Pass-through
    graph.add_node(GraphNode("k", "layernorm", "K", {}))
    
    graph.add_node(GraphNode("scores", "attention_scores", "Scores", {}))
    graph.add_edge("q", "scores")
    graph.add_edge("k", "scores")
    
    graph.add_node(GraphNode("mask", "causal_mask", "Mask", {"strict_square": True}))
    graph.add_edge("scores", "mask")
    
    try:
        tracker.propagate_shapes(graph, initial_shape=("B", 8, 128, 64))
        print("[PASS] Attention scores and valid causal mask succeeded.")
        assert graph.nodes[-1].output_shape == ("B", 8, 128, 128)
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")

    # Invalid Causal Mask (Not square)
    graph2 = ArchitectureGraph("test_invalid_mask")
    graph2.add_node(GraphNode("q", "layernorm", "Q", {}))
    # We need k to have a different shape, let's use a split_heads or something to change it, or just use sequence_pooling
    # Actually, we can create a custom pass-through node for test
    graph2.add_node(GraphNode("k", "reshape", "K", {"shape": ["B", 8, 64, 128]})) # 8*64*128 = 65536
    graph2.add_node(GraphNode("scores", "attention_scores", "Scores", {}))
    graph2.add_edge("q", "scores")
    graph2.add_edge("k", "scores")
    graph2.add_node(GraphNode("mask", "causal_mask", "Mask", {"strict_square": True}))
    graph2.add_edge("scores", "mask")

    try:
        tracker.propagate_shapes(graph2, initial_shape=("B", 8, 128, 64)) # 8*128*64 = 65536
        print("[FAIL] Invalid causal mask should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught invalid causal mask: {e}")

def test_cross_attention():
    print("\n=== Testing Cross-Attention ===")
    tracker = TensorTracker()
    
    # Valid Cross-Attention
    graph = ArchitectureGraph("valid_cross")
    # q_dec route:
    graph.add_node(GraphNode("q_root", "reshape", "Q_Root", {"shape": ["B", 2, 5]}))
    graph.add_node(GraphNode("q_dec", "linear", "Q", {"hidden_size": 768})) # (B, 2, 768)
    graph.add_edge("q_root", "q_dec")
    
    # kv_enc route:
    graph.add_node(GraphNode("kv_root", "reshape", "KV_Root", {"shape": ["B", 5, 2]}))
    graph.add_node(GraphNode("kv_enc", "linear", "KV", {"hidden_size": 768})) # (B, 5, 768)
    graph.add_edge("kv_root", "kv_enc")
    
    graph.add_node(GraphNode("cross", "cross_attention", "Cross", {"num_heads": 8, "causal": False}))
    graph.add_edge("q_dec", "cross")
    graph.add_edge("kv_enc", "cross")

    try:
        tracker.propagate_shapes(graph, initial_shape=("B", 1, 10))
        assert graph.nodes[-1].output_shape == ("B", 2, 768)
        print("[PASS] Valid cross-attention propagated successfully (Output follows Query sequence length).")
    except Exception as e:
        print(f"[FAIL] Valid cross attention failed: {e}")

    # Invalid Cross-Attention (Embed dim mismatch)
    graph2 = ArchitectureGraph("invalid_cross_dim")
    graph2.add_node(GraphNode("q_root", "reshape", "Q_Root", {"shape": ["B", 2, 5]}))
    graph2.add_node(GraphNode("q_dec", "linear", "Q", {"hidden_size": 512})) # Mismatch here
    graph2.add_edge("q_root", "q_dec")
    
    graph2.add_node(GraphNode("kv_root", "reshape", "KV_Root", {"shape": ["B", 5, 2]}))
    graph2.add_node(GraphNode("kv_enc", "linear", "KV", {"hidden_size": 768}))
    graph2.add_edge("kv_root", "kv_enc")
    
    graph2.add_node(GraphNode("cross", "cross_attention", "Cross", {"num_heads": 8, "causal": False}))
    graph2.add_edge("q_dec", "cross")
    graph2.add_edge("kv_enc", "cross")

    try:
        tracker.propagate_shapes(graph2, initial_shape=("B", 1, 10))
        print("[FAIL] Invalid cross-attention embed dim should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught cross-attention dim mismatch: {e}")

    # Invalid Autoregressive Flow (Causal cross-attention)
    graph3 = ArchitectureGraph("invalid_causal_cross")
    graph3.add_node(GraphNode("q_root", "reshape", "Q_Root", {"shape": ["B", 2, 5]}))
    graph3.add_node(GraphNode("q_dec", "linear", "Q", {"hidden_size": 768}))
    graph3.add_edge("q_root", "q_dec")
    
    graph3.add_node(GraphNode("kv_root", "reshape", "KV_Root", {"shape": ["B", 5, 2]}))
    graph3.add_node(GraphNode("kv_enc", "linear", "KV", {"hidden_size": 768}))
    graph3.add_edge("kv_root", "kv_enc")
    
    graph3.add_node(GraphNode("cross", "cross_attention", "Cross", {"num_heads": 8, "causal": True})) # Causal=True
    graph3.add_edge("q_dec", "cross")
    graph3.add_edge("kv_enc", "cross")

    try:
        tracker.propagate_shapes(graph3, initial_shape=("B", 1, 10))
        print("[FAIL] Causal cross-attention should have failed.")
    except TensorMismatchError as e:
        print(f"[PASS] Caught invalid autoregressive cross-attention flow: {e}")


def main():
    test_causal_mask_and_scores()
    test_cross_attention()

if __name__ == "__main__":
    main()
