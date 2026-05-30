"""Quick smoke-test for cross-attention event emission in TensorTracker."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.rag.tensor_tracker import TensorTracker
from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge

PASS, FAIL = "PASS", "FAIL"

# Build a minimal T5-style encoder-decoder graph with cross-attention
nodes = [
    GraphNode(id="enc_out",    label="Encoder Output",  type="transformer_encoder",
              params={"num_heads": 8, "embed_dim": 512}),
    GraphNode(id="dec_query",  label="Decoder State",   type="transformer_decoder",
              params={"num_heads": 8}),
    GraphNode(id="cross_attn", label="Cross-Attention", type="cross_attention",
              params={"num_heads": 8}),
    GraphNode(id="out",        label="Output",          type="linear",
              params={}),
]
edges = [
    GraphEdge(source="dec_query",  target="cross_attn"),
    GraphEdge(source="enc_out",    target="cross_attn"),
    GraphEdge(source="cross_attn", target="out"),
]
graph = ArchitectureGraph(name="T5-test", nodes=nodes, edges=edges)

tracker = TensorTracker()
try:
    tracker.propagate_shapes(graph, initial_shape=("B", 64, 512))
except Exception as e:
    print(f"propagation note: {e}")

events = graph.metadata.get("cross_attention_events", [])
print(f"\nCross-attention events captured: {len(events)}")
for ev in events:
    nid = ev["node_id"]
    qs  = ev["q_shape"]
    kvs = ev["kv_shape"]
    ss  = ev["score_shape"]
    sem = ev["semantic"]
    print(f"  node_id:     {nid}")
    print(f"  q_shape:     {qs}")
    print(f"  kv_shape:    {kvs}")
    print(f"  score_shape: {ss}")
    print(f"  semantic:    {sem}")
    print()

trace = graph.metadata.get("tensor_trace", [])
print("Tensor trace:")
for line in trace:
    print(" ", line)

# Validate
ok = len(events) >= 1
ev0 = events[0] if events else {}
ok = ok and ev0.get("semantic", {}).get("query")  == "Decoder Query"
ok = ok and ev0.get("semantic", {}).get("memory") == "Encoder Memory"
ok = ok and ev0.get("semantic", {}).get("fusion") == "Cross-Attention Fusion"
ok = ok and "score_shape" in ev0

status = PASS if ok else FAIL
print(f"\n[{status}] cross_attention_events schema correct")
sys.exit(0 if ok else 1)
