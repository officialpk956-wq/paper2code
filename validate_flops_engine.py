"""End-to-end validation of the FLOPs + TensorTracker integration."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.rag.flops_engine import FLOPsEngine
from core.rag.tensor_tracker import TensorTracker
from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge

eng = FLOPsEngine()

# ── Per-layer formula accuracy ──────────────────────────────
tests = [
    ("conv_1",  "conv2d",             ("B",3,224,224),  ("B",64,112,112), {"kernel_size":3}),
    ("lin_1",   "linear",             ("B",196,768),    ("B",196,768),    {"hidden_size":768}),
    ("mha_1",   "multiheadattention", ("B",196,768),    ("B",196,768),    {"num_heads":12}),
    ("ff_1",    "feedforward",        ("B",196,768),    ("B",196,3072),   {"ff_dim":3072}),
    ("cross1",  "cross_attention",    ("B",128,512),    ("B",128,512),    {"num_heads":8,"seq_len_k":512}),
    ("patch1",  "patchembedding",     ("B",3,224,224),  ("B",196,768),    {"patch_size":16,"embed_dim":768}),
    ("emb1",    "token_embedding",    ("B",512),        ("B",512,768),    {"vocab_size":30522,"embed_dim":768}),
    ("ln1",     "layernorm",          ("B",196,768),    ("B",196,768),    {}),
]

print()
print("Layer                FLOPs(MF)  Memory(MB)  Params(M)  Severity  Complexity")
print("-"*80)
all_ok = True
for nid, ntype, ins, outs, params in tests:
    r = eng.estimate(nid, ntype, ins, outs, params)
    nid_col = (nid + " (" + ntype[:10] + ")")[:24]
    print(f"{nid_col:<24} {r.flops_mflops:>8.2f}  {r.memory_mb:>9.3f}  {r.params_M:>8.4f}  {r.severity:<8}  {r.complexity}")
    if r.severity not in ("low","medium","high","critical"):
        print("  FAIL: invalid severity")
        all_ok = False

# ── TensorTracker integration ───────────────────────────────
print()
print("TensorTracker flops_events integration:")
nodes = [
    GraphNode(id="patch_emb", label="PatchEmb", type="patchembedding",     params={"patch_size":16,"embed_dim":768}),
    GraphNode(id="mha_1",     label="MHSA-1",   type="multiheadattention", params={"num_heads":12}),
    GraphNode(id="ff_1",      label="FFN-1",    type="feedforward",        params={"ff_dim":3072}),
    GraphNode(id="pool",      label="Pool",     type="sequence_pooling",   params={}),
    GraphNode(id="head",      label="Head",     type="linear",             params={"hidden_size":768}),
]
edges = [
    GraphEdge(source="patch_emb", target="mha_1"),
    GraphEdge(source="mha_1",     target="ff_1"),
    GraphEdge(source="ff_1",      target="pool"),
    GraphEdge(source="pool",      target="head"),
]
graph = ArchitectureGraph(name="ViT-FLOPs-Test", nodes=nodes, edges=edges)
tracker = TensorTracker()
tracker.propagate_shapes(graph, initial_shape=("B",3,224,224))

evs = graph.metadata.get("flops_events", [])
total_mf = sum(e["flops_mflops"] for e in evs)

print(f"  flops_events count: {len(evs)}  (expected {len(nodes)})")
print(f"  Total MFLOPs: {total_mf:.2f}")

for e in evs:
    nid  = e["node_id"]
    mf   = e["flops_mflops"]
    sev  = e["severity"]
    cpx  = e["complexity"]
    frm  = e["formula"][:60]
    print(f"    {nid:<14} {mf:>8.2f} MF  [{sev}]  {cpx}")
    print(f"                formula: {frm}")
    for w in e.get("warnings", []):
        print(f"                WARN: {w}")

# Validate all required keys present
required_keys = {"node_id","node_type","node_label","flops_mflops",
                 "params_M","memory_mb","formula","complexity","severity","warnings"}
for e in evs:
    missing = required_keys - set(e.keys())
    if missing:
        print(f"  FAIL: missing keys {missing} in event for {e.get('node_id')}")
        all_ok = False

integration_ok = len(evs) == len(nodes) and total_mf > 0
if not integration_ok:
    all_ok = False

print()
status = "PASS" if (all_ok and integration_ok) else "FAIL"
print(f"[{status}] FLOPs Engine + TensorTracker integration")
sys.exit(0 if (all_ok and integration_ok) else 1)
