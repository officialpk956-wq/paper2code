"""
Validate the upgraded KAG symbolic reasoning engine.
Tests: semantic roles, motif recognition, topology anomaly detection,
       and new explain() coverage for cross_attention / causal_attention / encoder / decoder.
"""
import sys, os
# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

from src.rag.knowledge_graph import KnowledgeGraph
from src.rag.semantic_explainer import SemanticExplainer
from src.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge

PASS, FAIL = "✅ PASS", "❌ FAIL"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}" + (f"\n         → {detail}" if detail else ""))

print("\n" + "="*60)
print("   KAG Symbolic Reasoning Engine — Validation Suite")
print("="*60)

kg = KnowledgeGraph()

# ── 1. Semantic Role Resolution ─────────────────────────────
print("\n[1] Semantic Role Resolution")
check("patchembedding → patch_embedding",   kg.get_semantic_role("patchembedding")   == "patch_embedding")
check("multiheadattention → token_mixer",   kg.get_semantic_role("multiheadattention") == "token_mixer")
check("cross_attention → token_mixer",      kg.get_semantic_role("cross_attention")  == "token_mixer")
check("causal_attention → token_mixer",     kg.get_semantic_role("causal_attention") == "token_mixer")
check("transformer_encoder → encoder",      kg.get_semantic_role("transformer_encoder") == "encoder")
check("transformer_decoder → decoder",      kg.get_semantic_role("transformer_decoder") == "decoder")
check("sequence_pooling → feature_aggregator", kg.get_semantic_role("sequence_pooling") == "feature_aggregator")
check("linear → classifier_head",           kg.get_semantic_role("linear")           == "classifier_head")

# ── 2. Architecture Rule Encoding ───────────────────────────
print("\n[2] Architecture Rule Encoding")
ctx = kg.get_context_for_terms(["conv2d", "linear"])
check("REQUIRES_FLATTEN rule present",  "REQUIRES_FLATTEN" in ctx or "Flatten" in ctx)

ctx2 = kg.get_context_for_terms(["batchnorm2d", "linear"])
check("INCOMPATIBLE rule present",      "INCOMPATIBLE" in ctx2)

ctx3 = kg.get_context_for_terms(["residual_add"])
check("REQUIRES_EQUAL_DIMS rule present", "REQUIRES_EQUAL_DIMS" in ctx3 or "identically sized" in ctx3)

# ── 3. Motif Recognition ────────────────────────────────────
print("\n[3] Motif Recognition")

def make_nodes(*types):
    nodes = []
    for i, t in enumerate(types):
        n = GraphNode(id=f"n{i}", label=t, type=t, params={})
        nodes.append(n)
    return nodes

# BERT-like: 12 transformerblock nodes, no decoder
bert_graph = ArchitectureGraph(name="BERT-test", nodes=make_nodes(*["transformerblock"]*12), edges=[])
motifs = kg.detect_motifs(bert_graph)
check("BERT motif detected",            "BERT-style Encoder Stack" in motifs, str(motifs))

# GPT-like: 12 causal_attention nodes, no encoder/cross
gpt_graph = ArchitectureGraph(name="GPT-test", nodes=make_nodes(*["causal_attention"]*12), edges=[])
motifs = kg.detect_motifs(gpt_graph)
check("GPT motif detected",             "GPT-style Autoregressive Decoder Stack" in motifs, str(motifs))

# ViT-like
vit_nodes = make_nodes("patchembedding", *["transformerblock"]*4, "sequence_pooling", "linear")
vit_graph = ArchitectureGraph(name="ViT-test", nodes=vit_nodes, edges=[])
motifs = kg.detect_motifs(vit_graph)
check("ViT motif detected",             "ViT Token Pipeline" in motifs, str(motifs))

# UNet-like
unet_nodes = make_nodes("conv2d", "maxpool2d", "conv2d", "convtranspose2d", "concat", "conv2d")
unet_graph = ArchitectureGraph(name="UNet-test", nodes=unet_nodes, edges=[])
motifs = kg.detect_motifs(unet_graph)
check("UNet motif detected",            "UNet-style Skip Structure" in motifs, str(motifs))

# ── 4. Topology Anomaly Detection ───────────────────────────
print("\n[4] Topology Anomaly Detection")

# cross_attention without an encoder → anomaly
cross_nodes = make_nodes("linear", "cross_attention", "linear")
cross_edges = [GraphEdge(source="n0", target="n1"), GraphEdge(source="n1", target="n2")]
cross_graph = ArchitectureGraph(name="BadCross", nodes=cross_nodes, edges=cross_edges)
anomalies = kg.verify_topology(cross_graph)
check("Cross-Attention without Encoder flagged", any("cross" in a.lower() for a in anomalies), str(anomalies))

# Clean ViT: no anomalies expected
clean_graph = ArchitectureGraph(name="CleanViT", nodes=vit_nodes, edges=[])
clean_anomalies = kg.verify_topology(clean_graph)
check("Clean ViT topology: no anomalies", len(clean_anomalies) == 0, str(clean_anomalies))

# ── 5. Semantic Explainer Coverage ──────────────────────────
print("\n[5] SemanticExplainer Coverage")

expl_cross  = SemanticExplainer.explain("cross_attention",     "token_mixer",   {})
expl_causal = SemanticExplainer.explain("causal_attention",    "token_mixer",   {})
expl_enc    = SemanticExplainer.explain("transformer_encoder", "encoder",       {"num_heads": 12, "embed_dim": 768})
expl_dec    = SemanticExplainer.explain("transformer_decoder", "decoder",       {"num_heads": 12})
expl_pool   = SemanticExplainer.explain("sequence_pooling",    "feature_aggregator", {})
expl_res    = SemanticExplainer.explain("residual_add",        "residual",      {})

check("cross_attention explanation non-generic",     "Encoder" in expl_cross or "Key/Value" in expl_cross,    expl_cross[:80])
check("causal_attention explanation non-generic",    "autoregressive" in expl_causal.lower(),                  expl_causal[:80])
check("transformer_encoder explanation with params", "768" in expl_enc or "12" in expl_enc,                   expl_enc[:80])
check("transformer_decoder explanation non-generic", "cross-attention" in expl_dec.lower(),                    expl_dec[:80])
check("sequence_pooling explanation non-generic",    "sequence" in expl_pool.lower(),                          expl_pool[:80])
check("residual_add explanation non-generic",        "identity" in expl_res.lower() or "skip" in expl_res.lower(), expl_res[:80])

# ── Summary ─────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for s, *_ in results if s == PASS)
total  = len(results)
print(f"   {passed}/{total} checks passed")
if passed == total:
    print("   🎉 KAG Reasoning Engine fully operational!")
else:
    print("   ⚠️  Some checks failed — review above output.")
print("="*60 + "\n")
sys.exit(0 if passed == total else 1)
