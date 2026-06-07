import sqlite3
import json
from collections import Counter

conn = sqlite3.connect('tensortonic_dev.db')
c = conn.cursor()

# ---- Step 1: Architecture node/edge structure fidelity ----
TARGETS = ["ResNet50", "U-Net", "Vision Transformer", "GoogLeNet", "ResNet18", "ResNet34", "DenseNet121", "MobileNetV2", "EfficientNet-B0", "FCN", "Transformer", "LeNet-5", "AlexNet", "VGG16", "VGG19"]

c.execute("SELECT id, title, architecture_graph, flops_analysis FROM papers")
papers = c.fetchall()

paper_map = {}
for pid, title, ag, fa in papers:
    ag_dict = json.loads(ag) if ag else {}
    fa_dict = json.loads(fa) if fa else {}
    paper_map[title] = {"id": pid, "graph": ag_dict, "flops": fa_dict}

print("=" * 60)
print("STEP 1: ARCHITECTURE PRESENCE CHECK")
print("=" * 60)
expected = ["LeNet-5","AlexNet","VGG16","VGG19","GoogLeNet","ResNet18","ResNet34","ResNet50","DenseNet121","MobileNetV2","EfficientNet-B0","FCN","U-Net","Transformer","Vision Transformer"]
for name in expected:
    exists = name in paper_map
    print(f"  {'PASS' if exists else 'FAIL'} | {name}")

print()
print("=" * 60)
print("STEP 2: SUPPORT LEVEL VERIFICATION")
print("=" * 60)
for name, data in paper_map.items():
    sl = data["graph"].get("support_level", "MISSING")
    status = "PASS" if sl == "verified" else "FAIL"
    print(f"  {status} | {name} | support_level={sl}")

print()
print("=" * 60)
print("STEP 3: ARCHITECTURE FIDELITY — NODE TYPES")
print("=" * 60)

def get_node_types(graph_dict):
    return [n.get("type") for n in graph_dict.get("nodes", [])]

def get_edge_types(graph_dict):
    return [e.get("type", "flow") for e in graph_dict.get("edges", [])]

# ResNet50 — must have Residual/bottleneck/conv/pool/linear
r50 = paper_map.get("ResNet50", {})
r50_types = get_node_types(r50.get("graph", {}))
r50_edges = get_edge_types(r50.get("graph", {}))
r34_types = get_node_types(paper_map.get("ResNet34", {}).get("graph", {}))
print(f"ResNet50 node types: {r50_types}")
print(f"ResNet34 node types: {r34_types}")
r50_pass = ("BottleneckBlock" in r50_types or "ResidualBlock" in r50_types) and "Conv2d" in r50_types and "Linear" in r50_types
print(f"ResNet50 has Bottleneck/Residual+Conv+Linear: {'PASS' if r50_pass else 'FAIL'}")
r50_r34_identical = r50_types == r34_types
print(f"ResNet50 == ResNet34 node structure (should differ): {'FAIL — IDENTICAL GRAPHS' if r50_r34_identical else 'PASS — different'}")

# Ensure ResNet50 uses BottleneckBlock (3-layer blocks), not plain ResidualBlock (2-layer)
r50_has_bottleneck = "BottleneckBlock" in r50_types
r34_has_plain = "ResidualBlock" in r34_types
print(f"ResNet50 uses BottleneckBlock (3-layer): {'PASS' if r50_has_bottleneck else 'FAIL'}")
print(f"ResNet34 uses ResidualBlock (2-layer): {'PASS' if r34_has_plain else 'FAIL'}")

# U-Net — must have skip connections
unet = paper_map.get("U-Net", {})
unet_edges = get_edge_types(unet.get("graph", {}))
unet_types = get_node_types(unet.get("graph", {}))
has_skip = "skip" in unet_edges
print(f"\nU-Net edge types: {unet_edges}")
print(f"U-Net has skip connections: {'PASS' if has_skip else 'FAIL'}")
print(f"U-Net has Upsample: {'PASS' if 'Upsample' in unet_types else 'FAIL'}")

# Vision Transformer — must have PatchEmbedding + MultiHeadAttention + Linear
vit = paper_map.get("Vision Transformer", {})
vit_types = get_node_types(vit.get("graph", {}))
has_patch = "PatchEmbedding" in vit_types
has_mhsa = "MultiHeadAttention" in vit_types
has_head = "Linear" in vit_types
print(f"\nViT node types (first 6): {vit_types[:6]}")
print(f"ViT PatchEmbedding: {'PASS' if has_patch else 'FAIL'}")
print(f"ViT MultiHeadAttention: {'PASS' if has_mhsa else 'FAIL'}")
print(f"ViT Classification Head (Linear): {'PASS' if has_head else 'FAIL'}")

# GoogLeNet — must have InceptionBlock
gnet = paper_map.get("GoogLeNet", {})
gnet_types = get_node_types(gnet.get("graph", {}))
has_inception = "InceptionBlock" in gnet_types
print(f"\nGoogLeNet node types: {gnet_types}")
print(f"GoogLeNet has InceptionBlock: {'PASS' if has_inception else 'FAIL'}")

print()
print("=" * 60)
print("STEP 4: CORPUS COVERAGE ANALYSIS")
print("=" * 60)

c.execute("SELECT paper_id, COUNT(*) as cnt FROM paper_modules GROUP BY paper_id ORDER BY cnt DESC")
module_counts = c.fetchall()
print("Modules per architecture:")
for pid, cnt in module_counts:
    # Find title
    title = next((t for t, d in paper_map.items() if d["id"] == pid), f"ID{pid}")
    fa = paper_map.get(title, {}).get("flops", {})
    params = fa.get("total_params_estimate", 0)
    flops_score = fa.get("total_flops_score", 0)
    print(f"  ID={pid} | {title:20s} | Modules={cnt} | Params={params:,} | FLOPs={flops_score}")

# Sanity check: LeNet should be smaller than DenseNet
lenet_params = paper_map.get("LeNet-5", {}).get("flops", {}).get("total_params_estimate", 0)
densenet_params = paper_map.get("DenseNet121", {}).get("flops", {}).get("total_params_estimate", 0)
lenet_flops = paper_map.get("LeNet-5", {}).get("flops", {}).get("total_flops_score", 0)
densenet_flops = paper_map.get("DenseNet121", {}).get("flops", {}).get("total_flops_score", 0)
print(f"\nLeNet-5 params={lenet_params:,} vs DenseNet121 params={densenet_params:,}")
print(f"LeNet-5 < DenseNet121 (params): {'PASS' if lenet_params < densenet_params else 'FAIL'}")
print(f"LeNet-5 flops={lenet_flops} vs DenseNet121 flops={densenet_flops}")
print(f"LeNet-5 < DenseNet121 (flops): {'PASS' if lenet_flops < densenet_flops else 'FAIL'}")

# FCN params check — 307M is suspicious
fcn_params = paper_map.get("FCN", {}).get("flops", {}).get("total_params_estimate", 0)
print(f"\nFCN params: {fcn_params:,} (expected: ~134M-500M range, PASS if > 100M)")
print(f"FCN params sanity: {'PASS' if fcn_params > 100_000_000 else 'FAIL'}")

# MobileNetV2 vs EfficientNet - identical?
mob_params = paper_map.get("MobileNetV2", {}).get("flops", {}).get("total_params_estimate", 0)
eff_params = paper_map.get("EfficientNet-B0", {}).get("flops", {}).get("total_params_estimate", 0)
mob_nodes = get_node_types(paper_map.get("MobileNetV2", {}).get("graph", {}))
eff_nodes = get_node_types(paper_map.get("EfficientNet-B0", {}).get("graph", {}))
print(f"\nMobileNetV2 params={mob_params:,} | EfficientNet-B0 params={eff_params:,}")
print(f"MobileNetV2 node types: {mob_nodes}")
print(f"EfficientNet-B0 node types: {eff_nodes}")
print(f"MobileNetV2 == EfficientNet-B0 (params): {'FAIL — IDENTICAL' if mob_params == eff_params else 'PASS'}")
print(f"MobileNetV2 == EfficientNet-B0 (nodes): {'FAIL — IDENTICAL GRAPH SHAPE' if mob_nodes == eff_nodes else 'PASS — differ'}")

print()
print("=" * 60)
print("STEP 5: DUPLICATE EXPLANATION DEEP DIVE")
print("=" * 60)

c.execute("SELECT paper_id, layer_name, explanation FROM paper_modules")
all_modules = c.fetchall()
expls = [r[2] for r in all_modules]
expl_counts = Counter(expls)
dups = [(expl[:80], cnt) for expl, cnt in expl_counts.most_common(10) if cnt > 1]
print(f"Total modules: {len(expls)}")
print(f"Unique explanations: {len(set(expls))}")
print(f"Duplicate ratio: {len(expls)-len(set(expls))}/{len(expls)} = {(len(expls)-len(set(expls)))/len(expls)*100:.1f}%")
print("Top duplicate explanations:")
for d, cnt in dups:
    print(f"  x{cnt}: {d}")

print()
print("=" * 60)
print("STEP 6: API SCHEMA CHECK (TestClient)")
print("=" * 60)

from fastapi.testclient import TestClient
from backend.server import app
client = TestClient(app)

resp = client.get("/api/papers")
data = resp.json()
has_stats = "statistics" in data
has_papers = "papers" in data
print(f"Has 'statistics' key: {'PASS' if has_stats else 'FAIL'}")
print(f"Has 'papers' key: {'PASS' if has_papers else 'FAIL'}")
if has_stats:
    stats = data["statistics"]
    print(f"statistics.total_papers = {stats.get('total_papers')} (expected 15): {'PASS' if stats.get('total_papers') == 15 else 'FAIL'}")
    print(f"statistics.total_modules = {stats.get('total_modules')} (expected 190): {'PASS' if stats.get('total_modules') == 190 else 'FAIL'}")
    print(f"statistics.architecture_categories = {stats.get('architecture_categories')}")
    print(f"Most categories resolved to 'Unknown': {'FAIL — 9/15 unknown' if stats.get('architecture_categories', {}).get('Unknown', 0) > 5 else 'PASS'}")
    print(f"statistics.largest_model = {stats.get('largest_model')}")
    print(f"statistics.most_complex_model = {stats.get('most_complex_model')}")

# Check per-paper support_level
if has_papers:
    papers_resp = data["papers"]
    missing_sl = [p for p in papers_resp if not p.get("support_level")]
    print(f"Papers missing support_level: {len(missing_sl)} (expected 0): {'PASS' if len(missing_sl) == 0 else 'FAIL'}")
    for p in papers_resp:
        sl = p.get("support_level")
        arch_type = p.get("architecture_type")
        print(f"  {p['title']:20s} | arch_type={arch_type:15s} | support_level={sl}")

print()
print("Done.")
