import sqlite3
import json
conn = sqlite3.connect('tensortonic_dev.db')
c = conn.cursor()
c.execute('SELECT id, title, architecture_graph, flops_analysis FROM papers')
papers = c.fetchall()

print(f"Total Papers: {len(papers)}")
for p in papers:
    pid, title, ag, fa = p
    ag_dict = json.loads(ag) if ag else {}
    fa_dict = json.loads(fa) if fa else {}
    sl = ag_dict.get('support_level', 'MISSING')
    nodes = ag_dict.get('nodes', [])
    edges = ag_dict.get('edges', [])
    params = fa_dict.get('total_params_estimate', 'MISSING')
    print(f"ID {pid} | Title: {title} | Support: {sl} | Nodes: {len(nodes)} | Edges: {len(edges)} | Params: {params}")

print("\n--- Modules ---")
c.execute('SELECT paper_id, layer_name, explanation, flops_context FROM paper_modules')
modules = c.fetchall()
print(f"Total Modules: {len(modules)}")

empty_expls = 0
duplicates = 0
nans = 0
expl_set = set()

for m in modules:
    pid, lname, expl, fc = m
    fc_dict = json.loads(fc) if fc else {}
    
    if not expl or expl.strip() == "":
        empty_expls += 1
    
    if expl in expl_set and expl and not expl.startswith("Structural module:"):
        # We allow "Structural module:" as it's the fallback
        duplicates += 1
    expl_set.add(expl)

    import math
    flops = fc_dict.get("total_flops_score", 0)
    p = fc_dict.get("total_params_estimate", 0)
    if isinstance(flops, float) and math.isnan(flops):
        nans += 1
    if isinstance(p, float) and math.isnan(p):
        nans += 1

print(f"Empty explanations: {empty_expls}")
print(f"Duplicate non-structural explanations: {duplicates}")
print(f"NaN values: {nans}")
