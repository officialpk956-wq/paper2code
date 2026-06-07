import sqlite3, json

conn = sqlite3.connect('tensortonic_dev.db')
c = conn.cursor()

# U-Net edges
c.execute("SELECT title, architecture_graph FROM papers WHERE title = 'U-Net'")
row = c.fetchone()
title, ag = row
g = json.loads(ag)
print("U-Net edges:")
for e in g.get('edges', []):
    print(f"  {e['source']} -> {e['target']} | type={e.get('edge_type','flow')}")

has_skip = any(e.get('edge_type') == 'skip' for e in g.get('edges', []))
print(f"\nHas 'skip' edge_type in architecture_graph: {has_skip}")

# Check what the pipeline/generator stored vs what was built
print("\nAll edge types found:", set(e.get('edge_type','flow') for e in g.get('edges',[])))

# ResNet50 vs ResNet34 - inspect block counts
c.execute("SELECT title, architecture_graph FROM papers WHERE title IN ('ResNet34', 'ResNet50')")
for row in c.fetchall():
    title, ag = row
    g = json.loads(ag)
    nodes = g.get('nodes', [])
    res_blocks = [n for n in nodes if n.get('type') == 'ResidualBlock']
    print(f"\n{title}: {len(res_blocks)} ResidualBlocks")
    for n in res_blocks:
        print(f"  {n['id']}: {n['label']}")
