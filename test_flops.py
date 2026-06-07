import sqlite3
import json

conn = sqlite3.connect('tensortonic_dev.db')
c = conn.cursor()

c.execute("SELECT metadata FROM papers WHERE title = 'MobileNetV2'")
row = c.fetchone()
if row:
    data = json.loads(row[0])
    for ev in data.get('flops_events', []):
        if ev['node_type'] == 'InvertedResidual':
            print(ev)
