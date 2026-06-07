import sqlite3
import json

conn = sqlite3.connect('tensortonic_dev.db')
c = conn.cursor()

c.execute("SELECT flops_analysis FROM papers WHERE title = 'MobileNetV2'")
row = c.fetchone()
if row:
    data = json.loads(row[0])
    print('MobileNetV2 params:', data.get('total_params_estimate'))

c.execute("SELECT flops_analysis FROM papers WHERE title = 'EfficientNet-B0'")
row = c.fetchone()
if row:
    data = json.loads(row[0])
    print('EfficientNet-B0 params:', data.get('total_params_estimate'))
