import sqlite3, json
conn = sqlite3.connect('tensortonic_dev.db')
c = conn.cursor()
c.execute("SELECT flops_analysis FROM papers WHERE title = 'LeNet-5'")
lenet = json.loads(c.fetchone()[0])
c.execute("SELECT flops_analysis FROM papers WHERE title = 'DenseNet121'")
densenet = json.loads(c.fetchone()[0])
print('LeNet FLOPs score:', lenet['total_flops_score'])
print('DenseNet FLOPs score:', densenet['total_flops_score'])
print('LeNet breakdown:')
for row in lenet['breakdown']:
    print(' ', row['node'], '->', row['flops_level'])
print('DenseNet breakdown:')
for row in densenet['breakdown']:
    print(' ', row['node'], '->', row['flops_level'])
