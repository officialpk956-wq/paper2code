import sys, json
sys.path.insert(0, r'c:\papper2code')
from dotenv import load_dotenv
load_dotenv(r'c:\papper2code\.env')
from backend.database import SessionLocal
from backend.models import PaperModule, Paper
import backend.server as srv

db = SessionLocal()
papers = db.query(Paper).all()

zero_flops = []
generic_exp = []
duplicates = {}
all_modules = []

for p in papers:
    modules = db.query(PaperModule).filter(PaperModule.paper_id == p.id).order_by(PaperModule.order_index).all()
    for m in modules:
        flops = srv.safe_dict(m.flops_context)
        score = flops.get('total_flops_score', 0)
        exp = m.explanation or ''
        
        all_modules.append({
            'paper': p.title[:30],
            'layer': m.layer_name,
            'flops_score': score,
            'exp_len': len(exp),
            'exp_preview': exp[:100].replace('\n', ' ')
        })

        key = exp[:80] if exp else ''
        if key:
            if key not in duplicates:
                duplicates[key] = []
            duplicates[key].append(f'{p.title[:15]}|{m.layer_name}')

        if score == 0:
            zero_flops.append(f'{p.title[:25]} | {m.layer_name}')
        if 'optimized for unknown' in exp or 'optimized for architectural consistency' in exp:
            generic_exp.append(m.layer_name)

print('=== SUMMARY ===')
print(f'Total modules: {len(all_modules)}')
print(f'Zero FLOPs:    {len(zero_flops)}')
print(f'Generic expl:  {len(generic_exp)}')
dup_groups = [(k, v) for k, v in duplicates.items() if len(v) > 1]
print(f'Dup groups:    {len(dup_groups)}')

print()
print('=== ZERO FLOPs (detail) ===')
for z in zero_flops:
    print(' ', z)

print()
print('=== DUPLICATE EXPLANATION GROUPS ===')
for k, names in dup_groups:
    print(f'  [{len(names)} modules] preview: {k[:70]}')
    for n in names:
        print(f'    - {n}')

print()
print('=== EXPLANATION QUALITY SAMPLE ===')
for m in all_modules:
    print(f'[{m["flops_score"]}] {m["layer"][:35]:35s} | {m["exp_preview"][:90]}')

db.close()
