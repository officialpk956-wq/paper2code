import re

def parse_architectures():
    with open('207 architectures across 13 categories..txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    archs = []
    for line in lines:
        line = line.strip()
        if line.startswith('|') and not line.startswith('|-') and not 'Name | Year' in line:
            parts = [p.strip() for p in line.split('|') if p.strip() or p == '']
            if len(parts) >= 8:
                archs.append(parts[0])
    print(f"Architectures: {len(archs)}")

def parse_papers():
    with open('200 Papers.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    papers = []
    for line in lines:
        line = line.strip()
        m = re.match(r'^\*\*#(\d+)\s*[—\-]\s*(.*?)\*\*$', line)
        if m:
            papers.append(int(m.group(1)))
    print(f"Papers: {len(papers)}, Missing: {set(range(1, 201)) - set(papers)}")

def parse_curriculum():
    with open('full 12-domain curriculum. Domain 11 Advanced + Expert and all of Domain 12.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    domains = set()
    for line in lines:
        m = re.match(r'^DOMAIN\s+(\d+)', line)
        if m:
            domains.add(int(m.group(1)))
    print(f"Curriculum Domains: {len(domains)}, Values: {sorted(list(domains))}")

def parse_system_design():
    with open('AI System Design Complete Curriculum.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    systems = set()
    for line in lines:
        m = re.match(r'^#\s*SYSTEM\s+(\d+)', line, re.IGNORECASE)
        if m:
            systems.add(int(m.group(1)))
    print(f"System Design Systems: {len(systems)}, Values: {sorted(list(systems))}")

parse_architectures()
parse_papers()
parse_curriculum()
parse_system_design()
