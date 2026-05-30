import os
import re

def fix_imports(directory):
    files_changed = 0
    imports_fixed = 0
    
    for root, _, files in os.walk(directory):
        # skip virtual envs and git
        if '.venv' in root or '.git' in root or '__pycache__' in root or 'frontend' in root:
            continue
            
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Regex to find 'from core' or 'import core'
            # Need to be careful with word boundaries
            new_content, count1 = re.subn(r'from core\b', 'from core', content)
            new_content, count2 = re.subn(r'import core\b', 'import core', new_content)
            new_content, count3 = re.subn(r'from core\.', 'from core.', new_content)
            new_content, count4 = re.subn(r'import core\.', 'import core.', new_content)
            
            # Additional check for things like src.something
            
            total_count = count1 + count2 + count3 + count4
            
            if total_count > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                files_changed += 1
                imports_fixed += total_count
                
    return files_changed, imports_fixed

if __name__ == '__main__':
    workspace_dir = 'c:\\papper2code'
    files_changed, imports_fixed = fix_imports(workspace_dir)
    print(f"Files changed: {files_changed}")
    print(f"Imports fixed: {imports_fixed}")
