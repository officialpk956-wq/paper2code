"""
Find ALL syntax errors in the JavaScript
"""

with open("C:/papper2code/static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

import re

match = re.search(r'<script>\s*(.*?)\s*</script>\s*<script src="/static/playground.js">', content, re.DOTALL)
if not match:
    print("Could not find script")
    exit(1)

script = match.group(1)
lines = script.split('\n')

print("COMPREHENSIVE SYNTAX ERROR SEARCH")
print("=" * 70)

# Look for common errors at critical positions
issues = []

for i, line in enumerate(lines, 1):
    # Skip empty lines and comments
    if not line.strip() or line.strip().startswith('//'):
        continue

    # Check for unclosed parentheses in critical areas
    if '.innerHTML =' in line and 'onclick=' in line:
        # This is critical - innerHTML with onclick needs careful quote handling
        open_parens = line.count('(')
        close_parens = line.count(')')

        # Count quotes properly
        dquotes = 0
        squotes = 0
        for c in line:
            if c == '"' and line.index(c) > 0 and line[line.index(c)-1] != '\\':
                dquotes += 1
            if c == "'" and line.index(c) > 0 and line[line.index(c)-1] != '\\':
                squotes += 1

        if open_parens != close_parens:
            print(f"[UNMATCHED PARENS] Line {i}: {open_parens} open vs {close_parens} close")
            print(f"                  {line.strip()[:120]}")
            issues.append(i)

# Look for mismatched quotes in critical areas
for i, line in enumerate(lines, 1):
    # Look specifically for onclick patterns
    if 'onclick=' in line:
        # Find the onclick attribute
        onclick_match = re.search(r'onclick=["\']([^"\']*)["\']', line)
        if onclick_match:
            onclick_value = onclick_match.group(1)
            if '${' in onclick_value:
                # This is using template literal interpolation inside onclick
                # Check if it has proper escaping

                # Look for JSON.stringify which would output quotes
                if 'JSON.stringify' in onclick_value:
                    print(f"[CRITICAL] Line {i}: onclick uses JSON.stringify inside template")
                    print(f"           {line.strip()[:120]}")
                    issues.append(i)

# Look for lines with mixed quote types in problematic contexts
for i, line in enumerate(lines, 1):
    if '.innerHTML' in line or 'onclick=' in line:
        # Count quote types
        has_backtick = '`' in line
        has_double = '"' in line
        has_single = "'" in line

        if has_backtick and (has_double or has_single):
            # This is a template literal with nested quotes - needs careful inspection
            # Look for broken patterns
            if '.innerHTML += `' in line and '`;' in line:
                # Multiline template start
                pass
            elif '.innerHTML = `' in line:
                # Check if backtick is closed on same line
                backtick_count = line.count('`')
                if backtick_count == 1:
                    # This line starts a multiline template
                    # Find where it closes
                    for j in range(i, min(i+100, len(lines))):
                        if '`;' in lines[j]:
                            break
                    else:
                        print(f"[MULTILINE TEMPLATE] Line {i} starts multiline, may not close properly")
                        print(f"                    {line.strip()[:100]}")
                        issues.append(i)

print(f"\nFound {len(issues)} potential issues")

# Look specifically around line 3810 area since we just fixed it
print("\n" + "=" * 70)
print("CHECKING AREA AROUND FIXED LINE (3810)")
print("=" * 70)

for i in range(3805, 3820):
    if i < len(lines):
        print(f"{i}: {lines[i][:120]}")

print("\n" + "=" * 70)
print("CHECKING FOR UNTERMINATED STRING LITERALS")
print("=" * 70)

# Look for incomplete template literals
in_template = False
template_start = 0

for i, line in enumerate(lines, 1):
    backtick_count = 0
    for c in line:
        if c == '`' and (i == 1 or line[line.index(c)-1] != '\\'):
            backtick_count += 1

    if backtick_count % 2 == 1:
        if not in_template:
            in_template = True
            template_start = i
        else:
            in_template = False
            template_start = 0

if in_template:
    print(f"[ERROR] Unclosed template literal starting at line {template_start}")
    if template_start < len(lines):
        print(f"        {lines[template_start-1][:100]}")
