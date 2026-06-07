"""
Find syntax errors by analyzing quote and bracket patterns
"""

with open("C:/papper2code/static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

import re

# Extract main script
match = re.search(r'<script>\s*(.*?)\s*</script>\s*<script src="/static/playground.js">', content, re.DOTALL)
if not match:
    print("Could not find script")
    exit(1)

script = match.group(1)
lines = script.split('\n')

print("SEARCHING FOR SYNTAX ERRORS - QUOTE/BRACKET ISSUES")
print("=" * 70)

# Look for lines with mismatched or problematic quote usage
for i, line in enumerate(lines, 1):
    # Skip comments
    if line.strip().startswith('//'):
        continue

    # Check for single quote within double quotes within single quotes
    if "'" in line and '"' in line:
        # Count them
        singles = line.count("'")
        doubles = line.count('"')

        # Look for patterns like onclick='...${...}..."...'
        if "onclick=" in line:
            print(f"[WARN] Line {i}: Complex onclick with mixed quotes")
            print(f"       {line.strip()[:120]}")

            # Check for unescaped quotes
            if line.count("'") % 2 != 0:
                print(f"       ^ ODD number of single quotes ({line.count("'")})")
            if line.count('"') % 2 != 0:
                print(f"       ^ ODD number of double quotes ({line.count('"')})")

# Look for template literals with complex nesting
print("\n" + "=" * 70)
print("TEMPLATE LITERAL ANALYSIS")
print("=" * 70)

in_template = False
template_start = 0
backtick_line = None

for i, line in enumerate(lines, 1):
    if line.strip().startswith('//'):
        continue

    # Count backticks on this line
    backtick_count = line.count('`')

    for j in range(backtick_count):
        if not in_template:
            in_template = True
            template_start = i
            backtick_line = i
        else:
            in_template = False

    # If we're in a template, check for problematic patterns
    if in_template and i > template_start:
        # Look for onclick with unbalanced quotes inside template
        if 'onclick=' in line:
            # Extract the onclick attribute
            match = re.search(r"onclick=['\"]([^'\"]*)['\"]", line)
            if match:
                onclick_content = match.group(1)
                if '${' in onclick_content:
                    print(f"Line {i}: Template literal with onclick containing template variable")
                    print(f"  {line.strip()[:100]}")

# Look for specific problematic character sequences
print("\n" + "=" * 70)
print("SPECIFIC PROBLEM PATTERNS")
print("=" * 70)

problematic_patterns = [
    (r"onclick=['\"].*\$\{.*?\}.*['\"]", "onclick with ${} variable"),
    (r"'[^']*\$\{[^}]*\}[^']*'", "Single-quoted string with template variable"),
    (r'"[^"]*\$\{[^}]*\}[^"]*"', "Double-quoted string with template variable"),
]

for pattern, desc in problematic_patterns:
    for i, line in enumerate(lines, 1):
        if i > 800 and i < 1000:  # Focus on area around line 907
            if re.search(pattern, line):
                print(f"Line {i}: {desc}")
                print(f"  {line.strip()[:120]}")

print("\n" + "=" * 70)
print("DETAILED CHECK OF LINE 907 AND SURROUNDING AREA")
print("=" * 70)

for i in range(900, 920):
    if i <= len(lines):
        line = lines[i-1]
        print(f"Line {i}: {line[:120]}")
