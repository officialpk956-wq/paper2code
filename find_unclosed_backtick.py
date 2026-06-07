"""
Find unclosed template literal in index.html
"""

import re

with open("C:/papper2code/static/index.html", "r", encoding="utf-8") as f:
    content = f.read()

# Extract main script block
match = re.search(r'<script>\s*(.*?)\s*</script>\s*<script src="/static/playground.js">', content, re.DOTALL)
if not match:
    print("Could not find script block")
    exit(1)

script = match.group(1)
lines = script.split('\n')

print("TRACKING TEMPLATE LITERAL BACKTICKS")
print("=" * 70)

# Track backtick state
backtick_stack = []
total_backticks = 0
errors = []

for line_num, line in enumerate(lines, 1):
    # Skip comments
    if line.strip().startswith('//'):
        continue

    # Count backticks, but be careful about escaping
    i = 0
    while i < len(line):
        # Check for escaped backtick
        if i > 0 and line[i-1] == '\\' and line[i] == '`':
            # Escaped backtick - skip it
            i += 1
            continue

        if line[i] == '`':
            # Found a backtick
            total_backticks += 1

            # Check if we're in a string or not
            # This is tricky because we need to track quote context
            in_single = False
            in_double = False

            # Simple heuristic: if there's an odd number of unescaped quotes before this backtick on this line
            before_backtick = line[:i]

            # Count quotes before
            single_quotes = 0
            double_quotes = 0
            j = 0
            while j < len(before_backtick):
                if before_backtick[j] == '"' and (j == 0 or before_backtick[j-1] != '\\'):
                    double_quotes += 1
                elif before_backtick[j] == "'" and (j == 0 or before_backtick[j-1] != '\\'):
                    single_quotes += 1
                j += 1

            if backtick_stack and backtick_stack[-1] == line_num:
                # This is a closing backtick for the current line's opening
                backtick_stack.pop()
            elif not backtick_stack:
                # This opens a template literal
                backtick_stack.append(line_num)
            else:
                # Check if we're still inside a template literal
                if backtick_stack:
                    backtick_stack.pop()
                else:
                    backtick_stack.append(line_num)

            if line_num % 100 == 0 or (line_num > 3750 and line_num < 3820):
                print(f"Line {line_num}: Backtick #{total_backticks}, Stack depth: {len(backtick_stack)}")
                if line.strip():
                    print(f"         {line.strip()[:100]}")

        i += 1

print()
print("=" * 70)
print(f"Total backticks: {total_backticks}")
print(f"Stack depth at end: {len(backtick_stack)}")

if backtick_stack:
    print(f"\nUNCLOSED TEMPLATE LITERALS - Started on lines: {backtick_stack}")
    for opened_line in backtick_stack:
        print(f"\n  Line {opened_line}:")
        if opened_line <= len(lines):
            print(f"  {lines[opened_line-1].strip()[:100]}")
else:
    print("\nAll template literals properly closed!")

# Check for specific problematic lines
print("\n" + "=" * 70)
print("CHECKING SPECIFIC AREAS NEAR END OF SCRIPT")
print("=" * 70)

for i in range(max(0, len(lines)-50), len(lines)):
    line = lines[i]
    if '`' in line:
        print(f"Line {i+1}: {line.strip()[:100]}")
