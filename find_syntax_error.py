"""
Find the exact syntax error in index.html JavaScript
"""

import re
from pathlib import Path

def extract_main_script():
    """Extract the main script block from index.html"""
    with open("C:/papper2code/static/index.html", "r", encoding="utf-8") as f:
        content = f.read()

    # Find the main script block (not the one with src attribute)
    match = re.search(r'<script>\s*(.*?)\s*</script>\s*<script src="/static/playground.js">', content, re.DOTALL)

    if match:
        script_content = match.group(1)
        return script_content, content.find(match.group(1))
    return None, None

def find_syntax_error():
    """Try to find syntax errors by checking various patterns"""

    script, offset = extract_main_script()
    if not script:
        print("Could not extract main script")
        return

    lines = script.split('\n')

    print("SYSTEMATIC SYNTAX ERROR SEARCH")
    print("=" * 70)
    print(f"Total lines in main script: {len(lines)}")
    print()

    # Check 1: Look for unclosed template literals
    print("CHECK 1: Template Literals (backticks)")
    print("-" * 70)
    backtick_count = script.count('`')
    if backtick_count % 2 != 0:
        print(f"[WARN] Odd number of backticks: {backtick_count}")
        # Find which line has the problem
        for i, line in enumerate(lines, 1):
            if '`' in line:
                count = line.count('`')
                if count % 2 != 0:
                    print(f"       Line {i}: {line.strip()[:80]}")
    else:
        print(f"[OK] Backticks balanced: {backtick_count}")
    print()

    # Check 2: Look for unescaped quotes in strings
    print("CHECK 2: String Quotes")
    print("-" * 70)
    issues = []
    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('//'):
            continue

        # Look for potential quote issues
        if 'localStorage' in line or 'fetch' in line:
            single_quotes = line.count("'")
            double_quotes = line.count('"')
            if single_quotes % 2 != 0:
                issues.append((i, line, f"Odd single quotes ({single_quotes})"))
            if double_quotes % 2 != 0:
                issues.append((i, line, f"Odd double quotes ({double_quotes})"))

    if issues:
        print(f"[WARN] Found {len(issues)} potential quote issues:")
        for line_num, line, issue in issues[:10]:
            print(f"       Line {line_num}: {issue}")
            print(f"              {line.strip()[:80]}")
    else:
        print("[OK] Quote balancing looks OK")
    print()

    # Check 3: Look for problematic characters
    print("CHECK 3: Special Characters")
    print("-" * 70)
    for i, line in enumerate(lines, 1):
        # Check for common problem characters
        if '\x00' in line:
            print(f"[ERROR] Null character at line {i}")
            issues.append((i, "Null character", line[:50]))
        if '—' in line or '–' in line:
            print(f"[WARN] Em-dash or en-dash at line {i}: {line.strip()[:80]}")
        if '…' in line:
            print(f"[WARN] Ellipsis character at line {i}")
    print()

    # Check 4: Look for regex issues
    print("CHECK 4: Regular Expressions")
    print("-" * 70)
    regex_lines = []
    for i, line in enumerate(lines, 1):
        if '/' in line and ('match' in line or 'test' in line or 'replace' in line):
            # Could be a regex
            if line.count('/') % 2 != 0:
                regex_lines.append((i, line))

    if regex_lines:
        print(f"[WARN] Found {len(regex_lines)} potential regex issues:")
        for line_num, line in regex_lines[:5]:
            print(f"       Line {line_num}: {line.strip()[:80]}")
    else:
        print("[OK] No obvious regex issues")
    print()

    # Check 5: HTML inside JavaScript
    print("CHECK 5: HTML Embedded in JavaScript")
    print("-" * 70)
    html_lines = []
    for i, line in enumerate(lines, 1):
        if '`<' in line or "'>'" in line or '"><' in line:
            html_lines.append((i, line))

    if html_lines:
        print(f"[INFO] Found {len(html_lines)} lines with embedded HTML")
        print("       Checking first few:")
        for line_num, line in html_lines[:5]:
            # Check for unescaped backticks in HTML strings
            print(f"       Line {line_num}: {line.strip()[:100]}")
            if '${' in line and '`' in line:
                print(f"              ^ Contains template literal")
    print()

    # Check 6: Braces/brackets balance by line
    print("CHECK 6: Brace Balance Issues")
    print("-" * 70)
    open_count = 0
    close_count = 0
    paren_issues = []
    for i, line in enumerate(lines, 1):
        # Count opening and closing
        line_open = line.count('{')
        line_close = line.count('}')
        if line_open != line_close:
            if abs(line_open - line_close) > 1:  # More than normal variance
                paren_issues.append((i, line, f"Open: {line_open}, Close: {line_close}"))
        open_count += line_open
        close_count += line_close

    print(f"Overall balance: {open_count} open vs {close_count} close")
    if paren_issues:
        print(f"[INFO] Lines with unusual brace counts:")
        for line_num, line, msg in paren_issues[:5]:
            print(f"       Line {line_num}: {msg}")
            print(f"              {line.strip()[:80]}")
    print()

    # Check 7: Look at playground.js reference
    print("CHECK 7: Script Closing and Next Script")
    print("-" * 70)
    if '</script>' in script:
        print("[ERROR] Found </script> tag INSIDE the main script block!")
        idx = script.find('</script>')
        line_num = script[:idx].count('\n') + 1
        print(f"        Located at line {line_num}")
    else:
        print("[OK] No </script> tag found inside main script")
    print()

    # Check 8: Try to find the actual syntax error by checking for common patterns
    print("CHECK 8: Known Problem Patterns")
    print("-" * 70)

    # Look for specific problematic patterns
    problem_patterns = [
        (r'\$\{[^}]*\$\{', "Nested template literal variables"),
        (r'return \(', "Return with unclosed parenthesis"),
        (r'\.then\([^)]*\$', "Promise chain with incomplete variable"),
        (r'`[^`]*$', "Unclosed template literal at end of line"),
    ]

    for pattern, description in problem_patterns:
        matches = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                matches.append((i, line))

        if matches:
            print(f"[WARN] Pattern: {description}")
            for line_num, line in matches[:3]:
                print(f"       Line {line_num}: {line.strip()[:80]}")
    print()

    # Final check: Look at around line 248-258 where we know fetch wrapper is
    print("CHECK 9: Known Problematic Area (fetch wrapper)")
    print("-" * 70)
    print("Examining lines 240-260:")
    for i, line in enumerate(lines[240:260], 241):
        print(f"{i:4d}: {line}")

if __name__ == "__main__":
    find_syntax_error()
