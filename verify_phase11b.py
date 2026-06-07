"""
Phase 11B fix verification — checks tensor shapes are populated (not '?') across all 7 architectures.
"""
import asyncio
from playwright.async_api import async_playwright

ARCHITECTURES = [
    ("LeNet-5",          1),
    ("ResNet18",         6),
    ("ResNet50",         8),
    ("DenseNet121",      9),
    ("U-Net",           13),
    ("Transformer",     14),
    ("Vision Transformer", 15),
]

BASE = "http://127.0.0.1:8000"

async def check_arch(page, name, pid):
    url = f"{BASE}/#/explorer/{pid}"
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(2500)

    # Click first timeline stage to ensure tensor journey is rendered
    stage_btn = await page.query_selector(".stage-timeline-item")
    if stage_btn:
        await stage_btn.click()
        await page.wait_for_timeout(1000)

    # Gather all .tensor-step-shape texts
    shapes = await page.evaluate("""
        Array.from(document.querySelectorAll('.tensor-step-shape')).map(el => el.textContent.trim())
    """)

    total = len(shapes)
    question_marks = shapes.count('?')
    populated = total - question_marks
    samples = shapes[:6]

    return {
        "name": name,
        "total": total,
        "question_marks": question_marks,
        "populated": populated,
        "samples": samples,
        "console_errors": errors,
        "pass": total > 0 and question_marks == 0,
    }

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("=" * 70)
        print("PHASE 11B VERIFICATION — Tensor Shape Fallback Fix")
        print("=" * 70)

        results = []
        for name, pid in ARCHITECTURES:
            r = await check_arch(page, name, pid)
            results.append(r)
            status = "PASS" if r["pass"] else "FAIL"
            print(f"\n--- {name} (#/explorer/{pid}) --- [{status}]")
            print(f"  Shapes total    : {r['total']}")
            print(f"  Populated (!=?) : {r['populated']}")
            print(f"  '?' count       : {r['question_marks']}")
            print(f"  Samples         : {r['samples']}")
            if r["console_errors"]:
                print(f"  Console errors  : {r['console_errors']}")

        await browser.close()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        all_pass = all(r["pass"] for r in results)
        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  {r['name']:<25} {status}  (populated={r['populated']}/{r['total']}, ?={r['question_marks']})")
        print()
        print("OVERALL PHASE 11B:", "PASS" if all_pass else "FAIL")

asyncio.run(main())
