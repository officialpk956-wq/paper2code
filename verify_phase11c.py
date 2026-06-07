"""
Phase 11C Compute Heatmap verification across all 7 architectures.
Checks: graph renders, heatmap colors update, legend updates, node details, stage summary, no console errors.
"""
import asyncio
from playwright.async_api import async_playwright

ARCHITECTURES = [
    ("LeNet-5",            1),
    ("ResNet18",           6),
    ("ResNet50",           8),
    ("DenseNet121",        9),
    ("U-Net",             13),
    ("Transformer",       14),
    ("Vision Transformer", 15),
]
BASE = "http://127.0.0.1:8000"
MODES = ["flops", "params", "memory"]


async def check_arch(page, name, pid):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    await page.goto(f"{BASE}/#/explorer/{pid}", wait_until="domcontentloaded")
    await page.wait_for_timeout(2800)

    result = {"name": name}

    # 1. Graph renders (cy nodes present)
    node_count = await page.evaluate("""
        (() => {
            const cy = window.currentExplorerCy;
            return cy ? cy.nodes().length : 0;
        })()
    """)
    result["graph_renders"] = node_count > 0
    result["node_count"] = node_count

    # 2. Heatmap buttons exist
    btn_ids = await page.evaluate("""
        ['hm-none','hm-flops','hm-params','hm-memory'].map(id => !!document.getElementById(id))
    """)
    result["hm_buttons"] = all(btn_ids)

    # 3. Stage Compute Summary present (highest cost layer text)
    heat_summary = await page.evaluate("""
        (() => {
            const panels = document.querySelectorAll('.stage-detail-panel');
            const first = panels[0];
            if (!first) return false;
            return first.textContent.includes('Highest Cost Layer');
        })()
    """)
    result["stage_summary"] = heat_summary

    mode_results = {}
    for mode in MODES:
        # Click heatmap button
        await page.evaluate(f"window.applyHeatmap('{mode}')")
        await page.wait_for_timeout(400)

        # Check button is active (has accent-transformer color)
        btn_active = await page.evaluate(f"""
            (() => {{
                const btn = document.getElementById('hm-{mode}');
                return btn ? btn.style.color.includes('accent') || btn.style.color !== '' : false;
            }})()
        """)

        # Check legend is visible
        legend_visible = await page.evaluate("""
            (() => {
                const l = document.getElementById('heatmap-legend');
                return l ? l.style.display !== 'none' : false;
            })()
        """)

        # Check legend has items
        legend_items = await page.evaluate("""
            document.getElementById('heatmap-legend-items')?.children.length || 0
        """)

        # Check node colors changed (not all #374151)
        colors = await page.evaluate("""
            (() => {
                const cy = window.currentExplorerCy;
                if (!cy) return [];
                return cy.nodes().map(n => n.style('background-color'));
            })()
        """)
        unique_colors = list(set(colors)) if colors else []

        mode_results[mode] = {
            "legend_visible": legend_visible,
            "legend_items": legend_items,
            "unique_colors": len(unique_colors),
            "color_sample": unique_colors[:4],
        }

    result["modes"] = mode_results

    # 4. Click a node and check node detail panel appears
    clicked_node = await page.evaluate("""
        (() => {
            const cy = window.currentExplorerCy;
            if (!cy || cy.nodes().length === 0) return false;
            const node = cy.nodes()[1] || cy.nodes()[0];
            node.emit('tap');
            return true;
        })()
    """)
    await page.wait_for_timeout(300)

    panel_visible = await page.evaluate("""
        (() => {
            const p = document.getElementById('node-detail-panel');
            return p ? p.style.display !== 'none' : false;
        })()
    """)
    node_detail_content = await page.evaluate("""
        document.getElementById('node-detail-content')?.textContent.trim().substring(0, 80) || ''
    """)
    result["node_detail_visible"] = panel_visible
    result["node_detail_text"] = node_detail_content

    # 5. Reset to None and check nodes revert
    await page.evaluate("window.applyHeatmap('none')")
    await page.wait_for_timeout(300)
    legend_hidden = await page.evaluate("""
        (() => {
            const l = document.getElementById('heatmap-legend');
            return l ? l.style.display === 'none' : true;
        })()
    """)
    result["none_hides_legend"] = legend_hidden

    result["console_errors"] = errors
    result["pass"] = (
        result["graph_renders"] and
        result["hm_buttons"] and
        result["stage_summary"] and
        result["node_detail_visible"] and
        result["none_hides_legend"] and
        all(m["legend_visible"] and m["legend_items"] == 4 and m["unique_colors"] > 1 for m in mode_results.values()) and
        len(errors) == 0
    )
    return result


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("=" * 72)
        print("PHASE 11C VERIFICATION — Compute Heatmap")
        print("=" * 72)

        results = []
        for name, pid in ARCHITECTURES:
            r = await check_arch(page, name, pid)
            results.append(r)
            status = "PASS" if r["pass"] else "FAIL"
            print(f"\n--- {name} (#/explorer/{pid}) --- [{status}]")
            print(f"  graph_renders       : {r['graph_renders']} ({r['node_count']} nodes)")
            print(f"  hm_buttons          : {r['hm_buttons']}")
            print(f"  stage_summary       : {r['stage_summary']}")
            print(f"  node_detail_visible : {r['node_detail_visible']}")
            print(f"  node_detail_text    : {r['node_detail_text'][:60]}")
            print(f"  none_hides_legend   : {r['none_hides_legend']}")
            for mode, mr in r["modes"].items():
                print(f"  [{mode:8}] legend={mr['legend_visible']} items={mr['legend_items']} unique_colors={mr['unique_colors']} samples={mr['color_sample'][:2]}")
            if r["console_errors"]:
                print(f"  ERRORS: {r['console_errors'][:3]}")

        await browser.close()

        print("\n" + "=" * 72)
        print("SUMMARY TABLE")
        print("=" * 72)
        print(f"{'Architecture':<25} {'Graph':>6} {'Btns':>6} {'Summary':>8} {'NodeDet':>8} {'FLOPs':>6} {'Params':>7} {'Mem':>5} {'Errors':>7} {'STATUS':>7}")
        print("-" * 87)
        all_pass = True
        for r in results:
            m = r["modes"]
            p = r["pass"]
            all_pass = all_pass and p
            print(
                f"{r['name']:<25}"
                f" {'✓' if r['graph_renders'] else '✗':>6}"
                f" {'✓' if r['hm_buttons'] else '✗':>6}"
                f" {'✓' if r['stage_summary'] else '✗':>8}"
                f" {'✓' if r['node_detail_visible'] else '✗':>8}"
                f" {'✓' if m['flops']['legend_visible'] else '✗':>6}"
                f" {'✓' if m['params']['legend_visible'] else '✗':>7}"
                f" {'✓' if m['memory']['legend_visible'] else '✗':>5}"
                f" {'0' if not r['console_errors'] else str(len(r['console_errors'])):>7}"
                f" {'PASS' if p else 'FAIL':>7}"
            )
        print()
        print("OVERALL PHASE 11C:", "PASS" if all_pass else "FAIL")

asyncio.run(main())
