"""
Phase 11 Browser Testing Script
Tests Phase 11A (Explorer) and Phase 11B (Tensor Journey) across all architectures
"""

import asyncio
import json
from playwright.async_api import async_playwright
import sys
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"

# Test data
TEST_CASES = [
    {"name": "LeNet-5", "id": 1, "modules": 7},
    {"name": "ResNet18", "id": 6, "modules": 8},
    {"name": "ResNet50", "id": 8, "modules": 8},
    {"name": "DenseNet121", "id": 9, "modules": 11},
    {"name": "U-Net", "id": 13, "modules": 14},
    {"name": "Transformer", "id": 14, "modules": 26},
    {"name": "Vision Transformer", "id": 15, "modules": 26},
]

# Results tracking
results = {
    "timestamp": datetime.now().isoformat(),
    "base_url": BASE_URL,
    "architectures": {}
}

async def test_explorer_page(page, architecture_id, arch_name):
    """Test the explorer page for a single architecture"""

    arch_results = {
        "name": arch_name,
        "id": architecture_id,
        "explorer": "UNTESTED",
        "tensor_journey": "UNTESTED",
        "graph": "UNTESTED",
        "toggles": {"math": "UNTESTED", "code": "UNTESTED"},
        "console_errors": [],
        "network_issues": [],
        "observations": []
    }

    try:
        # Navigate to explorer page
        url = f"{BASE_URL}/#/explorer/{architecture_id}"
        print(f"\n{'='*60}")
        print(f"Testing: {arch_name} (ID: {architecture_id})")
        print(f"URL: {url}")
        print('='*60)

        await page.goto(url, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)  # Allow JS to render

        # 1. Test Page Load
        print("[OK] Page loaded")
        arch_results["explorer"] = "PASS"

        # 2. Check for console errors
        console_messages = []
        def on_console(msg):
            if msg.type in ["error", "warning"]:
                console_messages.append(f"{msg.type}: {msg.text}")

        page.on("console", on_console)

        # Wait for page to settle
        await page.wait_for_timeout(1000)
        arch_results["console_errors"] = console_messages
        if console_messages:
            arch_results["console"] = "FAIL"
            print(f"⚠️  Console errors found: {console_messages}")
        else:
            arch_results["console"] = "PASS"
            print("✓ No console errors")

        # 3. Test Timeline Rendering
        timeline_items = await page.query_selector_all(".stage-timeline-item")
        if timeline_items:
            print(f"✓ Timeline renders ({len(timeline_items)} stages)")
            arch_results["observations"].append(f"Timeline has {len(timeline_items)} stages")
        else:
            print("✗ Timeline not found")
            arch_results["explorer"] = "PARTIAL"

        # 4. Test Stage Cards
        stage_details = await page.query_selector_all(".stage-detail-panel")
        if stage_details:
            print(f"✓ Stage details render ({len(stage_details)} panels)")
            arch_results["observations"].append(f"Stage details have {len(stage_details)} panels")
        else:
            print("✗ Stage details not found")
            arch_results["explorer"] = "PARTIAL"

        # 5. Test Tensor Journey
        tensor_journey = await page.query_selector("#tensor-journey-container")
        if tensor_journey:
            print("✓ Tensor Journey container found")
            arch_results["tensor_journey"] = "PASS"

            # Check for shapes
            tensor_steps = await page.query_selector_all(".tensor-step-shape")
            print(f"  - Found {len(tensor_steps)} tensor steps")

            # Check shape values
            shapes_with_data = 0
            for step in tensor_steps[:5]:  # Check first 5
                text = await step.text_content()
                if text and text.strip() != "?" and text.strip() != "":
                    shapes_with_data += 1
                    print(f"    Shape: {text.strip()}")

            if shapes_with_data > 0:
                print(f"✓ Tensor shapes display correctly")
            else:
                print("✗ Tensor shapes missing or showing '?'")
                arch_results["tensor_journey"] = "PARTIAL"

        else:
            print("✗ Tensor Journey not found")
            arch_results["tensor_journey"] = "FAIL"

        # 6. Test FLOPs Display
        flops_elements = await page.query_selector_all(".tensor-step-info")
        if flops_elements:
            print(f"✓ FLOPs/Params display found ({len(flops_elements)} elements)")

            # Check a few for actual values
            for i, elem in enumerate(flops_elements[:3]):
                text = await elem.text_content()
                if "FLOPs" in text or "params" in text or "M" in text or "G" in text:
                    print(f"  - Element {i}: Contains metric data")
                    arch_results["observations"].append(f"FLOPs/Params display has data")
        else:
            print("⚠️  FLOPs/Params display not found")

        # 7. Test Graph Rendering
        graph_container = await page.query_selector("#cy-explorer")
        if graph_container:
            print("✓ Graph container found")

            # Check if Cytoscape rendered
            cytoscape_node = await page.query_selector(".cy-container canvas, .cy-container svg, [data-cy], .cytoscape")
            if cytoscape_node:
                print("✓ Graph appears to be rendered (canvas/SVG found)")
                arch_results["graph"] = "PASS"
            else:
                print("⚠️  Graph container exists but may not be fully rendered")
                arch_results["graph"] = "PARTIAL"
                arch_results["observations"].append("Graph container found but rendering unclear")
        else:
            print("✗ Graph container not found")
            arch_results["graph"] = "FAIL"

        # 8. Test Stage Selection (click on second stage if available)
        if len(timeline_items) > 1:
            print("\nTesting stage selection...")
            await timeline_items[1].click()
            await page.wait_for_timeout(500)
            print("✓ Clicked second stage")

            # Check if content updated
            visible_stages = await page.query_selector_all(".stage-detail-panel[style*='display: block']")
            if visible_stages:
                print(f"✓ Stage content updated after click ({len(visible_stages)} visible panels)")
            arch_results["observations"].append("Stage selection click works")

        # 9. Test Math Toggle
        print("\nTesting Math toggle...")
        math_button = await page.query_selector("button:has-text('Show Math')")
        if math_button:
            await math_button.click()
            await page.wait_for_timeout(300)
            print("✓ Math toggle button clicked")

            # Check if math content visible
            math_elements = await page.query_selector_all(".tensor-math:not([style*='display: none'])")
            if math_elements:
                print(f"✓ Math content visible after toggle ({len(math_elements)} elements)")
                arch_results["toggles"]["math"] = "PASS"
            else:
                print("⚠️  Math content may not be visible")
                arch_results["toggles"]["math"] = "PARTIAL"
        else:
            print("⚠️  Math toggle button not found")
            arch_results["toggles"]["math"] = "PARTIAL"

        # 10. Test Code Toggle
        print("\nTesting Code toggle...")
        code_button = await page.query_selector("button:has-text('Show Code')")
        if code_button:
            await code_button.click()
            await page.wait_for_timeout(300)
            print("✓ Code toggle button clicked")

            # Check if code content visible
            code_elements = await page.query_selector_all(".tensor-code:not([style*='display: none'])")
            if code_elements:
                print(f"✓ Code content visible after toggle ({len(code_elements)} elements)")
                arch_results["toggles"]["code"] = "PASS"
            else:
                print("⚠️  Code content may not be visible")
                arch_results["toggles"]["code"] = "PARTIAL"
        else:
            print("⚠️  Code toggle button not found")
            arch_results["toggles"]["code"] = "PARTIAL"

        # 11. Test Network Calls
        print("\nChecking network calls...")
        network_issues = []

        def on_response(response):
            if response.status >= 400:
                network_issues.append(f"{response.status} {response.url}")

        page.on("response", on_response)

        # Make a request to check modules endpoint
        try:
            modules_response = await page.evaluate(f"""
                fetch('/api/papers/{architecture_id}/modules')
                    .then(r => r.json())
                    .then(data => ({{'status': 'ok', 'modules': data.modules?.length || 0}}))
                    .catch(e => ({{'status': 'error', 'message': e.message}}))
            """)

            if modules_response['status'] == 'ok':
                print(f"✓ API endpoint working (returned {modules_response.get('modules', 0)} modules)")
                arch_results["network"] = "PASS"
            else:
                print(f"✗ API endpoint failed: {modules_response.get('message', 'unknown error')}")
                arch_results["network"] = "FAIL"
        except Exception as e:
            print(f"✗ Network test error: {str(e)}")
            arch_results["network"] = "FAIL"

        arch_results["network_issues"] = network_issues

        print("\n" + "="*60)
        print(f"SUMMARY FOR {arch_name}")
        print("="*60)
        print(f"Explorer: {arch_results['explorer']}")
        print(f"Tensor Journey: {arch_results['tensor_journey']}")
        print(f"Graph: {arch_results['graph']}")
        print(f"Math Toggle: {arch_results['toggles']['math']}")
        print(f"Code Toggle: {arch_results['toggles']['code']}")
        print(f"Console: {arch_results['console']}")
        print(f"Network: {arch_results['network']}")

    except Exception as e:
        print(f"✗ Error testing {arch_name}: {str(e)}")
        arch_results["explorer"] = "FAIL"
        arch_results["error"] = str(e)

    return arch_results


async def main():
    """Main test runner"""

    print("\n" + "="*70)
    print("PAPER2CODE PHASE 11 BROWSER VERIFICATION")
    print("="*70)
    print(f"Testing URL: {BASE_URL}")
    print(f"Test Time: {datetime.now().isoformat()}")
    print("="*70)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Test each architecture
        for test_case in TEST_CASES:
            arch_results = await test_explorer_page(page, test_case["id"], test_case["name"])
            results["architectures"][test_case["name"]] = arch_results

        await browser.close()

    # Print final report
    print("\n\n" + "="*70)
    print("FINAL VERIFICATION REPORT")
    print("="*70)

    for arch_name, arch_data in results["architectures"].items():
        print(f"\n{arch_name}:")
        print(f"  Explorer: {arch_data.get('explorer', 'UNTESTED')}")
        print(f"  Tensor Journey: {arch_data.get('tensor_journey', 'UNTESTED')}")
        print(f"  Graph: {arch_data.get('graph', 'UNTESTED')}")
        print(f"  Math Toggle: {arch_data.get('toggles', {}).get('math', 'UNTESTED')}")
        print(f"  Code Toggle: {arch_data.get('toggles', {}).get('code', 'UNTESTED')}")
        print(f"  Console: {arch_data.get('console', 'UNTESTED')}")
        print(f"  Network: {arch_data.get('network', 'UNTESTED')}")

        if arch_data.get("console_errors"):
            print(f"  Errors: {arch_data['console_errors']}")
        if arch_data.get("network_issues"):
            print(f"  Network Issues: {arch_data['network_issues']}")

    # Save results to file
    with open("C:/papper2code/phase11_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to phase11_test_results.json")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
