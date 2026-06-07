import asyncio
from playwright.async_api import async_playwright

async def check_page_content():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("http://127.0.0.1:8000/#/explorer/1", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Get page content
        content = await page.content()

        # Check for key elements
        print("PAGE CONTENT ANALYSIS:")
        print("=" * 60)

        if "appContainer" in content:
            print("[OK] appContainer div found")
        else:
            print("[FAIL] appContainer div NOT found")

        if "Stage Progression" in content:
            print("[OK] 'Stage Progression' text found")
        else:
            print("[FAIL] 'Stage Progression' text NOT found")

        if "Tensor Journey" in content:
            print("[OK] 'Tensor Journey' text found")
        else:
            print("[FAIL] 'Tensor Journey' text NOT found")

        if "ARCHITECTURE EXPLORER" in content:
            print("[OK] 'ARCHITECTURE EXPLORER' badge found")
        else:
            print("[FAIL] 'ARCHITECTURE EXPLORER' badge NOT found")

        if "stage-timeline-item" in content:
            print("[OK] 'stage-timeline-item' class found in HTML")
        else:
            print("[FAIL] 'stage-timeline-item' class NOT in HTML")

        # Check for spinner
        if "spinner" in content:
            print("[WARN] Spinner element present (indicates loading not complete)")

        # Check page title
        title = await page.title()
        print(f"\nPage title: {title}")

        # Check console messages
        messages = []
        def on_console(msg):
            messages.append(f"{msg.type}: {msg.text}")

        page.on("console", on_console)
        await page.wait_for_timeout(500)

        if messages:
            print(f"\nConsole messages ({len(messages)} total):")
            for msg in messages[:10]:
                print(f"  {msg}")

        # Try to find what's actually in appContainer
        try:
            inner_html = await page.evaluate("""
                document.getElementById('app-container')?.innerHTML?.substring(0, 300) ||
                'NOT FOUND'
            """)
            print(f"\napp-container content (first 300 chars):\n{inner_html}")
        except Exception as e:
            print(f"\nCould not get app-container content: {e}")

        # Check router function
        try:
            router_status = await page.evaluate("""
                typeof window.router
            """)
            print(f"\nRouter function status: {router_status}")
        except:
            print("\nCould not check router function")

        # Check current hash
        try:
            hash_value = await page.evaluate("""
                window.location.hash
            """)
            print(f"Current hash: {hash_value}")
        except:
            print("Could not check hash")

        await browser.close()

asyncio.run(check_page_content())
