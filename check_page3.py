import asyncio
from playwright.async_api import async_playwright

async def check_page_content():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("http://127.0.0.1:8000/#/explorer/1", wait_until="load")
        await page.wait_for_timeout(5000)  # Longer wait for JS execution

        # Get the actual HTML source
        content = await page.content()

        # Check if the main script is in the page
        print("SCRIPT ANALYSIS:")
        print("=" * 60)

        if "<script" in content:
            script_count = content.count("<script")
            print(f"[OK] Found {script_count} script tags")

            if "window.router" in content:
                print("[OK] 'window.router' definition found in HTML")
            else:
                print("[FAIL] 'window.router' definition NOT found in HTML")

            if "renderExplorerPage" in content:
                print("[OK] 'renderExplorerPage' function found in HTML")
            else:
                print("[FAIL] 'renderExplorerPage' function NOT found in HTML")

            if "parseRoute" in content:
                print("[OK] 'parseRoute' function found in HTML")
            else:
                print("[FAIL] 'parseRoute' function NOT found in HTML")
        else:
            print("[FAIL] No script tags found")

        # Try evaluating router directly
        print("\nFUNCTION EXECUTION TEST:")
        try:
            # Try to manually call router
            result = await page.evaluate("""
                (function() {
                    const hash = window.location.hash;
                    return {
                        hash: hash,
                        hasRouter: typeof router !== 'undefined',
                        hasRenderExplorer: typeof renderExplorerPage !== 'undefined',
                        hasParseRoute: typeof parseRoute !== 'undefined',
                        documentReady: document.readyState
                    };
                })()
            """)

            print(f"Hash: {result['hash']}")
            print(f"router defined: {result['hasRouter']}")
            print(f"renderExplorerPage defined: {result['hasRenderExplorer']}")
            print(f"parseRoute defined: {result['hasParseRoute']}")
            print(f"Document state: {result['documentReady']}")

        except Exception as e:
            print(f"Error: {e}")

        # Check if there's a console-level error  preventing script execution
        try:
            # Try to read a simple function that should be in index.html
            result = await page.evaluate("""
                typeof window.formatNumber === 'function'
            """)
            print(f"\nformatNumber function available: {result}")
        except:
            pass

        # Check playground.js
        try:
            result = await page.evaluate("""
                typeof window.initPlayground === 'function'
            """)
            print(f"initPlayground function available: {result}")
        except:
            pass

        await browser.close()

asyncio.run(check_page_content())
