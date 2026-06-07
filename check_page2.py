import asyncio
from playwright.async_api import async_playwright

async def check_page_content():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto("http://127.0.0.1:8000/#/explorer/1", wait_until="networkidle")
        await page.wait_for_timeout(3000)  # Longer wait

        # Check what container IDs actually exist
        try:
            containers = await page.evaluate("""
                [
                    document.getElementById('app-container')?.innerHTML?.substring(0, 100) || 'NOT_FOUND',
                    document.getElementById('appContainer')?.innerHTML?.substring(0, 100) || 'NOT_FOUND',
                    document.querySelector('[id*="container"]')?.id || 'NOT_FOUND',
                    document.querySelector('[id*="app"]')?.id || 'NOT_FOUND'
                ]
            """)
            print("Container element checks:")
            print(f"  app-container: {containers[0]}")
            print(f"  appContainer: {containers[1]}")
            print(f"  Any element with 'container': {containers[2]}")
            print(f"  Any element with 'app': {containers[3]}")
        except Exception as e:
            print(f"Error checking containers: {e}")

        # Check if renderExplorerPage is defined
        try:
            result = await page.evaluate("""
                typeof renderExplorerPage
            """)
            print(f"\nrenderExplorerPage function: {result}")
        except:
            print("\nrenderExplorerPage function: ERROR checking")

        # Check if there's an error in the page
        try:
            result = await page.evaluate("""
                window.lastError || 'NO ERROR'
            """)
            print(f"Last error stored: {result}")
        except:
            pass

        # Check actual innerHTML of appContainer if it exists
        try:
            content = await page.evaluate("""
                document.getElementById('appContainer')?.innerHTML?.substring(0, 500) ||
                document.querySelector('main')?.innerHTML?.substring(0, 500) ||
                document.querySelector('#app')?.innerHTML?.substring(0, 500) ||
                'NOT FOUND'
            """)
            print(f"\nActual content in main container:\n{content}")
        except Exception as e:
            print(f"\nError getting content: {e}")

        # Get all script errors
        page_errors = []
        def on_console(msg):
            if msg.type == "error":
                page_errors.append(msg.text)

        page.on("console", on_console)
        await page.wait_for_timeout(1000)

        if page_errors:
            print(f"\nJavaScript errors found ({len(page_errors)}):")
            for err in page_errors[:5]:
                print(f"  {err}")
        else:
            print("\nNo JavaScript errors found")

        # Check what functions ARE available
        try:
            funcs = await page.evaluate("""
                Object.keys(window).filter(k => typeof window[k] === 'function').slice(0, 20)
            """)
            print(f"\nSome window functions available: {funcs}")
        except:
            pass

        await browser.close()

asyncio.run(check_page_content())
