"""
Diagnose Paper2Code app initialization failure
Trace startup sequence and identify where it fails
"""

import asyncio
from playwright.async_api import async_playwright
import json

async def diagnose_startup():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Capture all console messages
        console_messages = []
        def on_console(msg):
            console_messages.append({
                "type": msg.type,
                "text": msg.text,
                "location": str(msg.location)
            })

        page.on("console", on_console)

        # Capture errors
        page_errors = []
        def on_error(err):
            page_errors.append(str(err))

        page.on("pageerror", on_error)

        # Navigate to app
        print("PAPER2CODE APP INITIALIZATION DIAGNOSTIC")
        print("=" * 70)
        print("\n1. Navigating to application...")

        await page.goto("http://127.0.0.1:8000/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)  # Wait for JS execution

        # Check window functions
        print("\n2. Checking window functions...")
        functions_exist = await page.evaluate("""
            ({
                router: typeof window.router === 'function',
                renderLibraryPage: typeof window.renderLibraryPage === 'function',
                renderExplorerPage: typeof window.renderExplorerPage === 'function',
                renderOverviewPage: typeof window.renderOverviewPage === 'function',
                parseRoute: typeof window.parseRoute === 'function',
                formatNumber: typeof window.formatNumber === 'function',
                renderGraph: typeof window.renderGraph === 'function'
            })
        """)

        for func, exists in functions_exist.items():
            status = "[OK]" if exists else "[FAIL]"
            print(f"   {status} window.{func}: {exists}")

        # Check DOM elements
        print("\n3. Checking DOM elements...")
        dom_elements = await page.evaluate("""
            ({
                appContainer: document.getElementById('app-container') !== null,
                navActions: document.getElementById('nav-actions') !== null,
                header: document.querySelector('header') !== null,
                body: document.body !== null
            })
        """)

        for elem, exists in dom_elements.items():
            status = "[OK]" if exists else "[FAIL]"
            print(f"   {status} {elem}: {exists}")

        # Check app container content
        print("\n4. Checking app-container content...")
        app_content = await page.evaluate("""
            ({
                isEmpty: document.getElementById('app-container')?.innerHTML.trim().length === 0,
                hasSpinner: document.getElementById('app-container')?.innerHTML.includes('spinner'),
                hasContent: document.getElementById('app-container')?.innerHTML.length > 0,
                innerHTMLLength: document.getElementById('app-container')?.innerHTML.length || 0
            })
        """)

        print(f"   isEmpty: {app_content['isEmpty']}")
        print(f"   hasSpinner: {app_content['hasSpinner']}")
        print(f"   hasContent: {app_content['hasContent']}")
        print(f"   HTML length: {app_content['innerHTMLLength']} chars")

        # Check current hash
        print("\n5. Checking router state...")
        router_state = await page.evaluate("""
            ({
                currentHash: window.location.hash,
                hasRouter: typeof window.router === 'function'
            })
        """)

        print(f"   Current hash: {router_state['currentHash']}")
        print(f"   Router function exists: {router_state['hasRouter']}")

        # Console messages
        print("\n6. Console messages...")
        if console_messages:
            print(f"   Total messages: {len(console_messages)}")
            for i, msg in enumerate(console_messages[:20]):  # First 20 messages
                print(f"   [{i+1}] {msg['type'].upper()}: {msg['text'][:100]}")
        else:
            print("   No console messages")

        # Page errors
        print("\n7. Page errors...")
        if page_errors:
            print(f"   [ERROR] {len(page_errors)} error(s) found:")
            for err in page_errors:
                print(f"   {err}")
        else:
            print("   No page errors")

        # Try to manually call router
        print("\n8. Testing manual router call...")
        try:
            router_result = await page.evaluate("""
                (() => {
                    try {
                        if (typeof window.router === 'function') {
                            window.router();
                            return {success: true, message: 'router() executed'};
                        } else {
                            return {success: false, message: 'router is not a function'};
                        }
                    } catch(e) {
                        return {success: false, message: e.message, stack: e.stack};
                    }
                })()
            """)
            print(f"   Result: {json.dumps(router_result, indent=2)}")
        except Exception as e:
            print(f"   Error calling router: {e}")

        # Check if any libraries failed to load
        print("\n9. Checking library availability...")
        libraries = await page.evaluate("""
            ({
                cytoscape: typeof cytoscape !== 'undefined',
                Chart: typeof Chart !== 'undefined',
                Monaco: typeof monaco !== 'undefined'
            })
        """)

        for lib, available in libraries.items():
            status = "[OK]" if available else "[WARN]"
            print(f"   {status} {lib}: {available}")

        # Get the current app container state
        print("\n10. Current app container HTML (first 500 chars)...")
        html_content = await page.evaluate("""
            document.getElementById('app-container')?.innerHTML || 'NO ELEMENT FOUND'
        """)

        if html_content != 'NO ELEMENT FOUND':
            print(f"   {html_content[:500]}")
        else:
            print(f"   {html_content}")

        # Final diagnosis
        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)

        if not functions_exist['router']:
            print("\n[CRITICAL] Router function not defined!")
            print("   Root cause: JavaScript execution did not complete")

        if app_content['isEmpty']:
            print("\n[CRITICAL] App container is empty!")
            print("   Possible causes:")
            print("   1. Router was not called")
            print("   2. renderLibraryPage() did not execute")
            print("   3. Error prevented rendering")

        if page_errors:
            print("\n[ERROR] Errors found in page:")
            for err in page_errors:
                print(f"   {err}")

        if console_messages:
            errors = [m for m in console_messages if m['type'] == 'error']
            if errors:
                print(f"\n[ERROR] Console errors ({len(errors)}):")
                for msg in errors:
                    print(f"   {msg['text']}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(diagnose_startup())
