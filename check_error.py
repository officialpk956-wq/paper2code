import asyncio
from playwright.async_api import async_playwright

async def check_errors():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Capture ALL console messages
        all_messages = []
        def on_console(msg):
            all_messages.append({
                "type": msg.type,
                "text": msg.text,
                "location": msg.location
            })

        page.on("console", on_console)

        # Also capture page errors
        page_errors = []
        def on_error(error):
            page_errors.append(str(error))

        page.on("pageerror", on_error)

        await page.goto("http://127.0.0.1:8000/#/explorer/1", wait_until="load")
        await page.wait_for_timeout(5000)

        print("ALL CONSOLE MESSAGES:")
        print("=" * 60)
        if all_messages:
            for msg in all_messages:
                print(f"[{msg['type']}] {msg['text']}")
                if msg.get('location'):
                    print(f"  Location: {msg['location']}")
        else:
            print("No console messages")

        print("\nPAGE ERRORS:")
        print("=" * 60)
        if page_errors:
            for err in page_errors:
                print(f"ERROR: {err}")
        else:
            print("No page errors")

        # Try to manually evaluate the router function
        print("\nDIRECT EVAL TEST:")
        print("=" * 60)
        try:
            # Get the parseRoute function definition from the page
            func_def = await page.evaluate("""
                document.querySelector('script:not([src])')?.textContent?.substring(0, 500)
            """)
            print(f"Script content preview: {func_def}")
        except Exception as e:
            print(f"Error: {e}")

        # Try running router() directly
        try:
            result = await page.evaluate("""
                (() => {
                    try {
                        router();
                        return 'router() executed successfully';
                    } catch(e) {
                        return `router() error: ${e.message}`;
                    }
                })()
            """)
            print(f"\nDirect router() call: {result}")
        except Exception as e:
            print(f"Error calling router: {e}")

        await browser.close()

asyncio.run(check_errors())
