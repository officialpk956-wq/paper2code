import asyncio
from playwright.async_api import async_playwright
import json

async def test_js_error():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Capture page errors
        error_details = []
        def on_error(error_str):
            error_details.append(error_str)

        page.on("pageerror", on_error)

        # Goto the page
        await page.goto("http://127.0.0.1:8000/", wait_until="load")
        await page.wait_for_timeout(3000)

        print("PAGE ERRORS CAPTURED:")
        print("=" * 70)
        for err in error_details:
            print(f"ERROR: {err}\n")

        # Try to get a more detailed error
        try:
            result = await page.evaluate("""
                async function testCompile() {
                    const scriptContent = document.querySelector('script:not([src])').textContent;
                    try {
                        new Function(scriptContent);
                        return 'Script compiled successfully';
                    } catch(e) {
                        return {
                            message: e.message,
                            name: e.name,
                            stack: e.stack,
                            line: e.lineNumber,
                            column: e.columnNumber
                        };
                    }
                }
                testCompile();
            """)
            print("\nCOMPILATION TEST RESULT:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"\nCould not test compilation: {e}")

        # Try to find the exact error line by checking source
        try:
            result = await page.evaluate("""
                document.querySelector('script:not([src])').textContent.split('\\n').findIndex((line, i) => {
                    try {
                        new Function(line);
                    } catch(e) {
                        console.log(`Error on potential line: ${i+1}: ${line.substring(0,100)}`);
                        return true;
                    }
                    return false;
                });
            """)
            print(f"\nFirst problematic line around: {result}")
        except:
            pass

        await browser.close()

asyncio.run(test_js_error())
