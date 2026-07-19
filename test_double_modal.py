from playwright.sync_api import sync_playwright
import time

def test_double_modal():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("http://localhost:3000/dojo")
        
        # Wait for hydration and gate to appear
        page.wait_for_timeout(2000)
        
        # Click the Sign Up tab inside the AuthGuard form
        # We need to target the tab button specifically
        gate_div = page.locator('div.max-w-\\[420px\\]')
        if not gate_div.is_visible():
            print("Gate not visible!")
            browser.close()
            return
            
        signup_button = gate_div.locator("button:has-text('Sign Up')")
        signup_button.click()
        
        page.wait_for_timeout(1000)
        
        # Check how many forms/modals are visible
        modals = page.locator("form").count()
        print(f"Number of forms visible: {modals}")
        if modals > 1:
            print("Double modal issue confirmed!")
        else:
            print("Double modal issue NOT found.")
            
        browser.close()

test_double_modal()
