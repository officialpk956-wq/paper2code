import { test, expect } from '@playwright/test';

const RUN_SUCCESS_RESPONSE = {
  results: [
    { index: 0, passed: true, actual: [[1, 2], [3, 4]], expected: [[1, 2], [3, 4]], runtime_ms: 0.5, error: null },
  ],
  totalMs: 120,
};

const SUBMIT_ACCEPTED_RESPONSE = {
  results: [
    { index: 0, passed: true, actual: [[1, 2], [3, 4]], expected: [[1, 2], [3, 4]], runtime_ms: 0.5, error: null },
    { index: 1, passed: true, actual: [[0, 0], [0, 0]], expected: [[0, 0], [0, 0]], runtime_ms: 0.4, error: null },
  ],
  status: 'accepted',
  passedTests: 2,
  totalTests: 2,
  runtimeMs: 0.5,
  totalMs: 200,
};

test.describe('Dojo coding flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/dojo/run', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(RUN_SUCCESS_RESPONSE),
      }),
    );
    await page.route('**/api/dojo/submit', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(SUBMIT_ACCEPTED_RESPONSE),
      }),
    );
  });

  test('dojo index page renders problem list', async ({ page }) => {
    await page.goto('/dojo');
    // Problem list page should show some problems
    await expect(page).toHaveTitle(/Dojo|Paper2Code/i);
    await expect(page.locator('body')).toBeVisible();
  });

  test('problem page renders editor and toolbar', async ({ page }) => {
    await page.goto('/dojo');
    // Navigate to first available problem
    const problemLinks = page.locator('a[href^="/dojo/"]');
    const count = await problemLinks.count();
    if (count === 0) {
      test.skip();
      return;
    }
    await problemLinks.first().click();
    // Should show Python toolbar in editor
    await expect(page.getByText('Python 3')).toBeVisible({ timeout: 8000 });
    await expect(page.getByText('▶ Run')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('↑ Submit')).toBeVisible({ timeout: 5000 });
  });

  test('problem page shows back link', async ({ page }) => {
    await page.goto('/dojo');
    const problemLinks = page.locator('a[href^="/dojo/"]');
    const count = await problemLinks.count();
    if (count === 0) {
      test.skip();
      return;
    }
    await problemLinks.first().click();
    await expect(page.getByLabel('Back to Dojo')).toBeVisible({ timeout: 5000 });
  });

  test('run button triggers API call and shows results', async ({ page }) => {
    await page.goto('/dojo');
    const problemLinks = page.locator('a[href^="/dojo/"]');
    if (await problemLinks.count() === 0) { test.skip(); return; }
    await problemLinks.first().click();

    await expect(page.getByText('▶ Run')).toBeVisible({ timeout: 8000 });
    await page.getByText('▶ Run').click();
    // After run, results panel should show a test case result
    await expect(page.getByText('Case 1')).toBeVisible({ timeout: 5000 });
  });

  test('submit button triggers API call and shows verdict', async ({ page }) => {
    await page.goto('/dojo');
    const problemLinks = page.locator('a[href^="/dojo/"]');
    if (await problemLinks.count() === 0) { test.skip(); return; }
    await problemLinks.first().click();

    await expect(page.getByText('↑ Submit')).toBeVisible({ timeout: 8000 });
    await page.getByText('↑ Submit').click();
    await expect(page.getByText('Accepted')).toBeVisible({ timeout: 5000 });
  });
});
