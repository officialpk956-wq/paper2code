import { expect, test, type Page } from '@playwright/test';

// UI contract tests: auth and execution are fixtures, never live E2B calls.
test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('access_token', 'ui-fixture');
    localStorage.setItem('user_profile', JSON.stringify({ id: 1, name: 'Tester', email: 'test@example.com' }));
  });
  await page.route('**/api/**', route => route.fulfill({ json: { leaders: [], submissions: [] } }));
  await page.route('**/api/dojo/runs', route => route.fulfill({ json: { passed: true, stdout: 'fixture execution output', stderr: '' } }));
  await page.route('**/api/dojo/code-submissions', route => route.fulfill({ json: {
    passed: true, num_passed: 1, total: 1,
    cases: [{ kind: 'sample', name: 'Test 1', passed: true, got: 0.5, expected: 0.5, time_ms: 0.5 }],
  } }));
  if (process.env.E2E_NEGATIVE_CONTROL === '1') {
    await page.route('**/dojo', route => route.fulfill({ status: 404, body: 'Missing feature' }));
  }
});

async function openProblem(page: Page) {
  const response = await page.goto('/dojo');
  expect(response?.status()).toBe(200);
  const link = page.locator('a[href="/dojo/ml-sigmoid"]').first();
  await expect(link).toBeVisible();
  await link.click();
  await expect(page.getByRole('heading', { name: 'Sigmoid Activation', exact: true }).first()).toBeVisible();
}

test('dojo index exposes a real problem link', async ({ page }) => {
  const response = await page.goto('/dojo');
  expect(response?.status()).toBe(200);
  await expect(page.locator('a[href="/dojo/ml-sigmoid"]').first()).toBeVisible();
});

test('problem renders a usable editor and back link', async ({ page }) => {
  await openProblem(page);
  await expect(page.locator('.monaco-editor').first()).toBeVisible();
  await expect(page.getByRole('button', { name: 'Run', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Submit', exact: true })).toBeVisible();
  await expect(page.getByRole('link', { name: /Problem List/ })).toHaveAttribute('href', '/dojo');
});

test('run posts code and renders the returned output', async ({ page }) => {
  await openProblem(page);
  const request = page.waitForRequest(r => r.url().endsWith('/api/dojo/runs') && r.method() === 'POST');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const body = (await request).postDataJSON();
  expect(body.problem_id).toBe('ml-sigmoid');
  expect(body.code).toContain('def sigmoid');
  await expect(page.getByText('Ran Successfully', { exact: true })).toBeVisible();
  await expect(page.getByText('fixture execution output', { exact: true })).toBeVisible();
});

test('submit renders the returned grading verdict', async ({ page }) => {
  await openProblem(page);
  const request = page.waitForRequest(r => r.url().endsWith('/api/dojo/code-submissions') && r.method() === 'POST');
  await page.getByRole('button', { name: 'Submit', exact: true }).click();
  expect((await request).postDataJSON().problem_id).toBe('ml-sigmoid');
  await expect(page.getByText('Accepted', { exact: true })).toBeVisible();
  await expect(page.getByText('Test 1', { exact: true })).toBeVisible();
});

test('mobile tabs expose code and console without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 667 });
  await openProblem(page);
  const tabs = page.getByTestId('mobile-tab-switcher');
  await expect(tabs).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await tabs.getByRole('button', { name: 'Code', exact: true }).click();
  await expect(page.getByText('Python 3', { exact: true })).toBeVisible();
  await tabs.getByRole('button', { name: 'Console', exact: true }).click();
  await expect(page.getByRole('button', { name: /Test Case/i })).toBeVisible();
});

test('desktop exposes editor and problem together', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openProblem(page);
  await expect(page.getByTestId('mobile-tab-switcher')).toBeHidden();
  await expect(page.locator('.monaco-editor').first()).toBeVisible();
  await expect(page.getByRole('button', { name: /description/i })).toBeVisible();
  await expect(page.getByRole('button', { name: /Test Case/i })).toBeVisible();
});
