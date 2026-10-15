import { expect, test } from '@playwright/test';
import estimates from '../src/__tests__/fixtures/training-estimates.json';

test.beforeEach(async ({ page }) => {
  if (process.env.E2E_NEGATIVE_CONTROL === '1') {
    await page.route('**/labs', route => route.fulfill({ status: 404, body: 'Missing feature' }));
  }
  await page.addInitScript(() => {
    localStorage.setItem('access_token', 'ui-fixture');
    localStorage.setItem('user_profile', JSON.stringify({ name: 'Tester', email: 'test@example.com' }));
  });
  await page.route('**/api/**', route => route.fulfill({ json: {} }));
});

for (const [family, result] of Object.entries(estimates.cases)) {
  test(`estimator renders backend units: ${family}`, async ({ page }) => {
    page.on('console', message => { if (message.type() === 'error') console.log('BROWSER:', message.text()); });
    await page.route('**/api/training-estimator', route => {
      expect(route.request().postDataJSON()).toEqual({ architecture: family, ...estimates.request });
      return route.fulfill({ json: result });
    });
    await page.goto('/labs');
    await page.getByRole('button', { name: family, exact: true }).click();
    await page.getByRole('button', { name: 'Estimate Training Cost' }).click();
    await expect(page.getByText('Est. Time', { exact: true }).locator('..')).toContainText(result.training_hours.toFixed(1));
    await expect(page.getByText('Est. Cost', { exact: true }).locator('..')).toContainText(`$${result.compute_cost_usd.toFixed(2)}`);
    await expect(page.getByText('VRAM Required', { exact: true }).locator('..')).toContainText(result.gpu_memory_gb.toFixed(1));
    await expect(page.getByText('Training Steps', { exact: true }).locator('..')).toContainText(String(result.steps_total));
    await expect(page.getByText(/NaN|Infinity|Workspace Error/)).toHaveCount(0);
  });
}

test('invalid input never sends an estimator request', async ({ page }) => {
  let requests = 0;
  await page.route('**/api/training-estimator', route => { requests++; return route.fulfill({ json: estimates.cases.ResNet }); });
  await page.goto('/labs');
  await page.locator('input[type="number"]').first().fill('0');
  await page.getByRole('button', { name: 'Estimate Training Cost' }).click();
  await expect(page.getByText('Dataset size and batch size must be positive whole numbers.')).toBeVisible();
  expect(requests).toBe(0);
});

test('backend failure is a useful error and can be retried', async ({ page }) => {
  let requests = 0;
  await page.route('**/api/training-estimator', route => {
    requests++;
    return requests === 1
      ? route.fulfill({ status: 500, json: { detail: 'Estimator temporarily unavailable' } })
      : route.fulfill({ json: estimates.cases.ResNet });
  });
  await page.goto('/labs');
  await page.getByRole('button', { name: 'Estimate Training Cost' }).click();
  await expect(page.getByText('Estimator temporarily unavailable')).toBeVisible();
  await page.getByRole('button', { name: 'Estimate Training Cost' }).click();
  await expect(page.getByText('Est. Time', { exact: true })).toBeVisible();
});

test('malformed response fails visibly instead of inventing zeroes', async ({ page }) => {
  await page.route('**/api/training-estimator', route => route.fulfill({ json: { training_hours: 1 } }));
  await page.goto('/labs');
  await page.getByRole('button', { name: 'Estimate Training Cost' }).click();
  await expect(page.getByText('The estimator returned an invalid response. Please try again.')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Workspace Error' })).toHaveCount(0);
});
