import { expect, test } from '@playwright/test';

for (const feature of ['guided', 'dojo']) {
  test(`${feature} discloses local-only storage and restores it on reload`, async ({ page }) => {
    await page.addInitScript(() => {
      localStorage.setItem('access_token', 'ui-fixture');
      localStorage.setItem('user_profile', JSON.stringify({ name: 'Tester', email: 'test@example.com' }));
    });
    await page.route('**/api/**', route => route.fulfill({ json: {} }));
    await page.goto(feature === 'guided' ? '/papers/resnet/implement' : '/dojo/ml-sigmoid');
    await expect(page.getByText(/Saved only in this browser/)).toBeVisible();
    if (feature === 'guided') {
      await page.evaluate(() => localStorage.setItem('impl_resnet_v1', JSON.stringify({ completed: [0] })));
    } else {
      await page.evaluate(() => localStorage.setItem('code_ml-sigmoid', '# local draft fixture'));
    }
    await page.reload();
    await expect(page.getByText(/Saved only in this browser/)).toBeVisible();
    if (feature === 'guided') await expect(page.getByText(/1\/\d+ steps complete/)).toBeVisible();
    else await expect(page.locator('.monaco-editor')).toContainText('# local draft fixture');
  });
}
