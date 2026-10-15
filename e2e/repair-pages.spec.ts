import { expect, test } from '@playwright/test';

for (const route of [
  { path: '/learn', title: 'Curriculum', link: '/learn/mathematics' },
  { path: '/system-design', title: 'AI System Design', link: '/system-design/rag-systems' },
]) {
  for (const signedIn of [true, false]) {
    for (const navigation of ['direct', 'client']) {
      test(`${route.path} ${signedIn ? 'signed in' : 'signed out'} ${navigation}`, async ({ page }) => {
        const errors: string[] = [];
        page.on('pageerror', error => errors.push(error.message));
        page.on('console', message => { if (message.type() === 'error') { errors.push(message.text()); console.log('BROWSER:', message.text()); } });
        await page.route('**/api/**', r => r.fulfill({ json: {} }));
        if (signedIn) {
          await page.addInitScript(() => {
            localStorage.setItem('access_token', 'ui-fixture');
            localStorage.setItem('user_profile', JSON.stringify({ name: 'Tester', email: 'test@example.com' }));
          });
        }
        if (navigation === 'direct') {
          const response = await page.goto(route.path);
          expect(response?.status()).toBe(200);
        } else {
          await page.goto('/about');
          await page.locator('header nav button').filter({ hasText: 'Learn' }).hover();
          await page.locator(`header nav a[href="${route.path}"]`).click();
          await expect(page).toHaveURL(new RegExp(`${route.path}$`));
        }
        await expect(page.getByRole('heading', { name: route.title, exact: true }).first()).toBeVisible();
        if (signedIn) await expect(page.locator(`a[href="${route.link}"]`).first()).toBeVisible();
        else await expect(page.getByText('Sign in to continue', { exact: true })).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Workspace Error' })).toHaveCount(0);
        expect(errors).toEqual([]);
      });
    }
  }
}
