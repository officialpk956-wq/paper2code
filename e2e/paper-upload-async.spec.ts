import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const ENABLED = process.env.RUN_REAL_ASYNC_PAPER_E2E === '1';
const EMAIL = 'phase2_async_e2e@example.com';
const PASSWORD = 'Phase2AsyncPass123!';

test.describe('real asynchronous paper upload', () => {
  test.skip(!ENABLED, 'requires RUN_REAL_ASYNC_PAPER_E2E=1, Redis, Celery, and the backend');
  test.setTimeout(12 * 60 * 1000);

  test('shows processing in the UI and redirects after task completion', async ({ page, request }) => {
    await request.post(`${API_URL}/api/auth/register`, {
      data: { email: EMAIL, name: 'Phase 2 Async E2E', password: PASSWORD },
      failOnStatusCode: false,
    });
    const login = await request.post(`${API_URL}/api/auth/login`, {
      form: { username: EMAIL, password: PASSWORD },
    });
    expect(login.ok()).toBeTruthy();
    const auth = await login.json();

    await page.addInitScript((session) => {
      localStorage.setItem('access_token', session.access_token);
      if (session.refresh_token) localStorage.setItem('refresh_token', session.refresh_token);
      localStorage.setItem('user_profile', JSON.stringify(session.user));
    }, auth);

    await page.goto('/papers');
    await expect(page.getByText('My Workspace')).toBeVisible();
    await page.getByRole('checkbox').check();

    const fixturePath = path.resolve('tests/fixtures/phase1_architecture.pdf.b64');
    const pdf = Buffer.from(readFileSync(fixturePath, 'utf8').trim(), 'base64');
    await page.locator('input[type="file"]').setInputFiles({
      name: 'phase2-async-resnet.pdf',
      mimeType: 'application/pdf',
      buffer: pdf,
    });

    await expect(page.getByRole('dialog', { name: 'Paper upload progress' })).toBeVisible();
    await expect(page.getByText('Preparing your paper workspace')).toBeVisible();
    await expect(page).toHaveURL(/\/papers\/\d+$/, { timeout: 10 * 60 * 1000 });
    await page.getByRole('button', { name: 'Executable' }).click();
    await expect(page.getByText('implementation.py')).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText('Phase 1 verified')).toBeVisible();
  });
});
