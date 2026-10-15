import { test, expect, type Page } from '@playwright/test';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface } from 'node:readline';
import { existsSync } from 'node:fs';

// Real auth handlers and isolated database; only email/provider calls are replaced.
let child: ChildProcessWithoutNullStreams;
const waiting: ((value: any) => void)[] = [];
function command(message: object): Promise<any> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Auth bridge timed out')), 30000);
    waiting.push(value => { clearTimeout(timer); value.bridge_error ? reject(new Error(value.bridge_error)) : resolve(value); });
    child.stdin.write(JSON.stringify(message) + '\n');
  });
}
test.beforeAll(async () => {
  test.setTimeout(120000);
  const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : (existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python');
  child = spawn(python, ['-u', 'tests/auth_browser_bridge.py'], { stdio: 'pipe' });
  child.stderr.resume();
  await new Promise<void>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', code => reject(new Error(`Auth bridge exited: ${code}`)));
    createInterface({ input: child.stdout }).on('line', line => {
      if (!line.startsWith('REPAIR_JSON:')) return;
      const value = JSON.parse(line.slice('REPAIR_JSON:'.length));
      if (value.ready) resolve(); else waiting.shift()?.(value);
    });
  });
});
test.afterAll(() => { child?.stdin.end(); child?.kill(); });

async function connect(page: Page, expiredMfa = false) {
  await page.route('**/api/**', async route => {
    const req = route.request();
    const url = new URL(req.url());
    if (!url.pathname.startsWith('/api/auth/')) return route.fulfill({ json: {} });
    const response = await command({ action: 'request', method: req.method(), path: url.pathname + url.search, body: req.postData(), headers: req.headers() });
    if (expiredMfa && url.pathname === '/api/auth/login') {
      const data = JSON.parse(response.body);
      data.mfa_token = (await command({ action: 'expired_mfa' })).token;
      response.body = JSON.stringify(data);
    }
    await route.fulfill({ status: response.status, contentType: 'application/json', body: response.body });
  });
}

test('verification redeems signup token once and rejects reuse with unchanged database', async ({ page }) => {
  const user = await command({ action: 'seed' });
  expect((await command({ action: 'state' })).verified).toBe(false);
  await connect(page);
  await page.goto(`/auth/verified?token=${user.token}`);
  await expect.poll(async () => (await command({ action: 'state' })).verified).toBe(true);
  await expect(page.getByRole('heading', { name: 'Email verified', exact: true })).toBeVisible();
  const first = await command({ action: 'state' });
  expect(first.events).toBe(1);
  await page.reload();
  await expect(page.getByRole('heading', { name: 'Link already used' })).toBeVisible();
  expect(await command({ action: 'state' })).toEqual(first);
});

for (const state of ['expired', 'malformed', 'missing'] as const) {
  test(`verification truthfully rejects ${state} token`, async ({ page }) => {
    const user = await command({ action: 'seed', expired: state === 'expired' });
    await connect(page);
    await page.goto('/auth/verified' + (state === 'missing' ? '' : `?token=${state === 'expired' ? user.token : 'malformed-token'}`));
    await expect(page.getByRole('heading', { name: { expired: 'Link expired', malformed: 'Invalid verification link', missing: 'Verification token missing' }[state], exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Email verified', exact: true })).toHaveCount(0);
    expect((await command({ action: 'state' })).verified).toBe(false);
  });
}
test('verification stays pending while request is in flight', async ({ page }) => {
  let release!: () => void;
  const paused = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/api/auth/verify-email', async route => { await paused; await route.fulfill({ json: { status: 'ok' } }); });
  await page.goto('/auth/verified?token=pending-fixture');
  try { await expect(page.getByRole('status')).toHaveText('Verifying your email…'); }
  finally { release(); }
});
test('both mail senders link to the existing verification page', async ({ page }) => {
  const html: string[] = await command({ action: 'senders' });
  for (const body of html) {
    const link = new URL(body.match(/href="([^"]*token=[^"]*)"/)![1]);
    expect(link.pathname).toBe('/auth/verified');
    expect((await page.goto(link.pathname))?.status()).toBe(200);
    await expect(page.getByRole('heading', { name: 'Verification token missing' })).toBeVisible();
  }
});

for (const scenario of ['valid', 'wrong', 'expired', 'backup'] as const) {
  test(`MFA ${scenario}: no session before challenge, correct result after code`, async ({ page }) => {
    const user = await command({ action: 'seed', mfa: true });
    await connect(page, scenario === 'expired');
    await page.goto('/learn');
    await page.getByPlaceholder('Email', { exact: true }).fill(user.email);
    await page.getByPlaceholder('Password', { exact: true }).fill(user.password);
    const loginDone = page.waitForResponse(r => r.url().endsWith('/api/auth/login'));
    await page.locator('form button[type="submit"]').click();
    await loginDone;
    await expect.poll(() => page.evaluate(() => localStorage.getItem('access_token'))).toBeNull();
    await expect(page.getByRole('heading', { name: 'Two-step verification' })).toBeVisible();
    expect(await page.evaluate(() => localStorage.getItem('access_token'))).toBeNull();
    const code = scenario === 'backup' ? user.backup : scenario === 'wrong' ? 'not-a-code' : (await command({ action: 'code' })).code;
    await page.getByLabel('Authentication or backup code').fill(code);
    await page.getByRole('button', { name: 'Verify and sign in' }).click();
    if (scenario === 'wrong' || scenario === 'expired') {
      await expect(page.locator('form').getByRole('alert')).toContainText(scenario === 'wrong' ? 'Invalid MFA code' : 'Invalid or expired MFA token');
      expect(await page.evaluate(() => localStorage.getItem('access_token'))).toBeNull();
    } else {
      await expect(page.getByRole('heading', { name: 'Two-step verification' })).toHaveCount(0);
      const token = await page.evaluate(() => localStorage.getItem('access_token'));
      expect(token).toMatch(/^ey/);
      const me = await command({ action: 'request', method: 'GET', path: '/api/auth/me', headers: { Authorization: `Bearer ${token}` } });
      expect(me.status).toBe(200);
      expect(JSON.parse(me.body).email).toBe(user.email);
      if (scenario === 'backup') expect((await command({ action: 'state' })).backups).toBe(0);
    }
  });
}

test('ordinary modal login keeps a working session; MFA modal waits for verification', async ({ page }) => {
  await connect(page);
  for (const mfa of [false, true]) {
    const user = await command({ action: 'seed', mfa });
    await page.goto('/');
    await page.getByRole('button', { name: 'Sign In', exact: true }).first().click();
    await page.getByPlaceholder('Email', { exact: true }).fill(user.email);
    await page.getByPlaceholder('Password', { exact: true }).fill(user.password);
    await page.locator('form button[type="submit"]').click();
    if (mfa) {
      await expect(page.getByRole('heading', { name: 'Two-step verification' })).toBeVisible();
      expect(await page.evaluate(() => localStorage.getItem('access_token'))).toBeNull();
      await page.getByLabel('Authentication or backup code').fill((await command({ action: 'code' })).code);
      await page.getByRole('button', { name: 'Verify and sign in' }).click();
    }
    await expect(page.locator('form')).toHaveCount(0);
    const token = await page.evaluate(() => localStorage.getItem('access_token'));
    const response = await command({ action: 'request', method: 'GET', path: '/api/auth/me', headers: { Authorization: `Bearer ${token}` } });
    expect(response.status).toBe(200);
    expect(JSON.parse(response.body).email).toBe(user.email);
    await page.evaluate(() => localStorage.clear());
    await page.reload();
  }
});


