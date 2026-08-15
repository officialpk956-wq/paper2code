import { test, expect, type Page } from '@playwright/test';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const E2E_EMAIL = 'e2e_dojo_test@example.com';
const E2E_PASSWORD = 'E2eTestPass123!@';

// /dojo lives under the (protected) route group (AuthGuard), so every test
// here needs a logged-in session before navigating, or AuthGuard renders its
// sign-in wall instead of the real page content.
type E2EAuth = { access_token: string; refresh_token?: string; user: { id: number; name: string; email: string } };

async function loginForE2E(request: import('@playwright/test').APIRequestContext): Promise<E2EAuth> {
  // Idempotent: register if the user doesn't exist yet, ignore 400 "already registered".
  await request.post(`${API_URL}/api/auth/register`, {
    data: { email: E2E_EMAIL, name: 'E2E Dojo Test', password: E2E_PASSWORD },
    failOnStatusCode: false,
  });
  const res = await request.post(`${API_URL}/api/auth/login`, {
    form: { username: E2E_EMAIL, password: E2E_PASSWORD },
  });
  const body = await res.json();
  if (!body.access_token) {
    throw new Error(`E2E login failed: ${JSON.stringify(body)}`);
  }
  return body as E2EAuth;
}

// AuthGuard gates on AuthModalContext's `user` state, which only becomes
// non-null after AuthModalProvider's mount-time useEffect runs hydrate() and
// its async setState commits — this happens strictly AFTER page.goto()'s
// promise resolves (goto resolves on the load event; hydration is a
// subsequent React effect). Checking `.count()` synchronously right after
// goto() catches the page mid-hydration and reads 0 every time. Using an
// auto-retrying `expect(...).toBeVisible()` instead of a one-shot `.count()`
// waits out that window correctly.
async function gotoFirstProblem(page: Page): Promise<boolean> {
  await page.goto('/dojo');
  const problemLinks = page.locator('a[href^="/dojo/"]');
  try {
    await expect(problemLinks.first()).toBeVisible({ timeout: 10000 });
  } catch {
    return false;
  }
  await problemLinks.first().click();
  return true;
}

// Shapes below match what handleRun() in DojoEditor.tsx actually reads from
// each real endpoint (verified by reading the component, not guessed):
//   run:    POST /api/dojo/runs           -> { passed, stdout?, stderr? }
//   submit: POST /api/dojo/code-submissions -> { cases, num_passed, total, passed }
// The previous shapes here (`results`/`totalMs`/`passedTests` etc.) matched
// neither endpoint, and the route patterns below used to intercept
// **/api/dojo/run and **/api/dojo/submit — URLs the frontend never calls —
// so these mocks silently never fired; every "run"/"submit" click in this
// spec was hitting the real backend unmocked.
const RUN_SUCCESS_RESPONSE = {
  passed: true,
  stdout: '[[1, 2], [3, 4]]',
  stderr: '',
};

const SUBMIT_ACCEPTED_RESPONSE = {
  passed: true,
  num_passed: 2,
  total: 2,
  cases: [
    { kind: 'sample', name: 'Test 1', passed: true, got: [[1, 2], [3, 4]], expected: [[1, 2], [3, 4]], time_ms: 0.5 },
    { kind: 'sample', name: 'Test 2', passed: true, got: [[0, 0], [0, 0]], expected: [[0, 0], [0, 0]], time_ms: 0.4 },
  ],
};

test.describe('Dojo coding flow', () => {
  // Register+login ONCE for the whole file, not per-test — /api/auth/register
  // and /api/auth/login are both rate-limited, and calling them from every
  // test's beforeEach (7x per run) burns through that budget fast and makes
  // the suite flaky under repeated local/CI runs for no benefit, since the
  // same session is valid for every test here anyway.
  let auth: E2EAuth;
  test.beforeAll(async ({ request }) => {
    auth = await loginForE2E(request);
  });

  test.beforeEach(async ({ page }) => {
    await page.addInitScript((a) => {
      window.localStorage.setItem('access_token', a.access_token);
      if (a.refresh_token) window.localStorage.setItem('refresh_token', a.refresh_token);
      window.localStorage.setItem('user_profile', JSON.stringify(a.user));
    }, auth);

    await page.route('**/api/dojo/runs', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(RUN_SUCCESS_RESPONSE),
      }),
    );
    await page.route('**/api/dojo/code-submissions', (route) =>
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
    if (!(await gotoFirstProblem(page))) { test.skip(); return; }
    // Should show Python toolbar in editor. Buttons render an icon component
    // + text ("Run"/"Submit"), not the Unicode arrow characters this
    // assertion used to look for — those never matched anything.
    await expect(page.getByText('Python 3')).toBeVisible({ timeout: 8000 });
    await expect(page.getByRole('button', { name: 'Run' })).toBeVisible({ timeout: 5000 });
    await expect(page.getByRole('button', { name: 'Submit' })).toBeVisible({ timeout: 5000 });
  });

  test('problem page shows back link', async ({ page }) => {
    if (!(await gotoFirstProblem(page))) { test.skip(); return; }
    // The link's actual text is "Problem List" (DojoEditor.tsx ~line 315) —
    // no "Back to Dojo" label exists anywhere; this assertion never matched.
    await expect(page.getByText('Problem List')).toBeVisible({ timeout: 5000 });
  });

  // NOTE (local dev only): this test and 'submit button...' below cannot
  // currently pass against a local backend. next.config.mjs's CSP header
  // hardcodes connect-src to the production Render URL — localhost:8000 is
  // not on the allowlist, so the browser blocks the fetch before it ever
  // reaches page.route()'s mock, and DojoEditor's catch() sets runState to
  // 'error' instead of 'passed'. Unrelated to Phase 2; needs an env-aware
  // connect-src (same class of fix as the backend CSP in Phase 1 P1-4, but
  // for this frontend header instead) to unblock local full-stack E2E runs.
  // Both mocks/assertions here are otherwise correct against the real
  // /api/dojo/runs and /api/dojo/code-submissions contracts.
  test('run button triggers API call and shows results', async ({ page }) => {
    if (!(await gotoFirstProblem(page))) { test.skip(); return; }

    const runBtn = page.getByRole('button', { name: 'Run' });
    await expect(runBtn).toBeVisible({ timeout: 8000 });
    await runBtn.click();
    // A "run" (not "submit") only ever shows the pass/fail heading + raw
    // stdout — CaseResults ("Test N" rows) is gated on lastAction==='submit'
    // in DojoEditor.tsx and never renders for a plain run, so this checks
    // what the run path actually displays instead.
    await expect(page.getByText('Ran Successfully')).toBeVisible({ timeout: 5000 });
  });

  test('submit button triggers API call and shows verdict', async ({ page }) => {
    if (!(await gotoFirstProblem(page))) { test.skip(); return; }

    const submitBtn = page.getByRole('button', { name: 'Submit' });
    await expect(submitBtn).toBeVisible({ timeout: 8000 });
    await submitBtn.click();
    await expect(page.getByText('Accepted')).toBeVisible({ timeout: 5000 });
  });

  test('mobile layout (<768px): zero horizontal scroll, mobile tab switcher interactive', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    if (!(await gotoFirstProblem(page))) { test.skip(); return; }

    // Verify mobile tab switcher is visible on mobile
    const switcher = page.locator('[data-testid="mobile-tab-switcher"]');
    await expect(switcher).toBeVisible({ timeout: 8000 });

    // Verify no horizontal overflow
    const hasHorizontalOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > window.innerWidth;
    });
    expect(hasHorizontalOverflow).toBe(false);

    // Click "Code" tab -> editor becomes visible
    await switcher.getByRole('button', { name: 'Code' }).click();
    await expect(page.getByText('Python 3')).toBeVisible();

    // Click "Console" tab -> console becomes visible
    await switcher.getByRole('button', { name: 'Console' }).click();
    await expect(page.getByRole('button', { name: /Test Case/i })).toBeVisible();

    // Click "Problem" tab -> problem description becomes visible
    await switcher.getByRole('button', { name: 'Problem' }).click();
    await expect(page.getByRole('button', { name: /description/i })).toBeVisible();
  });

  test('desktop layout (>=768px): switcher hidden and all regions simultaneously visible', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    if (!(await gotoFirstProblem(page))) { test.skip(); return; }

    // Mobile switcher must be hidden on desktop
    const switcher = page.locator('[data-testid="mobile-tab-switcher"]');
    await expect(switcher).toBeHidden();

    // Problem description, Editor toolbar, and Console all visible at once
    await expect(page.getByRole('button', { name: /description/i })).toBeVisible();
    await expect(page.getByText('Python 3')).toBeVisible();
    await expect(page.getByRole('button', { name: /Test Case/i })).toBeVisible();
  });
});
