# Repair harness 2 — U1–U4

Scope: local working-tree repair only. No commit, push, reset, checkout, branch operation, deployment, credential rotation, real email, OAuth, paid LLM or E2B request. The extraction/generation files excluded by the harness were not edited. Earlier T1–T5 changes were preserved.

## Baseline

The unchanged application matched the baseline:

```text
python -m pytest -q
2034 passed, 9 skipped, 15 warnings in 512.81s (0:08:32)
ruff check backend core workers
All checks passed!
ruff format --check backend core workers
261 files already formatted
npx tsc --noEmit -p tsconfig.json
[isolated rerun: exit 0, no diagnostics]
npx vitest run
Test Files 9 passed (9); Tests 40 passed (40)
npm run build
[exit 0]
npx playwright test -c playwright.repair.config.ts
22 passed (35.2s), 1 skipped
```

Python commands use `.venv/Scripts/python.exe`. The initial separate TypeScript invocation collided with the concurrent Next build regenerating `.next/types` and emitted TS6053 missing-generated-file errors. It passed after the build completed with no source/configuration change; subsequent builds and standalone type checks run sequentially. This was a test-orchestration error, not a repaired application defect.

## U1 — fixed

**Selected handler:** POST `/api/auth/verify-email`, `backend.modules.auth.api.v1.verify_email` at line 251. A real registration call, with its email call captured, produced a token found by SHA256 in `verification_tokens`, not the raw-token `email_verification_tokens` table. The runtime route inventory showed exactly the new module's POST and GET at `/api/auth/verify-email`. `backend/routers/auth.py` is not mounted by the server; its duplicate GET is unreachable through this app.

**Reproduced:** piped `{"action":"probe_issuer"}` into `.venv/Scripts/python.exe -u tests/auth_browser_bridge.py` before backend edits:

```text
signup_token_store: verification_tokens SHA256
GET: 400
POST: 200
before: verified=false, verified_at=None, events=0
after: verified=true, events=1
```

The mounted no-body resend path also creates raw legacy tokens. Therefore merely changing the page to call the original POST handler would still break those links.

Production-browser command before rebuilding: `npx playwright test -c playwright.repair.config.ts auth-repair.spec.ts --workers=1`. The valid-link check expected the database verified flag to become true, but it stayed false. Expired, malformed and missing-link checks could not find their respective truthful headings. Pending-state check could not find its status. These five are genuine static-page failures. A separate sender test failed because the legacy sender did not emit `/auth/verified?token=fixture`.

**Root cause:** static success content in `src/app/auth/verified/page.tsx`; incompatible token stores in the GET/POST handlers; second sender's missing `/verify-email` page. Current redemption logic is in `backend/modules/auth/services/verification_service.py:36`.

**Changed:** kept one frontend page, with a small `VerifyEmail.tsx` client component under server metadata. It redeems via POST and shows pending, success, used, expired, invalid, missing, or unavailable states. Both email templates now target `/auth/verified`; the legacy sender's local default port matches the frontend. GET forwards safely encoded links to that page without mutating the database. The POST service recognizes both existing issuer stores, consumes tokens with a conditional update, and updates both account verification flags in the same transaction. No auth-module refactor or token migration.

**Acceptance:** focused backend checks initially yielded `6 passed in 1.31s`; including sender and CI contracts subsequently yielded `8 passed in 1.61s`. They assert flags, timestamp stability, exactly one verification audit event, expiry and reuse for both issuers, and the GET redirect without consumption. The final browser run also proves the actual page causes the database transition.

Production browser acceptance, using the actual local backend:

1. Signup link verifies the database row; reload rejects reuse and preserves timestamp/event count: `ok ... verification redeems signup token once and rejects reuse with unchanged database (2.0s)`.
2. Expired, malformed and missing parameters: all three browser cases passed with distinct headings and an unverified database row.
3. Pending response: the status remains visible while the response is held in flight; test passed.
4. Both captured senders point to the existing route; the page returns 200 and truthfully shows the missing-token state when visited without a token; test passed.
5. Both issuer types additionally pass direct POST valid/reused and expired cases. The frontend uses the same endpoint for either format.

**Negative:** `pytest tests/test_verification_path.py tests/test_ci_frontend_gate.py -q --tb=short` initially yielded `6 failed, 1 passed in 3.92s`. The already-correct modern-token expiry check was retained as an unchanged regression control; it is not claimed as a new fix. Sender-only negative: `1 failed in 3.07s`. Browser static-page negatives above cover the requested valid-account and invalid-state failures.

**Not covered:** live delivery, production database migration and deployment. Tests use real ASGI handlers, captured email calls and an isolated SQLite database. The bridge explicitly loads the repository `.env` then overrides database, secret, email and telemetry settings before importing the app. It never uses the real database for browser fixtures.

## U2 — fixed

**Reproduced:** `npx vitest run src/__tests__/auth-challenge.test.tsx` against the original hook:

```text
AssertionError: expected 'undefined' to be null
Tests 2 failed (2)
```

The second failure was a malformed login response resolving as a user instead of rejecting. Ordinary login succeeds in the first test before the MFA assertion fails. Corrected production-browser checks for valid, wrong, expired and backup-code scenarios all failed on the original bundle because no Two-step verification UI appeared.

**Root cause:** the original `src/components/useAuthActions.ts:41` stored the response unconditionally; both `AuthModal.tsx` and `AuthGuard.tsx` treated the returned user as authenticated. Current response validation is at `useAuthActions.ts:48`, with MFA exchange at line 80 and UI branches at `AuthModal.tsx:120` and `AuthGuard.tsx:101`. The common API helper also collapses 401 details to Unauthorized, so it cannot preserve distinct MFA code/expiry errors.

**Changed:** response validation distinguishes session tokens from a challenge. Challenge state is held in memory, any stale session is cleared, and callers do not sign in on a challenge. Both sign-in surfaces use a small shared `MfaChallenge.tsx`. Its request uses the existing endpoint but deliberately does not refresh/retry an unauthenticated MFA challenge; backend error details remain visible. Closing the modal or backing out clears the challenge. No backend MFA change or enrolment UI.

**Acceptance:** hook tests now pass; the full working frontend suite reported `42 passed (42)`. Final real-handler browser tests below cover storage being absent before MFA, valid TOTP, wrong code, expired challenge, backup consumption, ordinary authenticated `/me`, and both modal/guard entry points.

1. Ordinary modal login obtained a token that successfully authenticated `/api/auth/me` for the registered email; the same test then completed MFA through the modal: `ok ... ordinary modal login keeps a working session; MFA modal waits for verification (9.5s)`.
2. Valid MFA through the protected-page guard: no access token at the challenge screen, successful code exchange, then authenticated `/me`: passed `(2.7s)`.
3. Backup-code login: passed `(2.1s)`, with database backup count reduced to zero.
4. Wrong code and expired challenge: exact distinct messages and no access token; corrected scoped-selector rerun: `2 passed (48.3s)`.

The first working browser run had `33 passed, 1 skipped, 2 failed`: its generic alert locator matched both the correct form error and Next's route announcer. Scoping the locator to the form preserved the exact message assertion; no application fix or weakened expectation was needed. An additional negative rebuilt the original three auth files, ran the ordinary/MFA modal journey, and failed specifically on missing Two-step verification after ordinary login succeeded. All three repaired files were restored byte-for-byte afterward. One earlier post-install server startup timeout is excluded from product evidence.

**Negative:** literal undefined storage is asserted directly in the unit test. Four corrected browser scenarios fail without the UI fix. Initial browser harness errors (a misspelled `endsWith`, and a mocked sender capture) were corrected and rerun; those initial errors are excluded from product evidence.

**Not covered:** live OAuth, MFA enrolment, delivery, or authenticator-device provisioning. TOTP and signed expired tokens are generated locally against the real isolated backend.

## U3 — fixed with caveat (local job execution; no hosted CI run)

**Reproduced:** the new workflow-contract check failed with `AssertionError: Missing CI gate: npm ci`. The original workflow had no Vitest/Playwright steps and its sandbox job only echoed a simulation.

**Root cause:** original `.github/workflows/ci.yml:47` used an unlocked install, and line 62 merely printed a sandbox simulation. There were no frontend-test/browser jobs. Current real gates are at lines 60 and 71, invoking the production script at `package.json:29`.

**Changed:** `.github/workflows/ci.yml` retains Python lint/format and backend jobs, switches frontend installs to `npm ci`, adds explicit standalone TypeScript checking after the build, and adds separate unit and production-browser jobs. Removed the fake sandbox job. `package.json` adds `test:e2e:production`, which builds and then tests that artifact with `playwright.repair.config.ts`. The production config retains zero retries, reports skipped tests, forbids focused tests in CI, and retains failure screenshots/traces. The browser job installs Python dependencies for the isolated auth bridge, plus Chromium.

**Acceptance:** executed the browser-job build/test command locally with focused test selection for each deliberate defect. Both builds succeeded, both browser checks failed, both commands exited 1, and both injected changes were removed. The final full unfiltered production run follows restoration. No push or GitHub run is claimed.

RSC negative execution: temporarily replaced the client MotionDiv leaf with a direct server `motion.div` import in `/learn`. Ran `npm run test:e2e:production -- --grep "/learn signed in direct"` with `CI=true`, local API URL and live integration disabled. The build reported `Compiled successfully in 23.0s`, but the production browser reported `1 failed`: `getByRole('heading', { name: 'Curriculum', exact: true }).first()` was not visible. The job exited 1. This is an actual page failure after a successful build, not a deliberately failing assertion or a build failure.

The first direct Windows write used to restore the RSC source failed. Atomic replacement restored the client import and all four tags; inspection confirmed the diff contains exactly the pre-existing T3 changes (five additions/five removals relative to HEAD). Subsequent temporary changes use atomic writes and byte comparisons for restoration. No injected RSC defect is left in the tree.

Estimator negative execution: temporarily changed the UI/schema field `training_hours` to `training_time_hours`, leaving the real backend fixture unchanged. Ran `npm run test:e2e:production -- --grep "backend units: ResNet"` with the same CI environment. Output: `Compiled successfully in 23.1s`; `1 failed`; `getByText('Est. Time', { exact: true }).locator('..')` could not satisfy `Expected substring: "5.0"`. The job exited 1 and the source was restored byte-for-byte. Validation prevented a crash, but the acceptance gate still correctly rejected the unusable estimator.

**Negative:** the original workflow-contract test failed before configuration changes. The two actual source mutations above demonstrate runtime sensitivity after the configuration change; neither uses `E2E_NEGATIVE_CONTROL` nor changes the assertions to manufacture a failure.

**CI matrix:** ruff check YES; ruff format YES; pytest YES; tsc YES; vitest YES; playwright-production YES; build YES. Real service integration remains opt-in and explicitly skipped in normal output.

**Not covered:** GitHub-hosted Ubuntu execution, provider integration or deployment. Local command execution does not establish runner permissions, branch protection or webhook behavior.

## U4 — fixed

**Decision:** remove the duplicate disabled Implement-tab Run in Sandbox control. Keep the existing guided-implementation link and the separate executable-tab controls. Label local progress/drafts rather than introduce account persistence.

**Reproduced:** updated `WorkspacePaperClient.test.tsx` expected no Run in Sandbox control: `1 failed | 2 passed (3)` on the unfixed component. Production-browser `local-progress.spec.ts` failed for both guided and Dojo screens because neither showed the local-only disclosure.

Commands: `npx vitest run src/__tests__/WorkspacePaperClient.test.tsx`; `npx playwright test -c playwright.repair.config.ts auth-repair.spec.ts local-progress.spec.ts --grep 'MFA|both mail|discloses'`. The latter corrected negative run produced `7 failed`, including the two missing-disclosure cases.

**Root cause:** disabled promise at the former `WorkspacePaperClient.tsx:925`; browser-only storage in `ImplementClient.tsx:39–59` and Dojo draft/note storage without a sync disclosure.

**Changed:** removed only the duplicated disabled button; added visible text to guided implementation and Dojo explaining what is saved, that it is not account-synced, that clearing browser data removes it, and that others using the browser can see it. No server persistence or ownership claim was added.

**Acceptance:** the workspace component test now passes and the original executable-tab run test remains passing. Final browser tests additionally prove local completion/draft restoration after reload alongside the disclosure.

The actual production-browser checks passed: guided local completion restored after reload `(3.7s)`; Dojo's stored draft rendered in Monaco after reload `(4.5s)`. Both displayed the local-only disclosure. Existing mobile and desktop Dojo checks remained passing with the notice visible.

**Not covered:** cross-device/account persistence, live execution, or privacy isolation for local browser storage. The UI explicitly describes those storage limits.

## Execution notes

Logs are local ignored `.codex_u_*.log` files; no logs are deployment evidence. The final unfiltered production run passed all 35 enabled tests, including all U1–U4 checks, after restoring both deliberate mutations. The single opt-in real async paper test remained visibly skipped. `git diff --check` passed, and no diff remains in the generated content index or the excluded extraction/generation paths.

Installation evidence: `npm ci` exited 0, copied Monaco assets, and reported `added 932 packages, and audited 933 packages in 6m`. The lockfile was unchanged. Its audit summary reported 39 vulnerabilities (4 low, 8 moderate, 23 high, 4 critical); these were not triaged or fixed in this scoped repair. No `npm audit fix` was run. The local runner is Windows/Node 24; the workflow specifies Ubuntu/Node 20, which was not exercised remotely.

A later build hit a Windows `UNKNOWN` write error on the generated content index despite sufficient disk space. Replacing that generated file with identical bytes allowed the normal generator/build command to proceed; no generator behavior was changed. Timestamp-only generated changes are removed at the end.

## Final command outputs

All commands exited 0. The final production job invokes the displayed build command and then the unfiltered browser command against that same build. The standalone TypeScript check was repeated after restoration and after the final build completed.

```text
.venv/Scripts/python.exe -m pytest -q
2042 passed, 9 skipped, 15 warnings in 636.68s (0:10:36)

.venv/Scripts/python.exe -m ruff check backend core workers
All checks passed!

.venv/Scripts/python.exe -m ruff format --check backend core workers
261 files already formatted

npx tsc --noEmit -p tsconfig.json
[no diagnostics; exit 0]

npx vitest run
Test Files  10 passed (10)
     Tests  42 passed (42)

npm run build
✓ Compiled successfully in 23.0s
✓ Generating static pages (22/22)

npx playwright test -c playwright.repair.config.ts
1 skipped
35 passed (1.3m)
```
