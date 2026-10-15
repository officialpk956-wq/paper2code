# Repair report — 15 October 2026

Scope: the supplied repair harness, T1–T5. No commit, push, reset, checkout, branch change, deployment or credential rotation was performed. Changes remain in the working tree. The extraction and benchmark pipeline was not edited.

## Baseline gate

Executed before application edits:

```text
.venv/Scripts/python.exe -m pytest -q
2025 passed, 9 skipped, 15 warnings in 355.57s (0:05:55)

.venv/Scripts/python.exe -m ruff check backend core workers
All checks passed!

.venv/Scripts/python.exe -m ruff format --check backend core workers
261 files already formatted

npx tsc --noEmit -p tsconfig.json
[no diagnostics; exit 0]
```

The baseline matched. Test execution uses the existing Python 3.11 venv. Fresh-start probes explicitly load the repository `.env`, then override `DATABASE_URL` to their isolated database and disable email/telemetry credentials. They do not infer anything about the real database from a fallback SQLite file.

## T1 — fixed locally; deployed endpoint still exposed

**Reproduced:** `.venv/Scripts/python.exe -m pytest tests/test_celery_health_readiness.py -q --tb=short` against the original handler: `6 failed in 2.84s`. The anonymous response included the deliberately planted fake credential URL; missing-worker responses returned 200. The proposed details route returned 404 instead of enforcing authentication.

**Root cause:** `backend/routers/health.py`, original `health_celery`, returned raw environment/broker configuration without an auth dependency or worker check. The revised public handler starts at line 69; the protected details handler is at line 126.

**Changed:** application changes are confined to `backend/routers/health.py`. Public responses now contain exactly `ready`, `broker_reachable`, and `worker_responding`. The handler opens a broker connection without retrying and waits for an actual worker pong. Missing broker/worker yields 503. Exceptions are not included in the public response. `/api/health/celery/details` reuses the file's existing `get_current_admin` dependency.

**Acceptance:**

1. Planted `redis://:FAKE_SECRET_NEGATIVE_CONTROL@fake-broker:6379/0` in each relevant environment variable: public output contains no `://`, fake password, or fake host.
2. Reachable broker with no worker: 503, `ready=false`, `worker_responding=false`.
3. Unreachable broker: 503, both reachability flags false.
4. Mocked worker pong: 200, all readiness flags true.
5. Detailed endpoint: anonymous 401; ordinary user 403; administrator 200.
6. Result: `6 passed in 0.78s`. Included again in the complete backend suite.

**Negative:** all six new tests failed before the fix. Fake environment values are scoped with pytest's monkeypatch and restored after the test.

**Deployed check:** first request timed out. Second anonymous request to `https://paper2code-1-81y5.onrender.com/api/health/celery` returned:

```text
Anonymous HTTP status: 200
Diagnostic fields: ['celery_backend_url', 'REDIS_URL', 'CELERY_RESULT_BACKEND', 'CELERY_BROKER_URL']
Credential-bearing URL present: True
```

Credential values were not printed or copied into this report. **The deployed service is exposed; this working-tree fix does not repair that deployment.** Credential rotation is the user's decision.

**Not covered:** no live Celery worker was launched; readiness outcomes were tested with controlled connections/pongs. No deployment or rotation.

## T2 — fixed

**Reproduced:** fresh-interpreter startup tests for file SQLite and in-memory SQLite, through the real `backend.server` import and ASGI app, without pre-created model tables:

```text
verification_tokens present: False
register: 500 Internal Server Error
(sqlite3.OperationalError) no such table: verification_tokens
2 failed in 69.92s (0:01:09)
```

Commands: `.venv/Scripts/python.exe -m pytest tests/test_startup_auth_order.py -q --tb=short` and `.venv/Scripts/python.exe tests/startup_auth_probe.py sqlite:///./.repair_old_schema.db` before the fix.

**Root cause:** auth model registration occurred after `Base.metadata.create_all`. Test fixtures imported the finished app and consequently masked the ordering defect. The auth API also imports the server's `limiter`, so moving the import above limiter creation causes a circular import; that initial attempt was caught and corrected before acceptance.

**Changed:** only the auth-router import order in `backend/server.py` (now line 38, after limiter creation, before `create_all` at line 78). The column backfill and startup seeding are unchanged. New tests use a fresh subprocess to avoid an already-populated metadata registry. The in-memory test uses StaticPool solely so TestClient threads share the same database.

**Acceptance:**

1. Fresh file and in-memory startup: `2 passed in 70.46s (0:01:10)`.
2. Registration returns 201 without manual creation of auth tables.
3. Verification → login → `/api/auth/me` → refresh: `verify=200 login=200 me=200 refresh=200`.
4. Reused the file created by the unfixed code, retaining its missing auth schema and existing rows. Corrected startup adds the tables and the flow succeeds. A further check snapshots user identity/password-hash rows and a sentinel **before startup**, then asserts they remain unchanged afterward: `existing rows preserved: 3`; `register: 201`; `sentinel=preserved`.

**Negative:** both fresh-start regression cases failed with the missing-table error before the import moved. The same old-schema file first returned 500 and then 201 after the fix.

**Not covered:** these startup/upgrade probes use SQLite, not a production PostgreSQL migration. They invoke the actual ASGI application via TestClient, not a listening TCP server. Verification tokens are captured from the email-service call; no real email is sent. The unchanged best-effort column backfill is not a substitute for a production migration strategy.

## T3 — fixed

**Reproduced:** built the original frontend, then ran `npx playwright test -c playwright.repair.config.ts repair-pages.spec.ts` against `next start`:

```text
4 failed
4 passed (40.5s)
BROWSER: Error: Minified React error #130
```

The four failures were signed-in direct navigation and client-side navigation for both `/learn` and `/system-design`. The four signed-out checks already passed because the auth gate hid the invalid subtree; these remain unchanged-behavior controls, not evidence of a defect.

**Root cause:** server page modules directly used `motion.div` without a client boundary.

**Changed:** the two page files import `MotionDiv` from a three-line client component at `src/components/anim/MotionDiv.tsx`. Metadata and page composition stay server-side. No whole-page client conversion.

**Acceptance:** rebuilt production app; all eight tests passed (`8 passed (13.9s)`). Tests assert actual headings, content links or auth gate, no workspace error boundary, and no browser console/page errors. Client navigation clicks the app's navigation link from About. Build output retains `○ /learn` and `○ /system-design`.

**Negative:** the exact same four signed-in checks failed on the original production bundle, with error #130 recorded. No forced clicks or ignored console errors.

**Other occurrences:** scanned all `src/app/**/*.tsx`. Only these two pages matched the server-component/direct-motion pattern. Other matches are existing client modules (Dojo list/editor, Papers list, guided implementation).

**Not covered:** Chromium only. Signed-in page tests use a local session fixture to isolate rendering; account authentication is exercised separately in T2.

## T4 — fixed on the frontend side

**Reproduced:** `npx playwright test -c playwright.repair.config.ts repair-estimator.spec.ts` against the original estimator bundle, using responses captured from the real backend calculation:

```text
BROWSER: TypeError: Cannot read properties of undefined (reading 'toFixed')
7 failed
```

The suite now lives in `e2e/labs-flow.spec.ts`.

**Root cause:** the UI assumed four fields the API does not return. TypeScript's generic type parameter did not validate the runtime JSON.

**Changed:** only frontend production code in `LabsClient.tsx`. Backend consumers and calculations are unchanged. An installed Zod schema validates finite values without inventing zeroes or coercing strings. Positive whole-number inputs are checked before sending. The nonexistent Total Compute/PFLOPs card is replaced by actual Training Steps. The description states the 10-epoch assumption and uncertainty.

| Returned field | UI meaning | Computation in `core/implementation/cost_estimator.py` |
|---|---|---|
| `training_hours` | Estimated single-GPU wall-clock hours | Lines 305–321: profile FLOPs × batch × steps, forward/backward factor 3, throughput at 35% utilization, /3600; checkpointing adds 25%. Rounded to one decimal at line 380. |
| `compute_cost_usd` | Estimated GPU rental cost in USD | Line 324: unrounded wall-clock hours × configured hourly rate; two-decimal result at line 381. Not rounded displayed hours multiplied again. |
| `gpu_memory_gb` | Backend's approximate total VRAM estimate | Lines 282–300: FP32 weights + two Adam states + activation estimate; checkpointing halves activations; mixed precision divides total by 1.6. Rounded at line 379. |
| `steps_total` | Number of training steps | Lines 305–306: ceil(dataset size / batch size) × epochs. No FLOPs aliasing. |

The memory model uses binary byte conversion for weights while its heuristic activation/hardware profiles are labeled GB. The UI preserves the backend's approximate GB convention; this is not a newly validated physical-memory measurement. Cost, throughput and memory remain heuristic planning estimates, not benchmark results.

**Acceptance:** four families render the captured time/cost/memory/step values; invalid input sends no request; a 500 displays its message and retry succeeds; malformed responses show a useful error instead of crashing. A backend test checks that the handler's response continues to match the shared browser fixture. Expected values include ResNet 5.0 hours/$16.02/20.2 GB, Transformer 14.7/$46.90/40.5 GB, ViT 21.5/$68.78/50.6 GB, U-Net 66.8/$213.77/30.2 GB, each with 39,070 steps for the fixture inputs.

**Negative:** all seven browser checks failed before the change. The 500 itself was already displayed correctly; the retry-success portion of that combined check failed on the original response mismatch.

**Not covered:** no GPU profiling, price refresh or recalibration. No change to the backend's separate handling of direct invalid API requests; this task chose the frontend boundary.

## T5 — strengthened tests; removed obsolete route spec

**Reproduced:** the old block-viz smoke assertion could accept a 404. Requiring the actual response status and block content produced:

```text
Expected: 200
Received: 404
1 failed
```

**Decision:** delete `e2e/block-viz-flow.spec.ts`. Do not restore the obsolete route. Searches found no advertised `/block-viz` path in current `src` or README, so no current navigation link needed deletion. Backend block visualization was not removed. Replace coverage with current `/model-viz?g=1` behavior.

**Changed:**

- Labs specs now exercise the current estimator contract and error paths, replacing obsolete `/api/labs` mocks and old UI expectations.
- Model visualizer asserts successful graph response, actual rendered node count/content, FLOPs, search input, and selection.
- Dojo asserts an actual problem link, editor, code request payloads, returned execution output and grading verdict, plus responsive controls. Removed conditional `test.skip()` and real registration from these explicitly mocked UI tests. Live async upload remains a separate opt-in spec.
- `playwright.repair.config.ts` launches the production server on port 3001 through Playwright's server configuration; no development server was started from a raw shell.

**Negative:** `E2E_NEGATIVE_CONTROL=1` supplies absent pages/graph to representative tests in all three changed feature suites. Result: `3 failed`. These are intentional sensitivity checks, not suppressed normal failures.

**Additional reproduced failure:** the strengthened normal suite initially produced `21 passed, 1 skipped, 1 failed`. At 375px the Dojo list link could not be clicked: `input ... placeholder="Search problems..." ... intercepts pointer events`. A fixed 260px side column and shrinking filters squeezed the mobile list. A small responsive change to `DojoListClient.tsx` stacks the sidebar, allows filter wrapping and gives the list usable space. The mobile test itself was not weakened or forced through the obstruction. This is the sole extra production-file change needed by the newly enforced Dojo browser acceptance check.

**Acceptance:** after rebuilding, the same production-browser suite reports `22 passed (29.5s)`, `1 skipped`. All six Dojo checks, seven Labs checks, one model-visualizer check and eight route-rendering checks pass. The sole skip is the separate opt-in live async paper test. The unchanged mobile Dojo check passes after the responsive fix.

**Not covered:** mocked Dojo execution does not prove live E2B execution; mocked saved-graph retrieval does not prove upload/parsing. No live queue, email, OAuth, MFA, paid LLM or E2B work was attempted.

## Final verification

All six requested commands completed successfully:

```text
.venv/Scripts/python.exe -m pytest -q
2034 passed, 9 skipped, 15 warnings in 407.76s (0:06:47)

npx vitest run
Test Files  9 passed (9)
     Tests  40 passed (40)

npx tsc --noEmit -p tsconfig.json
[no diagnostics; exit 0]

.venv/Scripts/python.exe -m ruff check backend core workers
All checks passed!

.venv/Scripts/python.exe -m ruff format --check backend core workers
261 files already formatted

npm run build
✓ Compiled successfully in 38.4s
✓ Generating static pages (22/22)
[exit 0; /learn and /system-design remain static routes]
```

Additional production browser acceptance: `npx playwright test -c playwright.repair.config.ts`, with `RUN_REAL_ASYNC_PAPER_E2E=0`: `22 passed (29.5s)`, `1 skipped`, exit 0. The three deliberately absent-state controls separately produced `3 failed`, as required. Backend skips and warnings remain explicit; passing mocked browser contracts does not establish live provider integration.

## Evidence and reproduction

Ignored local logs in the repository root retain the exact runs: `.codex_repair_baseline_*`, `t1_before/after`, `deployed_retry`, `t2_before/after/old_before/upgrade_final`, `t3_build_before/console_before/after`, `t4_before`, `t5_block_negative`, `t5_negative`, and `final_*` (all prefixed `.codex_repair_` and suffixed `.log`). They are local artifacts, not committed deployment evidence.

Run the normal production-browser suite with `npm run build` followed by `npx playwright test -c playwright.repair.config.ts`. Run the sensitivity check with `E2E_NEGATIVE_CONTROL=1` and select `backend units: ResNet|index exposes|saved ONNX`; a nonzero exit is expected. The environment variable is never enabled in normal acceptance.
