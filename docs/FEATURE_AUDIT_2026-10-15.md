# Feature audit and completion plan

Date: 15 October 2026 (Asia/Kolkata). Branch: `main`, audited commit: `4eb287a`.

**Verdict: not everything works.** The build and automated suites pass, but real browser checks exposed broken landing pages and a broken estimator. A clean database exposed an account-registration failure. Do not treat the passing unit tests as release approval.

This was an audit, not a repair pass. Application source was not changed. The content generator's timestamp-only change was restored.

## What was tested

| Check | Fresh result | Limit |
|---|---|---|
| Frontend unit/component tests | 40 passed, 9 files | Predominantly mocked API responses |
| Backend suite, excluding live marker | 2,024 passed; 1 skipped; 9 deselected; 15 warnings | Test fixtures, in-memory databases and service mocks; not production integration |
| Lint | Exit 0; 7 warnings | Hook dependencies and two unused test arguments remain |
| TypeScript | Passed within content audit | Types do not validate actual API responses |
| Production build | Passed; 22 static pages generated | Successful build did not prevent runtime page failures |
| Content | 216 architecture articles, 94 curriculum lessons, 12 system-design articles, 195 paper articles compiled | Compilation is not a correctness review of educational claims |
| Internal Markdown links | 1,727 checked; zero broken | Does not prove interactive controls work |
| Existing browser suite | 1 passed, 8 failed, 10 did not run; stopped at 8 failures | Stale specs and failed login setup; the one pass accepts a visible 404 body |
| Additional browser inspection | 25 initial routes, then targeted interaction checks | Chromium; desktop plus narrow mobile samples, not every route/device combination |
| Local API smoke | 144 documented API paths inventoried; selected endpoints exercised | Not all 144 endpoints received every possible request |

The production frontend ran on localhost:3000. An isolated backend ran on 127.0.0.1:8011 with a disposable SQLite database and dotenv loading disabled. Browser API requests were redirected to that backend, because the frontend build used the repository's existing environment configuration. Thus these checks do **not** validate deployment routing, CORS, or production credentials.

Initial UI inspection used a synthetic local session solely to reach protected screens. Subsequent checks used a real local login. Registration first failed on the pristine database; auth tables were then explicitly initialized in that disposable database to enable downstream testing. That fixture setup does not fix application startup.

## Feature status

“Verified locally” means only the stated scope passed. “Partial” means a surface or component works while the complete user outcome is not verified.

| Feature | Status | Evidence and remaining work |
|---|---|---|
| Home, About, legal pages | Verified locally: rendering | Home, About, Terms, Privacy, Cookies, Security loaded. Full navigation, accessibility and all responsive layouts still need coverage. |
| Registration on a fresh installation | **Broken** | HTTP 500: missing `verification_tokens` table. Registration returned 201 only after initializing auth tables in the audit database. |
| Password login and refresh | Verified locally after schema setup | Login, `/api/auth/me`, and token refresh returned 200. Browser session reached protected editors. |
| Email confirmation | **Broken integration** | `/auth/verified?token=invalid` displays “Email verified.” Signup email points to this passive success page, which does not redeem the token. |
| Forgotten/reset password | Partial | Pages render; missing reset token shows a useful error. Actual email delivery and complete token-based password replacement not exercised. |
| Google OAuth | Unverified live | Requires real provider redirect/callback configuration. Backend tests alone do not verify the provider round trip. |
| MFA, sessions, account settings | Backend capability; frontend incomplete | API implements MFA login response and verification routes. `useAuthActions` expects ordinary access tokens and has no MFA challenge handling. No corresponding settings UI found in the route inventory. |
| Paper library and workspace list | Verified locally: rendering | `/papers` renders; empty local workspace shown. Golden-library filtering and all entries not manually exercised. |
| PDF upload and queued processing | Unverified end to end | Frontend tests cover upload requirements; backend tests cover pipeline components. Redis/worker/external-service chain not available in isolated setup. |
| Paper detail and evidence | Partial | ResNet page renders and honestly reports missing persisted chunks/generation evidence in the empty audit database. Uploaded-paper persistence and all tabs need a real ingestion run. |
| Generated executable code | Partial | Backend generation/validation tests passed. Fresh real PDF to persisted executable to sandbox run remains a release gate. |
| Paper AI tutor | Partial | Frontend/backend tests pass with mocks. Live model response, grounding, history, timeouts and cost behavior not verified. |
| Guided implementation | Partial | `/papers/resnet/implement` renders a visible Monaco editor. Progress uses browser local storage. Completion/execution and cross-device persistence not verified. |
| Implement-tab “Run in Sandbox” | **Incomplete** | Explicitly disabled with “Coming soon.” Separate executable-tab and guided-run controls exist; this particular button is unfinished. |
| Curriculum index | **Broken** | Signed-in `/learn` renders “Workspace Error” with React error #130. |
| Curriculum domain and lesson | Verified locally: sample + content compilation | Mathematics domain and canonical statistical-learning-theory lesson render; sampled lesson has 35 math elements. All 94 lessons compile. The guessed `/learn/mathematics/linear-algebra` URL is not canonical and was excluded from product failures. |
| Architecture library/detail | Verified locally: samples | Index and ResNet detail render; 216 articles compile. All individual 3D interactions not exercised. |
| Architecture comparison | Partial, data-dependent | ResNet/ViT selection reaches API and reports no analyzed paper backs those slugs in the empty database. Successful comparison requires seeded/processed papers and a further browser check. |
| System-design index | **Broken** | `/system-design` renders “Workspace Error” with React error #130. |
| System-design articles | Verified locally: sample + content compilation | RAG systems page renders; all 12 canonical articles compile. |
| Labs mutation tool | Verified locally: sampled mutation | ResNet Increase Width returns and displays a table: 6.6M to 14.4M parameters. Other mutation/family combinations not all exercised in browser. |
| Training-cost estimator | **Broken UI/API contract** | API returns 200 for four families, but clicking Estimate Training Cost crashes the UI into the workspace error boundary. |
| Dojo list and editor | Verified locally | Real local account opens sigmoid exercise; visible Monaco editor confirmed. Mobile sample had 390px viewport and 390px document width. |
| Dojo Run/Submit grading | Partial | Run reaches backend and clearly reports missing E2B key. Reference/grading backend tests pass; live sandbox execution and accepted/rejected submissions not verified. |
| Leaderboard | Partial | Public API returns 200 in empty audit database. Populated rankings and refresh after submission need end-to-end checks. |
| ONNX model visualizer | Verified locally: tiny model | Actual one-node ONNX upload parses, saves and reloads; `/model-viz?g=1` displays saved graph and accepts layer search. Large models and full selection/grouping behavior not exhaustively exercised. |
| PyTorch model visualizer | Unverified live | Requires configured E2B. Importability of the SDK is not proof the service is usable. |
| Extract Code page | Marketing entry point | Renders and links to paper workspace. “Any paper,” “complete,” and “production-ready in seconds” promises exceed what this audit verifies. |
| Adaptive learning, analytics, achievements, notifications | API-only/partial | Selected authenticated endpoints return 200. Frontend route/source inventory did not find dedicated views for these capabilities. Empty-data responses do not establish complete learning journeys. |
| Assessments, paper challenges, admin, announcements | Automated coverage/source only | Backend routes exist; no complete browser journey exercised. Define intended user-facing scope before adding pages. |
| Block visualization | Missing legacy frontend route | `/block-viz/resnet` is 404. Backend service exists and tests target the removed frontend route. Decide whether to restore the feature or retire that advertised/tested path. |

## Confirmed problems and repair order

### P0 — remove sensitive diagnostic output

`backend/routers/health.py:69` exposes an unauthenticated Celery diagnostics endpoint returning backend/broker URLs and environment values. This was confirmed locally using harmless localhost values. In a deployment with credential-bearing URLs, this can disclose credentials. It also returns 200 without checking a worker.

Fix: return a minimal sanitized readiness result; move detailed diagnostics behind administrator authorization. Check production exposure privately and rotate credentials if exposure is established. Acceptance: anonymous response contains no URL credentials, configuration secrets, or raw environment values; a missing worker cannot be reported as ready.

### P1 — repair visible crashes and account onboarding

1. **Fresh database registration:** `backend/server.py:77` calls `create_all` before importing the newer auth router at line 181 and its models. Unit fixtures import the completed application before creating their own tables, masking the startup-order failure. Fix with complete model registration before schema creation and a verified migration path. Acceptance: start from an empty database, register, verify, log in and refresh without manual schema setup; also exercise an existing database upgrade.
2. **Curriculum/system-design runtime failure:** both server-rendered page modules import and render `motion.div` directly. Browser consoles show React error #130 (undefined element type). Move animated sections into existing or minimal client components, retaining server metadata. Acceptance: signed-in and signed-out production-build visits render the intended page with no error boundary; test a client-side navigation too. This is the leading source-level cause, not a repair validated in this audit.
3. **Estimator contract mismatch:** frontend expects `training_time_hours`, `cost_estimate_usd`, `vram_required_gb`, and `total_flops`; backend returns `training_hours`, `compute_cost_usd`, `gpu_memory_gb`, and profile/breakdown fields. UI calls `.toFixed` on missing fields. Align one documented response shape and preserve units. Acceptance: all four families render finite results; invalid inputs and API failure give useful errors; displayed compute has a defined derivation rather than a renamed incompatible value.
4. **Email confirmation:** wire a real token-redemption flow before showing success. There are two email-service paths: signup links to `/auth/verified`, while another verification sender links to `/verify-email`, which is absent from the frontend route inventory. Consolidate URLs and token handling. Acceptance: valid, expired, reused, missing and invalid tokens each yield the correct state and database effect.

### P2 — make verification trustworthy and finish existing flows

5. Replace stale Labs/block-visualization browser specs with the intended current product journeys. A visible body and non-error title are insufficient: the existing block-viz smoke test passed on a 404. Dojo tests must fail, not skip, when the expected problem list is unavailable. Separate mocked UI tests from real integration tests.
6. Complete MFA challenge handling before offering MFA account setup to users. Handle the backend's `mfa_required` result instead of storing a missing access token. Acceptance: ordinary, MFA, bad code and backup-code login paths.
7. Resolve the unfinished implement-tab sandbox control and define persistence for guided progress. Keep unavailable operations clearly labeled until they are connected and tested.
8. Run the real core journey with configured staging services: signup → verification → PDF upload → task progress → extracted evidence → generated code → validation → sandbox execution → tutor answer → reload persisted workspace. Include failed jobs, unsupported papers, duplicate uploads, ownership boundaries and restart recovery.
9. Verify ONNX multi-node graphs, grouped blocks, node selection, sharing authorization and malformed/oversized uploads; then test supported TorchScript uploads through real E2B.
10. Populate comparison and leaderboard fixtures and test successful results, empty results and errors. Check small-screen layouts beyond the two samples. Investigate the Monaco data-font CSP warning observed in the browser; the editor itself did render.

### P3 — decide which backend capabilities become product features

Use current scope rather than adding every possible feature. Prioritize the core paper-to-code journey, then choose whether to expose learning analytics/adaptive plans, assessments, achievements/notifications, account controls and admin tooling. For each selected feature, define the route, user action, persisted outcome, access rules, empty/error states and one real acceptance journey. Remove or soften marketing claims until the measured pipeline supports them.

## Release acceptance plan

| Gate | Required evidence |
|---|---|
| Account and deployment | Fresh/upgrade database test; signup/verification/login/reset/logout/refresh; OAuth and MFA if enabled; sanitized readiness endpoints |
| Content browsing | Production-browser visits to both indexes, one domain and representative articles; compile all registered content; validate links |
| Paper processing | Real PDF fixture per supported family plus unsupported/malformed samples; task completion/failure; persisted citations/code; restart-safe state |
| Execution | Real sandbox accepted/rejected/error/timeout cases; runtime and output limits; no acceptance based solely on mocked results |
| Learning | Tutor grounding checks; guided step completion; refresh/persistence; assessment outcome if in scope |
| Visualization/labs | ONNX and TorchScript actual uploads; save/load; selection/search/grouping; mutations and estimator contract checks |
| Cross-cutting | Authorization, expired session, mobile navigation/editor, accessibility, network failures, production CORS/CSP and observability |

Exit criterion: all in-scope journeys have fresh passing evidence and no P0/P1 defects. Features not included in a release must be visibly unavailable or removed from claims. No overall “everything works” percentage is justified by the present evidence.

## Local evidence

The ignored `.codex_feature_*.log` files in the repository root contain the fresh runs: `unit`, `backend`, `lint`, `content`, `build`, `e2e`, `browser`, `landing_errors`, `api_read`, `authenticated`, `confirmations`, and `lesson`. `test-results/` contains browser failure screenshots/context. These are local audit outputs, not source changes or deployment verification.

The initial exploratory interaction log used “PASS” to mean a probe completed, including expected unavailable states. Only the explicit assertions in `confirmations`, actual HTTP outcomes, and observed screen contents are treated as feature evidence above. The first saved-graph exploratory URL used the wrong `id` query parameter; it was corrected to `g` and the graph was subsequently asserted visible. No successful comparison, live Dojo execution, or live email delivery is claimed.

Not checked: production deployment, real OAuth/email providers, paid/live LLM or E2B calls, queue recovery with a worker, load/concurrency, every content article in a browser, every API operation, all devices or all accessibility requirements. Backend and browser test output contains additional non-fatal warnings; they were not suppressed.
