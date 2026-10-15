# Remaining gaps and improvement plan

Reviewed against the current working tree on 15 October 2026. This is a source/configuration review after the T1–T5 repair, not another full runtime audit. Prior repair results remain 2,034 backend tests passed, 40 frontend tests passed, and 22 browser tests passed with one opt-in integration skip. Those results do not establish that external services work. No application source, deployment, credentials, or Git state was changed during this review.

## Priority and acceptance

| Priority | Remaining gap and current evidence | Improvement | Completion evidence |
|---|---|---|---|
| P0 | The previous repair check confirmed credential-bearing diagnostics on the deployed endpoint. Local protection is still an uncommitted change. Production was not rechecked in this review. | Release the reviewed health fix through the deployment process; decide rotation of exposed credentials and update affected services. | Anonymous response has no configuration values; no-worker status is unavailable; protected details enforce authorization; deployed revision matches the reviewed revision. |
| P1 | `src/app/auth/verified/page.tsx` always renders success. Signup's `backend/modules/auth/services/email_service.py:34` points there with a token. The page never redeems it. A second sender at `backend/services/email_service.py:52` points to `/verify-email`. | Establish one verification link and token-redemption path, with pending/success/error states. Verify compatibility with the token issuer before choosing between the backend's existing GET and POST implementations. | Valid token changes the account exactly once; invalid, expired, missing and reused tokens give truthful results. Test locally with captured mail before testing real delivery. |
| P1 | `backend/modules/auth/api/v1.py:125` returns `mfa_required` and `mfa_token`; `src/components/useAuthActions.ts` assumes access tokens and calls `setTokens` unconditionally. | Model ordinary login and MFA challenge as separate response types; retain challenge state without pretending login completed; connect code/backup-code verification. | Ordinary login, required challenge, bad/expired code, valid code and backup code; no missing token stored. Live OAuth remains a separate gate. |
| P1 | `.github/workflows/ci.yml` runs Python checks and a frontend build, but neither Vitest nor Playwright. Its sandbox job only echoes a simulation. The default Playwright configuration starts development mode, while the successful repair checks used production mode. | Make production-browser checks, frontend tests and content/link checks explicit CI jobs; use the lockfile for installation; label or remove the placeholder sandbox job. Keep mocked contracts and real integration gates distinct. | A deliberate route error or contract mismatch blocks CI. Build and browser tests use the same artifact/configuration. A skipped integration suite is visible as unverified. |
| P1 release gate | `e2e/paper-upload-async.spec.ts` is opt-in and was skipped. Even when enabled, it checks upload, redirect and executable labels, not actual execution, tutor grounding, ownership or restart recovery. | Extend a staging journey: upload → queue → evidence → generated code → validation → execution → reload. Include failure/timeout, duplicate uploads and ownership checks. | Real service-backed logs and persisted outputs for representative supported papers and an unsupported input. Run provider-dependent work only in an authorized staging setup. |
| P2 | `WorkspacePaperClient.tsx:925` explicitly disables the Implement tab's Run in Sandbox control. Other execution controls exist. | Decide whether to connect this editor's current code to execution or remove the duplicated unfinished action. | The edited code is what executes; output, failure and timeout states are clear. |
| P2 | Guided implementation completion is stored in localStorage in `ImplementClient.tsx:39–59`; Dojo drafts also use browser storage. | Decide what should sync per account, then persist progress/drafts with ownership and conflict rules. | Reload and second-device restore work; switching accounts does not reveal another user's saved work. This review did not test cross-account leakage. |
| P2 | `backend/server.py:77–111` uses create-all and a startup column backfill; comments explicitly say no Alembic is configured. SQLite startup probes passed, but production PostgreSQL upgrades were not exercised. | Add versioned migrations and a rehearsed production-database upgrade procedure. | Fresh install and upgrade from an existing database preserve data; concurrent startup and failed migration have defined outcomes. |
| P2 | Marketing still promises “production-ready code in seconds” at `extract-code/page.tsx:80`. A shared generic rollout paragraph occurs in 172 paper articles. | Align product promises with measured support and runtime; review educational content for paper-specific explanation, citations and examples. Repetition is an editorial signal, not proof of factual errors. | Claims have reproducible evidence; a reviewed sample of articles is accurate and useful, then extend review across the library. |

## Product improvements after release blockers

- Make job progress actionable: current stage, elapsed time, a clear failure reason and safe retry. Verify existing behavior before adding duplicate controls.
- Show provenance beside generated code: source passages, inferred choices, validation results and known unsupported elements.
- Test populated comparisons and leaderboards, larger ONNX graphs, malformed uploads, keyboard navigation and mobile editor workflows. Existing mocked happy paths do not establish these outcomes.
- Prioritize a complete paper-to-code journey before adding separate analytics, achievement, notification or admin screens. Earlier inventory identified backend capabilities; this review did not establish full frontend coverage for every one.

## Suggested sequence

1. Release and verify the health fix; decide credential rotation separately.
2. Repair email token redemption and MFA response handling with deterministic local tests.
3. Promote production browser and frontend tests into CI.
4. Run and expand the real staging paper-to-code journey, including recovery and authorization.
5. Finish editor execution and account progress, then migrations, accessibility and content quality.

The prior harness excluded provider work, pipeline changes, deployment and Git operations. This review proposes those future steps where relevant; it does not execute or imply authorization for them. No paid LLM, E2B, email or OAuth calls were made. No live readiness, load, security penetration or comprehensive accessibility testing was performed in this review.
