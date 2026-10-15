# Repair harness 2 — paper2code

Follows the T1–T5 repair. Tasks U1–U4 below. Same rules as before.

**Do not commit. No `git commit` / `push` / `reset` / `checkout --`.** Leave
changes in the working tree and report what you changed.

**Do not deploy, and do not rotate credentials.** Both are the user's decisions.

---

## 0. The rule, restated

This codebase's characteristic failure is **silent degradation** — something
that looks like a result and is not. Your own last pass found a browser spec
that passed against an HTTP 404, and U1 below is a page that reports "Email
verified" while verifying nothing.

> **Run every acceptance check against the unfixed code first and watch it
> fail.** If it passes before the fix, the check is wrong — rewrite the check.

> **Reproduced ≠ read.** Say which you did. "The code appears to…" is a
> hypothesis; an execution transcript is evidence.

Your T1–T5 report was verified independently against the tree before this
harness was written: the code changes, `2034 passed, 9 skipped`, the nine new
backend tests standalone, and a negative control (reverting
`backend/routers/health.py` → `6 failed`; restoring → `6 passed`). It held up.
Keep that standard.

Two things from that pass worth carrying forward: you caught your own circular
import in T2 before claiming acceptance, and you refused the T4 trap by
documenting what each cost field actually measures instead of aliasing a
renamed value. Both were right.

---

## 1. Environment and baseline

```bash
.venv/Scripts/python.exe -m pytest -q                               # expect 2034 passed, 9 skipped
.venv/Scripts/python.exe -m ruff check backend core workers
.venv/Scripts/python.exe -m ruff format --check backend core workers
npx tsc --noEmit -p tsconfig.json
npx vitest run                                                      # expect 40 passed
npm run build && npx playwright test -c playwright.repair.config.ts # expect 22 passed, 1 skipped
```

If the baseline differs, stop and report rather than building on it.

`.env` holds real credentials. A script that does not call
`load_dotenv("<repo>/.env")` silently uses `sqlite:///./tensortonic_dev.db`
instead of the real `test.db`.

---

## 2. Tasks

Each claim below was verified by reading the cited code on 15 October 2026.
Re-confirm line numbers before editing.

---

### U1 (P1) — email verification verifies nothing, and one of the two links is dead

**Verified, and worse than previously reported.**

- `src/app/auth/verified/page.tsx` contains **no token handling at all** — no
  `fetch`, no `useEffect`, no `searchParams`. It is a static page whose only
  content is the heading "Email verified". It renders that for any visitor,
  with any token or none.
- `backend/modules/auth/services/email_service.py:34` sends users to
  `{FRONTEND_URL}/auth/verified?token={token}` — the static page above.
- `backend/services/email_service.py:52` sends users to
  `{FRONTEND_URL}/verify-email?token={token}` — **and no `/verify-email` route
  exists in `src/app/`.** Those links 404.

So: one sender points at a page that lies, the other at a page that is missing.

The backend has **three** handlers already:

- `backend/modules/auth/api/v1.py:250` — `POST /verify-email` (body-based)
- `backend/modules/auth/api/v1.py:258` — `GET /verify-email` (query param)
- `backend/routers/auth.py:158` — `GET /verify-email` (a second router)

**Before writing any frontend code, determine which handler the signup token
issuer actually produces tokens for.** Two routers both claim this path; find
out which one is mounted at which prefix, which one the token format matches,
and whether the other is dead or reachable. Report what you find. A page that
redeems against the wrong handler will fail in a way that looks like an invalid
token.

**Fix boundary:** one verification page, one link in the senders, and only the
backend handler changes needed to make a single path coherent. Do not refactor
the auth module.

**Acceptance — reproduce each state, do not infer it:**
1. **Valid token:** account becomes verified in the database, exactly once.
   Assert the database row changed, not just the page text.
2. **Reused token:** second redemption does not re-verify and does not report
   plain success.
3. **Expired token**, **malformed token**, **missing token param**: each gives a
   truthful distinct state. "Email verified" must appear for none of them.
4. The page shows a pending state while the request is in flight.
5. Both senders point at the one surviving route; no sender emits a URL with no
   matching page.
6. **Negative control:** the current static page must fail checks 1 and 3.
   Paste that failure.

Capture tokens from the email-service call locally. **Send no real email.**

---

### U2 (P1) — MFA login stores a token that does not exist

**Verified.** `backend/modules/auth/api/v1.py:125` returns
`{"mfa_required": True, "mfa_token": temp_token}` — no access token.
`src/components/useAuthActions.ts:41` calls `setTokens(loginRes)`
unconditionally on the login response.

So an MFA-enabled account "logs in", stores `undefined` as its access token, and
fails later with an authentication error that points nowhere near the cause.
`verify_mfa_temp_token` and the verification route exist at
`backend/modules/auth/api/v1.py:144-154`.

**Fix boundary:** `useAuthActions.ts` and whatever minimal UI is needed to
enter a code. Do not build an MFA *enrolment* settings screen — that is a
separate product decision and is out of scope here.

**Do:** treat ordinary login and an MFA challenge as two distinct response
shapes. Validate which one arrived before storing anything. Hold the challenge
state without pretending login completed.

**Acceptance:**
1. Ordinary login: unchanged behaviour, tokens stored, session works.
2. MFA-required login: **no token is stored**, challenge state is entered.
   Assert `localStorage` has no `access_token` at this point — that assertion is
   the whole point of the task.
3. Valid code completes login and stores real tokens.
4. Wrong code, expired `mfa_token`, and a valid backup code each behave
   correctly and distinctly.
5. **Negative control:** against the current code, check 2 must fail.

---

### U3 (P1) — CI cannot fail for anything you fixed

**Verified.** `.github/workflows/ci.yml` has four jobs:
`lint-and-format` (ruff), `backend-tests` (`pytest tests/`),
`frontend-build` (`npm install && npm run build`), `docker-test`.

- **Zero** occurrences of `vitest` or `playwright` in the file.
- `docker-test` runs `echo "Sandbox execution simulation"` — a job that always
  passes and tests nothing.
- `frontend-build` uses `npm install`, not `npm ci`, so CI does not install the
  locked dependency set.

Every defect repaired in T3, T4 and T5 is invisible to this pipeline. `npm run
build` succeeded while `/learn` and `/system-design` crashed at runtime — the
build gate proves nothing about RSC runtime errors.

**Fix boundary:** `.github/workflows/ci.yml`, plus any `package.json` script or
Playwright config needed to run the production-mode browser suite in CI. Do not
weaken existing jobs.

**Do:** add frontend tests and the production-mode browser suite as real jobs.
Use `npm ci`. Either label `docker-test` honestly as a placeholder or delete it
— a job that cannot fail is worse than no job.

**Acceptance — this task is only done if you prove CI can fail:**
1. Introduce a deliberate RSC error on one route (the T3 pattern), push nothing,
   run the workflow locally or in a branch run, and show the browser job
   **failing**. Revert it.
2. Introduce a deliberate estimator contract mismatch (the T4 pattern) and show
   the relevant job failing. Revert it.
3. The browser job builds and serves in production mode — the same artifact and
   configuration your passing repair checks used, not `next dev`.
4. Opt-in integration suites that skip are **visible as skipped** in CI output,
   not silently absent.
5. State plainly whether CI now runs each of: ruff, pytest, tsc, vitest,
   playwright-production, build.

---

### U4 (P2) — decide the two unfinished controls

Lower priority and genuinely decisions, not repairs. Pick and justify; do not
build both.

- `WorkspacePaperClient.tsx:925` — Implement tab's "Run in Sandbox" is disabled
  with "Coming soon" while other execution controls exist. **Either** make the
  edited code the thing that executes, **or** remove the duplicated control.
  A permanently disabled button is a promise the product does not keep.
- `ImplementClient.tsx:39-59` — guided progress lives in `localStorage`; Dojo
  drafts too. **Either** define per-account persistence with ownership rules,
  **or** label it as local-only in the UI so users do not lose work expecting
  sync.

**Acceptance:** whichever you choose, the user-visible behaviour matches what
the UI claims, with a test. If you choose persistence, include a check that one
account cannot read another's saved work.

---

## 3. Out of scope — do not start

- **Deployment and credential rotation.** The deployed
  `/api/health/celery` still exposes broker credentials; I re-confirmed it
  independently. That is the user's action, not yours.
- **The real staging journey** (upload → queue → evidence → code → execution).
  It needs authorized credentials and a staging environment. Propose it; do not
  run it.
- Live email delivery, OAuth round trip, paid LLM or E2B calls.
- MFA *enrolment* UI (U2 covers login challenge handling only).
- Alembic migrations — real, but it needs a rehearsed Postgres procedure and a
  decision about downtime.
- `core/rag/`, `benchmarks/`, `core/fidelity.py`, `core/paper_to_code_generator.py`,
  `core/transformer_model_builder.py`, `core/classification.py` — the extraction
  and generation pipeline, separately measured and under active uncommitted work.
  **Touching these will collide.**
- Marketing-claim edits and the 172 repeated article paragraphs — editorial
  decisions for the user.

---

## 4. Report format

Per task:

```
U<n> — fixed / not fixed / fixed with caveat

Reproduced:   exact command and observed failure, verbatim
Root cause:   mechanism, with file:line
Changed:      files, and why that boundary
Acceptance:   each numbered check with real output
Negative:     proof the check fails without the fix
Not covered:  what you did not verify, explicitly
```

For U1, state up front which backend handler you selected and how you
established that the signup issuer's tokens match it.

Close with all six baseline commands plus the production browser suite, showing
real output lines.

If a claim here is wrong — the code does not say what this document states —
**say so and stop on that task.** This was written from a verification pass, and
a verification pass is not infallible. A reported disagreement is worth more
than a confident wrong fix.
