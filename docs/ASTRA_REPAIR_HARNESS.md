# Repair harness — paper2code

You are fixing confirmed defects in a Next.js + FastAPI application. Work through
the tasks in order. Each task has a reproduction, a fix boundary, and an
acceptance test you must actually run.

**Do not commit. No `git commit` / `push` / `reset` / `checkout --`.** Leave every
change in the working tree and report what you changed.

---

## 0. The one rule that matters here

This codebase has a documented history of **silent degradation**: failures that
look like results. Real examples from its own history —

- a benchmark metric that reported `null` for months because nothing ever fed it
- a verification step that replaced a correct extracted value (`1280`) with an
  invented one (`512`), accepted because its only check was "layer count did not
  decrease"
- a diffusion model generated as a plain segmentation U-Net: it ran, produced a
  tensor, and passed sandbox validation while being incapable of the task
- a smoke test that **passed against an HTTP 404**, because it only asserted that
  the body was visible and the title was not an error

So:

> **A test that cannot fail is worse than no test.** For every fix, run the
> acceptance check against the *unfixed* code first and watch it fail. If it
> passes before your fix, the check is wrong — rewrite the check, not the code.

And:

> **Never report a defect you have not reproduced, and never report a fix you
> have not re-run.** "The code looks like it does X" is a hypothesis. Executing
> it is evidence. State which one you have.

---

## 1. Environment

```bash
# Backend (Python 3.11 venv at .venv)
.venv/Scripts/python.exe -m pytest -q              # full suite, ~2025 tests, 6-7 min
.venv/Scripts/python.exe -m ruff check backend core workers
.venv/Scripts/python.exe -m ruff format --check backend core workers

# Frontend
npx tsc --noEmit -p tsconfig.json
npx vitest run
npm run build                                       # production build, catches RSC errors

# Servers (never run dev servers via a raw shell; use the launch config)
# backend: uvicorn backend.server:app --port 8000
# frontend: next dev --port 3001
```

`.env` holds real credentials. Scripts that do not call `load_dotenv("<repo>/.env")`
silently fall back to `sqlite:///./tensortonic_dev.db`, **not** the real
`test.db` — this has produced false "stale schema" diagnoses more than once.

**Baseline before you start:** `2025 passed, 9 skipped`, ruff clean, `tsc` clean.
If your first run does not match, stop and report the difference rather than
building on top of it.

---

## 2. Tasks

Each defect below was **independently verified** by reading the cited code. Line
numbers are from commit `4eb287a`; re-confirm before editing.

---

### T1 (P0) — `/api/health/celery` leaks broker credentials to anonymous callers

**Verified.** `backend/routers/health.py`, the `health_celery` handler:
- no `Depends(...)` — unauthenticated
- returns `celery_backend_url`, `broker_use_ssl`, and raw
  `os.getenv("CELERY_RESULT_BACKEND")` / `os.getenv("CELERY_BROKER_URL")`

Broker URLs in production carry credentials in the form
`redis://:PASSWORD@host:port`. It also returns 200 without contacting a worker,
so "ready" is not evidence a worker exists.

**Fix boundary:** `backend/routers/health.py` only.

**Do:** return a minimal sanitized readiness result anonymously (reachable
yes/no, worker responding yes/no). Move URL/environment detail behind the
existing admin dependency — find how other admin routes authorize and reuse it;
do not invent a new mechanism.

**Acceptance (write as tests, run them):**
1. Anonymous `GET /api/health/celery` response body contains no `://`,
   no value from `os.environ`, and no substring of either broker URL.
2. With no worker reachable, the anonymous response does **not** report ready.
3. The detailed variant returns 401/403 without admin credentials.
4. Negative control: temporarily put a fake credential URL in the env, assert it
   is absent from the anonymous body, then remove it.

**Report:** whether this endpoint is exposed on the deployed Render service. If
it is, say so plainly — credential rotation is the user's decision, not yours.

---

### T2 (P1) — registration fails on a fresh database

**Verified.** `backend/server.py:77` runs `Base.metadata.create_all(bind=engine)`.
The newer auth router is imported at line 181 (`from backend.modules.auth.api.v1
import router as new_auth_router`) and registered at 184 — **after** the tables
were created. Models reachable only through that import are therefore not in
`Base.metadata` when `create_all` runs, so `verification_tokens` is never
created. Registration then fails with HTTP 500.

Unit fixtures import the fully-constructed app before creating their own tables,
which is exactly why 2025 passing tests do not catch it.

**Fix boundary:** `backend/server.py` import/startup order, plus any model module
that must be imported for registration. Do not restructure the auth module.

**Acceptance:**
1. **Reproduce first.** Point `DATABASE_URL` at a brand-new empty SQLite file,
   start the app, `POST /api/auth/register` — observe the 500 and the missing
   table. Record the error.
2. After the fix, the same sequence returns 201 with no manual table creation.
3. Then: verify → login → `/api/auth/me` → refresh, all 200.
4. Existing-database upgrade path: run against a database created by the *old*
   code and confirm startup adds what is missing without data loss.
5. A regression test that builds a fresh in-memory database **through the real
   startup path** (not through a fixture that pre-creates tables) and registers.

> Note: `backend/server.py` already has a column-backfill mechanism below
> `create_all` (it `ALTER`s missing columns). Understand why that exists before
> changing the ordering — it is compensating for the same root cause and your
> fix should not break it.

---

### T3 (P1) — `/learn` and `/system-design` crash with React error #130

**Verified.** Both page modules use framer-motion directly and neither carries
a `"use client"` directive:

- `src/app/(protected)/learn/page.tsx` — no `"use client"`, 4 `motion.` usages
- `src/app/(protected)/system-design/page.tsx` — no `"use client"`, 2 usages

React error #130 is "element type is undefined" — the server component cannot
resolve `motion.div`.

**Fix boundary:** these two route files plus any small client component you
extract. Keep server-side metadata export where it is; do not convert whole
pages to client components if a leaf extraction will do.

**Acceptance:**
1. **Reproduce first** against a production build (`npm run build && npm start`),
   not dev — dev may mask it. Record the console error.
2. Both routes render the intended page, signed in **and** signed out, with no
   error boundary and no console error.
3. Client-side navigation into each route (click through from another page), not
   just a direct URL load.
4. `npm run build` still succeeds and the routes stay statically generated if
   they were before.
5. Grep the rest of `src/app/**` for the same pattern — server component
   importing `motion.*` — and list any others you find. Fix only these two;
   report the rest.

---

### T4 (P1) — training-cost estimator: frontend and backend disagree on every field

**Verified.** `src/app/(protected)/labs/LabsClient.tsx:29-32` declares:

```ts
total_flops: number;  training_time_hours: number;
cost_estimate_usd: number;  vram_required_gb: number;
```

`core/implementation/cost_estimator.py:345-346` returns `training_hours` and
`compute_cost_usd`; its docstring at :258 names `gpu_memory_gb`. **No field name
matches.** The UI calls `.toFixed()` on `undefined` and crashes into the error
boundary.

**Fix boundary:** pick ONE side and document the chosen response shape. Changing
the backend affects any other consumer — grep for them first.

**This is the trap:** renaming `compute_cost_usd` to `cost_estimate_usd` makes
the UI render a number. It does not make the number correct. `training_hours`
and `training_time_hours` may mean different things (wall-clock vs compute
hours), and `gpu_memory_gb` vs `vram_required_gb` may differ in whether
activations are included. **Establish what each backend value actually measures
before mapping it to a UI label**, and if the UI label has no backend
equivalent, say so rather than aliasing an incompatible value.

**Acceptance:**
1. Reproduce the crash in a browser for one family; record the console error.
2. All four model families render finite, unit-correct numbers.
3. A test asserting the response shape, so the contract cannot drift silently
   again.
4. Invalid input and a backend 500 both produce a useful message, not a crash.
5. State in your report, per field, what it measures and where that is computed.

---

### T5 (P2) — the browser specs cannot fail

**Verified.** `e2e/block-viz-flow.spec.ts` targets `/block-viz/...`; there is no
`src/app/(protected)/block-viz` route. The audit records that this spec
**passed against a 404**.

**Do:** make each spec assert something only the working feature can satisfy — a
specific element, a specific value, a specific network response — never "body is
visible" or "title is not an error". A spec for a removed feature must fail or
be deleted, never pass.

**Acceptance:** for each spec you touch, demonstrate it failing against the
broken/absent state, then passing against the working one. Paste both outcomes.

Decide and state: restore `/block-viz` as a route, or delete the spec and the
advertised path. Do not leave a test asserting a feature that does not exist.

---

## 3. Out of scope

Do not start these; if you find something, report it:

- the extraction/benchmark pipeline under `core/rag/`, `benchmarks/` — separately
  measured and currently passing its gate
- MFA frontend, OAuth round trip, email delivery — need real providers
- anything requiring live E2B or paid LLM calls
- committing, pushing, or branch operations

---

## 4. Report format

For each task:

```
T<n> — <one line: fixed / not fixed / fixed with caveat>

Reproduced:   <the exact command/steps and the failure you observed, verbatim>
Root cause:   <the mechanism, with file:line>
Changed:      <files, and why that boundary>
Acceptance:   <each numbered check, with the actual output>
Negative:     <proof the check fails without the fix>
Not covered:  <what you did not verify, explicitly>
```

Finish with: full backend suite, `vitest`, `tsc`, `ruff check`, `ruff format
--check`, and `npm run build` — all six, with their real output lines.

If a task turns out to be wrong — the code does not say what this document
claims — **say so and stop on that task.** This harness was written from a
verification pass, but a verification pass is not infallible, and a confidently
wrong fix is worse than a reported disagreement.
