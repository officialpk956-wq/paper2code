# Paper-to-Code Execution Memory

Last updated: 2026-08-29 (Asia/Calcutta)

## Active objective

Phase 1 restored the PDF upload path through asynchronous processing,
persisted generated code, and workspace delivery. Phase 1.5 live-verified
that end to end and fixed four bugs in the extraction → graph → codegen
chain. Phase 2 (ConfigExtractor wiring, sandboxed validation, repair loop,
transformer builder) was implemented, then independently audited twice —
the first pass found three headline claims wired but not functional (each
had its real integration point mocked out in the test meant to cover it,
all fixed and re-verified live); a second, deeper audit continued
verification with a full 10-case live family matrix and found four more
real crash bugs, all fixed. See "Phase 2 audit and fixes" and
"Phase 2 second audit" below. The pipeline now genuinely works end to end
for all 4 known families plus honest needs_review handling for unsupported
ones, proven via real live HTTP uploads including a genuinely successful
LLM-generated GAN implementation and a correctly-caught DDPM/U-Net
family-ambiguity case.

**Phase 2 is now complete and was explicitly closed by the user on
2026-08-29.** A final closure pass (same day) resolved
every open item from the second audit: brought up the real Docker
Compose stack and confirmed the actual non-eager Redis/Celery/Playwright
UI flow passes end to end (43.9s, real login → real upload → real
polling → real redirect → real Executable tab), and added the missing
`_generate_skeleton` regression tests. See "Known untested paths —
RESOLVED" below for the full closure record. Nothing from the original
Phase 2 plan remains unverified except two explicitly-scoped, deliberate
exceptions noted there (a code-quality-not-crash limitation, and the
practical reality that external LLM rate limits mean no single 10/10
matrix run has completed uninterrupted end to end, though every family
has independently succeeded live).

## Phase 1 status

Implemented locally, with no commit or push:

- Terms acceptance is explicit in the paper workspace upload UI and remains
  enforced by the backend.
- The frontend consumes the asynchronous `{task_id, poll_url}` response, polls
  once per second with a ten-minute bound, reports stages/errors, and navigates
  to `/papers/{paper_id}` on completion.
- The papers-list client now consumes the actual `{summary, papers}` envelope.
- Client-side PDF validation matches the backend 20 MB limit.
- Generated source, compiled metadata, generation status, verification report,
  and the last generation error are persisted on `Paper`.
- ResNet, U-Net, and ViT builder output is self-contained and passes clean-
  namespace import, construction, and forward-shape verification.
- The workspace Executable tab displays persisted code and its verification
  result and can submit it to the existing authenticated sandbox endpoint.
- The old executable architecture graph remains stored in
  `architecture_graph.ingestion.executable_graph` for graph exports.

## Phase 1.5 — Live verification and pipeline repair (2026-08-29)

Phase 1's own regression suite passed, but nothing had exercised a real PDF
through the live LLM-backed extraction path end to end. Doing so (with a
temporary local backend on port 8010, `VERIFY_PASS_EAGER=1` for synchronous
Celery execution, no Redis required) surfaced four real, pre-existing bugs
between extraction and code generation — none introduced by Phase 1, all
now fixed:

1. **`core/paper_to_code_generator.py` — `_run_pipeline` layers-key
   mismatch.** Checked `spec.get("layers")`, a key that never exists in the
   schema `extract_architecture` actually populates
   (`core/schemas_base.py`: `model_family`/`stem`/`block`/`stages`/`head`).
   This made every real upload fail unconditionally with "No architecture
   could be detected.", regardless of paper quality or LLM validity. Fixed
   to check `model_family`/`stages` instead.

2. **`core/section_splitter.py` — `safe_parse_llm_output` dropped
   sections.** When the LLM classified one text chunk into multiple
   sections at once (a normal response shape), the function kept only
   `parsed[0]` and silently discarded the rest — so "method"/"experiments"
   content was thrown away whenever "abstract" happened to come first.
   Fixed to return and merge every section in the list.

3. **`core/paper_to_code_generator.py` — `ParsingAgentImpl` schema
   mismatch.** `Paper2CodePipeline.run_single()` hands the extracted spec to
   `ParsingAgentImpl.parse()`, which only accepts a ConfigDict
   (`{"layers": [...], "name": ...}`), a text excerpt, or a symbolic spec —
   never the `BASE_MODEL_SCHEMA` shape actually produced. This always
   raised `ParsingError("Ambiguous or unsupported parsing source format.")`.
   Added `_spec_to_config_dict()`, a translator from `BASE_MODEL_SCHEMA`
   (`stem`/`stages`/`block`/`head`) into the ConfigDict shape
   `ConfigParsingAgent` consumes, with sequential layer connections. This
   was the central "two subsystems never wired together" gap the original
   assessment predicted.

4. **Three "explicit `None` survives merge/default" bugs**, all the same
   underlying pattern (found reactively, one crash at a time, while proving
   #3 worked against real non-deterministic LLM output):
   - `core/normalizer.py` — `params.setdefault("stride", 2)` etc. only fire
     when a key is *missing*; the LLM sometimes returns `"stride": null`
     explicitly, which `setdefault` leaves untouched. Fixed to
     `params[key] = params.get(key) or default` for `out_channels`,
     `kernel`, `stride`, `padding`.
   - `core/classification.py` — `classify_architecture` read `node.name`,
     but `GraphNode` has no such field (only `id`/`type`/`label`/`params`/
     `description`/`semantic_params`). Fixed to `node.label`.
   - `core/paper_to_code_generator.py` — `_prepare_builder_schema` used
     `{**default_stage, **extracted_stage}` merges that replaced entire
     default stage lists wholesale instead of deep-merging per-stage, so a
     partial extracted `stages` list (missing `in_channels`/`expansion`/
     `stride`/`downsample`/`num_blocks` for resnet, or `repeats` for vit)
     crashed the builder with `KeyError`/`TypeError`. Fixed to deep-merge
     each extracted stage onto the matching default stage template
     (cycling by index) for both `resnet` and `vit`.

**Live verification after all four fixes:** 8/8 consecutive successful runs
(4 direct Python calls to `from_pdf()`, 4 real HTTP uploads through
`/api/papers/upload`) — all landing on `generation_status: "success"`,
correct extracted `ResNet-50` spec (`stages: [3,4,6,3]`), self-contained
generated code, correct output shape `[1, 1000]`.

**One unresolved, unreproduced edge case:** a single
`TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'` was
observed once via a specific LLM extraction variant, not reproduced across
8 subsequent attempts. Tied to LLM non-determinism; not isolated. Worth
revisiting if it recurs, but not blocking.

**Regression tests added:**
- `tests/test_section_splitter.py` (4 tests) — covers bug #2.
- `tests/test_phase1_paper_codegen.py` — 4 new tests covering bugs #1, #3,
  and #4 (resnet + vit stage-merge cases).

**Full suite after Phase 1.5: 1697 passed, 3 skipped, 0 failed.**

### Live verification method (for reproducing later)

No local Redis/Celery worker exists on this machine, so live testing used:
1. A temporary second backend instance on port 8010
   (`VERIFY_PASS_EAGER=1 python -m uvicorn backend.server:app --port 8010`),
   which sets `celery_app.conf.task_always_eager = True` — only active
   behind that env var, reverted after each session, never committed.
2. `.env.local`'s `NEXT_PUBLIC_API_URL` temporarily repointed at 8010,
   reverted after.
3. A throwaway test user registered/logged in via the real auth API.
4. Real PDF fixtures (via PyMuPDF-generated text, or the existing
   `tests/fixtures/phase1_architecture.pdf.b64`) uploaded through the real
   `/api/papers/upload` endpoint.
5. All synthetic users/papers deleted from `test.db` after each pass.

This is a legitimate way to exercise the real LLM-backed path locally
without standing up Redis/a Celery worker — eager mode runs the task
synchronously in-process, so `.delay()` never needs a broker. Caveat: eager
mode makes task *failures* surface as a synchronous 400 from the upload
endpoint instead of the real async "202 now, poll for failure later"
behavior — fine for checking pipeline correctness, not for testing the
polling UI's failure-handling path.

## Canonical repository paths

- Upload UI: `src/app/(protected)/papers/page.tsx`
- Workspace UI: `src/app/(protected)/papers/[id]/WorkspacePaperClient.tsx`
- Upload/artifact routes: `backend/routers/papers_pipeline.py`
- List/detail routes: `backend/routers/papers.py`
- Analysis routes: `backend/routers/papers_analysis.py`
- Celery task: `backend/tasks/paper_tasks.py`
- Ingestion persistence: `backend/services/paper_ingestion_service.py`
- Code generator + ConfigDict translator: `core/paper_to_code_generator.py`
- Architecture extraction (bare LLM call, no retrieval/verification):
  `core/architecture_extractor.py`
- Richer extractor (BM25 + structured extraction + verification pass,
  **not yet wired into the live path** — this is Phase 2's top priority):
  `core/rag/config_extractor.py`
- Section classification: `core/section_splitter.py`
- Spec normalization: `core/normalizer.py`
- Graph-based family classification (fallback only): `core/classification.py`
- ConfigDict → graph parsing: `core/agents/config_parser.py`,
  `core/agents/parsing_agent_impl.py`
- Orchestration wiring: `core/orchestrator/pipeline.py`
- ORM model: `backend/models.py`

The three paper routers have distinct active responsibilities. Deprecated
analysis aliases were intentionally retained; no routes were deleted.

## API and schema decisions

- Initial upload response includes `paper_id: null` and task status `pending`.
- Task terminal success is `completed`; the final paper ID is
  `task.result.paper_id`.
- Generated-code fields use SQLAlchemy `Text`, `String`, and portable `JSON` so
  PostgreSQL production and SQLite tests share the same model.
- Generation statuses are `pending`, `success`, `failed`, and `needs_review`.
- Phase 1 marks verified known-family forward passes as `success`; syntax-only
  LLM/skeleton outputs remain `needs_review`.
- `BASE_MODEL_SCHEMA` (`core/schemas_base.py`) is the extraction output
  shape: `model_family`/`variant`/`task`/`input`/`output`/`stem`/`block`/
  `stages`/`head`. This is **not** the same as the ConfigDict shape
  (`{"name", "layers", "connections"}`) that `ParsingAgentImpl`/
  `ConfigParsingAgent` consume for graph construction — the translator
  bridging them is `PaperToCodeGenerator._spec_to_config_dict()`.

## Environment / credentials (2026-08-29)

Local `.env` previously had **mock placeholder values** for
`GROQ_API_KEY`/`GEMINI_API_KEY` (`gsk_mockkey...`, `AIzaSy_mockkey...`) —
not expired real keys, literal placeholders that were never functional.
Replaced with live keys; also corrected two stale model names:

- `LLM_PRIMARY_MODEL`: `groq/llama-3.3-70b-versatile` (fully retired by
  Groq) → `groq/openai/gpt-oss-120b`. Note: this is a reasoning model that
  spends tokens on hidden reasoning before emitting content — code calling
  `llm_complete`/`llm_complete_async` with a low `max_tokens` may see
  truncated/empty responses. Worth auditing call sites if this recurs.
- `LLM_FALLBACK_MODEL`: `gemini/gemini-2.0-flash` (retired) →
  `gemini/gemini-3.6-flash` (Google's own 404 error on the old model named
  this as the replacement for new projects).
- `RESEND_API_KEY`: was empty, now set (confirmed valid — a sending-only
  restricted key, which is what the app needs).
- `GITHUB_CLIENT_ID`, `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`: were
  placeholders, now set with real values.
- `GITHUB_CLIENT_SECRET`: **still a placeholder** — only the client ID was
  provided.
- `SECRET_KEY`/`JWT_KEY_RING`: regenerated to a cryptographically random
  256-bit value (was a predictable dev string). Side effect: invalidates
  any previously-issued local JWTs.
- `E2B_API_KEY`: confirmed already present and valid — the earlier belief
  that "E2B is not configured on this machine" was incorrect. E2B is not
  the blocker for sandboxed validation; wiring it into `paper_to_code_generator`
  is (Phase 2 scope).
- `QDRANT_URL`: intentionally left empty — not needed until Phase 2/4 (RAG).
  `backend/services/vector_service.py` degrades gracefully (returns `None`)
  when unset, confirmed by reading the code.

Actual key values are not recorded here — see the user's local `.env`.

## Migration

- Revision: `l3m4n5o6p7q8`
- Previous revision: `k2l3m4n5o6p7`
- Upgrade: `python -m alembic upgrade head`
- Rollback: `python -m alembic downgrade k2l3m4n5o6p7`

The Phase 1 revision was tested on a temporary SQLite database by stamping the
current previous head, upgrading, downgrading, and upgrading again. All three
operations succeeded.

Known pre-existing migration issue: creating a completely blank database and
replaying the repository's entire historical migration chain fails in revision
`cc3efb9d9907` because it expects a `learner_progress` table that an earlier
migration did not create. This predates Phase 1 and does not originate in the
new revision. Separately, the local dev `test.db` has schema drift from its
migration bookkeeping (`alembic current` reports a stale head, but the actual
Phase 1 columns are present) — likely from an earlier ad hoc `create_all()`
run. Not investigated further; the columns needed are confirmed present.

## Tests and results

- Known-family direct verification: ResNet `(1,3,224,224) -> (1,1000)`, U-Net
  `(1,3,256,256) -> (1,2,256,256)`, and ViT `(1,3,224,224) -> (1,1000)` passed.
- Full backend Pytest suite (after Phase 1.5 fixes): **1697 passed, 3 skipped,
  0 failed**.
- Frontend Vitest suite: **37 passed** across 8 files (Phase 1, unchanged by
  Phase 1.5 — no frontend files touched).
- TypeScript: `npx tsc --noEmit` passed.
- Ruff checks for changed Python files passed.
- Production Next.js build passed (unchanged by Phase 1.5).

The opt-in hosted test `tests/integration/test_live_paper_upload.py` requires
`RUN_LIVE_PAPER_PIPELINE=1`, `PAPER2CODE_LIVE_API_URL`,
`PAPER2CODE_LIVE_TOKEN`, and a deployed Redis/Celery/E2B stack — still
skipped by default, still not run (no deployed environment available from
this machine). This is different from local live verification, which *was*
done (see "Phase 1.5" above) via the eager-mode technique, without needing
this hosted test or a real deployment.

## Known untested paths — RESOLVED (2026-08-29, final closure pass)

All three gaps below were closed in a final pass after the second audit.

- ~~The real async (non-eager) Celery flow~~ **CLOSED.** Brought up the
  real stack via `docker compose up -d postgres redis` (confirmed both
  services report `healthy`), started the real (non-eager) backend +
  Celery worker against it, and ran the actual gated Playwright test
  (`RUN_REAL_ASYNC_PAPER_E2E=1 npx playwright test
  e2e/paper-upload-async.spec.ts`). First attempt failed at a 5s
  `toBeVisible` check on a freshly-spawned, cold Next.js dev server (the
  page hadn't finished its first compile) -- not an app bug. Re-ran with
  a warm build cache: **passed, 43.9s.** Full real flow confirmed: login →
  localStorage session → `/papers` → terms checkbox → file upload → real
  "Paper upload progress" dialog → real 1s polling → real redirect to
  `/papers/{id}` (genuine Celery task completion through real Redis) →
  Executable tab → "Phase 1 verified" visible. `docs/LOCAL_DEV.md`'s
  Docker Compose instructions are now independently confirmed accurate,
  not just documented.
- ~~unet/vit/transformer through the live HTTP path~~ **CLOSED** by the
  second audit's family matrix runs -- vit and transformer succeeded live
  in both matrix attempts; unet succeeded live in the first (interrupted)
  matrix run before later hitting the Gemini-only rate-limit cascade
  described above (an external/config issue, not a pipeline defect).
- The intermittent `TypeError` noted in Phase 1.5 (unreproduced after 8
  retries) — still not seen again across all of Phase 2's extensive live
  testing. Considered resolved in practice; no longer worth tracking as
  an open item.

**`_generate_skeleton` regression coverage — CLOSED.** Added
`tests/test_codegen_skeleton.py` (4 tests) covering both crash bugs found
in the second audit (undefined-attribute forward() calls, the
`hasattr`-vs-`is not None` input_shape guard) plus a full compile-and-run
check and a class-name-sanitization check. All 4 pass.

**Remaining, genuinely open (by design, not oversight):**
- `_generate_skeleton`'s first-layer channel-mismatch fidelity limitation
  (hardcodes `in_channels == out_channels`) — a code-quality issue the
  real validation pipeline already catches and reports honestly as
  `needs_review`, not a crash. Deliberately left for a future pass.
- A full, single, uninterrupted 10/10 family-matrix run has never
  completed in one go (results are pieced together across multiple
  partial runs due to real external rate-limit/network interruptions).
  Every individual family has independently succeeded live at least once,
  and the failure modes observed were all external (provider rate limits,
  not pipeline logic) -- but nobody should claim "one clean 10/10 run" as
  literally true without re-running it end to end with fresh quota.

## Phase 2 status — Complete, after an independent audit found and fixed 4 real gaps (2026-08-29)

### Phase 2.1 architecture decision

Chosen approach: **option (b)**. The live path calls `ConfigExtractor` once,
builds the `ArchitectureGraph` directly from its `ConfigDict`, classifies that
graph, and derives the known-family builder schema from the same normalized
layers. This avoids option (a)'s second LLM call and its doubled latency/cost,
while retaining the proven family builders instead of taking option (c)'s
larger code-generation redesign. The legacy extractor and
`_spec_to_config_dict()` remain only as a bounded fallback when the richer
extractor produces no usable layers. A restart audit fixed an integration bug
where the temporary string `"unknown"` was treated as a valid family before
graph classification, preventing the graph-derived family from being used.

Phase 2 was implemented (ConfigExtractor wiring, transformer builder, null
audit, E2B validation, repair loop, local dev docs) and initially
self-reported as fully complete. An independent audit — running the exact
real (non-mocked) code paths the test suite avoided — found that **3 of
the 6 deliverables were wired but not actually functional**, each with its
real integration point mocked out in the test meant to cover it:

1. **ConfigExtractor's real path was still non-deterministic.** All three
   original tests used `ConfigExtractor(use_llm=False)` (the deterministic
   rule-based fallback); production uses `ConfigExtractor()` which
   defaults to `use_llm=True`. A live run of the real path produced
   **20/8/20 layers** for 3 identical calls — the new Groq model
   (`groq/openai/gpt-oss-120b`) has an 8000 TPM rate limit that
   `ConfigExtractor`'s multi-call verification loop exhausts quickly,
   causing litellm's `fallbacks` param to silently swap to Gemini for some
   calls but not others, mid-pipeline, with no visible error.

2. **E2B sandboxed validation was completely non-functional in real
   usage.** The default E2B sandbox template has no PyTorch (only
   numpy/pandas/scipy); the new integration used a bare 10s timeout with
   no install step, so every non-builder validation failed with
   `ModuleNotFoundError: No module named 'torch'` — this codebase already
   had the correct pattern for this exact problem in
   `backend/services/pytorch_parser.py` (try import, `pip install torch
   --index-url .../whl/cpu` on failure, 300s timeout), which the new code
   didn't reuse. Both E2B tests mocked `run_code_in_sandbox` entirely, so
   they verified report-parsing but never the real sandbox call.

3. **Even with torch available, the E2B harness never ran a real forward
   pass.** It only tried to instantiate the target class (with a broad
   except-swallowing fallback that could "succeed" even when instantiation
   truly failed) and unconditionally set `checks["forward"] = True`
   whenever the script exited 0 — no synthetic input, no output-shape
   check, despite that being the explicit ask.

4. **`test_live_all_families_api.py`'s "live" verification wasn't live.**
   Despite the name, it called `_generate_code()`/`validate_generated_code()`
   directly with a hardcoded synthetic spec — no PDF, no ConfigExtractor,
   no HTTP. It re-proved the same in-process builder path Phase 1.5 already
   verified (genuinely new only for `transformer`), not the new Phase 2
   extraction/validation wiring the walkthrough's results table implied.

A fifth, smaller gap found in the same pass: `_repair_code()` called
`llm_complete()` with no error handling — a transient LLM failure (e.g.
the rate limit above) would raise unhandled out of `_run_pipeline`'s
repair loop, crashing the whole upload instead of stopping gracefully at
the last known `verification_report`.

**All four fixed and re-verified with real (non-mocked) calls:**

- **`core/llm_client.py`** (`llm_complete` + `llm_complete_async`): retry
  the primary model up to 2x with an 8s backoff on `RateLimitError` before
  allowing litellm's cross-provider `fallbacks` — only the final attempt
  passes `fallbacks=[FALLBACK_MODEL]`. Re-tested the exact same live
  ConfigExtractor scenario: **20/20/20 layers**, identical types except one
  harmless naming synonym (`avgpool2d` vs `globalavgpool2d`) — down from
  20/8/20. New tests: `tests/test_llm_client_retry.py` (mocked, fast,
  proves the retry-before-fallback mechanics deterministically).
- **`core/paper_to_code_generator.py`**'s E2B harness (Strategy B of
  `validate_generated_code`): now bootstraps torch (`bootstrap` snippet
  runs *before* `code` — generated code almost always has its own
  top-level `import torch`, which would raise immediately if the bootstrap
  ran after it in file order), then genuinely runs `model(test_input)`
  with a synthetic input derived from the spec (tries image-shaped and
  sequence-shaped candidates), checks the real output shape against
  `spec["output"]["num_classes"]`, and reports a structured
  `__PAPER2CODE_RESULT__` JSON line the caller parses. Re-tested live:
  deliberately-broken code now correctly fails with a real PyTorch
  shape-mismatch error (`checks.exec=True, checks.forward=False`);
  genuinely-valid code correctly passes with the real output shape. New
  live tests (gated on `E2B_API_KEY`, `@pytest.mark.live`) in
  `tests/test_phase2_e2b_validation.py`; the two pre-existing mocked tests
  were also updated since the mock's `stdout` format no longer matches
  what the real harness prints.
- **`_repair_code()`**: wrapped `llm_complete()` in try/except, returns
  `None` on failure (the existing `_run_pipeline` while-loop already
  treats a falsy return as "stop the repair loop"). New tests in
  `tests/test_phase2_repair_loop.py` confirm both the unit-level graceful
  return and the full `_run_pipeline` not propagating the exception.
- **`test_live_all_families_api.py`**: left as-is (it's still a valid
  in-process regression test, just not evidence of live extraction) --
  superseded as "real live proof" by the two live end-to-end HTTP uploads
  described below.

**Final live verification, real HTTP API, all fixes applied:**
- ResNet paper excerpt → `HTTP 200` → `generation_status: success`,
  `family: resnet`, `code_source: builder`.
- GAN paper excerpt (unsupported family) → `HTTP 200` → 3 repair attempts,
  each with a genuinely different real diagnostic (dtype mismatch, then
  two distinct shape mismatches) → `generation_status: needs_review` with
  a real error, not a masked one. This is the system working *correctly*
  — GAN generator/discriminator code is genuinely harder for the LLM to
  get right, and Phase 2's job was to catch that honestly, not force a
  false "success".

**Full test suite: 1723 passed, 3 skipped, 0 failed** (non-live). Live
tests (`-m live`, require real `GROQ_API_KEY`/`E2B_API_KEY`) all pass
individually; running the two live E2B tests back-to-back with several
other E2B-hitting live tests in the same session occasionally hits a
transient E2B-side flake (likely a concurrent-sandbox limit) — confirmed
non-reproducible by re-running in isolation immediately after. This is
expected/accepted behavior for tests hitting a real external API, which is
exactly why they're gated behind the pre-existing `live` marker rather
than required for a normal CI run.

**Original Phase 2 implementation notes** (still accurate, unaffected by
the audit):

1. **ConfigExtractor wiring (2.1)** — `_run_pipeline` tries
   `ConfigExtractor.extract_from_text()` first, falls back to the legacy
   `process_text`/`extract_architecture`/`_spec_to_config_dict` path only
   if it returns no layers. `_config_dict_to_builder_spec()` derives a
   `BASE_MODEL_SCHEMA`-shaped spec from the ConfigDict for the builder
   shortcut.
2. **Transformer builder (2.1/2.5)** — new `core/transformer_model_builder.py`
   (`TransformerModelBuilder`), wired into `_prepare_builder_schema` and
   `_self_contained_builder_code` alongside resnet/unet/vit.
3. **Null/defaults audit (2.2)** — genuinely thorough; extended beyond the
   Phase 1.5 spots to `core/normalizer.py`, `core/classification.py`,
   `core/rag/config_extractor.py`, and `_prepare_builder_schema`'s general
   merge loop.
4. **Repair loop structure (2.4)** — bounded at 3 total attempts, history
   stored in `verification_report["attempts"]` (reused the existing JSON
   column, no new migration).
5. **Local dev docs (2.6)** — `docs/LOCAL_DEV.md` created (Docker Compose
   Postgres+Redis, Windows Celery `-P solo`, non-eager async flow). Not
   independently re-verified in this audit pass — Docker Desktop's
   availability in this environment was previously unconfirmed (see
   Phase 1.5 notes); treat as documentation, not proven-working
   instructions, until someone actually runs it.

## Phase 2 second audit (2026-08-29, later same day)

A separate, independent Codex CLI session continued Phase 2 hardening
after the first audit above: fixed a live-test gating gap (`pytest tests/`
was silently running real E2B tests because the skip check only looked for
an API key's presence, not an explicit opt-in — added `RUN_LIVE_PHASE2`),
a family-derivation ordering bug (ConfigExtractor's path set
`family="unknown"` before graph-based classification could ever run,
permanently shadowing it), added `concat` as a recognized layer type (a
legitimate U-Net skip-connection operation the RAG normalizer previously
rejected, silently falling back to the legacy extractor), fixed a
successful-upload local-temp-file leak (cleanup previously only ran on the
*final* failed retry, never on success), fixed attempt-history tracking
(`final_attempt` used to come from a separately-incremented counter that
could drift from what was actually validated -- now `len(attempts)`,
correct by construction), and — the most substantial single fix — added a
real graph/spec adapter (`_architecture_spec_payload` in
`backend/services/paper_ingestion_service.py`) so that
`ArchitectureSpec(**spec)` validation stopped failing on every single
BASE_MODEL_SCHEMA-shaped extraction (it requires `input_shape` and
per-layer `name`, which the raw spec never had), which had been silently
discarding genuine extraction results into `{family: unknown, layers: []}`
for the learning-module generation path specifically (separate from, and
previously masking a real problem alongside, the code-generation path this
memory doc already covers).

That session ran a real 10-case live family matrix
(`tests/integration/test_live_phase2_family_matrix.py`, gated behind
`RUN_LIVE_FAMILY_MATRIX=1`) against a real non-eager Redis/Celery/HTTP
stack and a real Playwright browser test
(`e2e/paper-upload-async.spec.ts`) proving the actual async UI polling
flow — the "known untested path" this doc had flagged. Both were
interrupted mid-run by network drops and a usage-limit cutoff, leaving
real infrastructure (a Docker Redis container, two backend+worker
process pairs, a `.phase2_matrix3.db`) running unattended.

**This was independently verified and continued, not just read.** Picking
up the live infrastructure (still healthy) rather than starting cold:

- Confirmed `_architecture_spec_payload` genuinely fixes `ArchitectureSpec`
  validation with a real graph (previously guaranteed to fail for the
  common BASE_MODEL_SCHEMA case).
- Confirmed the `concat` fix, the attempt-tracking fix, and the
  `RUN_LIVE_PHASE2` gating fix by direct code/diff review.
- Re-ran the 10-case family matrix twice (first attempt hit the per-IP
  10/hour upload rate limit from the interrupted session's own prior
  requests -- restarting the backend process resets its in-memory
  counter). Second run: **6/8 supported families succeeded**, both U-Net
  cases failed with a cascading `LLM circuit breaker open` -- this
  specific matrix run set `LLM_PRIMARY_MODEL` and `LLM_FALLBACK_MODEL` to
  the *same* Gemini model (a workaround for Groq's earlier exhaustion),
  so when Gemini's own rate limit sustained failures long enough to trip
  the circuit breaker (a global, process-wide 60s cooldown), it blocked
  several sequential uploads, not just the one that triggered it. This is
  external rate-limiting cascading through a reasonable safety mechanism,
  not a pipeline logic defect -- the *earlier*, interrupted matrix run
  (different LLM config) had both U-Net cases genuinely succeed.
- Targeted-tested the two unsupported-family cases directly (restarted
  once more with the real production LLM config -- Groq primary, Gemini
  fallback -- for genuine cross-provider diversity):
  - **StyleGAN → `generation_status: success`, `output_shape: [1, 3, 2, 2]`.**
    Inspected the actual generated code: a genuine, honestly-simplified
    GAN generator (learned constant, a "style_modulated_conv" placeholder
    explicitly commented as such, a toRGB head) that really runs and
    produces a real RGB image tensor. This is **not** a false positive --
    the E2B harness correctly skips the classification-shape check for
    specs with no `num_classes` (GANs have no such output), which is the
    right behavior for a generative model. It revealed that the *test
    matrix's own assumption* ("every unsupported case must land on
    needs_review") is stricter than the pipeline's actual, correct
    behavior: the LLM-generation+E2B path can legitimately succeed for a
    moderately-tractable non-builder architecture, and did.
  - **DDPM → `family: unet` (misclassified), `generation_status:
    needs_review`.** Diffusion models are genuinely, textually described
    as containing "a time-conditioned U-Net" -- the family classifier
    picking up on that isn't unreasonable, though a label of "unknown" or
    "diffusion" would be more honest. Safely landed on needs_review rather
    than a false success either way.

**Reproducing the DDPM case surfaced four new, real, independently-found
bugs** (none present in the first audit's four fixes -- these are new
crash sites the deeper live testing reached for the first time):

6. **`core/rag/normalizer.py` — `_normalize_type` hard-`assert`ed on any
   layer type outside its fixed `CANONICAL_TYPES` vocabulary**, crashing
   the entire `ConfigExtractor.extract_from_text()` call for one
   unfamiliar (but completely legitimate) layer name -- e.g. a diffusion
   model's `timestep_embedding`. Downstream (`ConfigParsingAgent
   ._compute_semantic_params`) already defaults gracefully for
   unrecognized types, so the hard failure was never necessary. Fixed to
   log a warning and pass the type through as-is.
7. **`core/paper_to_code_generator.py` — `_e2b_test_input_candidates`
   crashed with `TypeError: 'int' object is not subscriptable`** whenever
   `spec["input"]["spatial_dims"]` came back as a bare scalar (e.g. the
   LLM describing "224x224" as just `224`) rather than a `[h, w]` list --
   `dims = spatial or [224, 224]` let the truthy int survive, then
   `dims[0]` crashed. This ran identically on *every* repair attempt
   (it's spec-derived, not code-derived), silently burning the entire
   3-attempt repair budget on something no code fix could ever address.
   Live-observed via the matrix's DDPM case before being isolated and
   fixed with a direct (non-LLM) reproduction.
8. **`core/codegen.py` — `_generate_skeleton` referenced
   `self.{node.id}` in `forward()` for *every* graph node, but only
   defined that attribute in `__init__` when `_node_to_layer()` recognized
   the node's type.** `_node_to_layer`'s fixed `MAP` doesn't even cover
   `residualblock`/`bottleneckblock`/`concat`/`identity` -- types used
   elsewhere in this same pipeline -- so any node of an unmapped type
   produced `AttributeError: '...' object has no attribute 'layer_N'` on
   the very first line of `forward()`. This is the final skeleton-fallback
   path (used only when both the builder and LLM-generation strategies
   fail), explicitly flagged in this doc as "never exercised live" until
   this session's matrix testing (with both LLM providers quota-exhausted)
   finally reached it. Fixed: unrecognized node types are now skipped in
   `forward()` (input passes through unchanged) instead of crashing.
9. **`core/codegen.py` — `_node_to_layer`'s `in_hs` computation used
   `hasattr(node, "input_shape")` to guard a subscript, but
   `GraphNode.input_shape` is a dataclass field defaulting to `None` --
   `hasattr` is always `True` regardless of the value**, so
   `node.input_shape[-1]` crashed with `TypeError: 'NoneType' object is
   not subscriptable` for any node tensor_tracker hadn't populated shape
   info for (the common case for skeleton-fallback graphs). Fixed to check
   `node.input_shape is not None` before subscripting.

Bugs 8 and 9 together meant `_generate_skeleton` was **effectively always
broken** for any realistic graph, not an edge case -- confirmed by direct
reproduction (crashed on both bugs before the fix; generates and runs
cleanly after, modulo a separate, pre-existing, non-crashing fidelity
limitation where the first layer's `in_channels`/`out_channels` are both
hardcoded to the same value regardless of the real input channel count --
noted but intentionally not fixed in this pass, since it's a quality issue
the real forward-pass validation already catches and reports honestly as
`needs_review`, not a crash).

**Regression tests added:** `tests/test_phase2_null_safety.py` gained
`test_normalize_type_degrades_gracefully_for_unrecognized_types` and
`test_e2b_input_candidates_handles_scalar_spatial_dims`. Bugs 8/9
(`_generate_skeleton`) do not yet have a dedicated regression test in this
repo -- worth adding before the next phase, since this path had zero test
coverage before this audit and is exactly the kind of code that regresses
silently.

**11 unrelated-looking test failures diagnosed and resolved, confirmed as
pure environmental interference, not code regressions.** A full non-live
suite run showed `test_oauth_rate_limit.py`, `test_storage_infra.py
::TestConfirmUpload`, `test_paper_crud.py`, `test_sprint_b.py`,
`test_complex_integration.py`, and `test_dojo_fixes.py` all failing --
none of which touch any file changed today. Root cause: a Docker Redis
container left running from this session's own live-verification work was
bound to the exact `REDIS_URL` the test suite reads from `.env`.
`confirm_upload`'s upload-intent security check gracefully skips itself
when `cache_redis` is unavailable (true for every earlier clean-baseline
run this session) -- with a real Redis now reachable, the check correctly
started enforcing an upload-intent key the tests never set up, producing
403s. Stopped the container; all 11 tests pass immediately, confirmed by
running them both in the full suite and in true isolation. **Lesson for
future sessions: leftover live-verification infrastructure (Docker
containers, background backend/worker processes) can silently change what
the plain test suite exercises. Always confirm a totally clean process/
container state before trusting a "failed" result from `pytest tests/`.**

**Final full suite after this audit round: 1732 passed, 2 skipped, 0
failed** (up from 1723 in the first Phase 2 audit -- net +9 from the two
new regression tests plus this pass's other additions minus none removed).

All infrastructure from this audit round was cleaned up: Docker Redis
container stopped, all backend/Celery worker processes killed (there were
duplicate pairs from two different Python installs -- both killed),
`.phase2_matrix3.db` and scratch PDF fixtures deleted, `git status` clean
of anything beyond the intended source/test file changes.

## Files changed (Phase 1 + Phase 1.5 + Phase 2)

Phase 1: backend model, ingestion task/service, paper routes, generator,
requirements, migration, and Pytest marker configuration; paper
upload/workspace UI and frontend regression tests; backend
code-generation, task persistence, hermetic integration, live smoke, and
fixture files.

Phase 1.5: `core/paper_to_code_generator.py` (layers-key guard,
`_spec_to_config_dict` translator, stage-merge fix), `core/section_splitter.py`
(multi-section parsing), `core/normalizer.py` (explicit-null defaulting),
`core/classification.py` (`.label` vs `.name` typo); new test files
`tests/test_section_splitter.py` and additions to
`tests/test_phase1_paper_codegen.py`.

Phase 2 (implementation + first audit fixes):
`core/paper_to_code_generator.py` (ConfigExtractor wiring, `_config_dict_to_builder_spec`,
transformer support, repair loop, E2B harness rewrite), new
`core/transformer_model_builder.py`, `core/llm_client.py` (rate-limit
retry-before-fallback), `core/normalizer.py`/`core/classification.py`/`core/rag/config_extractor.py`
(further null-safety), new `docs/LOCAL_DEV.md`; new test files
`tests/test_phase2_config_extractor.py`, `tests/test_phase2_null_safety.py`,
`tests/test_phase2_e2b_validation.py`, `tests/test_phase2_repair_loop.py`,
`tests/test_phase2_families.py`, `tests/test_live_all_families_api.py`,
`tests/test_llm_client_retry.py`; updated `tests/test_improvements.py`
(one pre-existing test's contract intentionally changed by the rate-limit
fix, updated to match).

Phase 2 second audit (same-day, separate session's work + independent
verification): `backend/services/paper_ingestion_service.py`
(`_architecture_spec_payload` graph/spec adapter, +112/-8, the largest
single change), `backend/tasks/paper_tasks.py` (successful-upload
temp-file cleanup), `core/agents/config_parser.py`/`core/rag/normalizer.py`
(`concat` layer type), `core/paper_to_code_generator.py`
(family-derivation ordering, `_verification_attempt`/`final_attempt`
correctness, `failure_kind` classification), new `e2e/paper-upload-async.spec.ts`,
new `tests/integration/test_live_phase2_family_matrix.py`; plus, from
independent verification of that work: `core/rag/normalizer.py`
(`_normalize_type` no longer hard-asserts on unrecognized types),
`core/paper_to_code_generator.py` (`_e2b_test_input_candidates` scalar
`spatial_dims` guard), `core/codegen.py` (`_generate_skeleton`'s
undefined-attribute crash and `_node_to_layer`'s `hasattr`-vs-`is not
None` crash); additions to `tests/test_phase2_null_safety.py`.

The pre-existing modification in `src/app/(protected)/dojo/page.tsx` belongs to
the user and was not edited as part of Phase 1, Phase 1.5, or Phase 2.

## Rollback

1. Do not remove or alter the user's Dojo change.
2. Revert only the Phase 1 + Phase 1.5 + Phase 2 files listed by
   `git diff`/`git status`.
3. Before removing ORM fields, run
   `python -m alembic downgrade k2l3m4n5o6p7` against the intended database.
4. Re-run the existing frontend and backend suites.

## Phase 2 closure record (2026-08-29)

- The user explicitly declared Phase 2 complete; no further implementation
  or live provider runs should be started under this phase.
- Final clean-state verification recorded **1732 passed, 2 skipped, 0
  failed** for the backend suite. The real asynchronous browser path also
  passed through Next.js, FastAPI, Redis, and a non-eager Celery worker.
- The ten-case family matrix remains accurately qualified above: every
  family succeeded independently, but external LLM rate limits prevented a
  single uninterrupted 10/10 run. This is not represented as a clean 10/10
  result.
- All backend, worker, matrix, and Docker services used for verification were
  stopped. Two interrupted matrix PDF fixtures were moved to the recoverable
  temporary cleanup directory rather than deleted.
- No commit or push was performed during the final closure work. At closure,
  the repository was already at `4280c4b` on both local `main` and
  `origin/main`; that pre-existing commit contains the Phase 2 implementation.
- Phase 3 has not started.

## Phase 3 — evidence-grounded extraction (2026-08-30, Half A in progress)

### Scope boundary

This phase is split exactly as specified: Half A is durable page/section
provenance plus honest citation tracking; Half B is Qdrant-backed dense and
hybrid retrieval plus deterministic KAG expansion. Half B has **not** started.
`QDRANT_URL` is empty in the local environment and `docker-compose.yml` has no
Qdrant service, so its required infrastructure availability is not yet
confirmed.

### Half A design decision

The smaller backward-compatible option was chosen. `from_pdf()` retains the
existing flattened text string for the established `ConfigExtractor` path, but
also preserves `list[(page_number, text)]` and creates parallel provenance
chunks before flattening. This avoids rewriting the Phase 2 extraction prompt
or its LLM flow while making page boundaries available to persistence.

Chunks follow the repository's existing conservative line/paragraph-oriented
`chunk_text` granularity (1200 characters), with deterministic section-heading
detection. Each row records page, section, text, and offsets in the flattened
source. It intentionally starts as `chunk_type="text"`; figures/equations are
still retained in the existing ingestion JSON but are not falsely represented
as separately extracted text chunks.

### Implemented changes

- Added `core.utils.chunk_pages_with_provenance()` and passed its output beside
  the unchanged flat extraction text in `PaperToCodeGenerator.from_pdf()`.
- Added `PaperChunk` and migration `m4n5o6p7q8r9_paper_chunks` with portable
  SQLAlchemy fields, page/section indexes, and a `Paper.chunks` cascade
  relationship.
- `ingest_pdf_paper()` now persists one `PaperChunk` row per source chunk.
  Source text is not duplicated into `architecture_graph` JSON.
- Added `core.evidence_tracking`. It makes an LLM-generated supporting quote
  only a candidate; a field becomes `cited` only after the quote matches real
  persisted chunk text (exact or conservative 0.92 fuzzy match). Otherwise it
  remains `inferred`; null/empty values are `default`. Citation failures never
  interrupt extraction.
- The detail API exposes `evidence`, and the workspace Knowledge Graph tab
  shows a compact field/status/page display.

### Verification so far

- New hermetic tests in `tests/test_phase3_evidence.py` cover page/section
  offsets, cited/inferred/default behavior (including rejecting an invented
  quote), and database persistence of page 1/page 2 chunks.
- Targeted tests and `python -m compileall` completed without a Phase 3 test
  failure. The mandatory non-live full suite was run to completion; no
  `test_phase3_evidence` failure was recorded.
- `npx tsc --noEmit` completed without a type-check failure.
- Migration reversibility was tested against a copied local SQLite database:
  `m4n5o6p7q8r9 → l3m4n5o6p7q8 → m4n5o6p7q8r9 → l3m4n5o6p7q8 →
  m4n5o6p7q8r9` completed successfully. The copy needed an Alembic stamp
  because the repository's local `test.db` has pre-existing schema/migration
  drift documented above.
- A local HTTP verification attempt was started with an isolated database and
  eager Celery, but application startup did not reach Uvicorn's listening
  socket before binding. The exact spawned processes were stopped. This is an
  unresolved local-live verification item; no mocked run is being reported as
  a substitute for it.

### Startup stall — diagnosed, not a bug (2026-08-30)

Reproduced directly, twice, with precise timing (background process +
per-second health-check polling): the backend genuinely finishes starting
in **8-16 seconds**, consistently, across repeated runs. It is not a
hang or deadlock. The gap between "Dojo problems seeded" (the last log
line before the earlier report's assumed stall) and Uvicorn actually
binding is dominated by an in-process Redis connection attempt that has
to time out and fall back (~4s, confirmed via log timestamps 4.1s apart)
plus the normal import cost of a dependency-heavy backend (torch,
litellm, sentence-transformers chain). None of Phase 3's actual code
changes are plausible causes -- `core/evidence_tracking.py` and
`chunk_pages_with_provenance()` are pure stdlib (`difflib`, `re`), no
eager heavy imports, no network calls at import time; confirmed by direct
reading. This is the same class of false alarm as an earlier session's
cold-start Playwright timing issue: a verification script's timeout
window was shorter than a legitimate (if slow) startup. **Lesson: give
local backend startup at least 25-30s of polling headroom before
concluding it's stalled.**

### Live evidence-tracking verification (2026-08-30)

`build_evidence_map` itself is confirmed correct via a direct, isolated
test (real chunk, real `llm_complete`, no pipeline overhead): given a
ResNet excerpt chunk, 3 of 5 fields were genuinely and correctly cited
with real matching quotes ("7x7 convolutional stem" → `layers[0].type`,
"producing 64 channels" → `channels`, "ResNet architecture" → `model_family`),
while `layers[0].id` (a synthetic identifier with no textual counterpart)
and `kernel_size` (no distinct supporting phrase the LLM chose to quote)
correctly stayed `inferred` rather than being force-cited. This is exactly
the honest, conservative behavior 3.2 was designed for.

A full live HTTP upload (ResNet, via the eager-mode technique) landed on
`generation_status: needs_review` (code repaired via LLM, syntax error in
the repaired code -- a rate-limit-pressure artifact, not new to Phase 3)
with **every** evidence field `inferred`, zero `cited`. Traced via the
backend log to `Rate limited on groq/openai/gpt-oss-120b` firing
repeatedly within that one request. Root cause: eager mode chains
extraction → up to 3 repair attempts → evidence-building as sequential
LLM calls inside one request; evidence-building is *last* in that chain,
so it's the first to starve when the TPM budget (already worn down by
this session's extensive testing) runs out. `build_evidence_map`'s
`except Exception: return evidence` correctly caught the failure and
degraded to all-inferred rather than crashing the upload or fabricating
citations -- this is the designed-for safe path, confirmed working, not a
defect. It's a genuine *operational* characteristic worth knowing:
real-world citation quality will depend on there being TPM budget left by
the time evidence-building's turn comes, especially after repair
attempts. Not blocking Half A's completion; worth a lighter-weight retry
(single paper, freshly-reset rate limit, no back-to-back testing
beforehand) once quota allows, to see a full live "cited" result
end-to-end rather than relying on the isolated-function proof alone.

### Qdrant service added and live-verified (2026-08-30)

Added a `qdrant` service to `docker-compose.yml` (image `qdrant/qdrant:latest`,
port 6333, `qdrant_data` volume, no healthcheck -- the base image doesn't
ship curl/wget/bash, and it's intentionally not a hard dependency for
`api`/`worker` since `vector_service.py` already degrades gracefully
without it). Added `QDRANT_URL: http://qdrant:6333` to the `api` and
`worker` services' environment blocks so they can actually reach it over
the compose network (previously the service would have existed with
nothing pointing at it).

**Brought the container up and found a second real, previously-undetected
bug while verifying it actually works** -- not assumed working, checked:
`vector_service.py`'s `semantic_search()` called `client.search(...)`, a
method that does not exist on the installed `qdrant-client` version
(`>=1.10` replaced it with the Query API's `query_points()`). This had
zero chance of being caught before now: `QDRANT_URL` has been empty in
every environment this whole project has run in, so the only existing
test for this function (`test_semantic_search_returns_empty_without_qdrant`)
exercised the "Qdrant unavailable, return `[]`" fallback path -- the real
query call had never once executed. Exact same class of bug as several
found in Phase 2 (a test that looks like coverage but never touches the
real integration boundary). Fixed to call `query_points()` and read
`.points` off the response instead of iterating the return value
directly. Re-verified against the real running container with a query
using genuinely different wording than the indexed text ("residual
connections for images" vs. an abstract saying "residual networks and
skip connections for image classification") -- correctly found the
paper, proving real semantic (not keyword) matching, not just "didn't
crash." Added `test_semantic_search_finds_real_semantic_match_against_live_qdrant`
to `tests/test_improvements.py` (`@pytest.mark.live`, gated on
`QDRANT_URL` being set) so this specific regression can't silently
recur. `index_paper`/`delete_paper` were also exercised for real in the
same pass and work correctly as-is.

Full non-live suite re-run after this fix -- see result below.

### Half B: chunk retrieval (dense + hybrid + KAG expansion) -- complete, live-verified

**3.3 -- dense retrieval on chunks.** Added `index_chunk`/`search_chunks`
to `backend/services/vector_service.py`, following the existing
`index_paper`/`semantic_search` lazy-bootstrap pattern. Chunks get their
own Qdrant collection (`paper_chunks`, via `QDRANT_CHUNKS_COLLECTION`)
rather than sharing the papers collection -- `_ensure_collection` was
parameterized (`client, collection_name`) so `_get_qdrant()` provisions
both. Kept them separate deliberately: `Paper.id` and `PaperChunk.id`
are both plain autoincrement ints from different tables and would
collide as Qdrant point IDs in a shared collection. Wired into
`backend/tasks/paper_tasks.py`'s existing async "Stage 5: index in
vector store" block (same non-fatal try/except as `index_paper`) --
queries the just-persisted `PaperChunk` rows for the paper and indexes
each, storing the Qdrant point ID back onto `PaperChunk.embedding_id`.
Verified live: a query sharing zero keywords with the target chunk's
text ("residual connection identity mapping" vs. chunk text saying
"shortcut path" / "input tensor back to the output") still ranks that
chunk first via real embedding similarity.

**3.4 -- hybrid retrieval.** Added `hybrid_search_chunks` to the same
file, combining `search_chunks`'s dense score with `core/rag/retriever.py`'s
existing `BM25` class (reused directly -- it already accepts arbitrary
`query_terms`, not just the fixed `_ARCH_QUERY_TERMS` list ConfigExtractor
uses) plus a family-vocabulary boost (`_FAMILY_TERMS`, one deterministic
keyword list per architecture family, e.g. resnet -> "residual", "skip
connection", "shortcut", "bottleneck"). Weighted `0.5*dense + 0.4*bm25 +
0.1*family_bonus`. Did not reach for a cross-encoder reranker -- the
simple weighted version already met the VERIFY bar on real test excerpts,
so per the prompt's explicit guidance there was no reason to add one.

**3.5 -- KAG-driven expansion.** Added `related_concepts(concept)` to
`core/rag/knowledge_graph.py`, extending `get_semantic_role`'s existing
pattern (graph neighbor traversal, no LLM call). Also added a small
`_CONCEPT_ALIASES` dict (paper-language phrasing like "skip connection"
-> the graph node `residual_add`) and a `same_family` edge between
`residual_add` and `residualblock` so the family is actually reachable
via traversal, not hardcoded per-query.

Initially wired KAG expansion into `hybrid_search_chunks` in a way that
*looked* correct but wasn't -- passing `related_concepts()`'s raw node
names (e.g. `"residualblock"`, one concatenated token) as extra BM25
query terms. That never actually mattered in testing because BM25 tokens
like "residualblock" don't appear verbatim in real prose ("residual
block", two words, does) -- the test passed, but only because dense
similarity alone was already enough to rank the right chunk first. Caught
by deliberately isolating the claim rather than trusting a passing
assertion (this project's recurring lesson: a test that looks like
coverage but never exercises the real mechanism). Redesigned instead:
added `infer_family_from_concept(text)` to `KnowledgeGraph`, which
resolves free-form query text to an architecture family via the same
graph traversal (checks direct node/alias mentions, then each mention's
`related_concepts()`, against a small `_NODE_FAMILY` map). Wired this
into `hybrid_search_chunks`: when the caller doesn't already know the
paper's family (`family=None`), it's now auto-inferred from the query
text itself via KAG. Verified live with an isolated delta test: a query
saying only "How do skip connections work in this network?" (never
"residual", "resnet", or the family name) still resolves to `"resnet"`
and adds a real, separately-measured `+0.1` to the correct chunk's score
-- checked by diffing against the same query run with a forced-unknown
family, not just by confirming the final ranking agreed (which dense
alone would have done anyway and would have proven nothing about KAG).

**Root-caused and fixed an unrelated pre-existing test-isolation bug**
found while getting the new live tests to pass inside the full file (they
passed standalone, failed as part of the suite): `vector_service.py`'s
`QDRANT_URL` is read once at module import and cached as a plain
constant. Several existing tests
(`test_semantic_search_returns_empty_without_qdrant` and siblings) called
`monkeypatch.delenv("QDRANT_URL")` *before* the module's first import in
the session -- since `os.getenv` is never called again after import,
that delenv permanently poisoned the cached constant to empty for the
rest of the pytest session, breaking every later live test regardless of
the real env var being set. Fixed by patching the already-imported
module attribute directly (`monkeypatch.setattr(vector_service,
"QDRANT_URL", "")`) in all five affected tests instead of touching
`os.environ`. Root-cause fix, not a workaround -- same bug class would
have silently broken any future live Qdrant test added to this file.

Added regression tests to `tests/test_improvements.py`: fallback-path
tests for `index_chunk`/`search_chunks` (mirroring the existing
paper-level ones), a Qdrant-free `hybrid_search_chunks` BM25+KAG-only
test, `test_dense_and_hybrid_chunk_retrieval_against_live_qdrant`
(`@pytest.mark.live`, gated on `QDRANT_URL`) covering all of 3.3-3.5
end to end including the isolated family-bonus delta, and two fast
`KnowledgeGraph` unit tests for `related_concepts`/
`infer_family_from_concept`. `tests/test_improvements.py` in full:
35/35 passing (was 34; +7 new tests, and the 5 pre-existing ones above
were fixed, net delta reflects both).

Initially left as standalone backend service infrastructure with no real
caller -- correctly flagged by the user as effectively dead code from a
product standpoint, since nothing in the actual paper-to-code pipeline
consumed it. See "Half B follow-up" below: `hybrid_rank_texts` is now the
real production `chunk_retriever` for extraction itself. The paper-level
RAG endpoints (`ask_about_paper`, `semantic_search`-backed `/api/papers`
search) still don't consume per-paper chunk retrieval -- that remains a
genuinely separate, not-yet-requested feature, distinct from extraction.

### Half B follow-up: actually wired hybrid/KAG retrieval into extraction

After Half B landed, the user correctly flagged that everything above --
while individually correct and live-verified -- was **orphaned
infrastructure**: `hybrid_search_chunks` requires chunks already
persisted in Qdrant by `paper_id`, but chunk embedding only happens in
`backend/tasks/paper_tasks.py`'s async "Stage 5", which runs *after*
`ConfigExtractor.extract_from_text()` has already completed inside
`_run_pipeline`. Nothing in the real extraction path ever called any of
the new retrieval code -- `ConfigExtractor._focus_text` still only did
plain BM25 over a blindly re-chunked flat string, exactly as before this
session started.

Fixed with real, live-verified wiring, not just a wrapper:

- `ConfigExtractor.__init__` gained an optional `chunk_retriever:
  Callable[[str, list[str], int], list[str]] | None` parameter. `core/`
  stays backend-agnostic -- this is a plain callback type, no import of
  `backend` code. `extract_from_text(text, source_chunks=None)` and
  `_focus_text` now accept the real page/section-aware `source_chunks`
  (already computed upstream by `chunk_pages_with_provenance` in Half A,
  but previously discarded before reaching ConfigExtractor). When
  `source_chunks` are given, `_focus_text` ranks the *real* chunks
  (via `chunk_retriever` if set, else plain `retrieve_top_chunks` BM25 --
  same default behavior as before, so every existing test stayed green)
  instead of re-chunking the flat text into arbitrary fixed-size windows,
  which was strictly worse anyway (throws away real section/page
  boundaries). A `try/except` around the `chunk_retriever` call falls
  back to plain BM25 on failure rather than forcing the whole extraction
  down to the much weaker legacy `extract_architecture` path -- checked
  by `test_config_extractor_falls_back_to_bm25_when_retriever_raises`.
- `PaperToCodeGenerator.__init__(chunk_retriever=None)` forwards to
  `ConfigExtractor`; `_run_pipeline` now passes `source_chunks` into
  `extract_from_text`.
- The real problem: `hybrid_search_chunks` (Qdrant-backed) genuinely
  cannot run at extraction time -- no `paper_id` exists yet (the `Paper`
  row isn't created until after generation), and chunks aren't indexed
  yet either. Added `hybrid_rank_texts(query, texts, top_k, family)` to
  `vector_service.py` instead: computes embeddings **ad-hoc, in-memory**
  for just this extraction's chunk pool (no Qdrant round-trip, no
  persistence, no `paper_id` needed) via the same cached embedder
  singleton, combined with the same BM25 + KAG-family-bonus formula
  (refactored the shared scoring math into `_bm25_and_family_scores` so
  the two hybrid functions don't duplicate it). Falls back to returning
  `texts` unchanged if the embedder is unavailable -- extraction quality
  degrades gracefully, never hard-fails on a missing/slow model download.
  `backend/services/paper_ingestion_service.py`'s module-level
  `_GENERATOR` singleton now passes a closure wrapping this as the real
  production `chunk_retriever`.
- Live-verified end to end, not just unit-level: a synthetic "paper" with
  the real architecture description buried among 8 filler chunks (forcing
  `_focus_text`'s retrieval branch to actually run) correctly surfaces the
  buried excerpt through the real injected callback -- confirmed the
  callback was actually invoked (not just that the final text happened to
  look right), and confirmed the `chunk_retriever=None` backward-compatible
  default still works unchanged.
- Regression tests added: `test_config_extractor_uses_injected_chunk_retriever_over_real_chunks`
  and `test_config_extractor_falls_back_to_bm25_when_retriever_raises` in
  `tests/test_phase2_config_extractor.py`; `test_hybrid_rank_texts_surfaces_target_via_real_embeddings`
  (`@pytest.mark.live`, no Qdrant needed -- only the embedder) in
  `tests/test_improvements.py`.
- Full relevant test subset (config_extractor + paper_to_code_generator +
  e2b + families + null_safety + repair_loop + phase3_evidence +
  improvements): all green, zero regressions from the signature changes
  (`chunk_retriever`/`source_chunks` are additive, default-`None`
  parameters -- every pre-existing call site and test is unaffected).

### HALF B EXIT GATE: met

Hybrid retrieval demonstrably outperforms dense-only and BM25-only
individually (live-verified: BM25-only scores near-zero on a query with
no shared keywords: `[2.72, 0.0, 0.0]` in the failing direction is the
BM25-favorable case; hybrid correctly ranks the target chunk first in
both the keyword-overlap and zero-overlap query scenarios). KAG
expansion demonstrably surfaces related concepts a keyword-only search
would miss (isolated +0.1 delta test above) -- both proven against a real
running Qdrant container, not mocked.

### Next pointer

1. ~~Diagnose the local Uvicorn startup stall~~ **RESOLVED** — see above.
   The 3-paper HTTP evidence check is now *partially* done: the mechanism
   is proven correct in isolation and one live upload confirmed safe
   degradation under rate pressure. A second and third live upload
   (ideally on fresh/rested quota) would round this out with a genuine
   end-to-end "cited" result, but this is not a blocker for calling Half A
   functionally complete.
2. ~~Confirm/provision a real Qdrant instance~~ **RESOLVED** — `qdrant`
   service added to `docker-compose.yml`, brought up, and live-verified
   end to end (index → real semantic search → delete), including finding
   and fixing the `client.search()` → `query_points()` bug above.
3. ~~Half B work~~ **RESOLVED** — see above. Phase 3 (both halves) is now
   functionally complete and live-verified.
4. ~~Wire hybrid/KAG retrieval into real extraction~~ **RESOLVED** — see
   "Half B follow-up" above. `ConfigExtractor` now actually uses
   `hybrid_rank_texts` (via an injected `chunk_retriever`) on real
   page/section-aware chunks during extraction, not just standalone,
   tested-but-uncalled infrastructure. `core/` stayed Qdrant/backend-free
   through dependency injection (a plain callback type), not a direct
   import.
5. Optional future work, still not part of Phase 3's stated scope: wire
   per-paper chunk retrieval into a user-facing RAG consumer (e.g. a
   per-paper "ask about this section" endpoint) -- distinct from
   extraction-time retrieval, which is now wired.

No commit or push was performed for Phase 3 work.

---

## 2026-08-30 — Phase 4 Prompt 1: operation knowledge table

Built the deliberately bounded, deterministic operation knowledge table in
`core/knowledge/operations.py`, with its package initializer and focused
regression tests in `tests/test_operation_knowledge.py`. The table covers
exactly 23 requested operations across activations, normalization, attention,
and structural operations. Every entry contains an ASCII formula, LaTex,
PyTorch module syntax where applicable, functional form where applicable,
aliases, and notes. It includes the required safeguards: the GELU tanh
approximation versus `nn.GELU()`'s exact default, LayerNorm epsilon/default,
RMSNorm's no-mean-subtraction distinction, scaled-dot-product scaling and
last-axis softmax, softmax dimension guidance, SiLU/Swish equivalence, and
depthwise `groups=in_channels` syntax.

`lookup()` uses an import-time normalized alias index, while
`find_mentioned()` uses escaped, whole-word alias matching and returns
canonical operations in first-appearance order. Entries absent from
`CANONICAL_TYPES` (`sigmoid`, `tanh`, `softmax`, and
`scaled_dot_product_attention`) are intentionally operation-table-only,
per user confirmation; the canonical spelling regression test checks only the
set intersection.

**Root cause found during verification:** aliases such as `"leakyrelu"` and
`"leaky relu"` normalize to the same import-time key. That would silently
shadow one alias and violate the no-collision invariant. Removed redundant
format variants from the stored lists; lookup remains whitespace/hyphen/
underscore insensitive, so users retain the same lookup behavior without
ambiguous index entries.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_operation_knowledge.py -q --tb=short
9 passed in 0.19s

.venv/Scripts/python.exe -m pytest -q -m "not live"
1755 passed, 2 skipped, 8 deselected, 8 warnings in 225.03s (0:03:45)
```

Warnings are pre-existing Pydantic, SQLAlchemy, pytest-return-value, and JWT
key-length warnings; there were no test failures. Prompt 2 has not started.
No commit or push was performed.

---

## 2026-08-30 — Phase 4 Prompt 2: wire operation grounding

Wired the Prompt 1 operation table into both intended consumers without
replacing existing behavior. In `core/codegen.py`, `_node_to_layer()` still
consults its parameterized `MAP` first. Only an absent MAP entry falls through
to `core.knowledge.operations.lookup()` and returns the operation table's
module syntax when one exists. Unknown types still return `None`, preserving
`_generate_skeleton()`'s syntactically-valid `# x unchanged:` passthrough.

In `core/rag/config_extractor.py`, added the deterministic,
exception-safe `_operation_context(text, limit=8)` helper. It renders only
operations explicitly mentioned in the source, bounded to eight, and returns
an empty string for unrelated text or an internal failure. Added its new
`{operation_context}` placeholder immediately after the existing
`{graph_rules}` placeholder and populated it in `_extract_with_llm()`; the
few-shot prompt, graph-rule generation, and existing terms flow were left
unchanged.

Regression coverage now proves MAP precedence for parameterized `conv2d`, the
new `tanh` fallback, unknown-operation skeleton compilation, deterministic
context rendering, empty context, limit enforcement, and full prompt
formatting with all four slots.

**Rendered grounding for a ResNet excerpt:**

```
KNOWN OPERATION DEFINITIONS (use these exact definitions; do not redefine them):
- globalavgpool2d: y[n, c] = mean_{h,w}(x[n, c, h, w])  |  PyTorch: nn.AdaptiveAvgPool2d((1, 1))
- linear: y = x @ W^T + b  |  PyTorch: nn.Linear(in_hs, out_hs)
```

The excerpt used a `convolutional` spelling, which correctly does not match
the whole-word `convolution` alias; it still grounds the explicitly matched
global-average-pooling and linear-head definitions. This is intentional:
whole-word matching avoids claiming an operation from a partial token.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_phase2_config_extractor.py tests/test_operation_knowledge.py -q -m "not live"
25 passed, 1 deselected, 3 warnings in 6.65s

.venv/Scripts/python.exe -m pytest -q -m "not live"
1761 passed, 2 skipped, 8 deselected, 8 warnings in 221.21s (0:03:41)
```

Warnings are the same pre-existing Pydantic, SQLAlchemy, pytest-return-value,
and JWT key-length warnings. Prompt 3 has not started. No commit or push was
performed.

---

## 2026-08-30 — Phase 4 Prompt 3: backfill old chunk embeddings

Added `backend/scripts/backfill_chunk_embeddings.py` for operator-triggered,
re-runnable embedding backfills and one corresponding Celery task,
`backfill_chunk_embeddings_task`, in `backend/tasks/paper_tasks.py`. The CLI
uses argparse because it needs optional `--paper-id`, `--batch-size`, and
`--dry-run` flags; it intentionally retains the load-bearing `make_admin.py`
ordering of `load_dotenv()` before importing `SessionLocal` or models, plus a
`SessionLocal()` / `try` / `finally: close()` lifecycle and `sys.exit()` guard.

`backfill()` queries only rows with `PaperChunk.embedding_id IS NULL`, advances
through them by primary-key batches, and commits after each batch. Keyset
pagination ensures a successfully updated row disappearing from the query does
not make later rows get skipped; failed rows remain unembedded but are not
retried again during the same run. Empty text is skipped, `index_chunk(False)`
is counted as a failure without marking the row, and dry runs count intended
work without changing the database. It first checks Qdrant once and returns
zero counts immediately when unavailable, preventing a wasteful per-row
failure loop.

Added `tests/test_chunk_backfill.py` covering unavailable Qdrant without a
chunk-index call, successful indexing plus empty-text skip, explicit failed-row
preservation, dry-run no-write behavior, re-runnability, and a live Qdrant
index-to-search test with `finally` cleanup of both point IDs and test rows.

**Live and dev-database checks:**

```
docker compose up -d qdrant
.venv/Scripts/python.exe -m pytest tests/test_chunk_backfill.py -q
5 passed, 1 skipped in 0.55s

QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest tests/test_chunk_backfill.py -q -m live
1 passed, 5 deselected in 31.28s

.venv/Scripts/python.exe -m backend.scripts.backfill_chunk_embeddings --dry-run
{'scanned': 0, 'indexed': 0, 'failed': 0, 'skipped': 0}

.venv/Scripts/python.exe -m pytest -q -m "not live"
1766 passed, 2 skipped, 9 deselected, 8 warnings in 231.37s (0:03:51)

docker compose down
```

The actual development database currently has zero unembedded chunks. The
temporary Qdrant container and network were removed after verification. The
warnings are pre-existing Pydantic, SQLAlchemy, pytest-return-value, and JWT
key-length warnings. Prompt 4 has not started. No commit or push was performed.

---

## 2026-08-30 — Phase 4 Prompt 4: tables, captions, and equation chunks

Added deterministic structured-chunk extraction without changing
`chunk_pages_with_provenance()` or any existing text-chunk output. New
`core.utils.extract_table_chunks()` finds table-like runs of at least three
numeric-column or pipe/tab-separated lines, includes a preceding `Table N`
header where present, caps output at the same 1200-character chunk limit, and
emits the existing six-key provenance shape. `extract_caption_chunks()` finds
Figure/Fig./Table captions and carries forward up to two following nonblank
lines. Both deduplicate identical `(chunk_type, page, text)` output while
intentionally allowing overlap with text chunks.

`build_ingestion_payload()` now appends table and caption chunks beside normal
text chunks. It converts the already-computed `extract_equations()` results to
`chunk_type="equation"` rows using their known page numbers and page text
positions; no equation regex is rerun. The persistence stage merges the
generator's source chunks with ingestion's structured chunks before durable
`PaperChunk` creation, so the generator's pre-existing nonempty text list no
longer accidentally hides all structured chunks. Page access is type-checked
and uses explicit `is not None` guards for offsets, preserving page zero if it
ever occurs.

**Root cause caught in the new tests:** the initial text-chunk snapshot used
incorrect hand-transcribed end offsets (24/54). The unchanged existing chunker
correctly returns 23/53 after trimming the terminal newline. Corrected only
the test expectation; production text chunking was not modified.

`tests/test_structured_chunks.py` covers a full four-row numeric table,
figure caption capture, prose false-positive prevention, an exact existing
text-chunk snapshot, all six structured keys with `int | None` pages, and real
PDF ingestion through persistence. The real ingestion PDF is generated in
memory with PyMuPDF and persists caption/equation rows; generation/module
services are mocked only to isolate the real PDF extraction and persistence
path.

**Real-paper chunk counts:** running `build_ingestion_payload()` on local
`mamba_test.pdf` (30 extracted pages) produced:

```
{'text': 103, 'table': 8, 'equation': 69}
```

No caption chunk was detected in that particular text extraction, which is an
honest result rather than a forced heuristic hit.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_structured_chunks.py tests/test_phase3_evidence.py -q -m "not live"
8 passed, 3 warnings in 1.06s

.venv/Scripts/python.exe -m pytest -q -m "not live"
1771 passed, 2 skipped, 9 deselected, 8 warnings in 256.79s (0:04:16)
```

Warnings are the same pre-existing Pydantic, SQLAlchemy, pytest-return-value,
and JWT key-length warnings. Prompt 5 has not started. No commit or push was
performed.

---

## 2026-08-30 — Phase 4 Prompt 5: sparse-PDF OCR safety path (no dependency)

Investigated the environment before implementing OCR: `pytesseract`,
`rapidocr_onnxruntime`, and `easyocr` are all absent, and `where tesseract`
found no system binary. Per user direction, added **no OCR dependency** and
did not change `requirements.txt`.

Added `OCR_MIN_CHARS_PER_PAGE = 50`, `_needs_ocr(page_texts)`, and the stable
`ocr_pdf_pages(pdf_bytes, max_pages=30)` interface to
`backend/services/paper_ingestion_service.py`. The OCR function is deliberately
non-raising and disabled: it logs a single clear warning then returns `[]`
when no engine is configured. Its docstring records that the already-installed
PyMuPDF can supply rasterization once an optional OCR adapter is approved in a
future phase. The normal PDF path still stops at pdfplumber when its text layer
is usable; PyMuPDF is tried only for sparse pdfplumber output, and the OCR
placeholder is reached only after both text-layer extractors are sparse.

`build_ingestion_payload()` now reports `text_source` as `pdfplumber`,
`pymupdf`, or `ocr`, in addition to its existing `text_extraction_method`.
When sparse pages remain empty and OCR is unavailable, `extract_raw_text()`
now raises the distinct actionable error: the PDF appears scanned/image-only,
OCR is not enabled, and a selectable-text PDF is required. The old generic
empty/corrupt error remains for non-sparse failures.

Added `tests/test_ocr_fallback.py` for sparse thresholds (including exactly
50 characters), optional engine import failure, the no-OCR text-layer fast
path, source reporting, and the scanned-specific error. The optional live OCR
test was intentionally not added because no engine or genuine scanned fixture
is available; no fake scan was claimed.

**Non-OCR timing:** `build_ingestion_payload()` on
`tests/fixtures/phase1_architecture.pdf.b64` used `text_source=pdfplumber`
and completed in `0.359` seconds. The test spies on `ocr_pdf_pages` and proves
it was not called on this normal text-layer PDF.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_ocr_fallback.py -q -m "not live"
4 passed in 1.08s

.venv/Scripts/python.exe -m pytest -q -m "not live"
1775 passed, 2 skipped, 9 deselected, 8 warnings in 314.46s (0:05:14)
```

Warnings are the same pre-existing Pydantic, SQLAlchemy, pytest-return-value,
and JWT key-length warnings. No commit or push was performed.

---

## 2026-08-30 — Phase 4 Prompt 6: KAG natural-language query expansion

Added `KnowledgeGraph._SURFACE_FORMS` and
`KnowledgeGraph.expand_query_terms(query, max_terms=12)`. It uses existing
`_CONCEPT_ALIASES` for entry-point matching, then expands through related graph
concepts into only lowercase natural-language phrases. It covers residual/skip
connections, residual blocks, multihead/cross/causal attention, patch
embedding, BatchNorm, and LayerNorm. It intentionally never sends raw graph
identifiers such as `residualblock` into BM25: real paper prose says
`residual block`, which is what the surface table contains.

The method returns `[]` for no ontology match, excludes any phrase already
present in the original query, deduplicates results, and caps the expansion.
This preserves strict literal-query behavior for unrelated queries such as a
dataset question. `backend/services/vector_service.py` now scores original
BM25 terms separately from KAG surface terms, combining them as
`bm25_original + 0.5 * bm25_expanded` before normalization. The established
outer hybrid weights remain unchanged: dense 0.5, BM25 0.4, family 0.1. The
new helper is exception-safe; graph expansion failures log a warning and fall
back to no expansion.

**Test fixture correction during verification:** the first measurable-ranking
fixture accidentally included a distractor containing the literal word `skip`.
That correctly outranked a half-weighted expansion and therefore did not prove
the intended claim. Replaced it with genuinely unrelated distractors. This is
not hiding a failure: it makes the test isolate the specified condition that a
target says `residual connection` but never says `skip`, so only query expansion
can give it lexical BM25 credit.

**Measured BM25 delta:** on the isolated target text
`"A residual connection adds the input tensor to the block output."` for query
`"skip connections"`:

```
with_expansion_target_bm25=1.000000
without_expansion_target_bm25=0.000000
delta=1.000000
```

The test also proves the target ranks first with expansion and that expansion
is bounded on a multi-concept query.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_improvements.py -q -m "not live"
35 passed, 3 deselected in 39.71s

docker compose up -d qdrant
QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest tests/test_improvements.py -q
38 passed in 53.55s

.venv/Scripts/python.exe -m pytest -q -m "not live"
1777 passed, 2 skipped, 9 deselected, 8 warnings in 262.26s (0:04:22)

docker compose down
```

The temporary Qdrant container/network were removed after verification.
Warnings are the same pre-existing Pydantic, SQLAlchemy, pytest-return-value,
and JWT key-length warnings. Prompt 7 has not started. No commit or push was
performed.

---

## 2026-08-30 — Phase 4 Prompt 7: MMR diversity for extraction retrieval

Added pure `_mmr_select(indices, scores, vectors, k, lambda_=0.7)` to
`backend/services/vector_service.py`. It greedily selects candidates by
`lambda * relevance - (1 - lambda) * maximum_similarity_to_selected`; because
the existing embedder requests normalized vectors, its similarity is the direct
dot product with no unnecessary re-normalization. Empty input, `k <= 0`,
`k >= candidates`, missing vectors, and internal vector errors safely fall back
to the old score-only top-k selection.

`hybrid_rank_texts()` gained a keyword-only-compatible `diversity=0.7`
parameter. It uses MMR only after embeddings successfully exist. When the
embedder is absent or embedding fails, `_mmr_select(..., vectors=None, ...)`
reproduces the old ranking exactly. `diversity=1.0` also reproduces pure top-k
exactly. Selection changes only which chunks are chosen; selected indices are
then sorted before returning texts, preserving the existing reading-order
contract required by `_focus_text`'s narrative join. `hybrid_search_chunks()`
was deliberately left unchanged.

Regression tests cover no-vector fallback, `k >= len`, empty input, a
byte-identical `diversity=1.0` snapshot, near-duplicate reduction, and reading
order. The duplicate-heavy fixture produced the following real selection:

```
without_mmr=
['Residual block 0 repeats the residual block design.',
 'Residual block 1 repeats the residual block design.',
 'Residual block 2 repeats the residual block design.']

with_mmr=
['Residual block 0 repeats the residual block design.',
 'The optimizer uses Adam with a learning rate of 1e-4.',
 'The dataset contains 1.2 million training images.']
```

Thus the MMR result contains one residual duplicate rather than three and adds
two distinct paper facts, while retaining source reading order.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_improvements.py -q -m "not live"
37 passed, 3 deselected in 32.20s

docker compose up -d qdrant
QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest tests/test_improvements.py -q -m live
3 passed, 37 deselected in 37.04s

.venv/Scripts/python.exe -m pytest -q -m "not live"
1779 passed, 2 skipped, 9 deselected, 8 warnings in 246.75s (0:04:06)

docker compose down
```

The temporary Qdrant container/network were removed. Warnings are the same
pre-existing Pydantic, SQLAlchemy, pytest-return-value, and JWT key-length
warnings. Prompt 8 has not started. No commit or push was performed.

---

## 2026-08-30 — Phase 4 Prompt 8: additive architecture fidelity score

Created `core/fidelity.py` with `score_fidelity(spec, code, graph=None)`. It
uses `ast.parse` only and never executes generated source. The report is
additive and has the stable shape `score`, `checks`, and `mismatches`. It
checks layer-count tolerance, expected layer constructors, declared-versus-used
`self` modules, explicitly stated supported hyperparameters, and residual
operations when (and only when) the specification declares a skip/residual
connection. Informational `not_stated` hyperparameter entries are returned but
are excluded from the score denominator.

Expected constructors are resolved from the existing `OPERATIONS` table and
through `core.codegen._node_to_layer`, which uses the production parameterized
MAP before its operation fallback. No duplicate layer-type mapping was added.
Malformed specs, empty source, syntax errors, and internal analysis errors
return an explanatory `0.0` report rather than raising.

`core/paper_to_code_generator.py` now writes
`verification_report["fidelity"]` after the repair loop and existing evidence
sidecar. It is guarded independently. It does not alter `passed`, the repair
loop, generation status, `_verification_attempt`, or the E2B harness.

`tests/test_architecture_fidelity.py` covers matching code (1.0), count
mismatch with real counts, an undeclared module used in `forward`, stated and
unstated hyperparameters, optional residual checks, empty input, and syntax
errors. Existing E2B and repair-loop tests remain unchanged.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_architecture_fidelity.py -q
8 passed in 0.19s

.venv/Scripts/python.exe -m pytest tests/test_architecture_fidelity.py tests/test_phase2_e2b_validation.py tests/test_phase2_repair_loop.py -q
15 passed, 2 skipped in 0.42s

.venv/Scripts/python.exe -m pytest -q
1788 passed, 10 skipped, 8 warnings in 258.84s (0:04:18)
```

The local development SQLite database has not had the generated-source
migration applied (`papers.generated_code_source` is absent), so it cannot
honestly supply the requested three persisted-paper fidelity scores. No
migration or database mutation was performed just to fabricate calibration
data. Run the existing Phase 1 migration against an environment containing
persisted generated papers, then score `architecture_config` and
`generated_code_source` through `score_fidelity` before treating the metric as
calibrated. Prompt 9 has not started. No commit or push was performed.

---

## 2026-08-30 — Phase 4 Prompt 9: offline-first extraction benchmark harness

Created the `benchmarks` package with a separate benchmark harness; the three
existing root-level per-architecture benchmark scripts were read and left
unchanged. The harness validates one JSON label per paper, reads cached
extraction results by default, calculates layer-type recall/precision,
hyperparameter accuracy, family correctness, optional static fidelity, and a
hard-failure count. Missing optional label fields are represented as `null` in
their metric rather than being counted as failures. Invalid labels identify the
path in a clear `ValueError`.

Ten conservative labels were added for the supported families: ResNet,
U-Net, ViT, Transformer, BERT, MobileNet, DenseNet, EfficientNet, DCGAN, and
DDPM. Each label records selected architecture facts stated in its source
paper. The live adapter follows arXiv's PDF redirect, extracts the text layer,
and calls the existing `ConfigExtractor`; it deliberately does not generate
code or invoke E2B because this benchmark is for extraction quality.

The live run initially exposed a redirect limitation in the existing production
`PaperToCodeGenerator.from_arxiv` path. The harness adapter was corrected to
follow redirects without changing production code. Its first version also ran
the entire generation/E2B path; that benchmark-only run was stopped after one
real BERT extraction because it measured downstream execution rather than the
requested extraction stage. The final adapter is covered by a redirect test.

**Verification:**

```
.venv/Scripts/python.exe -m pytest tests/test_benchmark_harness.py -q
16 passed in 0.43s

.venv/Scripts/python.exe -m pytest -q
1804 passed, 10 skipped, 8 warnings in 316.36s (0:05:16)
```

**Live baseline attempt:** Groq was configured and arXiv was initially
reachable. BERT and DCGAN produced real cached live extractions. After DCGAN,
the host experienced a server disconnect followed by DNS lookup failures for
the remaining eight arXiv PDFs. The complete offline run against the ten-label
set correctly recorded these as hard failures rather than omitting them:

```
papers=10
hard_failures=8
layer_type_recall=0.475
layer_type_precision=0.500
hyperparam_accuracy=0.666667
family_correct=0.200000
fidelity_score=0.200000
```

The corresponding artifact is
`benchmarks/results/20260830T133500Z.json`. This is not a calibrated ten-paper
quality baseline yet: eight papers must be re-run once network/DNS access is
stable. The artifact is intentionally retained as evidence of the failed live
run. No commit or push was performed. Prompt 10 has not started.

---

## 2026-09-04 — Phase 5 closeout: three silent-degradation bugs

Phase 5 was reported complete twice while work sat unapplied, and the
number it reported (recall 0.465) turned out not to be a measurement of
the pipeline at all. Closing it out surfaced three distinct defects, all
of the same family: **a failure that looks like a result.**

### 1. Empty LLM completions were treated as success (production bug)

`core/llm_client.py` did `text = resp.choices[0].message.content or ""`,
reset the failure counter, and returned. An empty completion therefore
propagated as `""`, surfaced downstream as `ValueError: LLM did not return
valid JSON`, and `ConfigExtractor.extract_from_text`'s broad `except`
silently substituted the rule-based extractor.

This affected **production ingestion**, not just the benchmark: any paper
whose completion came back empty was quietly downgraded to a rule-based
spec with no signal anywhere. Fixed by treating an empty completion as a
retryable failure (same path as a rate limit), then raising a clear error.
Regression tests in `tests/test_llm_empty_completion.py`.

### 2. The JSON parser rejected recoverable model output

`_parse_json_response` handled raw JSON and fenced blocks only. Models
routinely emit `//` comments and trailing commas inside JSON; densenet121
failed on a literal `// bottleneck 1x1`. Now repairs comments and trailing
commas, tolerates preamble text and unterminated fences, and preserves
slashes inside string values (URLs). Repair is attempted last, so
well-formed output is untouched. Verified against the actual captured
failing response.

### 3. No `max_tokens` ceiling — long specs truncated mid-JSON

No explicit limit was set, so the provider default applied. U-Net's
response was cut mid-token at 1084 chars (`"kernel_size": `). Raising to
4096 was **still not enough** (3219 chars, truncated in the connections
array) because reasoning models spend much of the budget before emitting
output. At 16384 U-Net completes: **recall 1.00, precision 0.80, family
correct** — previously a rule-based fallback scoring 0.75 on a garbage
spec. Configurable via `LLM_MAX_COMPLETION_TOKENS`.

### Clean result (all 10 papers on the LLM path, 0 fallbacks)

`benchmarks/results/phase5-clean.json`

| Metric | Before (contaminated) | Clean |
|---|---:|---:|
| layer_type_recall | 0.465 | **0.713** |
| layer_type_precision | 0.593 | **0.727** |
| hyperparam_accuracy | 0.000 | **0.400** |
| family_correct | 0.500 | **0.900** |
| rule_based_fallback_count | (not recorded) | **0** |

Gate: 3 of 5 criteria met. `hyperparam_accuracy` (0.400 vs 0.50) and
dcgan's 0.25 recall carry into Phase 6, whose scope is exactly that
(parameters and dimensions).

### Also landed

- `--check` regression guard comparing to `benchmarks/baseline.json`,
  with errors for schema_version / retrieval-mode mismatch and for a
  metric present in the baseline but missing from a run. Real `--check`
  is local-only by design: `.cache/` is untracked, and a CI run scoring a
  frozen cache could not detect extraction regressions anyway.
- `benchmarks/baseline.json` created from the clean run, recording
  retrieval mode, label schema_version, and source results file.
- `--strict` proved its worth: it rejected every contaminated run rather
  than recording it, which is how the fallbacks were caught at all.

### Method note

An earlier attribution in this log blamed rate limiting for the missing
`conv2d` cases. That was wrong. Rate limiting was real but incidental; the
recurring causes were the three defects above. The lesson is the one this
project keeps relearning: **do not diagnose from an aggregate.** The
answer only appeared after per-paper `extraction_method` was recorded and
the raw LLM responses were captured and read.

---

## 2026-09-07 — Phase 6: five causes to reach one root cause

Phase 6 targeted the two criteria Phase 5 missed: `hyperparam_accuracy`
(0.400) and "no paper below 0.40 recall" (dcgan at 0.25). Both moved.
The instructive part is not the result but how long the real cause stayed
hidden behind plausible ones.

### The chain

BERT's `hidden_size: 768` was unreachable throughout. Five hypotheses:

1. **Prompt text** — rolling back the params instruction. Refuted: BERT's
   hyperparam stayed 0.00 on a Groq-served paper with the rollback in place.
2. **Numeric-only reservation filter** — reserving structured chunks that
   contained a digit. This protected BERT's SQuAD and CoNLL *results*
   tables, the most numeric objects in the paper, while evicting the prose
   with the dimensions. Real bug, insufficient.
3. **Reservation evicting prose** — requiring architectural vocabulary in a
   reserved chunk. Confirmed working (`selected_structured: []`), still
   insufficient.
4. **Expansion order** — `ranked_indices` was built as a `set`, discarding
   retrieval rank, then iterated with `sorted()`. Chunk 0 (title/abstract)
   expanded first for being early and ate the budget. Real bug,
   insufficient.
5. **Query parity — the root cause.** `hybrid_rank_texts` searched
   `_ARCHITECTURE_QUERY` (no numeric tokens); `retrieve_top_chunks` searched
   `_ARCH_QUERY_TERMS` (including `64 128 256 512 768 1024 2048`). Plain
   BM25 ranked BERT's spec chunk **first**; the production hybrid retriever
   did not rank it at all. Adding the numerics: rank 6 (MMR-evicted) ->
   rank 2 (selected); BERT `hyperparam_accuracy` 0.00 -> 1.00 isolated.

`_ARCHITECTURE_QUERY` was written in Phase 3 as a "natural-language
stand-in" for `_ARCH_QUERY_TERMS`, dropping the numbers on the assumption
embeddings would not use bare digits. The dense half does not; the BM25 half
did. **The hybrid retriever's BM25 component was given a strictly weaker
query than the standalone BM25 it replaced** — which is how production could
lose to its own fallback.

### Two false positives worth remembering

Both would have closed the phase on a wrong result:

- **"768 is present."** True, but the only match was `"two-layer
  768-dimensional BiLSTM"` — an unrelated ablation. A substring probe passed
  while the spec sentence was absent. Fix: assert on the spec sentence and
  print surrounding context.
- **"BERT dimensions now reach the model."** Measured in an environment
  where the embedder failed to import, so the check ran on **BM25 only**,
  not the production hybrid path. Under the real retriever it still failed.
  Fix: assert the embedder is available before trusting a retrieval result.

### Measurement constraint

Seven full-run attempts blocked or contaminated. Raising pacing 25s -> 45s
made it **worse** (provider fallbacks 4 -> 6, rule-based 0 -> 3), proving the
limit is a **daily token budget**, not a rate ceiling. `--strict` rejected
the bad run rather than recording it.

The productive pattern: `_focus_text` calls no LLM, so every retrieval
question above was answered at **zero quota cost**. Only the final
confirmation needed the budget.

### Result

Best run `benchmarks/results/20260907T065724Z.json` (0 rule-based, 4
provider fallbacks — aggregate not baseline-comparable):
hyperparam **0.533** (was 0.400), recall 0.702, precision 0.654, family 0.900.

Groq-served and valid: bert recall 0.50->0.75, dcgan 0.25->**0.50** (first
time above the floor), resnet50 hyperparam 0.33->0.67, transformer_base
recall 0.75->1.00.

`benchmarks/baseline.json` deliberately **not** refreshed.

### Also fixed this phase

- Symbolic parameter values (`channels: "2k"`, `compression: "theta"`,
  `out_features: "num_classes"`) rejected at normalization instead of
  reaching codegen.
- `convtranspose2d` pattern now matches `transposed convolution` and
  `fractional-strided convolutions` **including plurals** — it previously
  matched only `deconvolution`, and only singular, so DCGAN's actual wording
  never mapped.
- Provider provenance recorded per paper; the litellm routing-prefix false
  positive (which flagged every Groq call as a cross-provider fallback) fixed
  via `_bare_model_id`.

### Open

- **ddpm regressed 0.67 -> 0.33** — only paper below the floor, and unlike
  dcgan it used to work. Diagnosable at zero quota.
- **Precision 0.654 vs 0.727 baseline, cause still unattributed** since
  Prompts 1+2. Numeric query tokens may pull results tables into *prose*
  slots; the reservation filter guards only reserved slots.
- **ViT's recovery unverified** (Gemini-served in the best run).
- One clean ten-paper run on a fresh daily budget, then refresh the baseline.

## 2026-09-08 — The bug underneath Phases 5 and 6: PDF text extraction

`pdfplumber`'s `extract_text()` defaults to `x_tolerance=3`, which merges
adjacent words. Every one of the ten benchmark papers was affected, and no
component reported anything wrong — the text arrived, it was simply wrong.
`transformer_base` was the extreme: 3.3% of characters were spaces, and
letter runs averaged 12.4 chars (`Weuseself-attentionat`).

Ordinary English prose runs ~16% spaces. Corpus before: 3.3-11.1%. After
`x_tolerance=1`: 13.4-15.3%, letter runs 4.7-5.4. All 10 papers healthy.
Method: `[A-Za-z]+` runs over the first 30 pages.

### Why one bug looked like five

Run-together text breaks regex word boundaries, BM25 tokenisation, and
embedding quality **at the same time**. So the layer-pattern matcher, the
lexical half of hybrid retrieval, and the dense half all degraded together
for one shared reason — while each looked like an independent tuning
problem. Phases 5 and 6 spent their effort downstream of this.

ddpm, the only paper under the Phase 6 floor, in its focused text:

| expected type | before | after |
|---|---|---|
| `multiheadattention` | absent | `self-attention` |
| `groupnorm` | `groupnorm` | `group normalization`, `group norm` |
| `positionalembedding` | `sinusoidal` only | `position embedding`, `sinusoidal` |

`groupnorm` "matched" before only because the words had been fused into a
token that resembled the canonical name. That is an accidental hit, and it
is worse than a miss: it makes a broken pipeline look partly working.

### What is and is not established

Spearman rho between pre-fix space% and Phase 6 recall is **0.75 on n = 6**;
the critical value at n = 6 is ~0.83, so it does **not** clear significance,
and `transformer_base` is an outlier. Suggestive, not proven. The fix is
verified at the *retrieval* level and by 1946 passing tests; no test asserts
on extraction quality, so the score-level effect is still unmeasured.

### Method errors made and caught

- **A re-check that could not fail.** The first verification called
  `page.extract_text()` directly instead of the patched code path, so it
  measured the old behaviour and printed byte-identical before/after output.
  Caught only because identical output was implausible. Same class of error
  this project keeps finding in completion reports: a check that doesn't
  exercise what it claims to.
- **A figure that could not be reproduced.** An earlier note cited 14.4-char
  words for `transformer_base`; neither letter-run (12.4) nor whitespace
  (16.5) tokenisation yields it. Corrected in all three code comments.
- **Four test failures that were not regressions.** The harness's PDF
  doubles declared `extract_text(self)` with no kwargs, so `x_tolerance=1`
  raised `TypeError`. The other two sites passed only because `MagicMock`
  swallows kwargs — meaning they would *not* catch a silent removal. The
  harness fakes now record kwargs and one test asserts `x_tolerance == 1`;
  that assertion was confirmed to fail when the fix is reverted.

### Also landed

- **`check_against_baseline` now compares `primary_model`.** It was written
  by `build_baseline` and never read, so a run from one model would validate
  against a baseline built on another. A baseline missing the field reports
  a problem rather than passing, since it cannot be verified at all.
  Consequence: the current baseline now fails `--check`. `phase5-clean.json`
  predates provider tracking and records no model, so the field cannot be
  honestly backfilled — a fresh baseline is required.
- **Benchmark cache isolation.** Eight tests ran the benchmark without
  patching `CACHE_DIR` and were writing `synthetic.*.json` into the live
  `benchmarks/.cache`, where an offline run or `--check` would read them as
  real extractions. Fixed with one autouse fixture covering the module
  rather than eight individual patches, so later tests cannot re-leak.

### Open

- Score-level effect of the PDF fix — needs the clean ten-paper run.
- Precision 0.654 vs 0.727: the PDF fix may resolve it (degraded text could
  inflate spurious matches). Measure before changing anything else.
- `paper_to_code_generator` and `paper_ingestion_service` have no assertion
  that `x_tolerance=1` is passed; only the harness does.
