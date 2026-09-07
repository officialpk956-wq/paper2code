OBJECTIVE: Independently verify that Phase 1, Phase 1.5, and Phase 2
(implementation + two audit rounds + final closure) are genuinely
complete and working -- not just that the code exists, but that it
actually runs correctly when exercised for real. Do not commit or push
anything. This is a verification pass, not a feature-development pass.

READ FIRST: docs/PAPER_TO_CODE_EXECUTION_MEMORY.md, in full. It is the
detailed, honest record of everything built and every bug found across
three rounds of work on this codebase today. It names specific bugs,
specific fixes, specific files, and specific test files. Do not
re-derive this history -- use it as your map.

================================================================================
WHY THIS VERIFICATION PASS EXISTS -- READ THIS BEFORE YOU START
================================================================================

This codebase has a proven, repeated history *today* of automated tests
passing while the real functionality was broken, because the test mocked
exactly the boundary that mattered. Concrete examples already found and
fixed once each:

- ConfigExtractor's "consistency" test used `use_llm=False` (the
  deterministic rule-based path) and never exercised the actual
  production config (`use_llm=True`), which was non-deterministically
  broken (20/8/20 layers for identical input across 3 real calls).
- Two E2B validation tests mocked `run_code_in_sandbox` entirely. The
  real integration always failed with `ModuleNotFoundError: No module
  named 'torch'` (no install step in a template that doesn't have it),
  and even after that was fixed, the harness never ran a real forward
  pass -- it just checked instantiation and unconditionally reported
  `forward: True`.
- `test_live_all_families_api.py`, despite its name, never made an HTTP
  request or touched ConfigExtractor -- it called internal functions
  directly with a hardcoded spec.
- A full non-live suite run showed 11 unrelated-looking failures (oauth,
  storage, paper visibility) that turned out to be caused by a leftover
  Docker Redis container silently changing what a security check
  enforced -- not a code bug, but real enough to produce a wrong
  conclusion if not investigated.

**Do not repeat this pattern.** Wherever this prompt asks you to "verify"
something, that means: actually run it, with real (non-mocked) inputs
where the memory doc indicates a real integration point exists, and look
at the actual output -- not "the test passed" as a proxy for "it works,"
if the test doesn't actually exercise the real path.

================================================================================
STEP 0 -- CONFIRM A CLEAN STARTING STATE
================================================================================

Before running anything, check for leftover infrastructure from a prior
session that could silently change test behavior (this exact thing
happened today):

- `docker ps` -- any project-related containers (postgres, redis) still
  running from a previous session should be noted. If present, either
  intentionally use them (fine for live/async testing) or stop them
  before running the plain `pytest tests/` baseline, since a real Redis
  being reachable changes what some security checks enforce.
- Check for stray backend/Celery worker processes already bound to ports
  8000/8010 (Windows: `Get-CimInstance Win32_Process | Where-Object {
  $_.CommandLine -like '*celery*' -or $_.CommandLine -like '*uvicorn*'
  }`). Kill anything unexpected before starting your own.
- `git status` -- confirm the working tree matches what the memory doc's
  "Files changed" section describes. If it doesn't, STOP and report the
  discrepancy before proceeding -- don't verify against a moving target.

================================================================================
VERIFICATION CHECKLIST
================================================================================

**1. Full non-live test suite (baseline).**
```
python -m pytest tests/ -q -m "not live"
```
Expected: 1736 passed, 2 skipped, 5 deselected, 0 failed (per the memory
doc's last recorded run). If you get a different number, investigate
before concluding regression -- check Step 0's environmental causes
first (leftover Docker/processes), since that exact scenario produced 11
false failures earlier today.

**2. Live LLM-dependent tests, if you have real credentials.**
Requires `GROQ_API_KEY` in `.env` and `RUN_LIVE_PHASE2=1`:
```
python -m pytest tests/test_phase2_config_extractor.py tests/test_llm_client_retry.py -v -m live
```
The `test_config_extractor_real_llm_path_is_consistent` test hits the
real Groq/Gemini APIs and costs real tokens. If it fails, check whether
the failure is a genuine layer-count/type inconsistency (a real
regression) or an API error (rate limit / quota exhausted -- Gemini's
free tier is 20 requests/day and may already be exhausted depending on
what else has run against it today). Report which one it is; don't
conflate them.

**3. Live E2B validation tests, if you have `E2B_API_KEY`.**
```
python -m pytest tests/test_phase2_e2b_validation.py -v -m live
```
These hit the real E2B API and include a ~2-4 minute cold-start pip
install of torch (the sandbox template doesn't have it preinstalled).
Don't kill it early thinking it's hung.

**4. The real non-eager async flow, end to end, through the actual UI.**
This is the highest-value check in this list -- it's the one thing that
was hardest to prove and was only closed in the final pass today.
```
docker compose up -d postgres redis
docker compose ps          # confirm both report "healthy"
```
Then, WITHOUT setting any eager-mode override, start the real backend and
a real Celery worker (see docs/LOCAL_DEV.md section 3 for exact commands
-- `-P solo` on Windows). Confirm `curl http://localhost:8000/health`
reports `{"redis": "healthy"}` (a *real* Redis, not eager mode).

Then run the actual gated Playwright test:
```
$env:RUN_REAL_ASYNC_PAPER_E2E='1'
npx playwright test e2e/paper-upload-async.spec.ts --project=chromium --reporter=list
```
KNOWN QUIRK (not a bug): the first run after starting a fresh Next.js dev
server can fail on a 5-second `toBeVisible` timeout because the page
hasn't finished its first cold compile yet. If this happens, simply
re-run the same command once more (the build cache is now warm) before
concluding there's a real problem. A genuine pass looks like: `1 passed`
with the full flow -- upload dialog, polling, redirect to `/papers/{id}`,
Executable tab, "Phase 1 verified" -- all visible.

**5. Spot-check specific bugs that were found and fixed today, to
confirm none of them regressed.** These are all fast, no LLM/E2B needed:
```
python -m pytest tests/test_phase2_null_safety.py tests/test_codegen_skeleton.py tests/test_phase1_paper_codegen.py tests/test_section_splitter.py -v
```
If you want to go further, directly reproduce a couple of the specific
crash conditions the memory doc documents (e.g. call
`PaperToCodeGenerator()._e2b_test_input_candidates({"input": {"channels":
3, "spatial_dims": 224}})` directly -- a scalar `spatial_dims` used to
crash this with `TypeError: 'int' object is not subscriptable`; it
should now return a candidate list cleanly).

**6. Live family matrix (optional, expensive -- only if you have budget
for it).** `tests/integration/test_live_phase2_family_matrix.py`, gated
behind `RUN_LIVE_FAMILY_MATRIX=1`, uploads 10 real papers (8 supported +
2 unsupported) through the real HTTP/Celery stack and can take 10+
minutes plus real LLM cost. The memory doc is explicit that no single
uninterrupted 10/10 run has ever completed (external rate limits kept
interrupting it), though every individual family has independently
succeeded live at least once. If you run this and it partially fails,
check the actual error before concluding a regression -- rate-limit
cascades (see the memory doc's account of the U-Net failure caused by
setting primary=fallback=the same Gemini model) are a known, external,
non-code cause.

================================================================================
HOW TO REPORT
================================================================================

For each numbered check above: pass/fail, and if fail, a root-cause
classification -- "genuine regression in [file/function]" vs "external
(rate limit/quota/network)" vs "environmental (leftover process/
container)" vs "test infrastructure issue (e.g. cold-start timing)".
Don't just report a pass/fail count without the classification -- an
unclassified failure is not useful information given this session's
history of failures that turned out to be nothing, and passes that
turned out to be nothing.

If you find something the memory doc doesn't mention -- a new bug, a
new inconsistency -- treat it exactly like the audits described in the
memory doc did: reproduce it directly, isolate the root cause, and only
then decide whether it's worth fixing in this pass or flagging for later.

================================================================================
SAFETY / CLEANUP
================================================================================

- No commits, no pushes.
- Whatever you start (Docker containers, backend/worker processes,
  Playwright's own dev server), stop and remove before finishing. Confirm
  with `docker ps`, `git status`, and a listen-port check that nothing is
  left running or changed beyond intentional test/source files.
- Delete any synthetic test users/papers you create against the local
  SQLite DB during live verification.
- If you had to fix something, follow the same regression-test discipline
  the memory doc's fixes all followed: a test that would have caught the
  bug, not just a fix.
