OBJECTIVE: Replace the unreliable single-shot architecture extraction with the
existing richer extractor, close the null-handling gaps that pattern-match
across the schema-normalization code, extend real sandboxed validation to
LLM-generated code (currently never executed), add a bounded repair loop, and
prove all four supported families work live — not just the three verified so
far.

PROJECT: paper2code (Next.js + FastAPI/Celery backend)
CURRENT STATE: Phase 1 + Phase 1.5 complete. The full chain (PDF upload ->
section splitting -> architecture extraction -> ConfigDict translation ->
graph construction -> known-family code generation -> in-process validation
-> persistence -> workspace display) works end to end and was live-verified
8/8 consecutive times for the resnet family via the real HTTP API. Full
regression suite: 1697 passed, 3 skipped, 0 failed. Read
docs/PAPER_TO_CODE_EXECUTION_MEMORY.md in full before starting -- it has the
exact bugs found and fixed, the live-verification method (no local Redis
needed), and why this pipeline behaves the way it does today.

SCOPE: Phase 2 covers items 1-6 below. Formal orchestration
(state-machine/versioning/per-stage timeouts) is explicitly OUT OF SCOPE for
this pass -- defer it until 1-6 are solid and live-verified, same as how
Phase 1 was scoped tightly rather than attempting everything at once.

DO NOT COMMIT OR PUSH. Preserve the existing Dojo modification in
src/app/(protected)/dojo/page.tsx untouched. Do not regress the 1697-passing
baseline -- run the full suite after every numbered section below, not just
at the end.

================================================================================
BACKGROUND YOU NEED (read the code, don't take this as gospel)
================================================================================

Two extractors exist side by side, doing the same job at wildly different
quality levels:

- core/architecture_extractor.py `extract_architecture(sections, paper_name)`
  -- the one the live path currently uses. One bare LLM call, no retrieval,
  no verification pass. This is the direct cause of most bugs found in
  Phase 1.5: it non-deterministically returns different shapes call to call
  (sometimes proper "stages", sometimes data stuffed into "block.params",
  sometimes explicit nulls where a key is expected).

- core/rag/config_extractor.py `ConfigExtractor.extract_from_text(text) ->
  ConfigDict` -- section-aware focusing, BM25 retrieval for oversized text,
  LLM extraction with few-shot prompting, a self-correction verification
  loop, and normalization. Already built. Never called from the live upload
  path. Returns the ConfigDict shape directly ({"name", "layers",
  "connections"}) -- the same shape core/agents/parsing_agent_impl.py's
  ParsingAgentImpl.parse() already natively accepts for its "text"/
  "pdf_extract" format_hint branch.

This means routing through ConfigExtractor could ELIMINATE the need for the
_spec_to_config_dict() translator added in Phase 1.5 for the graph-building
half of the pipeline -- ConfigExtractor already speaks ConfigDict.

BUT: core/paper_to_code_generator.py's _generate_code() picks the
known-family builder shortcut off spec.get("model_family") and
spec.get("stages") (BASE_MODEL_SCHEMA shape), which a ConfigDict does not
have. You cannot just swap one extractor for the other without deciding how
the builder path gets its family/stage info afterward. Three real options,
in order of how much they change:

  (a) Run both extractors (redundant LLM calls, wasteful, but zero
      redesign of _generate_code).
  (b) Derive model_family from the ConfigDict's graph via
      core/classification.py's classify_architecture(graph) (already fixed
      in Phase 1.5 to read node.label correctly), and derive per-stage
      builder info by grouping/counting the graph's layer nodes by type.
  (c) Redesign _generate_code to work primarily off the graph/ConfigDict,
      treating BASE_MODEL_SCHEMA as a legacy shape only produced when
      falling back to the old extractor.

Investigate all three before picking one. Do not default to (a) just
because it's least work -- it doubles LLM cost and latency on every upload,
which matters at the volumes this product wants. Write a short paragraph in
docs/PAPER_TO_CODE_EXECUTION_MEMORY.md explaining which you chose and why
before writing the implementation.

================================================================================
PHASE 2 BREAKDOWN
================================================================================

**2.1 Wire ConfigExtractor into the live extraction path**

ACTION:
  a) Read core/rag/config_extractor.py in full -- understand what
     use_llm/use_section_splitter/use_retriever/verify actually do, and
     what its self-correction verification loop checks for.
  b) Pick and document one of options (a)/(b)/(c) above.
  c) Implement it in core/paper_to_code_generator.py's _run_pipeline().
  d) Keep extract_architecture() and _spec_to_config_dict() in place (do
     not delete) unless your chosen option makes them fully unreachable --
     if so, remove the dead code and its now-unused imports, and delete or
     update the tests that exercised it (tests/test_phase1_paper_codegen.py
     has tests against _spec_to_config_dict and the old extractor's guard
     condition -- update them to match reality rather than leaving them
     testing dead code).
  e) Add a test: feed ConfigExtractor a real paper excerpt (not the sparse
     3-sentence fixture -- use something closer to a real Methods section,
     e.g. the ResNet-50 excerpt in Phase 1.5's verification notes) and
     confirm it returns a populated, consistent ConfigDict across at least
     3 repeated calls (LLM non-determinism is the whole problem you're
     fixing -- prove it's actually more consistent now).

VERIFY:
  - Direct Python reproduction first (same technique as Phase 1.5): call
    the updated _run_pipeline()/from_pdf() directly with load_dotenv(),
    not through the HTTP API, to get fast iteration before wiring the full
    stack.
  - Then the live HTTP path: temporary backend on a spare port (8010 is
    already used to convention in this session -- check nothing else has
    it bound first) with VERIFY_PASS_EAGER=1 set in backend/celery_app.py
    (same pattern as Phase 1.5 -- gate behind that exact env var, revert
    the file after testing, never commit it).
  - Run 8+ consecutive uploads of the same real excerpt; report the
    consistency of extracted model_family/stages across all of them.

--

**2.2 Systematic null/defaults audit**

PROBLEM: Three of Phase 1.5's four bugs were the same pattern: a dict key
is explicitly present with value None (not missing), and .setdefault() or
a {**default, **override} spread doesn't catch that, so None survives into
code that assumes a real value and crashes.

ACTION:
  a) Grep for `.setdefault(` and dict-spread merge patterns
     (`{**`) across core/normalizer.py, core/classification.py,
     core/paper_to_code_generator.py, and any other file that touches
     extracted spec/stage/config dicts (core/rag/normalizer.py is a
     DIFFERENT file with a similarly-named function -- check it too, it
     backs the ConfigExtractor path you're now wiring in via 2.1).
  b) For each site found, determine: can this key legitimately be None
     from the extractor/LLM? If yes, does the current code handle that,
     or would it crash three functions later like the stage-merge bug
     did? Fix with `value = extracted.get(key) or default` instead of
     `.setdefault(key, default)`, or an explicit `if value is None:
     value = default` where the value could legitimately be falsy-but-
     valid (e.g. 0, False -- setdefault-style "or" fallbacks are wrong
     for those, use an explicit None check).
  c) Add one test per fix, following the existing pattern in
     tests/test_phase1_paper_codegen.py
     (test_builder_schema_fills_in_missing_stage_fields) --
     construct a spec/stage dict with an explicit None in the field you
     just fixed, confirm it no longer crashes.

VERIFY: `pytest tests/ -k "normalizer or classification or builder_schema" -v`

--

**2.3 Extend real execution validation to LLM-generated code via E2B**

PROBLEM: core/paper_to_code_generator.py's validate_generated_code()
currently only executes code for code_source == "builder" (known
families). For code_source == "llm" or "skeleton" (unknown families, or
when the builder path fails), it skips execution entirely and returns
status "needs_review" unconditionally -- that code has literally never
been run, ever, by anyone, automated or not. This is also a safety
consideration: LLM-generated code is untrusted and should not be exec()'d
in-process the way builder code currently is (builder code is composed
from this repo's own trusted source via inspect.getsource(), not LLM
output -- exec()'ing that in-process is a reasonable, already-shipped
choice; exec()'ing raw LLM output in-process is not).

ACTION:
  a) Find the existing E2B sandbox integration the Phase 1 workspace
     Executable tab already calls (per the execution memory doc: "can
     submit it to the existing authenticated sandbox endpoint"). Read
     that endpoint and its E2B client code -- don't build a second E2B
     integration from scratch.
  b) Extend validate_generated_code() (or add a sibling function it calls
     for non-builder code_source) to submit LLM/skeleton-generated code to
     that same E2B sandbox for the same checks builder code gets
     in-process: import, instantiate, synthetic forward pass, output
     shape check against spec.get("output"). Keep the builder path's
     in-process exec() as-is (it's fine, don't fix what isn't broken).
  c) Handle sandbox timeout, network-denial, and resource-limit failures
     as structured diagnostics (stage="sandbox", not a bare exception
     string) -- these are expected/common failure modes for arbitrary
     LLM-generated code, not edge cases.
  d) Do NOT let this call block the upload response indefinitely -- E2B
     sandbox calls have real latency; keep this inside the existing async
     Celery task, not the synchronous upload handler.

VERIFY:
  - Unit test with a deliberately broken LLM-style code string (e.g. wrong
    import, syntax that passes ast.parse but fails at runtime) confirming
    it reaches "needs_review" with a real E2B-sourced error, not a skipped
    check.
  - Unit test with valid unknown-family generated code (craft one from an
    architecture family the builder path doesn't support, e.g. a GAN or
    a paper with no clean family match) confirming it can now reach
    "success" if it's actually correct.

--

**2.4 Bounded repair loop**

PROBLEM: When generation_status is "needs_review" or "failed", nothing
happens -- the user sees the broken result and that's the end of it. The
verification_report already has everything a repair attempt needs: stage,
error, expected/actual shape (see the "checks"/"error"/"input_shape"/
"output_shape" fields already being populated).

ACTION:
  a) Add a repair function that takes the current code, the
     verification_report diagnostic, and the original architecture spec,
     and asks the LLM to fix specifically the failure described (not
     "regenerate from scratch" -- give it the diagnostic, the current
     code, and ask for a targeted fix, per the original plan's repair
     design).
  b) Wire it into the Celery task (backend/tasks/paper_tasks.py) or
     _run_pipeline(): on a failed/needs_review verification result,
     retry up to 3 times total (including the first attempt), re-running
     validate_generated_code() after each repair attempt, stopping early
     on first success.
  c) Persist every attempt, not just the last one. This needs a schema
     decision: either widen Paper.verification_report to hold a list of
     per-attempt reports (JSON already supports that with zero migration
     if you keep the same column, just change what's written into it),
     or add a new generation_attempts JSON column via a new Alembic
     migration (matching the original plan's PaperPipelineAttempt design
     more closely, but more work). Pick the smaller change unless you can
     show a real reason attempt history needs to be queried
     independently of the paper -- it likely doesn't yet.
  d) Never loop more than 3 attempts. Log/expose which attempt number
     produced the final result.

VERIFY:
  - Test with a spec that fails on attempt 1 (inject one of the bugs from
    2.2 you haven't fixed everywhere, or craft one on purpose) and
    confirm repair improves it to attempt 2 or 3.
  - Test that a spec which will never succeed (deliberately impossible)
    stops at exactly 3 attempts and returns "needs_review", not an
    infinite loop or a 4th attempt.

--

**2.5 Live-test the untested paths**

PROBLEM: Every live verification in Phase 1.5 used resnet. unet and vit
were only verified via direct unit-level calls
(test_known_family_source_is_self_contained_and_executable), never through
the live HTTP upload path with real extraction. transformer and the
unknown-family fallback path have zero verification of any kind this
session.

ACTION:
  a) Using the same technique as Phase 1.5 (PyMuPDF-generated realistic
     excerpt PDF, live HTTP upload via the eager-mode backend), run one
     real live upload for each of: unet, vit, transformer, and one
     deliberately-unsupported family (e.g. a GAN or diffusion model
     excerpt) to exercise the skeleton/LLM-generation fallback.
  b) For the unsupported-family case, confirm the result is an honest
     "needs_review"/"unsupported" outcome with a clear reason, not a
     silent wrong answer presented as "success".
  c) Fix whatever breaks. Given the pattern this session, expect at least
     one more null-handling or schema-shape surprise -- that's normal,
     not a sign anything is wrong with your approach.

VERIFY: All 4 new live-upload cases produce a coherent result state
(success, needs_review, or a clear unsupported message) -- none should
produce a raw Python traceback or an ungraceful 500.

--

**2.6 Local dev infrastructure for the async path**

PROBLEM: Every live test this session and Phase 1.5 needed a
VERIFY_PASS_EAGER hack because no Redis/Celery worker exists locally.
This is fine for checking pipeline correctness but never exercises the
real "202 now, poll later" behavior, including how failures surface
through polling (Phase 1.5's memory doc flagged this gap explicitly:
eager mode collapses async failures into a synchronous 400, which is NOT
what production does).

ACTION:
  a) Add a docker-compose.yml (or extend one if it exists) with a Redis
     service for local dev. Check if Docker Desktop is actually usable in
     this environment first (it was NOT running during Phase 1.5's
     session -- confirm before assuming this works).
  b) Document (in docs/PAPER_TO_CODE_EXECUTION_MEMORY.md or a new
     docs/LOCAL_DEV.md) how to run: `docker compose up redis`, then a
     real Celery worker (`celery -A backend.celery_app worker --pool=solo`
     on Windows), then the backend without VERIFY_PASS_EAGER.
  c) Run at least one real (non-eager) upload through this real stack.
     Confirm the upload endpoint returns immediately (not 6-10 seconds
     later like eager mode did), the polling UI shows intermediate
     states, and a deliberately-failing upload surfaces its error via
     polling rather than a synchronous 400.

VERIFY: One real async upload, watched through the actual frontend UI (not
curl), from "Preparing your paper workspace" through to either the
workspace redirect or a clean failure message with a working Retry button.

================================================================================
ACCEPTANCE CRITERIA (Phase 2 exit gate)
================================================================================

- ConfigExtractor (or your chosen integration) is live, with a documented
  rationale for which option (a/b/c) was chosen.
- 10 real/realistic paper excerpts -- at least 2 per family (resnet, unet,
  vit, transformer) plus 2 deliberately-unsupported -- uploaded through the
  live HTTP path. At least 80% of the 8 supported-family uploads reach
  generation_status "success" without manual intervention. Both
  unsupported-family uploads produce an honest "needs_review"/"unsupported"
  result, never a silent wrong answer.
- The repair loop demonstrably improves at least one case from
  needs_review/failed to success, and demonstrably stops at 3 attempts on
  an unfixable case.
- LLM-generated (non-builder) code now gets real E2B execution validation,
  proven via a unit test with a deliberately broken example.
- At least one real async (non-eager, real Redis/worker) upload verified
  through the actual frontend UI.
- Full test suite still green: `pytest tests/ -q` shows 0 failures, and the
  passing count only goes up (new tests added, nothing removed except
  tests that were exercising code you deliberately deleted -- and those
  removals are explained in the memory doc, not silent).
- docs/PAPER_TO_CODE_EXECUTION_MEMORY.md updated with what changed, what
  was tested, what's still known-broken, and the exact next phase --
  same discipline as Phase 1.5's update.

================================================================================
SAFETY / ROLLBACK
================================================================================

- No commit or push.
- backend/celery_app.py's eager-mode addition must stay gated behind
  VERIFY_PASS_EAGER and be reverted (git checkout --) before you consider
  any section done -- do not leave it modified between sections.
- Any new migration (if you choose the generation_attempts column route in
  2.4) must be reversible; test upgrade -> downgrade -> upgrade on a throwaway
  SQLite copy before treating it as done, same as Phase 1's migration was
  verified.
- Delete all synthetic test users/papers/scratch files created during
  verification before finishing each section -- don't let them accumulate
  across sections into a messy final git status.
- If E2B usage in 2.3 has a per-request cost, note the approximate cost of
  your test runs in the memory doc -- don't run it in a loop without
  noticing what it costs.

================================================================================
REFERENCES
================================================================================

- docs/PAPER_TO_CODE_EXECUTION_MEMORY.md -- read this first, in full.
- Extraction: core/architecture_extractor.py (old), core/rag/config_extractor.py
  (new target), core/rag/normalizer.py (different file from core/normalizer.py
  -- don't confuse them).
- Graph/ConfigDict: core/agents/parsing_agent_impl.py, core/agents/config_parser.py,
  core/orchestrator/pipeline.py.
- Codegen + validation: core/paper_to_code_generator.py.
- Sandbox: find the existing E2B client via the workspace Executable tab's
  "Run" button in src/app/(protected)/papers/[id]/WorkspacePaperClient.tsx
  and trace it back to its backend route.
- Tests: tests/test_phase1_paper_codegen.py, tests/test_section_splitter.py
  (extend these, follow their existing style rather than inventing a new one).

================================================================================
QUESTIONS FOR CLARIFICATION
================================================================================

Pause and ask before proceeding if any of the following is unclear:
1. Which ConfigExtractor integration option (a/b/c) fits the product's
   actual latency/cost budget per upload -- this is a product decision as
   much as a technical one.
2. Whether generation_attempts should be a new migration or reuse the
   existing verification_report JSON column as a list -- lean toward reuse
   unless there's a clear reason not to.
3. Whether E2B has a cost/quota ceiling this session should respect during
   testing.
4. Whether Docker Desktop is actually available/usable for 2.6 in this
   environment -- confirm before planning around it.

================================================================================
START HERE
================================================================================

1. Read docs/PAPER_TO_CODE_EXECUTION_MEMORY.md in full.
2. Read core/rag/config_extractor.py and core/architecture_extractor.py
   side by side -- understand exactly what each produces.
3. Run `pytest tests/ -q` to confirm the 1697-passing baseline before
   touching anything.
4. Work through 2.1 first (it's the highest-leverage fix and everything
   else builds on it being solid) using the direct-Python-then-live-HTTP
   verification rhythm from Phase 1.5 -- fast iteration first, full stack
   second.
5. After each numbered section, run the full suite and update the memory
   doc's running notes before moving to the next section.
