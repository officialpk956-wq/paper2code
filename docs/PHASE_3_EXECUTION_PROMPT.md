OBJECTIVE: Ground the extraction pipeline in real evidence. Every extracted
architecture parameter must either cite the specific paper chunk(s) that
justified it, or be honestly labeled as inferred/default. Build the
persistent chunk store with real page/section provenance that citation
requires, wire actual dense retrieval into the existing (currently unused)
Qdrant infrastructure, and let a lightweight knowledge graph expand what
gets retrieved rather than only checking a graph after the fact.

PROJECT: paper2code (Next.js + FastAPI/Celery backend)
CURRENT STATE: Phase 1, Phase 1.5, and Phase 2 (implementation + two audit
rounds + final closure) are complete and committed (commit a15e365).
Extraction (ConfigExtractor, primary), code generation (known-family
builders + LLM fallback), sandboxed validation (E2B, real forward pass),
and a bounded repair loop all genuinely work, proven via repeated live
HTTP uploads. Read docs/PAPER_TO_CODE_EXECUTION_MEMORY.md in full before
starting -- it has the exact bugs found and fixed across three rounds of
work, and the live-verification method that keeps finding real problems
mocks would have hidden. Read docs/PAPER_TO_CODE_MASTER_PLAN.md's Phase 4
section -- this prompt scopes that phase concretely against what's
actually built today, not what was assumed at the start.

SCOPE: This phase is intentionally split into two halves with a hard
boundary between them. Do NOT start half B until half A is fully
live-verified. Half A needs no new infrastructure. Half B needs a real
Qdrant instance, which may not exist yet -- see QUESTIONS FOR
CLARIFICATION before touching it.

- Half A (3.1 + 3.2): chunk persistence with real page/section provenance,
  citation tracking through extraction. No new infra.
- Half B (3.3 + 3.4 + 3.5): dense retrieval, hybrid retrieval, KAG-driven
  expansion. Needs Qdrant.
- 3.6 (minimal provenance surface in the API/UI) can be done incrementally
  alongside either half, once there's real citation data to surface.

DO NOT COMMIT OR PUSH. Preserve the Dojo modification in
src/app/(protected)/dojo/page.tsx untouched. Do not regress the current
passing baseline -- run the full suite after every numbered section, not
just at the end.

================================================================================
BACKGROUND YOU NEED (verified by direct code inspection, not assumed)
================================================================================

**The page-provenance data is already being thrown away today.**
`core/paper_to_code_generator.py`'s `from_pdf()` extracts text page by
page via pdfplumber (`for page in pdf.pages[:30]: text = page.extract_text()`),
then immediately destroys the page boundaries:
`raw_text = "\n\n".join(text_pages)`. Every downstream consumer
(`process_text`, `ConfigExtractor.extract_from_text`) only ever sees one
flat blob of text with no way to know which page anything came from. This
is the actual, concrete reason "page" provenance doesn't exist yet -- it's
not that nobody built a chunk store, it's that the *page numbers
themselves* never survive past this one join. Fixing this join (keep a
parallel list of (page_number, text) instead of flattening early) is a
prerequisite for any real page-level citation, and it touches a
function every existing test already exercises -- be careful and re-run
the full suite immediately after this specific change before doing
anything else.

**A working local embedding pipeline already exists and is barely used.**
`backend/services/vector_service.py` has `embed_text(text)` (sentence-
transformers, `BAAI/bge-small-en-v1.5`, 384-dim, cosine distance),
`index_paper(paper_id, title, abstract, authors)`,
`semantic_search(query, limit)`, and `delete_paper(paper_id)`. All of it
degrades gracefully to no-ops when `QDRANT_URL` is unset (confirmed by
reading `_get_qdrant()` -- returns `None`, callers check for that). Do not
introduce a second embedding model or a second vector store client. Reuse
this file; add functions to it (e.g. `index_chunk`, `search_chunks`) rather
than duplicating its Qdrant/embedder bootstrapping.

**ConfigExtractor's consistency was hard-won -- do not casually rewrite
its core prompt or flow.** Phase 2's audits spent significant effort
tracing a 20/8/20-layer inconsistency back to a rate-limit-driven
cross-provider fallback in `core/llm_client.py` (now fixed with a
retry-before-fallback mechanism). Citation tracking must be *additive* to
`ConfigExtractor.extract_from_text()` -- a new step that runs after or
alongside existing extraction, not a rewrite of the extraction prompt
itself, or you risk reintroducing the exact non-determinism that was just
closed out. If you do need to touch the core extraction prompt, re-run
`tests/test_phase2_config_extractor.py::test_config_extractor_real_llm_path_is_consistent`
(gated behind `RUN_LIVE_PHASE2=1` + real `GROQ_API_KEY`) before and after
your change and compare.

================================================================================
HALF A -- CHUNKING, PERSISTENCE, CITATION (no new infra)
================================================================================

**3.1 Section-aware chunks with real provenance**

ACTION:
  a) Fix the page-boundary loss described above: `from_pdf()` should keep
     `text_pages` as `list[tuple[int, str]]` (page number, text) instead
     of flattening to one string before handing off to `_run_pipeline`.
     `_run_pipeline` and everything downstream currently takes a single
     `text: str` -- decide whether to thread the per-page structure all
     the way through, or to chunk immediately after PDF extraction (before
     `_run_pipeline` runs) and pass chunks alongside the flattened text
     for backward compatibility with the existing extraction path. The
     latter is very likely less invasive -- investigate both, pick the
     smaller diff, and say which you chose and why in the memory doc
     before implementing.
  b) Add a `PaperChunk` SQLAlchemy model to `backend/models.py`: paper_id
     (FK), section (str -- reuse the existing section vocabulary from
     `core/section_splitter.py`'s `classify_section`: abstract,
     introduction, related_work, method, experiments, results, discussion,
     conclusion, other), page (int, nullable -- some chunks may span pages
     or come from a source where page tracking wasn't possible), chunk_type
     (str: text/equation/table/caption -- start with "text" only if
     equation/table/caption extraction isn't already separated elsewhere;
     check `build_ingestion_payload` in
     `backend/services/paper_ingestion_service.py` first, since it may
     already separate these), text (Text), source_offset_start/end (int,
     nullable), embedding_id (str, nullable -- the Qdrant point ID once
     Half B indexes it), created_at. New Alembic migration, tested
     upgrade→downgrade→upgrade on a throwaway SQLite copy before treating
     it as done (same discipline as every prior migration this project).
  c) Persist chunks during `ingest_pdf_paper` (or wherever you decide in
     (a) chunking should happen) -- one row per chunk, not one giant blob.

VERIFY: Direct Python reproduction first (load a real fixture PDF,
confirm chunks persist with correct page numbers), then a live HTTP
upload, then check the DB directly that rows exist with real page
numbers, not all `NULL`.

**3.2 Citation tracking through extraction**

PROBLEM: Nothing today tracks which chunk justified `num_heads: 8` vs.
which value was just a schema default. The master plan's exit gate for
this whole phase is literally this capability.

ACTION:
  a) Design decision, investigate before implementing: how does a
     citation get attached to an extracted field? Recommended approach
     (cheaper and more hallucination-resistant than asking the LLM to
     output chunk IDs directly, which it cannot know and would invent):
     ask the LLM to additionally return a short supporting quote/phrase
     per extracted field it's confident about, then verify that phrase
     is actually a substring (or close fuzzy match -- decide a reasonable
     threshold) of one of the real chunks from 3.1. Only mark a field as
     "cited" if that verification passes; otherwise mark it "inferred" or
     "default" (matching the master plan's own three-way distinction).
     This keeps citations honest by construction instead of trusting the
     LLM's own claim.
  b) This needs a schema addition to what `extract_from_text`/
     `extract_architecture` return -- a parallel `evidence: dict[str, {
     status: "cited"|"inferred"|"default", chunk_ids: list[int] }]`
     structure alongside the existing spec. Store it in
     `verification_report` (the existing JSON column already used for
     repair-attempt history -- reuse the same field, add a new top-level
     key, no new migration needed for this part) or a new column if you
     have a clear reason the existing one is the wrong place -- default to
     reuse unless you can justify otherwise.
  c) Don't touch ConfigExtractor's core extraction call for this -- add
     it as a follow-up step that takes the already-extracted spec plus
     the chunks and produces the evidence map. Keep it separable so a
     failure in citation-checking degrades to "everything inferred"
     rather than breaking extraction itself.

VERIFY: Direct reproduction with a real paper excerpt where you know which
sentence justifies which field (e.g. the same ResNet-50 excerpt already
used throughout Phase 1.5/Phase 2 testing -- "Stage 1 has 3 bottleneck
blocks with 64 channels" should cite a specific chunk for `stages[0]`).
Confirm fields with no textual support are honestly marked, not silently
cited to the nearest unrelated chunk.

------------------------------------------------------------------------------
HALF A EXIT GATE: every field in a real extraction result has an evidence
entry (cited-with-chunk-ids, inferred, or default) -- verified against at
least 3 different real paper excerpts, not one cherry-picked case. Full
test suite still green. Update the memory doc with what you built, the
chunking-location decision from 3.1(a), and results before starting Half B.
------------------------------------------------------------------------------

================================================================================
HALF B -- DENSE + HYBRID RETRIEVAL, KAG EXPANSION (needs Qdrant)
================================================================================

Do not start this section until Half A is verified and Qdrant availability
is confirmed (see QUESTIONS FOR CLARIFICATION). If Qdrant isn't available
in this environment, stop and report that rather than guessing at a
workaround -- the existing code already degrades gracefully with it
absent, so there's no pressure to fake it.

**3.3 Real dense retrieval on chunks**

ACTION:
  a) Extend `backend/services/vector_service.py` with `index_chunk(chunk_id,
     paper_id, text, section, page)` and `search_chunks(query, limit,
     paper_id=None, section=None)` -- reuse `embed_text()` and the
     existing Qdrant bootstrapping (`_get_qdrant()`, `_ensure_collection()`),
     following the same pattern as `index_paper`/`semantic_search`. A
     second Qdrant collection for chunks (distinct from the existing
     "papers" collection) is probably right -- confirm by reading
     `_ensure_collection`'s current single-collection assumption before
     deciding whether to parameterize it or add a second one.
  b) Index every chunk from 3.1 at persistence time (or backfill existing
     ones with a one-off script -- don't require a full re-upload of
     every paper already in the DB).
  c) Store the returned Qdrant point ID back on the `PaperChunk` row's
     `embedding_id`.

VERIFY: A semantic query that has no exact keyword overlap with the
target chunk's text (proving it's genuinely dense/semantic, not just BM25
in disguise) returns that chunk in the top results.

**3.4 Hybrid retrieval**

ACTION: Combine 3.3's dense search with the existing BM25 mechanism
already inside `ConfigExtractor` (read how it currently decides "text is
too large, use BM25" before extending it) plus architecture-family
filtering (`classify_architecture`/`infer_family_from_name`, both already
built and tested in `core/classification.py`) plus a lightweight rerank.
Keep the rerank simple -- a weighted combination of dense score + BM25
score + family match is a reasonable start; don't reach for a separate
cross-encoder reranker model unless the simple version demonstrably
under-performs on real test cases.

VERIFY: Compare hybrid retrieval's top-k against dense-only and BM25-only
on the same 3+ real paper excerpts from Half A's verification -- hybrid
should not be worse than either alone on any of them.

**3.5 KAG-driven expansion**

PROBLEM: `core/rag/knowledge_graph.py`'s `KnowledgeGraph` class only
validates a graph *after* it's built (`detect_motifs`, `verify_topology`).
The master plan's actual ask is for KAG to expand *retrieval* -- e.g.
extraction evidence mentions "residual connection," KAG should know to
also retrieve/consider "skip connection" and "shape-preserving projection"
as related concepts, the same way `get_semantic_role` already maps node
*types* to semantic roles.

ACTION: Read `get_semantic_role`'s existing rule structure first -- it's
very likely the right pattern to extend (a lookup table of concept →
related concepts) rather than building something new. Add a
`related_concepts(concept: str) -> list[str]` function using the same
deterministic, rule-based style (no new LLM calls for this -- the master
plan is explicit that KAG should not become "an additional opaque LLM
layer"). Wire it into 3.4's retrieval: when a chunk or extracted field
matches a known concept, also search for its related concepts.

VERIFY: A paper excerpt describing "skip connections" without ever using
the word "residual" should still surface content tagged with the
"residual" concept family when a resnet-family query expansion runs.

------------------------------------------------------------------------------
HALF B EXIT GATE: hybrid retrieval demonstrably outperforms dense-only and
BM25-only individually on the same test excerpts; KAG expansion
demonstrably surfaces related concepts a keyword-only search would miss.
------------------------------------------------------------------------------

================================================================================
3.6 -- MINIMAL PROVENANCE SURFACE (incremental, either half)
================================================================================

Once real evidence data exists (end of Half A onward), extend the paper
detail API response (`backend/routers/papers.py`, same pattern as the
`generated_code_source`/`verification_report` fields already added there)
to include the evidence map, and add a minimal display in the workspace UI
-- check the existing tab structure in
`src/app/(protected)/papers/[id]/WorkspacePaperClient.tsx` first; the
Knowledge Graph or Blueprint tab is likely the right home for this rather
than a new tab. Keep this small: a field's value plus "cited (page N)" or
"inferred" is enough for this phase. Richer citation UI (jump-to-source,
highlighted PDF viewer) is out of scope here.

================================================================================
ACCEPTANCE CRITERIA (Phase 3 exit gate)
================================================================================

- `PaperChunk` persists with real, non-null page numbers for the common
  case (a normal, non-scanned PDF).
- Every extracted field in a real extraction result carries an honest
  evidence status (cited/inferred/default), verified against 3+ real
  paper excerpts covering at least 2 different architecture families.
- Hybrid retrieval (dense + BM25 + family filter) outperforms either
  signal alone on the same test set.
- KAG expansion surfaces at least one genuinely related concept a
  keyword-only search would have missed, on a real example.
- Full test suite green throughout -- run it after every numbered
  section, not just at the end. New regression tests for each piece
  (a chunk-persistence test, a citation-verification test with a known
  cited/inferred/default split, a hybrid-vs-single-signal retrieval
  test) -- follow the existing `tests/test_phase2_*.py` naming and
  `@pytest.mark.live` + `RUN_LIVE_*` gating conventions for anything that
  needs real Groq/Qdrant/E2B access, so normal `pytest tests/` stays
  hermetic.
- docs/PAPER_TO_CODE_EXECUTION_MEMORY.md updated with what changed, the
  Half A chunking-location decision and why, live verification results,
  and the exact next-phase pointer -- same discipline as every prior
  phase.

================================================================================
SAFETY / ROLLBACK
================================================================================

- No commit, no push.
- The new `PaperChunk` migration must be reversible -- test
  upgrade→downgrade→upgrade on a throwaway SQLite copy before treating it
  as done.
- Watch for the exact bug pattern that caused most of Phase 2's fixes:
  "explicit `None` survives `setdefault`/dict-merge/hasattr-guard" --
  apply the same scrutiny to new code in this phase (chunk fields that
  might be `None` -- page number, offsets -- are exactly the kind of
  value that pattern bites). Add a defensive test the moment you touch
  anything that could receive `None` where a real value is expected.
- Verify claims with real, non-mocked calls wherever the memory doc
  indicates a real integration point exists -- this session's whole
  history is tests passing while the real thing was mocked at exactly the
  boundary that mattered. Don't repeat it in this phase's own new tests.
- Clean up: any Qdrant collection created during testing, any synthetic
  test papers/users in the local DB, any background processes/containers
  started for live verification -- stopped/removed before finishing.

================================================================================
QUESTIONS FOR CLARIFICATION
================================================================================

Pause and ask before proceeding if any of the following is unclear:
1. Is a real Qdrant instance available for Half B? (Cloud free tier at
   cloud.qdrant.io, or self-hosted via the existing `docker-compose.yml`
   -- check whether it already has a qdrant service defined; if not,
   adding one is reasonable.) If not available, Half A alone is still a
   complete, valuable, shippable phase -- don't block on this.
2. Confirm `sentence-transformers` and its model weights
   (`BAAI/bge-small-en-v1.5`) are actually installed/downloadable in this
   environment -- `_get_embedder()` already degrades gracefully if not,
   but Half B's exit gate needs it working for real, not degraded.
3. Chunk granularity: paragraph-level, sentence-level, or fixed-token-
   window? The master plan doesn't specify. Paragraph-level is a
   reasonable default (matches how `classify_section` already chunks
   text via `core/utils.py`'s `chunk_text` -- check its current chunking
   strategy before inventing a new one).
4. Whether citation verification's fuzzy-match threshold should be
   configurable or hardcoded for this pass -- lean toward hardcoded with
   a named constant unless you find a real reason it needs to vary.

================================================================================
START HERE
================================================================================

1. Read docs/PAPER_TO_CODE_EXECUTION_MEMORY.md in full.
2. Read docs/PAPER_TO_CODE_MASTER_PLAN.md's Phase 4 section.
3. Read `core/paper_to_code_generator.py`'s `from_pdf()` and confirm the
   page-boundary-loss finding above still matches current code.
4. Read `core/utils.py`'s `chunk_text` and `backend/services/
   paper_ingestion_service.py`'s `build_ingestion_payload` -- both likely
   already do partial versions of what 3.1 needs; extend rather than
   duplicate.
5. Run `pytest tests/ -q -m "not live"` to confirm the current clean
   baseline before touching anything.
6. Work Half A in order (3.1 then 3.2), verifying live after each step
   before moving to the next.
