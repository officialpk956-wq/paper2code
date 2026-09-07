# Phase 4 — Execution Prompts (RAG/KAG hardening + operation grounding)

**How to use this file.** Each prompt below is standalone and copy-pasteable.
Give Codex **ONE prompt at a time**, in the run order in §2. Always paste
**§0.5 (Universal Preamble)** immediately above the prompt you're running —
it is the anti-hallucination guard and it is not optional.

Do not paste two prompts at once. Do not skip the verification gate at the
end of a prompt. If a gate fails, fix it before moving on.

---

## §0 — STALE GAP-TABLE CORRECTION (read this first)

The gap table that motivated this file was written **before** Phase 3 Half B
landed. Several rows in it are **already implemented, tested, and live-verified**.
If you hand Codex that table as-is, it will rebuild working code and break it.

| Gap-table row | Reality as of now |
|---|---|
| Dense retrieval: `index_chunk()` / `search_chunks()` | **DONE.** Both exist in `backend/services/vector_service.py`, wired into `backend/tasks/paper_tasks.py` Stage 5. |
| Dense retrieval: *backfill existing papers* | **STILL MISSING** → Prompt 3 |
| BM25: search persisted chunks + combine with dense | **DONE.** `hybrid_search_chunks()` + `_bm25_and_family_scores()`. |
| Hybrid ranking: dense + BM25 + family match | **DONE.** Weighted `0.5*dense + 0.4*bm25 + 0.1*family`. |
| Hybrid ranking: *diversity selection* | **STILL MISSING** → Prompt 7 |
| KAG: `related_concepts()` | **DONE.** Plus `infer_family_from_concept()`. |
| KAG: *query expansion into retrieval terms* | **STILL MISSING** → Prompt 6 (current code infers a *family*, it does not expand *query terms*) |
| Structured extraction consumes chunk bundles | **DONE.** `ConfigExtractor.extract_from_text(text, source_chunks=...)` + injected `chunk_retriever`. |
| Evidence: retrieval supplies chunks before extraction | **DONE.** Via `chunk_retriever` in `_focus_text`. |
| Evidence: *multi-paper live verification* | **STILL MISSING** → Prompt 9 |
| PDF→chunks: *table/caption/equation as first-class chunks* | **STILL MISSING** → Prompt 4 |
| PDF→chunks: *OCR for scanned PDFs* | **STILL MISSING** → Prompt 5 |
| Code generation: *architecture-fidelity checks* | **STILL MISSING** → Prompt 8 |
| Evaluation: *labeled benchmark* | **STILL MISSING** → Prompt 9 |
| Agent layer: bounded coordinator | **DEFERRED ON PURPOSE.** Do not build. Measure hybrid RAG first. |
| Workspace transparency UI | **STILL MISSING** → Prompt 10 |

New work not in that table at all: **the operation knowledge table**
(formula + syntax + aliases) → Prompts 1 and 2. This is the highest-value
item in the file; run it first.

**Current baseline: 1746 passed, 2 skipped, 0 failed** (`pytest -m "not live"`).
Any prompt that ends with fewer than 1746 passing has broken something.

---

## §0.5 — UNIVERSAL PREAMBLE (paste this above EVERY prompt)

```
GROUND RULES — read before writing any code.

1. REALITY CHECK FIRST. This prompt states exact file paths, function
   signatures, and line ranges. Before editing, open each named file and
   confirm what the prompt claims is actually there. If ANY signature,
   constant, or file path in this prompt does not match the real repo,
   STOP and report the mismatch. Do not "fix" the discrepancy by
   inventing a plausible alternative.

2. DO NOT INVENT APIs. Do not call functions, kwargs, model fields, env
   vars, or library methods that you have not read with your own eyes in
   this repo. If you need something that does not exist, say so instead
   of writing a call to it and hoping.

3. NO REWRITES. Make the smallest change that satisfies the task. Do not
   refactor, rename, reformat, reorder imports, or "clean up" adjacent
   code. Do not touch files outside the ALLOWED FILES list.

4. NO NEW DEPENDENCIES unless the prompt explicitly names one. Prefer
   stdlib, then something already in requirements.txt.

5. TESTS ARE NOT OPTIONAL. Every prompt has a VERIFY block with real
   commands. Run them. Paste the real output in your report. Never
   claim a test passed without showing its output. A test that mocks
   the thing it is supposed to be testing does not count as coverage —
   this project has been bitten by that repeatedly.

6. NULL SAFETY. This codebase has a recurring bug class: an explicit
   `None` surviving a merge / `setdefault` / `hasattr` guard and
   crashing downstream. When you add a field that can legitimately be
   None (page numbers, offsets, shapes), guard with
   `x is not None`, never `hasattr(obj, "x")` and never a bare truthiness
   check that would also swallow `0`.

7. GRACEFUL DEGRADATION. Anything touching an external service (Qdrant,
   the embedder, OCR, an LLM) must degrade to a working fallback, never
   hard-fail the pipeline. Follow the existing pattern in
   backend/services/vector_service.py: log a warning, return an empty /
   passthrough result.

8. DO NOT COMMIT. Do not run `git commit`, `git push`, `git reset`, or
   `git checkout --`. Leave all changes in the working tree. The user
   commits manually.

9. REPORT HONESTLY. If something does not work, say it does not work and
   show the error. Do not report partial work as complete. If you ran out
   of context or had to skip a step, say exactly which step.

ENVIRONMENT
  Windows, PowerShell primary, Git Bash available. Python venv at
  .venv/Scripts/python.exe . Project root C:\papper2code .
  Qdrant for live tests:  docker compose up -d qdrant
  then export QDRANT_URL=http://localhost:6333
  Full suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
  Baseline that must not regress: 1746 passed, 2 skipped.
```

---

## §1 — GROUND TRUTH: what already exists

Give Codex this section verbatim whenever a prompt references these. These
are real, current signatures — not aspirational.

**`backend/services/vector_service.py`**
```python
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "papers")
CHUNKS_COLLECTION = os.getenv("QDRANT_CHUNKS_COLLECTION", "paper_chunks")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
VECTOR_DIM        = 384

_get_qdrant()                      -> client | None   # lazy; None if QDRANT_URL unset
_get_embedder()                    -> SentenceTransformer | None
_ensure_collection(client, collection_name)
embed_text(text)                   -> list[float] | None
index_paper(paper_id, title, abstract, authors="")            -> bool
semantic_search(query, limit=10)                              -> list[int]
index_chunk(chunk_id, paper_id, text, section="other", page=None) -> bool
search_chunks(query, limit=10, paper_id=None, section=None)   -> list[dict]
_resolve_family(query, family)                                -> str | None
_bm25_and_family_scores(query, texts, family)  -> (bm25_norm: list[float], family_bonus: list[float])
hybrid_search_chunks(query, chunks, limit=10, paper_id=None, family=None) -> list[dict]
hybrid_rank_texts(query, texts, top_k=6, family=None)         -> list[str]
delete_paper(paper_id)                                        -> bool
_FAMILY_TERMS: dict[str, list[str]]   # family -> characteristic vocabulary
```
Scoring formula (both hybrid fns): `0.5*dense + 0.4*bm25_norm + 0.1*family_bonus`.
`query_points()` is the correct Qdrant call — **`client.search()` does not exist**
on the installed client version.

**`core/rag/knowledge_graph.py`** — class `KnowledgeGraph`, `networkx.DiGraph`
```python
_CONCEPT_ALIASES: dict[str, str]   # "skip connections" -> "residual_add"
_NODE_FAMILY:     dict[str, str]   # "residual_add" -> "resnet"
related_concepts(concept)          -> list[str]
infer_family_from_concept(text)    -> str | None
get_semantic_role(node_type)       -> str | None
get_context_for_terms(terms)       -> str      # emitted into the extraction prompt
identify_terms(text)               -> list[str]
detect_motifs(arch_graph)          -> list[str]
verify_topology(arch_graph)        -> list[str]
```

**`core/rag/config_extractor.py`**
```python
_ARCHITECTURE_QUERY: str          # natural-language stand-in for BM25 arch terms
_LAYER_PATTERNS:     list[tuple[str, str]]   # regex -> canonical type
_LLM_EXTRACTION_PROMPT: str       # has {few_shot} {graph_rules} {text} slots
_VERIFICATION_PROMPT:   str

class ConfigExtractor:
    __init__(use_llm=True, use_section_splitter=True, use_retriever=True,
             verify=True, max_context_chars=10_000, chunk_retriever=None)
    extract_from_text(text, source_chunks=None) -> ConfigDict
    _focus_text(text, source_chunks=None)       -> str
    _extract_with_llm(text)                     -> dict   # builds graph_rules via self.ontology
    _verify_extraction(original_text, extracted)-> dict
    _extract_rule_based(text)                   -> dict
```
`chunk_retriever` signature: `(query: str, texts: list[str], top_k: int) -> list[str]`.
Production impl is `_chunk_retriever` in `backend/services/paper_ingestion_service.py`,
wrapping `vector_service.hybrid_rank_texts`.

**`core/codegen.py`** — `_generate_skeleton(graph)`, `_node_to_layer(node)`.
`MAP` dict lives at **lines 128–147**; `return MAP.get(node.type, None)` at line 149.
Unrecognized types emit a `# x unchanged:` passthrough comment, never a crash.

**`core/rag/normalizer.py`** — `CANONICAL_TYPES: set[str]` (~37 entries),
`_SYNONYM_MAP: dict[str,str]`, `_normalize_type(layer_type)`.
`_normalize_type` **must not raise** on unknown types — it logs and passes through.

**`core/rag/retriever.py`** — `class BM25(corpus, k1=1.5, b=0.75)` with
`.score(query_terms, doc_idx)` and `.get_top_k(query_terms, top_k)`;
`_tokenize(text)`, `retrieve_top_chunks(chunks, top_k, query_terms=None)`,
`retrieve_and_merge(...)`, `_ARCH_QUERY_TERMS: list[str]`.

**`backend/models.py`** — `PaperChunk`: `id, paper_id (FK cascade), section,
page (nullable), chunk_type (default "text"), text, source_offset_start,
source_offset_end, embedding_id (nullable), created_at`.
Indexes on `(paper_id, section)` and `(paper_id, page)`.

**`core/utils.py`** — `chunk_pages_with_provenance(pages: list[tuple[int,str]], max_chars=1200) -> list[dict]`
returning dicts with `text / section / page / chunk_type / source_offset_start / source_offset_end`.

**`core/evidence_tracking.py`** — `build_evidence_map(spec, chunks, complete=None)`,
`FUZZY_QUOTE_THRESHOLD = 0.92`. Quotes are verified against real persisted chunk
text before being marked `cited`; unverified stays `inferred`.

**`backend/services/paper_ingestion_service.py`** — `extract_figures(pdf_bytes, page_texts)`,
`extract_equations(page_texts)`, `_chunk_retriever(...)`, `_GENERATOR = PaperToCodeGenerator(chunk_retriever=_chunk_retriever)`.
Text extraction uses pdfplumber with a **PyMuPDF (`fitz`) fallback**.

**`verification_report`** keys: `passed, error, entrypoint_class, attempts,
total_attempts, final_attempt, evidence`.

**Test conventions** — `pytest.ini` registers exactly one custom marker: `live`.
Live tests gate on `@pytest.mark.live` plus a `skipif` on the relevant env var.
**Never** use `monkeypatch.delenv("QDRANT_URL")` — module-level constants are
cached at import and delenv poisons them for the whole session. Use
`monkeypatch.setattr(vector_service, "QDRANT_URL", "")`.

---

## §2 — RUN ORDER & LOOP PROTOCOL

Run in this order. Later prompts assume earlier ones landed.

| # | Prompt | Depends on | Risk |
|---|---|---|---|
| 1 | Operation knowledge table | — | low |
| 2 | Wire op table into codegen + extraction prompt | 1 | medium |
| 3 | Backfill chunk embeddings for existing papers | — | low |
| 4 | Table / caption / equation chunks | — | medium |
| 5 | OCR fallback for scanned PDFs | 4 | medium |
| 6 | KAG query-term expansion | — | medium |
| 7 | Diversity selection in hybrid ranking | 6 | low |
| 8 | Architecture-fidelity checks | — | high |
| 9 | Labeled benchmark + multi-paper live verification | 1–8 | medium |
| 10 | Workspace transparency UI | 3,4,8 | low |

**Loop protocol — after EVERY prompt, in this order:**
1. Run the prompt's VERIFY block. Paste real output.
2. Run the full gate: `.venv/Scripts/python.exe -m pytest -q -m "not live"`.
   Must be ≥ 1746 passed, 0 failed.
3. Append a dated section to `docs/PAPER_TO_CODE_EXECUTION_MEMORY.md`:
   what was built, what was found, exact root cause of any bug, what is
   still open. Match the existing style in that file.
4. Report to the user: what landed, what the gate said, what you skipped.
5. **Stop.** Wait for the next prompt. Do not chain into the next one.

---

## PROMPT 1 — Operation Knowledge Table (formula + syntax + aliases)

> Copy from here ↓

**CONTEXT — why this exists.**
`core/codegen.py` lines 128–147 already has a `MAP` dict from canonical layer
names to PyTorch constructor syntax (`"sigmoid": "nn.Sigmoid()"`,
`"gelu": "nn.GELU()"`, `"layernorm": f"nn.LayerNorm({in_hs})"`, …). It is used
as a deterministic fallback so the LLM is not trusted to write boilerplate for
well-known ops.

Two things are missing:
1. **The formula half.** The map knows the *syntax* but not the *mathematics*.
2. **It runs too late.** `MAP` is only consulted at codegen time — *after* the
   LLM has already extracted the spec. It does nothing to stop the LLM
   hallucinating an operation's definition *during* extraction.

This prompt builds the data structure. Prompt 2 wires it in. Build only the
table here.

**Design intent (do not deviate):** a single hand-curated table
`{canonical_name: {formula, syntax, aliases, notes}}` for operations common
enough to be worth curating. This raises the floor on common ops; a paper's
novel custom operation still falls through to the LLM as today. Do **not**
attempt to cover every op — a wrong entry is worse than a missing one.

**ALLOWED FILES**
- create `core/knowledge/__init__.py`
- create `core/knowledge/operations.py`
- create `tests/test_operation_knowledge.py`

**DO NOT TOUCH** `core/codegen.py`, `core/rag/config_extractor.py`,
`core/rag/normalizer.py`, or anything under `backend/` in this prompt.

**TASK**

1. Read `core/codegen.py` lines 105–150 (the `MAP` dict and `_node_to_layer`)
   and `core/rag/normalizer.py` lines 17–57 (`CANONICAL_TYPES`).

   **`OPERATIONS` and `CANONICAL_TYPES` are deliberately different sets.**
   They answer different questions and neither contains the other:
   - `CANONICAL_TYPES` = "what can be a **node in the architecture graph**"
   - `OPERATIONS` = "what **mathematical operations** do we have a grounded
     definition for"

   `sigmoid`, `tanh`, and `softmax` are legitimate operations that are not
   currently graph node types. `scaled_dot_product_attention` is the
   mechanism *inside* `multiheadattention` and is never its own node. All
   four belong in `OPERATIONS` and must **not** be added to
   `CANONICAL_TYPES` in this prompt.

   The only rule that binds: **where a name appears in BOTH sets, the
   spelling must be identical.** That is exactly what VERIFY test 4 checks
   (it parametrizes over the intersection, so entries absent from
   `CANONICAL_TYPES` are simply not checked). Do not add, remove, or edit
   anything in `core/rag/normalizer.py` — it is not in ALLOWED FILES.

2. Create `core/knowledge/operations.py` with a module-level dict:

```python
OPERATIONS: dict[str, dict] = {
    "sigmoid": {
        "formula": "sigma(x) = 1 / (1 + exp(-x))",
        "latex": r"\sigma(x) = \frac{1}{1 + e^{-x}}",
        "syntax": "nn.Sigmoid()",
        "functional": "torch.sigmoid(x)",
        "aliases": ["sigmoid", "logistic", "logistic function"],
        "output_range": "(0, 1)",
        "notes": "Elementwise. Do not use as a final layer with BCEWithLogitsLoss.",
    },
    ...
}
```

Every entry MUST have: `formula` (plain ASCII), `latex`, `syntax` (the
`nn.*` constructor exactly as `core/codegen.py`'s `MAP` would write it, or
`None` if the op has no module form), `functional` (the `torch.*` /
`F.*` call, or `None`), `aliases` (lowercase strings), `notes`.
`output_range` is optional — include for activations only.

3. Cover exactly these operations, no more:

*Activations:* `sigmoid`, `relu`, `leakyrelu`, `gelu`, `silu`, `swish`,
`tanh`, `softmax`.
*Normalization:* `batchnorm2d`, `layernorm`, `groupnorm`, `rmsnorm`.
*Attention:* `scaled_dot_product_attention`, `multiheadattention`.
*Structural:* `linear`, `conv2d`, `depthwise_conv2d`, `dropout`,
`residual_add`, `concat`, `flatten`, `globalavgpool2d`, `patchembedding`.

Correctness requirements (get these exactly right — they are the whole point):
- `gelu` — give the exact tanh approximation, and note that
  `nn.GELU()` defaults to the exact erf form, `nn.GELU(approximate="tanh")`
  for the approximation.
- `layernorm` — the formula MUST include the `eps` term in the denominator,
  and `notes` must state the PyTorch default `eps=1e-5`.
- `rmsnorm` — must be the no-mean-subtraction form; note it is not
  `layernorm`.
- `scaled_dot_product_attention` — formula must include the `1/sqrt(d_k)`
  scaling and the softmax over the last axis.
- `softmax` — `notes` must state that `dim` is required in practice and that
  PyTorch warns when it is omitted.
- `silu` and `swish` — same math; `notes` on `swish` must say it is an alias
  of SiLU (beta=1) and that `CANONICAL_TYPES` carries both.
- `depthwise_conv2d` — `syntax` must show the `groups=in_channels` form.

4. Add a lookup function:

```python
def lookup(term: str) -> dict | None:
    """Resolve a free-form term to its operation entry via exact canonical
    name or alias match. Case/whitespace/hyphen insensitive. Returns None
    for unknown terms -- callers fall back to the LLM."""
```
Normalize with `re.sub(r"[\s\-_]+", "", term.lower())` on both sides, matching
the style already used by `_normalize_type` in `core/rag/normalizer.py`.
Build the alias index **once at import** into a module-level dict; do not
rescan `OPERATIONS` on every call.

5. Add:
```python
def find_mentioned(text: str) -> list[str]:
    """Return canonical names whose alias appears in `text`, as whole words.
    Deterministic, no LLM. Ordered by first appearance in the text."""
```
Use `re.escape` on aliases and word boundaries. An alias containing a space
must still match. Do not return duplicates.

**VERIFY** — create `tests/test_operation_knowledge.py` with these tests, and
they must genuinely assert content, not just "the dict is non-empty":

1. Every entry has all required keys, and `aliases` is a non-empty list of
   lowercase strings.
2. No alias is claimed by two different canonical names (assert the alias
   index size equals the total alias count — a collision means one silently
   shadows the other).
3. Every `syntax` value that is not `None` starts with `"nn."`, and every
   `functional` that is not `None` starts with `"torch."` or `"F."`.
4. Every canonical name that is also in
   `core.rag.normalizer.CANONICAL_TYPES` uses the identical spelling
   (parametrize over the intersection).
5. `lookup("Logistic Function")`, `lookup("logistic_function")` and
   `lookup("sigmoid")` all return the same entry.
6. `lookup("some_novel_op_xyz")` is `None`.
7. `find_mentioned("We apply layer normalization then a GELU activation.")`
   returns `["layernorm", "gelu"]` in that order.
8. `find_mentioned("")` returns `[]`.
9. Assert the specific correctness requirements from step 3: `"eps"` appears
   in `layernorm`'s formula, `"sqrt"` in `scaled_dot_product_attention`'s,
   `"groups"` in `depthwise_conv2d`'s syntax.

Run:
```
.venv/Scripts/python.exe -m pytest tests/test_operation_knowledge.py -q
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — new test file fully green; full suite ≥ 1746 passed, 0 failed
(this prompt adds no wiring, so nothing else may change).

> Copy to here ↑

---

## PROMPT 2 — Wire the operation table into codegen + extraction

> Copy from here ↓

**PREREQ:** Prompt 1 landed; `core/knowledge/operations.py` exists with
`OPERATIONS`, `lookup()`, `find_mentioned()`.

**CONTEXT.** Two consumers, two different jobs:
- **Codegen (already partly solved):** `core/codegen.py`'s `MAP` at lines
  128–147 covers ~17 types; `_node_to_layer` returns `None` for anything else
  and `_generate_skeleton` emits a `# x unchanged:` passthrough. Ops that
  `OPERATIONS` knows but `MAP` does not (`tanh`, `softmax`, `groupnorm`,
  `rmsnorm`, `silu`, `flatten`, …) currently become silent no-ops.
- **Extraction (the real win):** `_LLM_EXTRACTION_PROMPT` in
  `core/rag/config_extractor.py` already has a `{graph_rules}` slot fed by
  `KnowledgeGraph.get_context_for_terms(terms)`. Nothing tells the LLM what
  the mentioned operations actually *are*. Adding grounded formula+syntax
  context for ops the paper mentions is the point of this prompt.

**ALLOWED FILES**
- `core/codegen.py`
- `core/rag/config_extractor.py`
- `tests/test_operation_knowledge.py` (extend)
- `tests/test_phase2_config_extractor.py` (extend)

**DO NOT** change `_LLM_EXTRACTION_PROMPT`'s existing rules text, few-shot
examples, or the `{graph_rules}` mechanism. You are **adding** a slot, not
rewriting the prompt. This prompt has been tuned; a rewrite will regress
extraction quality and is explicitly out of scope.

**TASK — part A: codegen fallback**

1. In `core/codegen.py`, in `_node_to_layer`, keep `MAP` as the **first**
   lookup (it is parameterized with real channel/head values — `OPERATIONS`
   is not, and must never override it). Only when `MAP.get(node.type)` is
   `None`, consult `core.knowledge.operations.lookup(node.type)` and return
   its `syntax` if that is not `None`.
2. Import at module top, not inside the function.
3. `_generate_skeleton`'s passthrough comment branch must still handle the
   case where both lookups miss. Do not remove it.

**TASK — part B: extraction grounding**

4. In `core/rag/config_extractor.py`, add a module-level helper:

```python
def _operation_context(text: str, limit: int = 8) -> str:
    """Grounding block for operations explicitly mentioned in `text`.

    Deterministic (no LLM). Returns "" when nothing matches, so the prompt
    slot collapses to nothing rather than emitting an empty header.
    """
```
It calls `find_mentioned(text)`, takes at most `limit` entries (papers that
mention 20 ops would otherwise blow the context budget), and renders:

```
KNOWN OPERATION DEFINITIONS (use these exact definitions; do not redefine them):
- layernorm: y = (x - mean) / sqrt(var + eps) * gamma + beta  |  PyTorch: nn.LayerNorm(normalized_shape)
- gelu: ...
```

5. Add a `{operation_context}` slot to `_LLM_EXTRACTION_PROMPT`, placed
   **immediately after `{graph_rules}`** and before `### Now extract from this text:`.
6. In `_extract_with_llm`, populate it: `operation_context=_operation_context(text)`.
   Leave the `terms` / `graph_rules` lines exactly as they are.
7. `_operation_context` must never raise. Wrap the body in try/except and
   return `""` on failure — a grounding-block bug must not take down
   extraction.

**VERIFY**

Codegen (fast, no LLM):
1. Build a minimal graph node whose `.type` is `"tanh"` (in `OPERATIONS`,
   not in `MAP`) and assert `_node_to_layer` now returns `"nn.Tanh()"`.
2. Node type `"conv2d"` still returns the **`MAP`** parameterized string
   (assert the channel number appears in it) — proving `OPERATIONS` did not
   shadow `MAP`.
3. Node type `"timestep_embedding_xyz"` still returns `None`, and
   `_generate_skeleton` on a graph containing it still produces syntactically
   valid Python. Assert with `compile(code, "<gen>", "exec")` — actually
   compile it, do not eyeball it.

Extraction (fast, no LLM — do **not** call the real LLM in these tests):
4. `_operation_context("We use layer normalization and a GELU activation.")`
   contains `"layernorm"`, `"gelu"`, `"eps"`, and `"nn.LayerNorm"`.
5. `_operation_context("This paper is about datasets.")` returns exactly `""`.
6. `_operation_context` on text mentioning 15+ ops returns at most `limit`
   entries.
7. Assert `"{operation_context}"` is present in `_LLM_EXTRACTION_PROMPT`
   **and** that `_LLM_EXTRACTION_PROMPT.format(...)` still works with all
   four slots — a missing slot raises `KeyError` at extraction time and would
   only surface in production.
8. Re-run the existing extractor tests untouched:
```
.venv/Scripts/python.exe -m pytest tests/test_phase2_config_extractor.py tests/test_operation_knowledge.py -q -m "not live"
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — all of the above green; full suite ≥ 1746 passed, 0 failed.
Report the *actual* rendered `_operation_context` output for a real ResNet
excerpt so the user can eyeball the grounding text quality.

> Copy to here ↑

---

## PROMPT 3 — Backfill chunk embeddings for existing papers

> Copy from here ↓

**CONTEXT.** `index_chunk()` is wired into `backend/tasks/paper_tasks.py`
Stage 5, so **new** uploads get their `PaperChunk` rows embedded into the
`paper_chunks` Qdrant collection and `PaperChunk.embedding_id` set. Papers
ingested *before* that landed have `PaperChunk` rows with
`embedding_id IS NULL` and nothing in Qdrant. Hybrid search silently returns
worse results for them — no error, just quietly degraded. That is exactly the
failure mode this project keeps getting bitten by.

**ALLOWED FILES**
- create `backend/scripts/backfill_chunk_embeddings.py`
- `backend/tasks/paper_tasks.py` (add one Celery task only)
- create `tests/test_chunk_backfill.py`

**TASK**

1. Read `backend/scripts/make_admin.py` first. It is the only script in
   `backend/scripts/`, so it defines the house style. Match these four
   things exactly:
   - a module docstring whose first line is a `Usage:` line;
   - **`load_dotenv()` called BEFORE importing `backend.database` /
     `backend.models`.** This ordering is load-bearing, not cosmetic:
     `SessionLocal` is constructed at import time and needs `DATABASE_URL`
     already in the environment. Keep the same "imports after `load_dotenv()`"
     layout even though it trips linters' E402;
   - `db = SessionLocal()` with `try: ... finally: db.close()`;
   - an `if __name__ == "__main__":` guard, and `sys.exit(1)` on error.

   **Use `argparse` for this script.** `make_admin.py` uses a manual
   `len(sys.argv) != 2` check because it takes exactly one positional
   argument; that does not generalize to three optional flags. Argparse is
   the correct choice here and is an intentional divergence, not a style
   violation. Do not copy the manual `sys.argv` check.
2. Write `backfill_chunk_embeddings.py` exposing:

```python
def backfill(db, batch_size: int = 100, paper_id: int | None = None,
             dry_run: bool = False) -> dict:
    """Embed PaperChunk rows that have no embedding_id yet.

    Returns {"scanned": int, "indexed": int, "failed": int, "skipped": int}.
    """
```
Requirements:
- Query only `PaperChunk.embedding_id.is_(None)`. Do **not** re-embed rows
  that already have one (this is the whole point — it must be re-runnable).
- Process in batches of `batch_size`; commit per batch, not per row and not
  once at the end. A crash halfway must leave completed batches persisted.
- On `index_chunk()` returning `False`, count it in `failed` and **continue**.
  Do not abort the run and do not set `embedding_id`.
- Skip rows whose `text` is empty/whitespace; count in `skipped`.
- `dry_run=True` counts what *would* happen and writes nothing.
- If `vector_service._get_qdrant()` returns `None`, log a clear error and
  return the zeroed dict immediately — do not loop over thousands of rows
  calling a function that can only fail.
3. CLI: `--paper-id`, `--batch-size`, `--dry-run`. Print the result dict.
4. Add a Celery task in `backend/tasks/paper_tasks.py` named
   `backfill_chunk_embeddings_task`, following the existing task style in
   that file (`@celery_app.task`, `SessionLocal()`, try/finally `db.close()`).
   **Do not** add it to Celery Beat — this is operator-triggered, not
   scheduled. Do not modify any existing task.

**VERIFY**

Non-live (must pass with no Qdrant running):
1. `backfill` with `_get_qdrant()` patched to return `None` returns zeroed
   counts and does **not** iterate (assert via a spy that the chunk query was
   never executed, or that `index_chunk` was never called).
2. With `index_chunk` patched to return `True`, a session containing 3
   chunks (one with empty text) yields `{"scanned":3,"indexed":2,"failed":0,"skipped":1}`
   and the two real rows have non-null `embedding_id`.
3. With `index_chunk` patched to return `False`, `failed == 2` and
   `embedding_id` stays `None` on both — assert this explicitly; silently
   marking failures as indexed is the bug most likely to be written here.
4. `dry_run=True` leaves every `embedding_id` `None`.
5. Running `backfill` twice in a row: the second run reports `indexed == 0`.

Live (Qdrant up, `@pytest.mark.live` + skipif on `QDRANT_URL`):
6. Insert 2 real chunks, run the real `backfill`, then assert
   `search_chunks(<query matching chunk 1>, paper_id=...)` returns chunk 1.
   Clean up the Qdrant points and DB rows in a `finally`.

```
docker compose up -d qdrant
.venv/Scripts/python.exe -m pytest tests/test_chunk_backfill.py -q
QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest tests/test_chunk_backfill.py -q -m live
.venv/Scripts/python.exe -m pytest -q -m "not live"
docker compose down
```

**EXIT GATE** — all green; full suite ≥ 1746. Report the real dry-run output
against the actual dev database (how many chunks are currently unembedded).

> Copy to here ↑

---

## PROMPT 4 — Table / caption / equation chunks as first-class source records

> Copy from here ↓

**CONTEXT.** `core/utils.py:chunk_pages_with_provenance()` produces chunks
with `chunk_type` defaulting to `"text"` — and today **everything** is
`"text"`. Meanwhile `backend/services/paper_ingestion_service.py` already
extracts figures (`extract_figures`, PyMuPDF image xrefs) and equations
(`extract_equations`, regex over page text) — but those live in the Paper JSON
payload and are **never turned into `PaperChunk` rows**, so retrieval cannot
reach them. Architecture hyperparameters in papers live overwhelmingly in
**tables** and **figure captions**. Right now retrieval is blind to both.

`PaperChunk.chunk_type` already exists and is already persisted. Use it.

**ALLOWED FILES**
- `core/utils.py`
- `backend/services/paper_ingestion_service.py`
- `tests/test_phase3_evidence.py` (extend) or a new `tests/test_structured_chunks.py`

**DO NOT** change `chunk_pages_with_provenance`'s existing text-chunking
behavior or its return-dict keys. Existing chunks must come out byte-identical.
Add new chunks alongside them.

**TASK**

1. Read `core/utils.py:chunk_pages_with_provenance` and
   `paper_ingestion_service.extract_figures` / `extract_equations` fully
   before editing. Confirm the exact dict keys each produces.
2. In `core/utils.py`, add:
```python
def extract_table_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """Detect table-like regions in page text and emit chunks with
    chunk_type='table'. Deterministic, no LLM."""

def extract_caption_chunks(pages: list[tuple[int, str]]) -> list[dict]:
    """Emit chunk_type='caption' for figure/table caption lines."""
```
Each returns dicts with the **same keys** as `chunk_pages_with_provenance`
(`text/section/page/chunk_type/source_offset_start/source_offset_end`) so
downstream persistence needs no special-casing.

Table heuristic (keep it boring and deterministic):
- A run of ≥3 consecutive lines each containing ≥2 runs of whitespace-separated
  numeric-ish tokens, **or** ≥2 pipe/tab separators.
- Include the line immediately above the run if it matches
  `^\s*Table\s+\d+`.
- Cap each table chunk at the same `max_chars` used elsewhere.

Caption heuristic:
- Lines matching `^\s*(Figure|Fig\.|Table)\s+\d+[.:)]?\s+\S` — capture that
  line plus up to the next 2 lines until a blank line.

3. In `paper_ingestion_service`, after the existing
   `chunk_pages_with_provenance(...)` call, append the table and caption
   chunks to the same `source_chunks` list. Equations: convert the existing
   `extract_equations` output into chunks with `chunk_type="equation"`,
   reusing its already-computed page numbers — do **not** re-run the regex.
4. `page` may legitimately be `None` for a chunk. Guard every consumer with
   `is not None`, never `hasattr`, never bare truthiness (page 0 is falsy).
   The persistence loop already does this — match it.
5. Deduplicate: a caption line that is also inside a text chunk is fine
   (overlap is acceptable and useful for retrieval), but two *identical*
   caption chunks are not. Dedupe on `(chunk_type, page, text)`.

**VERIFY**
1. A synthetic page containing a clear 4-row numeric table yields exactly one
   `chunk_type="table"` chunk whose `text` contains all 4 rows.
2. A page with `"Figure 3: The residual block adds the input..."` yields a
   `chunk_type="caption"` chunk containing that sentence.
3. Prose with no tables/captions yields **zero** table/caption chunks
   (assert the false-positive rate — a heuristic that fires on everything is
   worse than none).
4. `chunk_pages_with_provenance` output on a fixed input is **unchanged** —
   snapshot its result before and after your edit and assert equality.
5. Every emitted chunk has all six required keys and `page` is `int` or `None`.
6. End-to-end: run the real ingestion path on a fixture PDF and assert the
   persisted `PaperChunk` rows include at least one non-`"text"` `chunk_type`.
```
.venv/Scripts/python.exe -m pytest tests/test_structured_chunks.py tests/test_phase3_evidence.py -q -m "not live"
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — green; full suite ≥ 1746. Report, for one real paper PDF, the
count of chunks by `chunk_type`.

> Copy to here ↑

---

## PROMPT 5 — OCR fallback for scanned PDFs

> Copy from here ↓

**PREREQ:** Prompt 4 landed.

**CONTEXT.** `paper_ingestion_service` extracts text with pdfplumber and falls
back to PyMuPDF (`fitz`). Both are **text-layer** extractors: a scanned/image-only
PDF yields empty or near-empty text, and `core/paper_to_code_generator.from_pdf`
then raises `"Could not extract any text from the PDF. It might be corrupted,
empty, or image-only."` The user gets a hard failure with no recourse.

**IMPORTANT — do not install a new dependency without checking.** OCR needs
either `pytesseract` (+ a system Tesseract binary, which may not be present on
this Windows box) or `rapidocr-onnxruntime` (pure pip, no system binary).
**Step 1 is to determine what is actually available and report back.** Do not
assume.

**ALLOWED FILES**
- `backend/services/paper_ingestion_service.py`
- `requirements.txt` (only if the user approves a dependency)
- create `tests/test_ocr_fallback.py`

**TASK**

1. **First, investigate and STOP for confirmation.** Check whether
   `pytesseract`, `rapidocr-onnxruntime`, or a system `tesseract` binary is
   already available:
   ```
   .venv/Scripts/python.exe -c "import importlib.util as u; print({m: bool(u.find_spec(m)) for m in ['pytesseract','rapidocr_onnxruntime','easyocr']})"
   where tesseract
   ```
   Report the result and your recommendation, then **wait for approval**
   before adding anything to `requirements.txt`.

2. Once approved, add to `paper_ingestion_service`:
```python
OCR_MIN_CHARS_PER_PAGE = 50   # below this, a page is "probably scanned"

def _needs_ocr(page_texts: list[str]) -> bool:
    """True when the text layer is too sparse to be a real text PDF."""

def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 30) -> list[tuple[int, str]]:
    """Rasterize + OCR. Returns [(page_number, text), ...].
    Returns [] if OCR is unavailable -- never raises."""
```
3. Wire it as a **third** fallback, only when `_needs_ocr()` is true after both
   existing extractors have run. Never run OCR on a PDF that already has a
   usable text layer — it is orders of magnitude slower.
4. Respect the existing 30-page cap (`pdf.pages[:30]` in
   `core/paper_to_code_generator.from_pdf`). Do not raise it.
5. Record what happened in the ingestion summary dict (there is already an
   `ingestion` summary with `detected_components` / `figure_count` /
   `equation_count`) — add `text_source: "pdfplumber" | "pymupdf" | "ocr"`
   so the UI and the user can see that a paper was OCR'd (OCR text is noisier
   and downstream confidence should reflect that).
6. OCR must degrade gracefully: missing library, missing binary, or a
   rasterization error → log a warning, return `[]`, and let the existing
   `"Could not extract any text"` error surface as it does today.

**VERIFY**
1. `_needs_ocr(["", "", ""])` is `True`; `_needs_ocr(["<800 chars of prose>"])`
   is `False`. Test the boundary at exactly `OCR_MIN_CHARS_PER_PAGE`.
2. With the OCR module patched to raise `ImportError`, `ocr_pdf_pages`
   returns `[]` and does not propagate.
3. A text-layer PDF fixture does **not** trigger OCR — assert with a spy that
   `ocr_pdf_pages` was never called. This is the important one: an accidental
   always-on OCR path would silently make every upload 50x slower.
4. `text_source` is correctly reported for the non-OCR path.
5. Live/optional (mark `@pytest.mark.live`, skip if OCR unavailable): a real
   scanned-PDF fixture produces non-empty text. If you cannot produce a
   scanned fixture, say so — do not fake one by deleting a text layer and
   claiming it is equivalent.
```
.venv/Scripts/python.exe -m pytest tests/test_ocr_fallback.py -q -m "not live"
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — green; full suite ≥ 1746; and a timing number for the
non-OCR path proving you did not regress normal ingestion speed.

> Copy to here ↑

---

## PROMPT 6 — KAG query-term expansion into retrieval

> Copy from here ↓

**CONTEXT — read carefully, this is a precision task.**
`KnowledgeGraph.related_concepts("skip connections")` returns
`["residualblock"]`, and `infer_family_from_concept()` maps a query to an
architecture *family* which then contributes a `+0.1` bonus in
`_bm25_and_family_scores`. That is family **inference**, not query **expansion**.

What is missing: when a query says "skip connections", the BM25 half of the
hybrid score still only searches for the literal tokens
`["skip","connections"]`. A chunk that says *"residual connection"* or
*"projection shortcut"* and never says "skip" gets **zero** BM25 contribution.
The graph knows they are the same concept; retrieval does not use that.

A trap to avoid (this exact mistake was made once already): do **not** feed
raw graph node names like `"residualblock"` into BM25 as query terms. Real
prose says *"residual block"* (two words); `"residualblock"` as a single
token matches nothing, so the expansion looks wired but does nothing measurable.
Expansion terms must be **natural-language surface forms**.

**ALLOWED FILES**
- `core/rag/knowledge_graph.py`
- `backend/services/vector_service.py`
- `tests/test_improvements.py` (extend)

**TASK**

1. In `KnowledgeGraph`, add a surface-form table and an expansion method:
```python
_SURFACE_FORMS: dict[str, list[str]] = {
    "residual_add":  ["residual connection", "skip connection",
                      "shortcut connection", "projection shortcut",
                      "identity mapping"],
    "residualblock": ["residual block", "bottleneck block"],
    ...
}

def expand_query_terms(self, query: str, max_terms: int = 12) -> list[str]:
    """Expand a natural-language query with surface forms of graph-related
    concepts. Deterministic, no LLM. Returns lowercase phrases, deduped,
    original query terms first, never more than max_terms."""
```
Cover at minimum the residual/skip family, attention family
(`multiheadattention`, `cross_attention`, `causal_attention`), patch embedding,
and normalization. Reuse `_CONCEPT_ALIASES` for entry-point matching — do not
duplicate it.

2. `expand_query_terms` must:
   - return `[]` for a query matching nothing (callers then behave exactly as
     today — this must be a strict superset of current behavior);
   - never return a term already present in the query;
   - be capped at `max_terms` (an unbounded expansion turns BM25 into noise
     and would *reduce* precision).

3. In `vector_service._bm25_and_family_scores`, tokenize
   `query + " " + " ".join(expanded_terms)` instead of just `query`, but weight
   expanded terms **lower** than original terms. Simplest correct approach:
   score original and expanded separately with the existing `BM25` instance and
   combine as `bm25_orig + 0.5 * bm25_expanded`, then normalize the sum. Do
   **not** change the outer `0.5/0.4/0.1` weights — they are tuned and verified.

4. `expand_query_terms` failing must not break retrieval: wrap the call,
   fall back to no expansion.

**VERIFY — the gate here is a measurable delta, not "it ran".**
1. `expand_query_terms("how do skip connections work")` includes
   `"residual connection"` and does **not** include `"skip connection"`
   (already in the query) and does not include `"residualblock"`
   (not a surface form).
2. `expand_query_terms("the dataset has 1.2M images")` returns `[]`.
3. Cap: a query hitting many concepts returns exactly `max_terms`.
4. **The real test.** Build a chunk pool where the target chunk says
   *"residual connection"* and never *"skip"*, plus ≥2 distractors. Query
   `"skip connections"`. Assert:
   - with expansion, the target's BM25 component is **> 0**;
   - with expansion disabled, it is **== 0**;
   - the target ranks first with expansion.
   Assert the numeric delta, not just the final ordering — ordering alone can
   be produced by the dense half and would prove nothing about expansion.
5. Existing hybrid tests still pass unchanged (`test_dense_and_hybrid_chunk_retrieval_against_live_qdrant`,
   `test_hybrid_search_chunks_falls_back_to_bm25_and_family_without_qdrant`).
```
docker compose up -d qdrant
QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest tests/test_improvements.py -q
.venv/Scripts/python.exe -m pytest -q -m "not live"
docker compose down
```

**EXIT GATE** — the numeric BM25 delta in test 4 is demonstrated in your
report with real numbers; full suite ≥ 1746.

> Copy to here ↑

---

## PROMPT 7 — Diversity selection in hybrid ranking

> Copy from here ↓

**PREREQ:** Prompt 6 landed.

**CONTEXT.** `hybrid_rank_texts()` returns the top-k by score. On a real paper
the top 6 chunks are frequently 6 near-duplicate restatements of the same
architecture sentence (papers repeat the architecture in abstract, intro,
method, and figure caption). The LLM then sees the same fact six times and
learns nothing about the rest of the model. Diversity selection fixes the
*composition* of the retrieved set, not the scoring.

**ALLOWED FILES**
- `backend/services/vector_service.py`
- `tests/test_improvements.py` (extend)

**TASK**

1. Add Maximal Marginal Relevance selection:
```python
def _mmr_select(indices: list[int], scores: list[float],
                vectors, k: int, lambda_: float = 0.7) -> list[int]:
    """Greedy MMR: iteratively pick the candidate maximizing
    lambda_*relevance - (1-lambda_)*max_similarity_to_already_selected.
    `vectors` may be None -- then fall back to plain top-k by score."""
```
Pure function, no I/O. Cosine similarity between already-normalized embedding
vectors is a dot product — the embedder is called with
`normalize_embeddings=True`, so do **not** re-normalize.

2. Use it in `hybrid_rank_texts` **only when embeddings were computed**
   (`_get_embedder()` returned a model). When the embedder is unavailable
   there are no vectors to diversify against — fall back to today's behavior
   exactly.
3. Preserve the existing contract: `hybrid_rank_texts` returns texts in
   **reading order** (`sorted(idx for idx, _ in scored[:top_k])`), not score
   order. MMR changes *which* are selected, not the output ordering. Do not
   silently change this — `_focus_text` joins them into a narrative for the LLM.
4. Add `diversity: float = 0.7` as a keyword arg with that default; `1.0`
   must reproduce pure top-k exactly (assert this).
5. Do **not** apply MMR in `hybrid_search_chunks` in this prompt — that path
   returns scored IDs to a different consumer. Scope it to `hybrid_rank_texts`.

**VERIFY**
1. `diversity=1.0` returns byte-identical output to the pre-change function on
   a fixed pool (snapshot before your edit).
2. Pool of 5 near-identical "residual block" sentences + 2 distinct chunks
   (optimizer, dataset), `top_k=3`: with `diversity=0.7` the result contains
   **at most 2** of the near-duplicates. Without MMR it contains 3.
3. `_mmr_select(..., vectors=None, ...)` equals plain top-k.
4. `_mmr_select` with `k >= len(indices)` returns everything, no crash.
5. Empty input returns `[]`.
6. Output is still in reading order (indices ascending).
```
.venv/Scripts/python.exe -m pytest tests/test_improvements.py -q -m "not live"
QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest tests/test_improvements.py -q -m live
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — test 2 demonstrated with the real selected texts printed;
full suite ≥ 1746.

> Copy to here ↑

---

## PROMPT 8 — Architecture-fidelity checks (beyond "does it run")

> Copy from here ↓

**CONTEXT — highest-risk prompt in this file. Scope discipline matters.**
`verification_report` currently answers *"did the generated code execute in the
sandbox without raising?"* (`passed`, `error`, `entrypoint_class`, `attempts`).
A model that instantiates cleanly but has 3 transformer layers where the paper
says 12 is reported as a **success**. That is the single biggest correctness
gap in the pipeline.

You are adding a **separate, additive** fidelity score. You are **not**
changing what `passed` means, not changing the repair loop's exit condition,
and not gating generation on fidelity. Existing consumers of
`verification_report["passed"]` must keep working identically.

**ALLOWED FILES**
- create `core/fidelity.py`
- `core/paper_to_code_generator.py` (one call site + report key only)
- create `tests/test_architecture_fidelity.py`

**DO NOT** modify `_verification_attempt`, the repair loop, `passed`,
`generation_status`, or E2B harness code.

**TASK**

1. Create `core/fidelity.py`:
```python
def score_fidelity(spec: dict, code: str, graph=None) -> dict:
    """Compare the extracted spec against the generated code.

    Returns:
      {"score": float,            # 0.0-1.0
       "checks": [{"name": str, "passed": bool, "detail": str}, ...],
       "mismatches": [str, ...]}

    Static analysis only -- parse `code` with `ast`, never exec it.
    Never raises: a malformed spec or unparseable code returns
    score 0.0 with an explanatory check, not an exception.
    """
```
2. Implement exactly these checks, no more (each one must be genuinely
   verifiable from the AST — do not add a check you cannot implement honestly):
   - `layer_count` — number of `nn.*` constructor calls in the class body vs.
     `len(spec["layers"])`. Tolerance ±20%; report the real numbers.
   - `layer_types_present` — every distinct canonical type in the spec has at
     least one plausible corresponding `nn.*` call. Map via the `MAP` dict in
     `core/codegen.py` and `core.knowledge.operations.OPERATIONS` (Prompt 1)
     — **reuse**, do not re-hardcode a third mapping.
   - `declared_vs_used` — every `self.X = nn...` assigned in `__init__` is
     referenced in `forward`, and every `self.X` used in `forward` was
     assigned. (This catches the exact skeleton bug fixed earlier in the
     project.)
   - `key_hyperparams` — for hyperparameters explicitly present in the spec
     (`num_heads`, `hidden_size`, `channels`, `num_layers`, `kernel_size`),
     check the literal value appears as an argument somewhere in the code.
     Only check params the spec actually has — **never** penalize a param the
     paper never stated. Report `"not_stated"`, not a failure.
   - `residual_present` — if `spec["connection_types"]` contains
     `skip`/`residual`, `forward` must contain an addition or a `torch.cat`.
     Skip this check entirely when the spec declares no such connections.
3. `score` = passed checks / applicable checks. Checks that are not applicable
   (e.g. residual on a plain CNN) are excluded from the denominator, **not**
   counted as passes. State this in the docstring.
4. In `core/paper_to_code_generator.py`, after the repair loop finishes and
   next to where `verification_report["evidence"]` is set (~line 234), add
   `verification_report["fidelity"] = score_fidelity(...)` inside a try/except
   that logs and sets `None` on failure. **Nothing else changes.**

**VERIFY**
1. Hand-write a spec + matching code → score `1.0`.
2. Same spec, code with half the layers → `layer_count` fails, score < 1.0,
   and `mismatches` names the real counts.
3. Code with `self.conv2` used in `forward` but never assigned →
   `declared_vs_used` fails.
4. Spec with `num_heads: 8`, code with `num_heads=4` → `key_hyperparams` fails
   and the mismatch string contains both `8` and `4`.
5. Spec with **no** `num_heads` → that param is `not_stated`, and the score is
   **not** penalized. Assert the denominator excluded it.
6. Spec with no skip connections → `residual_present` is absent from
   `checks`/denominator entirely.
7. `score_fidelity({}, "")` returns `score == 0.0` and does not raise.
8. `score_fidelity(spec, "def broken(:")` (SyntaxError) returns `0.0` with an
   explanatory check, does not raise.
9. **Regression guard:** assert `verification_report["passed"]` is unchanged
   for a fixture that previously passed. Run the existing
   `tests/test_phase2_e2b_validation.py` and `tests/test_phase2_repair_loop.py`
   untouched and green.
```
.venv/Scripts/python.exe -m pytest tests/test_architecture_fidelity.py tests/test_phase2_e2b_validation.py tests/test_phase2_repair_loop.py -q -m "not live"
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — green; full suite ≥ 1746. Report real fidelity scores for 3
already-generated papers so the user can judge whether the scoring is
calibrated or just always returns ~0.9.

> Copy to here ↑

---

## PROMPT 9 — Labeled benchmark + multi-paper live verification

> Copy from here ↓

**PREREQ:** Prompts 1–8 landed.

**CONTEXT.** `benchmark_vit_pipeline.py`, `benchmark_bert_pipeline.py`, and
`benchmark_gpt_pipeline.py` exist at the repo root — read all three first;
they are per-architecture scripts with hand-written expectations, not a shared
harness. There is no way to answer *"did last week's change make extraction
better or worse?"* That question is unanswerable today, which means every
quality claim in this project so far has been anecdotal.

Scope: build the harness and label **10 papers**, not 25. A 25-paper set that
is half-wrong is worse than a 10-paper set that is right, and labeling is the
expensive part. Scaling to 25→200 comes later.

**ALLOWED FILES**
- create `benchmarks/__init__.py`, `benchmarks/harness.py`, `benchmarks/labels/*.json`
- create `tests/test_benchmark_harness.py`
- **do not** modify the three existing root-level `benchmark_*.py` scripts

**TASK**

1. Define the label schema (one JSON per paper, in `benchmarks/labels/`):
```json
{
  "paper_id": "resnet50",
  "source": "arxiv:1512.03385",
  "family": "resnet",
  "expected": {
    "layer_types": ["conv2d", "batchnorm2d", "residualblock", "avgpool2d", "linear"],
    "key_hyperparams": {"kernel_size": 7, "channels": 64, "num_classes": 1000},
    "min_layers": 5,
    "has_residual": true
  },
  "notes": "stem is 7x7/2, four stages"
}
```
Only label facts **explicitly stated in the paper**. If a value is not stated,
omit it — never guess. A wrong label silently penalizes correct extractions
forever and is the most damaging possible artifact here.

2. `benchmarks/harness.py`:
```python
def run_benchmark(label_paths, extractor=None, fidelity=True) -> dict:
    """Run extraction over labeled papers, score against labels.
    Returns {"per_paper": [...], "aggregate": {...}}."""
```
Metrics: `layer_type_recall`, `layer_type_precision`, `hyperparam_accuracy`,
`family_correct` (bool), plus `fidelity_score` from Prompt 8 when available.
Aggregate = mean per metric + a count of papers that hard-failed.

3. It must run **offline by default** against cached extraction outputs, with
   `--live` to re-run real extraction (LLM calls cost money and rate-limit —
   this project has already hit Groq quota exhaustion mid-verification).
   Cache extraction results to `benchmarks/.cache/` keyed by paper id.
4. Print a table and write `benchmarks/results/<UTC-timestamp>.json` so runs
   are comparable over time. Take the timestamp from the caller/CLI.
5. Label **10 papers** spanning families already supported by
   `core/classification.py`: resnet, unet, vit, transformer, bert_gpt,
   mobilenet, densenet, efficientnet, gan, diffusion — one each.

**VERIFY**
1. Harness on a synthetic label + a **known-correct** synthetic extraction
   scores 1.0 on every metric.
2. Harness on a deliberately wrong extraction (missing layer type, wrong
   hyperparam) produces recall/accuracy **< 1.0** with the specific miss named.
3. A label file missing an optional key does not crash and does not count
   against the score.
4. A malformed label JSON produces a clear error naming the file, not a
   `KeyError` traceback.
5. Offline mode makes **zero** LLM calls — assert with a patched `llm_complete`
   that raises if called. This is the check that keeps the benchmark runnable.
6. Every label file validates against the schema (parametrized test over
   `benchmarks/labels/*.json`).
```
.venv/Scripts/python.exe -m pytest tests/test_benchmark_harness.py -q
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — harness green offline; **then** run it live once against the 10
labeled papers and report the real aggregate table. That number is the Phase 4
baseline. If it is bad, report it as bad — a benchmark that flatters the system
is worthless.

> Copy to here ↑

---

## PROMPT 10 — Workspace transparency UI

> Copy from here ↓

**PREREQ:** Prompts 3, 4, 8 landed.

**CONTEXT.** `src/app/(protected)/papers/[id]/WorkspacePaperClient.tsx` shows
minimal evidence. The backend now knows far more than the UI reveals:
retrieved chunks with page/section provenance, per-field citation status
(`cited` vs `inferred`) from `build_evidence_map`, generation attempt count and
per-attempt errors (`verification_report.attempts`), and (after Prompt 8) a
fidelity score with named mismatches. A user currently cannot tell a
well-grounded extraction from a confident guess. Surfacing this honestly is
the difference between a demo and a tool.

**ALLOWED FILES**
- `src/app/(protected)/papers/[id]/WorkspacePaperClient.tsx`
- new components under `src/components/` as needed
- the paper detail API route, **only** if a needed field is not already serialized

**TASK**

1. Read `WorkspacePaperClient.tsx` fully and check what the API already returns
   before adding backend fields. Prefer surfacing existing data.
2. Add an **Evidence** panel:
   - per spec field: value + status badge (`cited` green / `inferred` amber)
     + page number + the verified quote when `cited`.
   - honest empty state: *"No verified citations — values were inferred from
     the model's reading of the paper."* Do not imply grounding that is not
     there. This project deliberately distinguishes verified from unverified
     quotes; the UI must not blur it.
3. Add a **Generation** panel: attempts used (`total_attempts` / 3), per-attempt
   error strings, `code_source` (`builder` / `llm` / `skeleton`), and the
   fidelity score with its failed check names.
4. Show `chunk_type` on retrieved chunks (`text` / `table` / `caption` /
   `equation`) and `text_source` when a paper was OCR'd, with a caveat that
   OCR'd text is noisier.
5. Every field must be null-safe. Papers ingested before these features exist
   have `evidence: null`, no `fidelity`, `page: null`. Render a graceful
   fallback for each — do **not** crash the page and do not render "undefined".
   Use `x != null` checks; `page` can legitimately be `0`.

**VERIFY**
1. `npm run build` (or the project's build script) succeeds with no new TS errors.
2. Existing frontend tests pass: `npm test` — report the real count.
3. Render with: a fully-cited paper, a fully-inferred paper, a paper with
   `evidence: null`, and a paper with `total_attempts: 3` and a failure.
   All four render without crashing.
4. Screenshot or describe each of the four states in your report.

**EXIT GATE** — build clean, frontend tests green, four states verified.
Backend suite must still be ≥ 1746 (you should not have touched it).

> Copy to here ↑

---

## §3 — APPENDIX: commands cheat sheet

```bash
# Qdrant (needed for live retrieval tests)
docker compose up -d qdrant
curl http://localhost:6333/healthz
docker compose down

# Full gate — must be >= 1746 passed, 0 failed
.venv/Scripts/python.exe -m pytest -q -m "not live"

# Live tests (Qdrant + embedder)
QDRANT_URL=http://localhost:6333 .venv/Scripts/python.exe -m pytest -q -m live

# Single file
.venv/Scripts/python.exe -m pytest tests/<file>.py -q

# Run from repo root; if imports fail:
PYTHONPATH=/c/papper2code .venv/Scripts/python.exe <script>
```

**Never run:** `git commit`, `git push`, `git reset`, `git checkout --`.
The user commits manually.
