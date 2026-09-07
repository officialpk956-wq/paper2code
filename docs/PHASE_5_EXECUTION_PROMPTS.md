# Phase 5 — Execution Prompts (extraction quality)

**How to use this file.** One prompt at a time, in the run order in §2.
Always paste **§0.5 (Universal Preamble)** immediately above the prompt you
are running. Do not chain prompts. Do not skip a VERIFY gate.

Phase 5's goal is a single number moving: **layer-type recall from 0.502 to
≥ 0.70**, without gaming it. Every prompt here either moves that number or
makes it honestly measurable.

---

## §0 — WHY THIS PHASE EXISTS (read first)

The 2026-08-31 live benchmark over 10 labeled papers
(`benchmarks/results/20260830T201959Z.json`, 0 hard failures):

| Metric | Value |
|---|---:|
| layer_type_recall | **0.502** |
| layer_type_precision | 0.710 |
| family_correct | 0.700 |
| hyperparam_accuracy | **0.000** |
| fidelity_score | null (structurally unmeasurable) |

**The headline discovery: 20% of expected layer types are unreachable by
construction.** `_LLM_EXTRACTION_PROMPT` in `core/rag/config_extractor.py`
tells the model *"type must be one of:"* and then lists **14** types.
`CANONICAL_TYPES` in `core/rag/normalizer.py` has **39**. The model is
forbidden from emitting the other 25 — so it cannot possibly extract them,
no matter how clearly the paper describes them.

Measured against the actual labels, 8 of 10 papers are affected:

| Paper | Expected types the model is forbidden to emit |
|---|---|
| bert_base | `positionalembedding`, `gelu` |
| dcgan | `convtranspose2d` |
| densenet121 | `concat` |
| efficientnet_b0 | `depthwise_conv2d` |
| mobilenet_v1 | `depthwise_conv2d` |
| transformer_base | `feedforward`, `softmax` |
| unet | `concat` |
| vit_base | `positionalembedding` |

**10 of 50 expected types (20%) are blocked. Maximum achievable recall
today is 0.80, and we are at 0.502.**

This also plausibly explains the two collapse cases. EfficientNet-B0
extracted **1 layer** — and EfficientNet is built from MBConv blocks, i.e.
`depthwise_conv2d`, which is blocked. DCGAN's whole generator is
`convtranspose2d`, blocked.

Two supporting findings:

- **`softmax` is a label bug, not a pipeline bug.** It is not in
  `CANONICAL_TYPES` at all (it lives only in `core/knowledge/operations.py`).
  The other 6 blocked types are legitimate canonical types simply not
  offered. Fix the label, not the vocabulary.
- **`hyperparam_accuracy` is genuinely 0.000.** Verified directly: the
  ResNet-50 spec has 2 layers, both `params: {}`. Separately, the harness
  looks for label keys like `num_classes`, but `_PARAM_PATTERNS` in
  `config_extractor.py` only produces `kernel_size, channels, stride,
  padding, hidden_size, num_heads, num_layers` — so `num_classes` is
  unreachable too. Same class of bug as the type enum: **the label asks for
  vocabulary the pipeline never produces.**

**Do not read precision alone.** ResNet-50 scored precision **1.00** by
emitting two layers that happened to be correct. Precision rewards
extracting nothing. Recall and layer count are the honest signals.

---

## §0.5 — UNIVERSAL PREAMBLE (paste above EVERY prompt)

```
GROUND RULES — read before writing any code.

1. REALITY CHECK FIRST. This prompt states exact file paths, signatures,
   and line numbers. Open each named file and confirm before editing. If
   ANY claim in this prompt does not match the real repo, STOP and report
   the mismatch. Do not invent a plausible substitute. (Two Phase 4
   prompts had errors caught exactly this way -- this rule works.)

2. DO NOT INVENT APIs. Do not call functions, kwargs, model fields, or
   library methods you have not read in this repo.

3. NO REWRITES. Smallest change that satisfies the task. Do not refactor,
   rename, reformat, or touch files outside ALLOWED FILES.

4. NO NEW DEPENDENCIES. Not one, in any prompt in this phase.

5. MEASURE, DO NOT ASSERT. This phase is about one number: layer-type
   recall. Every prompt ends by running the benchmark. Paste the real
   before/after aggregate. "Should improve extraction" is not a result.
   A change that does not move a metric gets REVERTED, not rationalized.

6. NEVER GAME THE METRIC. Do not add layer types to the output just to
   inflate recall. Do not special-case a benchmark paper by name, by
   title, or by any identifying string. Do not tune thresholds against
   the 10 labeled papers until they pass. If you catch yourself writing
   anything paper-specific, stop -- that is overfitting to the test set
   and it destroys the benchmark's value permanently.

7. NULL SAFETY. Recurring bug class in this repo: an explicit `None`
   surviving a merge / setdefault / hasattr guard. Guard with
   `x is not None`, never `hasattr`, never bare truthiness (0 is falsy).

8. GRACEFUL DEGRADATION. Anything touching Qdrant, the embedder, or an
   LLM degrades to a working fallback and never hard-fails extraction.

9. DO NOT COMMIT. No git commit / push / reset / checkout --. Leave
   everything in the working tree.

10. REPORT HONESTLY. If a number got worse, say so and show it. A
    faithful negative result is worth more than a flattering claim.

ENVIRONMENT
  Windows, PowerShell primary, Git Bash available.
  Python: .venv/Scripts/python.exe    Root: C:\papper2code
  Full suite:      .venv/Scripts/python.exe -m pytest -q -m "not live"
  MUST NOT REGRESS: 1803 passed, 2 skipped, 9 deselected, 0 failed.
    The suite has 1814 tests total. This baseline is COMMAND-SPECIFIC:
    with `-m "not live"` the live tests are DESELECTED (9). With a bare
    `pytest -q` they are instead COLLECTED and SKIPPED, giving
    "1804 passed, 10 skipped" -- the same suite, a different invocation.
    Always use the `-m "not live"` form above so the numbers compare.
  Benchmark offline: .venv/Scripts/python.exe -m benchmarks.harness
  Benchmark live:    .venv/Scripts/python.exe -m benchmarks.harness --live
  Qdrant (live retrieval tests): docker compose up -d qdrant
                                 QDRANT_URL=http://localhost:6333
  BASELINE TO BEAT: recall 0.502 | precision 0.710 | hyperparam 0.000
                    | family 0.700 | hard_failures 0
```

---

## §1 — GROUND TRUTH

**`core/rag/config_extractor.py`**
```python
_ARCHITECTURE_QUERY: str
_LAYER_PATTERNS: list[tuple[str, str]]     # regex -> canonical type, 26 entries
_PARAM_PATTERNS: list[tuple[str, list[str]]]
   # keys produced: kernel_size, channels, stride, padding,
   #                hidden_size, num_heads, num_layers   (NO num_classes)
_LLM_EXTRACTION_PROMPT   # slots: {few_shot} {graph_rules} {operation_context} {text}
_VERIFICATION_PROMPT
_operation_context(text, limit=8) -> str   # Phase 4; never raises

class ConfigExtractor:
    __init__(use_llm=True, use_section_splitter=True, use_retriever=True,
             verify=True, max_context_chars=10_000, chunk_retriever=None)
    extract_from_text(text, source_chunks=None) -> ConfigDict
    _focus_text(text, source_chunks=None) -> str
    _extract_with_llm(text) -> dict
    _extract_rule_based(text) -> dict
    _verify_extraction(original_text, extracted) -> dict
```
The allowed-type line, verbatim, is inside `_LLM_EXTRACTION_PROMPT`:
```
- "type" must be one of: conv2d, conv1d, linear, maxpool2d, avgpool2d,
  multiheadattention, transformerblock, batchnorm2d, layernorm, relu,
  dropout, upsample, residualblock, patchembedding
```

**`core/rag/normalizer.py`** — `CANONICAL_TYPES` (39), `_SYNONYM_MAP`,
`_normalize_type` (passes unknown types through with a warning, never raises).
Assertion at ~line 257: every `_SYNONYM_MAP` value must be in `CANONICAL_TYPES`.

**`benchmarks/harness.py`**
```python
load_label(path)
_live_extractor(label) -> {"spec": ..., "family": ...}   # NO "code" key
_hyperparameters(value, wanted, found=None) -> dict[str, list]  # recursive key search
_score_label(label, extraction, fidelity) -> dict
run_benchmark(label_paths, extractor=None, fidelity=True) -> dict
main(argv=None) -> int      # --live, --timestamp, positional label paths
LABELS_DIR / benchmarks/labels/*.json ; cache benchmarks/.cache/<id>.json
```
Fidelity guard at ~line 184:
`if fidelity and isinstance(extraction.get("code"), str) and extraction["code"].strip():`
`_live_extractor` never returns `code`, so this is always False.

**`backend/services/vector_service.py`** — `hybrid_rank_texts(query, texts,
top_k=6, family=None, diversity=0.7) -> list[str]`, `_mmr_select`,
`_bm25_and_family_scores`, `_FAMILY_TERMS`, `index_chunk`, `search_chunks`,
`hybrid_search_chunks`.

**`core/utils.py`** — `chunk_pages_with_provenance(pages, max_chars=1200)`,
`extract_table_chunks(pages)`, `extract_caption_chunks(pages)`.

**`core/knowledge/operations.py`** — `OPERATIONS` (23), `lookup(term)`,
`find_mentioned(text)`.

**Production wiring reference** (`backend/services/paper_ingestion_service.py`):
```python
def _chunk_retriever(query, texts, top_k):
    from backend.services.vector_service import hybrid_rank_texts
    return hybrid_rank_texts(query, texts, top_k=top_k)

_GENERATOR = PaperToCodeGenerator(chunk_retriever=_chunk_retriever)
```

---

## §2 — RUN ORDER

| # | Prompt | Why this order |
|---|---|---|
| 1 | Unblock the type vocabulary | Largest known win, fully evidenced, may fix both collapse cases |
| 2 | Re-baseline + label audit | Measure P1 honestly before diagnosing anything else |
| 3 | Wire benchmark to production retrieval | Phase 3/4 retrieval is currently unmeasured |
| 4 | Parameter extraction | Independent workstream; hyperparam is 0.000 |
| 5 | Diagnose residual collapse cases | Only what P1–P4 did not fix |
| 6 | Fidelity reporting + CI guard | Close the loop, prevent regression |

**After every prompt:** run the full suite (≥1803), run the benchmark, paste
before/after, append to `docs/PAPER_TO_CODE_EXECUTION_MEMORY.md`, then STOP.

---

## PROMPT 1 — Unblock the extraction type vocabulary

> Copy from here ↓

**CONTEXT.** `_LLM_EXTRACTION_PROMPT` offers the model 14 layer types.
`CANONICAL_TYPES` has 39. The model cannot emit what it is not offered, so
**10 of 50 expected types across the benchmark (20%) are unreachable by
construction** — including `depthwise_conv2d` (the whole of MobileNet and
EfficientNet), `convtranspose2d` (DCGAN's generator), `concat` (U-Net and
DenseNet skip merges), and `positionalembedding` + `gelu` (BERT and ViT).

This is the single highest-value fix in Phase 5 and it is a prompt/vocabulary
change, not an algorithm change.

**ALLOWED FILES**
- `core/rag/config_extractor.py`
- `core/codegen.py` (task 2b only)
- `benchmarks/labels/transformer_base.json`
- `tests/test_phase2_config_extractor.py` (extend)

**DO NOT** touch `core/rag/normalizer.py`, `CANONICAL_TYPES`, or
`_SYNONYM_MAP`. Do not rewrite the few-shot examples or the rules text
beyond the specific edits below.

**EXPECTED WORKING-TREE STATE:** `core/rag/config_extractor.py` and
`tests/test_phase2_config_extractor.py` already carry uncommitted
modifications. This is **expected, not an anomaly** — it is Phase 4
Prompt 2's operation-grounding work (`_operation_context` and its tests),
which has not been committed yet. Build on top of it; do not revert it.

**TASK**

1. Read the current `_LLM_EXTRACTION_PROMPT` and
   `core/rag/normalizer.py`'s `CANONICAL_TYPES` in full.

2. **Expand the allowed-type list deliberately — not to all 39.** Add types
   that (a) are in `CANONICAL_TYPES` and (b) a paper would plausibly
   describe in prose. Add at minimum:
   `depthwise_conv2d`, `convtranspose2d`, `concat`, `positionalembedding`,
   `gelu`, `feedforward`, `globalavgpool2d`, `flatten`, `residual_add`,
   `groupnorm`, `leakyrelu`, `silu`, `clstoken`, `sequence_pooling`.

   **Renderability is deliberately NOT a criterion.** Six of these
   (`convtranspose2d`, `positionalembedding`, `feedforward`, `clstoken`,
   `concat`, `residual_add`) currently have no `nn.*` syntax in
   `core/codegen.py`'s `MAP` or in `OPERATIONS`. Include them anyway:
   extraction fidelity and codegen coverage are **separate concerns**. A
   paper that uses transposed convolutions must produce a spec that says
   so — that is *correct extraction*. Suppressing the type to keep codegen
   tidy makes the spec lie about the paper and blinds `score_fidelity` to
   the gap. `_generate_skeleton` already degrades safely on unrenderable
   types (emits a `# x unchanged:` passthrough, never crashes) — that is
   deliberate, tested Phase 2 behavior.

   Deliberately **exclude** internal decomposition types a paper never names
   as a layer: `query_projection`, `key_projection`, `value_projection`,
   `attention_merge`. State in your report why each excluded type was
   excluded. Blindly pasting all 39 is the wrong answer.

2b. **Close the cheap codegen gaps** so newly-allowed types are not silently
   dropped. Add to `core/codegen.py`'s `MAP` (it is parameterized with real
   channel/head values, so it belongs there, not in `OPERATIONS`):
   - `convtranspose2d` → `nn.ConvTranspose2d(...)` using the same `ch` / `k`
     locals the neighbouring entries use. **DCGAN's entire generator is
     this type** — highest value of the group.
   - `feedforward` → an `nn.Sequential` of two `nn.Linear` layers with an
     activation between, sized from the existing `in_hs` / `out_hs` locals.
     Transformer papers describe this constantly.

   Leave `concat` and `residual_add` **unrenderable on purpose**: they are
   structural graph operations (`torch.cat`, `+`), not modules, and the
   sequential skeleton cannot express them — a limitation already documented
   in `_generate_skeleton`'s header comment. They still earn their place in
   the allowed list because they populate `connections` /
   `connection_types` and feed fidelity scoring. `positionalembedding` and
   `clstoken` are `nn.Parameter`-based and need shape context; leave them
   for a later prompt rather than guessing a shape here.

3. Keep the list readable — group it by category with line breaks, the way
   the current one wraps. This string goes into every extraction call; a
   sprawling wall of tokens has its own cost.

4. **Fix the `softmax` label bug.** `benchmarks/labels/transformer_base.json`
   expects layer type `softmax`, which is not in `CANONICAL_TYPES` (it exists
   only in `core/knowledge/operations.py` as an operation). The pipeline
   cannot legitimately produce it as a layer type. Remove `softmax` from that
   label's `layer_types`. **Do not** add `softmax` to `CANONICAL_TYPES` to
   make the label pass — that is fixing the test by breaking the design.

5. Check `_LAYER_PATTERNS` for the newly-allowed types. The rule-based
   fallback should be able to produce what the LLM can now emit. Add regex
   patterns only for types that are clearly missing and unambiguous — e.g.
   depthwise / separable convolution, transposed / deconvolution,
   concatenation. Do not add ambiguous single-word patterns that would
   misfire on ordinary prose.

**VERIFY**

1. Assert every type named in the prompt's allowed list is in
   `CANONICAL_TYPES` (parse the string, parametrize the test). This is the
   guard that stops the two lists drifting apart again — it is the most
   important test in this prompt.
2. Assert the four projection/merge types are NOT in the allowed list.
3. `_LLM_EXTRACTION_PROMPT.format(...)` still renders with all four slots.
4. Existing extractor tests pass unchanged.
5. No label file expects a type outside `CANONICAL_TYPES` (parametrize over
   `benchmarks/labels/*.json`) — this catches the `softmax` class of bug for
   every future label.
6. **Run the live benchmark and paste the full before/after table.**
```
.venv/Scripts/python.exe -m pytest tests/test_phase2_config_extractor.py -q -m "not live"
.venv/Scripts/python.exe -m pytest -q -m "not live"
.venv/Scripts/python.exe -m benchmarks.harness --live
```

**EXIT GATE** — full suite ≥1803, 0 failed. Benchmark re-run with the real
table pasted. Expect recall to rise materially from 0.502; report the actual
number whatever it is. If recall does **not** move, say so plainly — that
would mean the enum was not the binding constraint and Prompt 5's diagnosis
becomes the priority.

> Copy to here ↑

---

## PROMPT 2 — Re-baseline and audit the labels

> Copy from here ↓

**PREREQ:** Prompt 1 landed.

**CONTEXT.** Prompt 1 changed what the model is allowed to say. Before
diagnosing anything further, establish a clean new baseline and make sure the
labels themselves are not the thing being measured. A wrong label
permanently penalizes correct extraction — the `softmax` case proved this
class of bug is already present.

**ALLOWED FILES**
- `benchmarks/labels/*.json`
- `benchmarks/harness.py` (validation only)
- `tests/test_benchmark_harness.py` (extend)

**TASK**

1. Re-run the live benchmark and record the post-Prompt-1 aggregate as the
   new working baseline.
2. **Audit all 10 labels against their actual papers.** For each expected
   layer type and hyperparameter, confirm the paper explicitly states it.
   Remove anything that is inferred, conventional-but-unstated, or an
   artifact of how you would implement it rather than what the paper says.
   Report every change with a one-line justification.
3. **Audit the hyperparameter keys specifically.** Labels use keys like
   `num_classes`, but `_PARAM_PATTERNS` only produces `kernel_size,
   channels, stride, padding, hidden_size, num_heads, num_layers`. For each
   label key, decide and report: is this key one the pipeline *should*
   produce (→ leave it, Prompt 4 will add it), or one the label invented
   (→ fix the label)? Do not silently delete a key just to raise the score.
4. Add a `schema_version` field to every label file and validate it in
   `load_label`. Labels will keep changing; unversioned ground truth becomes
   unfalsifiable.

**VERIFY**
1. Every label validates: required keys present, `schema_version` correct,
   no expected type outside `CANONICAL_TYPES`, no duplicate types.
2. A label with an unknown `schema_version` raises a clear error naming the
   file.
3. Benchmark re-run; paste the table.
```
.venv/Scripts/python.exe -m pytest tests/test_benchmark_harness.py -q
.venv/Scripts/python.exe -m benchmarks.harness --live
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — labels audited with a written changelog, all validating,
new baseline recorded. Report the number even if the audit *lowered* it —
a lower honest number is the point of this prompt.

> Copy to here ↑

---

## PROMPT 3 — Make the benchmark measure the real pipeline

> Copy from here ↓

**PREREQ:** Prompts 1–2 landed.

**CONTEXT.** `_live_extractor` calls
`ConfigExtractor().extract_from_text(text)` — no `source_chunks`, no
`chunk_retriever`. In `_focus_text`, `chunk_texts` is therefore empty, the
hybrid/KAG branch is skipped entirely, and it falls back to plain BM25 over
a blindly re-chunked flat string.

**All of Phase 3's retrieval work and Phase 4's Prompts 6–7 (KAG query
expansion, MMR diversity) are currently unmeasured.** The benchmark is
scoring a code path production does not use.

**ALLOWED FILES**
- `benchmarks/harness.py`
- `tests/test_benchmark_harness.py` (extend)

**TASK**

1. Read `backend/services/paper_ingestion_service.py`'s `_chunk_retriever`
   and `_GENERATOR` construction — that is the production wiring to mirror.
   Read `core/rag/config_extractor.py:_focus_text` to confirm exactly what
   `source_chunks` changes.
2. In `_live_extractor`, extract **page-aware** text (keep `(page_number,
   text)` tuples, as `core/paper_to_code_generator.from_pdf` does) and build
   `source_chunks` via `chunk_pages_with_provenance`. Also append
   `extract_table_chunks` and `extract_caption_chunks` output, matching
   production.
3. Pass a `chunk_retriever` that wraps `vector_service.hybrid_rank_texts`,
   and pass `source_chunks` into `extract_from_text`.
4. **Support measuring both paths.** Add a `--retrieval {production,legacy}`
   flag defaulting to `production`. `legacy` keeps today's behavior. The
   point is a controlled A/B: the same 10 papers scored both ways is the
   only way to know whether the retrieval work helped.
5. Record which mode produced a result in the results JSON, and key the
   cache separately per mode so the two do not overwrite each other.

**VERIFY**
1. `_live_extractor` in `production` mode actually invokes the retriever —
   assert with a spy, not by inspecting output. A silently-inert retriever
   is exactly the bug this prompt exists to fix.
2. `legacy` mode does not invoke it.
3. Cache keys differ between modes; running both leaves two cache entries.
4. Offline mode still makes zero LLM calls (patched `llm_complete` that
   raises if called).
5. **Run the benchmark BOTH ways and paste both tables side by side.**
```
.venv/Scripts/python.exe -m benchmarks.harness --live --retrieval legacy
.venv/Scripts/python.exe -m benchmarks.harness --live --retrieval production
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — both tables reported. This finally answers whether Phase 3/4
retrieval improved extraction. **If production mode is not better, report
that honestly** — it is a genuinely valuable finding and must not be buried.

> Copy to here ↑

---

## PROMPT 4 — Parameter extraction

> Copy from here ↓

**PREREQ:** Prompts 1–3 landed.

**CONTEXT.** `hyperparam_accuracy` is **0.000** — not low, zero. Verified
directly: the ResNet-50 spec contains 2 layers, both `params: {}`, for a
paper that explicitly states a 7×7 stem, 64 channels, and 1000 classes. Even
richly-extracted papers carry few params (DenseNet-121: 9 layers, 1
parameterized). This is a distinct failure from layer-type recall and needs
its own work.

Two candidate causes, and they need different fixes:
- The prompt says *`"params": ONLY extract values EXPLICITLY stated in the
  text. Do NOT guess.`* That is strongly discouraging phrasing with no
  worked guidance, and the few-shot examples may under-demonstrate params.
- The focused text reaching the LLM may not contain the parameter-bearing
  sentences at all (a retrieval problem, which Prompt 3 may already have
  improved).

**ALLOWED FILES**
- `core/rag/config_extractor.py`
- `tests/test_phase2_config_extractor.py` (extend)

**TASK**

1. **Diagnose first.** For ResNet-50 and DenseNet-121, dump the focused text
   actually passed to `_extract_with_llm` and check whether the parameter
   sentences ("7×7 convolution with 64 filters", "growth rate k=32") are
   present. Report the finding before changing anything. If the text is
   missing, this is a retrieval problem and the prompt fix is cosmetic.
2. Keep the "do not guess" constraint — it exists to prevent fabricated
   hyperparameters, which is a worse failure than missing ones. Reword to
   be *permissive about stated values* while still forbidding invention.
   Something in the spirit of: extract every value the text explicitly
   states; omit the key entirely when it is not stated; never infer a
   conventional default.
3. Strengthen the few-shot examples to demonstrate richer `params`. The
   ResNet example already shows `kernel_size`/`channels`/`stride` — check
   whether the others do, and make at least one example show a densely
   parameterized layer.
4. Add `num_classes` to `_PARAM_PATTERNS` **only if** Prompt 2's audit
   concluded it is a key the pipeline should produce. Patterns like
   `(\d+)[\s-]*way classification`, `(\d+) classes`, `num_classes\s*[:=]\s*(\d+)`.
   Follow the existing list's structure exactly.

**VERIFY**
1. Rule-based extraction on a text stating "7×7 convolution with 64 filters,
   stride 2" produces `kernel_size=7`, `channels=64`, `stride=2`.
2. If `num_classes` was added: "1000-way classification" yields
   `num_classes=1000`; text with no class count yields **no** `num_classes`
   key at all (absent, not `None` — an explicit `None` surviving into params
   is this repo's recurring bug).
3. No fabrication: text stating only a layer type yields `params: {}`, not
   invented defaults. Assert this explicitly — it is the failure mode the
   original wording was protecting against and must not regress.
4. Existing extractor tests unchanged and green.
5. Benchmark re-run; paste before/after with attention to
   `hyperparam_accuracy`.
```
.venv/Scripts/python.exe -m pytest tests/test_phase2_config_extractor.py -q -m "not live"
.venv/Scripts/python.exe -m benchmarks.harness --live
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — `hyperparam_accuracy` > 0.0 with the real number reported,
and no regression in recall/precision. Report the step-1 diagnosis in full
regardless of outcome.

> Copy to here ↑

---

## PROMPT 5 — Diagnose the residual collapse cases

> Copy from here ↓

**PREREQ:** Prompts 1–4 landed and re-baselined.

**CONTEXT — REDIRECTED 2026-08-31 after reviewing the Prompt 1–4 results.**
ResNet-50 is fixed (2 → 33 layers, 23 parameterized). EfficientNet-B0 is
still collapsed (1 → 2 layers) but is **no longer the priority.** Two
larger problems surfaced in the run history and take precedence:

**(a) Nobody has established the noise floor.** Two `--retrieval production`
runs over identical code and identical labels produced recall **0.490** and
**0.465** — a 0.025 spread from LLM nondeterminism alone. Any conclusion
resting on a difference smaller than that is unsupported. Every causal claim
in Phase 5 so far is n=1.

**(b) Production retrieval appears to be destroying hyperparameters.**
Same labels, same run session:

| Mode | recall | hyperparam_accuracy |
|---|---:|---:|
| legacy | 0.543 | **0.133** |
| production (run 1) | 0.490 | **0.000** |
| production (run 2) | 0.465 | **0.000** |

Per-paper parameter counts point the same way: ddpm **23 → 0**,
transformer_base **5 → 0**, mobilenet_v1 **3 → 0**. A consistent direction
across papers is much harder to explain as noise than the recall gap is.

Plausible mechanism to test, not assume: `hybrid_rank_texts` selects 6
chunks by architecture-query relevance with MMR diversity. Parameter values
often live in tables, captions, and experimental-setup prose that score low
on an architecture query — and MMR actively evicts near-duplicate chunks,
which is exactly where repeated hyperparameter mentions live. Legacy's
`get_architecture_text` path may retain more contiguous parameter-bearing
text.

**Confound to respect:** Prompt 2's audit changed the labels (50 → 45
expected types), so current numbers are **not** comparable to the 0.502
baseline. Only same-label comparisons are valid — legacy vs production is
valid; today vs baseline is not.

**ALLOWED FILES**
- create `benchmarks/diagnose.py`
- `core/rag/config_extractor.py` (only if the diagnosis proves a fix there)
- tests as needed

**TASK**

1. **Instrument first — the cache cannot answer this.** `_normalise_extraction`
   stores only `{spec, family, code}`, so intermediate stages are gone.
   Extend the live path to persist a **companion** diagnostic file per paper
   per mode (e.g. `benchmarks/.cache/<id>.<mode>.diag.json`): raw text
   length, chunk count by `chunk_type`, the focused text actually sent to
   the LLM, the raw LLM response, the spec pre-normalization, and the spec
   post-normalization. Keep it a *separate* file so the existing cache
   format and its tests are untouched. Diagnostics are written only on live
   runs; offline scoring must not require them.
2. Write `benchmarks/diagnose.py` to read and compare those files. It is an
   instrument, not a fix — offline, no LLM calls, no new dependencies.
3. **Establish the noise floor before any causal claim.** Run the same mode
   over the same papers **3 times** and report per-metric spread. Every
   later comparison must be stated against that floor: a difference inside
   it is not a finding. Quota is limited — if 3 full runs are too expensive,
   run 3× on a 4-paper subset including `ddpm` and `transformer_base`, and
   say that is what you did.
4. **Target the parameter loss, not EfficientNet.** Diagnose `ddpm`
   (23 params legacy → 0 production) and `transformer_base` (5 → 0) by
   diffing their focused text between modes. The question to answer with
   evidence: *does the production focused text still contain the sentences
   stating the hyperparameters?* If it does not, this is retrieval
   (`hybrid_rank_texts` / MMR evicting parameter-bearing chunks). If it
   does, it is the prompt or parsing.
5. **Report before fixing.** State the attribution, quote the actual focused
   text on both sides, and stop for approval. Retrieval, prompt, and parsing
   need very different fixes; guessing wastes the phase.

If the evidence shows production retrieval is genuinely worse, that is a
legitimate outcome and must be reported plainly — Phase 4's retrieval work
being a net negative on extraction is exactly the kind of finding this
benchmark exists to surface. Do not soften it, and do not quietly switch the
default back without saying so.

**VERIFY**
1. `diagnose.py` runs offline against the cache with no LLM calls.
2. Its output contains all six stages listed above.
3. The attribution is evidence-backed: quote the actual focused text and
   actual LLM response in your report, do not paraphrase.

**EXIT GATE** — a written, evidence-backed attribution for each remaining
collapse case, and a proposed fix awaiting approval. No speculative fixes.

> Copy to here ↑

---

## PROMPT 6 — Fidelity reporting and CI regression guard

> Copy from here ↓

**PREREQ:** Prompts 1–5 landed.

**CONTEXT.** Two loose ends.

`fidelity_score` is `null` for every paper on the live path, permanently:
`_live_extractor` returns `{"spec", "family"}` with no `code`, so
`_score_label`'s guard `isinstance(extraction.get("code"), str)` is always
False. The aggregate advertises a metric it cannot populate. A
permanently-null metric is worse than an absent one — it reads as
"measured, scored zero."

And nothing stops extraction quality silently regressing between phases.

**ALLOWED FILES**
- `benchmarks/harness.py`
- `tests/test_benchmark_harness.py` (extend)
- CI config, if the repo has one

**TASK**

1. Pick ONE and justify it in your report:
   - **(a)** Generate code in the live path so fidelity is real. Truthful but
     slow, and pulls codegen + possibly E2B into every benchmark run.
   - **(b)** Drop `fidelity_score` from the live aggregate and report it only
     in offline runs where cached code exists. Honest and cheap.

   **(b) is recommended** — the benchmark's job is measuring *extraction*,
   and mixing in generation makes a regression harder to localize. If you
   choose (a), it must be behind a `--fidelity` flag, off by default.

2. Add a `--check` mode: run offline against cached extractions, compare to
   a committed `benchmarks/baseline.json`, exit non-zero if any metric drops
   more than a stated tolerance (start at 0.05 absolute).
3. Commit the current post-Phase-5 aggregate as `benchmarks/baseline.json`.
4. Wire `--check` into CI if a config exists; otherwise document the command
   in the harness docstring and report that CI wiring was not possible.

**VERIFY**
1. `--check` against an unchanged baseline exits 0.
2. `--check` against a synthetically degraded result exits non-zero and names
   the metric that dropped.
3. A metric that *improves* does not fail the check.
4. A metric present in the baseline but missing from the run is an error, not
   a silent pass — this is the failure mode that lets a metric quietly
   disappear.
5. `--check` makes zero LLM calls.
```
.venv/Scripts/python.exe -m pytest tests/test_benchmark_harness.py -q
.venv/Scripts/python.exe -m benchmarks.harness --check
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — `--check` works both directions, baseline committed, full
suite ≥1803. Report the final Phase 5 aggregate against the 2026-08-31
baseline (recall 0.502 / precision 0.710 / hyperparam 0.000 / family 0.700)
and state plainly whether the Phase 5 exit gate in
`docs/PAPER_TO_CODE_MASTER_PLAN.md` §8 was met.

> Copy to here ↑

---

## §3 — PHASE 5 EXIT GATE (from the master plan)

- layer-type recall ≥ **0.70** mean, **and no single paper below 0.40**
- `hyperparam_accuracy` ≥ **0.50** (from 0.000)
- no paper extracting fewer than **5 layers** without a written explanation
- precision not below **0.65** (guards against dumping types to inflate recall)
- a written attribution for every remaining miss

Report each of these five explicitly at the end. Partial success stated
honestly is the expected outcome; a claim of full success without the
numbers behind it is not acceptable.
