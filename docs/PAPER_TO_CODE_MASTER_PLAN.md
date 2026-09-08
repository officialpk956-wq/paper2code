# Reliable Paper-to-Code Pipeline — Master Plan

> **Status: living roadmap. Last updated 2026-09-07.**
>
> This began as the original 7-phase plan written before any implementation.
> The original assessment, root-cause findings, and phase definitions are
> preserved below **unedited** — they are the baseline everything since has
> been measured against, and rewriting them would destroy that reference.
>
> What has been added: a **§0 Status** section recording where the system
> actually is, per-phase completion markers, and a **§8 Next Phases**
> section that supersedes the original Phases 5–7 with evidence-driven
> scope.
>
> `docs/PAPER_TO_CODE_EXECUTION_MEMORY.md` remains the detailed engineering
> log (every bug, root cause, and verification run). This file is the
> altitude above it: what is done, what is left, and what to do next.

---

## §0 Status — 2026-09-08

**Delivered:** original Phases 1–4 complete; Phases 5 and 6 (as re-scoped in
§8) executed, each meeting most but not all of its gate. A post-Phase-6
finding — **silently degraded PDF text extraction across all ten papers** —
is fixed but not yet scored (§8 → Post-Phase 6).
**Gate:** **1946** backend tests passing (10 skipped, 0 failed) + 38 frontend.
**Uncommitted:** everything from Phase 3 onward is in the working tree.
**Current quality** (best clean-ish measurement, see §8 → Phase 6 — RESULT):
recall 0.702 | precision 0.654 | hyperparam 0.533 | family 0.900.
**Blocking constraint:** Groq **daily token budget**, not rate limiting.

### Phase mapping (planned → as executed)

| Original phase | As executed | Status |
|---|---|---|
| P1 Restore end-to-end path | Phase 1 | ✅ done |
| P2 Orchestration pipeline | Phase 2 (absorbed P2+P3) | ✅ done |
| P3 Compiler and repair system | Phase 2 | ✅ done |
| P4 Evidence-grounded RAG + KAG | Phase 3 (Half A + Half B) | ✅ done |
| P5 Benchmark corpus → 200 papers | Phase 4 (harness + 10 labels) | ◐ harness done, corpus not scaled |
| P6 Transparent workspace results | Phase 4 (Prompt 10) | ✅ done |
| P7 Hardening, rollout, continuous eval | — | ⬜ not started |

Phase 4 also delivered work not in the original plan: a curated
**operation knowledge table** (`core/knowledge/operations.py`, 23 ops with
formula + PyTorch syntax + aliases) that grounds the extraction prompt and
backs codegen, and **architecture-fidelity scoring** (`core/fidelity.py`)
that answers *"does the code match the paper?"* rather than only *"does it
run?"*.

### Maturity re-scored against the original baseline

| Area | Then | Now | What moved it |
|---|---:|---:|---|
| Upload, storage, Celery jobs | 70% | 90% | Contract fixed; async flow verified live |
| PDF parsing | 50% | 70% | Page/section provenance chunks; table/caption/equation chunks. **OCR still absent** (seam only) |
| Architecture extraction | 40% | 65% | ConfigExtractor is now the production path; operation grounding in-prompt |
| RAG | 20% | 75% | Chunk-level dense + BM25 + family hybrid, wired *into extraction* |
| KAG / knowledge graph | 30% | 60% | `related_concepts`, family inference, query expansion. Ontology still narrow |
| Code generation | 25% | 50% | Builders self-contained; op-table fallback. **Fidelity now measured and low** |
| Execution and repair | 15% | 70% | E2B wired, bounded 3-attempt repair, structured diagnostics |
| Evaluation corpus | 10% | 30% | Harness + 10 labels exist; corpus not scaled |
| Product progress/reporting | 30% | 65% | Workspace evidence + generation panels |

### First real benchmark baseline (2026-08-31, all 10 papers, live)

`benchmarks/results/20260830T201959Z.json` — 0 hard failures.

| Metric | Value |
|---|---:|
| layer_type_recall | **0.502** |
| layer_type_precision | 0.710 |
| family_correct | 0.700 |
| hyperparam_accuracy | **0.000** |
| fidelity_score | not measurable (see below) |

Per-paper extraction volume, which is the more diagnostic view:

| Paper | Layers extracted | With params |
|---|---:|---:|
| transformer_base | 54 | 10 |
| unet | 45 | 27 |
| dcgan | 25 | 12 |
| vit_base | 14 | 6 |
| densenet121 | 9 | 1 |
| ddpm | 8 | 5 |
| bert_base | 5 | 3 |
| mobilenet_v1 | 5 | 3 |
| **resnet50** | **2** | **0** |
| **efficientnet_b0** | **1** | **0** |

### The honest headline

Infrastructure is now substantially ahead of output quality. Retrieval,
repair, sandboxing, provenance, and transparency are real and tested.
Extraction is the bottleneck, and the failure mode is **bimodal, not
uniformly weak**: some papers extract richly (U-Net 45 layers with 27
parameterized), while ResNet-50 collapsed to 2 layers with zero
parameters and EfficientNet-B0 to a single layer. Those are silent
near-total failures — they did not error, they produced a plausible-looking
empty spec.

Three specific findings that shape the next phase:

1. **`hyperparam_accuracy` is genuinely 0.000, not a scoring bug.** Verified
   directly: the ResNet-50 spec contains two layers, both with `params: {}`.
   The paper explicitly states a 7×7 stem, 64 channels, and a 1000-way
   classifier. None were captured. Parameter extraction is effectively
   not working, distinct from layer-type extraction which partly works.
2. **High precision here is a trap.** ResNet-50 scored precision 1.00 —
   because it emitted two layers and both happened to be right. Precision
   rewards extracting almost nothing. Recall and layer count are the honest
   signals; precision must never be read alone.
3. **`fidelity_score` is structurally unmeasurable on the live path.**
   `_live_extractor` deliberately does not generate code, so
   `_score_label`'s `isinstance(extraction.get("code"), str)` guard is
   always False and fidelity is always `None`. The aggregate advertises a
   metric it can never populate live. Either wire code generation into the
   benchmark or drop the field from the live report — reporting a
   permanently-null metric is worse than not reporting it.

Building more infrastructure on top of this compounds the problem. The plan
below reflects that.

> **Superseded 2026-09-04.** The table above is the 2026-08-31 snapshot,
> kept as the historical baseline. It is now known to be **invalid as a
> measurement**: several papers had silently fallen back to the rule-based
> extractor, so 0.502 was a blend of two different extractors. On a clean
> run where all 10 papers reach the LLM, recall is **0.713** and
> hyperparam_accuracy **0.400**. See **§8 → Phase 5 — RESULT** for the
> three defects behind that gap and **Phase 6** for what remains.

---

## Assessment

The idea is strong, but the 200-paper RAG/knowledge-graph corpus should not be the first fix. The immediate problem is that existing components are disconnected and several contracts are broken.

Recommended architecture:

`PDF → evidence extraction → architecture specification → retrieval/KG enrichment → code generation → sandbox validation → bounded repair → verified result`

Use one orchestrated workflow with specialist stages, not an uncontrolled swarm of agents.

## Current State and Root Causes

Approximately **30–35% of a reliable end-to-end system exists; 65–70% remains**.

| Area | Current maturity | Finding |
|---|---:|---|
| Upload, storage, Celery jobs | 70% | Infrastructure exists, but the frontend/backend upload contract is broken |
| PDF parsing | 50% | Text, sections, figures, and equations exist; OCR and evidence quality are limited |
| Architecture extraction | 40% | LLM and structured extraction exist, but production bypasses the better parser |
| RAG | 20% | Qdrant exists but indexes only title, authors, and abstract |
| KAG/knowledge graph | 30% | Deterministic concepts and topology checks exist but coverage is narrow |
| Code generation | 25% | Builders, LLM generation, and skeleton fallback exist, but outputs are not reliably runnable |
| Execution and repair | 15% | E2B execution exists separately; paper generation only checks syntax |
| Evaluation corpus | 10% | 218 educational pages exist, but only 2 local PDFs and 15 methodology tracks |
| Product progress/reporting | 30% | Task stages exist but do not represent verification attempts or quality |

### Immediate core-feature failures

1. **Upload is currently broken at the contract level.**
   - The frontend sends `file`, `title`, and `visibility`.
   - The active backend endpoint requires `terms_accepted=true`.
   - The frontend therefore receives a 400 response.
   - The frontend expects a `Paper`, while the backend returns `{task_id, status, poll_url}` and expects polling.

2. **Generated code is not reliably persisted with the paper.**
   - It is stored mainly inside the transient task result.
   - The paper workspace reads architecture artifacts but has no durable verified-code artifact and validation report.

3. **Known-family code generation is broken.**
   - ResNet, U-Net, and ViT builder output fails immediately because the returned class source omits required imports and dependencies.
   - Transformer returns an architecture graph builder rather than a runnable `torch.nn.Module`.
   - A direct probe produced:
     - ResNet: `NameError: nn is not defined`
     - U-Net: `NameError: nn is not defined`
     - ViT: `NameError: nn is not defined`
     - Transformer: executes, but it is not the intended model implementation.

4. **The existing ingestion agent is not usable in production.**
   - It imports `generate_pytorch_code`, which does not exist.
   - Tests hide this by dynamically mocking that missing function.
   - Production does not invoke this agent anyway.

5. **Validation means syntax only.**
   - LLM code receives `ast.parse`.
   - It is not imported, instantiated, given synthetic input, run forward/backward, or checked against extracted shapes.
   - E2B already provides the required sandbox but is not connected to paper generation.

6. **The repair loop fixes only syntax.**
   - Runtime errors, missing dependencies, tensor mismatches, wrong constructors, and incorrect output shapes are not fed back to the LLM.
   - Celery retry reruns the job; it does not perform an evidence-based code repair.

7. **RAG is too shallow.**
   - Qdrant stores one abstract-level vector per paper.
   - There are no full-paper chunks, section/page provenance, equations, captions, implementation evidence, sparse+dense fusion, or reranking.

8. **Existing keyword/KAG work is disconnected.**
   - `ConfigExtractor` already contains BM25 retrieval, concept mapping, KAG context, structured extraction, and one verification pass.
   - The live PDF pipeline instead calls the older architecture extractor directly.

The 88 focused backend tests currently pass, but they validate isolated plumbing. They do not test a real PDF through generation, sandbox execution, repair, and verified delivery.

## Seven-Phase Implementation Plan

### Phase 1 — Restore the existing end-to-end path

- Align the upload request and response contracts.
- Add an explicit Terms acceptance control and send `terms_accepted`.
- Treat upload as an asynchronous operation:
  - receive `task_id`;
  - poll `/api/tasks/{id}`;
  - show stage/progress;
  - navigate to the paper after `paper_id` is available.
- Persist generated code, source, generation status, and errors with the paper.
- Add an end-to-end test covering upload → task → persisted paper → workspace.
- Remove or consolidate overlapping paper routes after confirming active consumers.

**Exit gate:** a PDF can be uploaded and the completed paper workspace opens without manual intervention.

### Phase 2 — Create one production orchestration pipeline

Replace the disconnected generators with a single versioned state machine:

1. Validate and store PDF.
2. Extract text, sections, tables, equations, captions, and page references.
3. Select implementation-relevant evidence.
4. Extract a typed architecture specification.
5. Verify the specification against source evidence.
6. Retrieve similar implementations and architecture concepts.
7. Generate a self-contained PyTorch project.
8. Validate it in the sandbox.
9. Repair it up to three times.
10. Persist the best result and verification report.

Reuse `ConfigExtractor`, `ParsingAgentImpl`, the graph compiler, tensor tracker, LiteLLM client, Celery, and E2B. Retire the unused LangGraph path or rebuild it around the same production services.

Every stage must have a timeout, structured output, retry policy, token budget, and checkpoint.

**Exit gate:** there is exactly one code path used by uploads, tests, API calls, and future corpus evaluations.

### Phase 3 — Build the compiler and repair system

Generate a project bundle containing:

- `model.py`
- `config.json`
- `smoke_test.py`
- `requirements.txt`
- `README.md`
- evidence/provenance manifest

Validation ladder:

1. AST and forbidden-code checks.
2. Dependency allowlist.
3. Import generated module.
4. Discover and instantiate the model.
5. Generate synthetic input from the extracted specification.
6. Run a forward pass.
7. Validate output shapes and architecture invariants.
8. Run a minimal backward pass where applicable.
9. Check NaN/Inf and deterministic smoke-test behavior.
10. Enforce sandbox time, memory, filesystem, network, and output limits.

On failure, return a structured diagnostic containing stage, exception, traceback, expected shape, actual shape, relevant source evidence, and previous attempts. The repair agent receives only this diagnostic, the current code, and relevant evidence.

Maximum: **three generation/repair attempts**. Never loop indefinitely.

**Exit gate:** supported generated projects pass sandbox import, instantiation, forward, shape, and backward checks.

### Phase 4 — Implement evidence-grounded RAG + KAG

Create section-aware chunks from full papers:

- abstract and introduction;
- method and architecture;
- equations;
- tables;
- figure captions;
- training details;
- appendix and implementation notes.

Each chunk stores:

- paper/version ID;
- section;
- page;
- chunk type;
- source offsets;
- architecture family;
- concepts/entities;
- embedding version;
- licensing/provenance metadata.

Retrieval will combine:

- dense Qdrant similarity;
- BM25/keyword search;
- architecture-family filtering;
- knowledge-graph expansion;
- reranking;
- diversity selection.

The knowledge graph should initially use Postgres/JSON plus typed edges. A separate Neo4j-style service is unnecessary until graph-query scale proves it is needed.

KAG is used to expand and verify retrieval—for example, connecting residual blocks to skip connections and shape-preserving projections—not as an additional opaque LLM layer.

**Exit gate:** every extracted parameter and generated architectural decision can cite supporting paper chunks or clearly identify itself as an inference/default.

### Phase 5 — Build the benchmark corpus, then expand to 200 papers

Do not immediately call the 218 educational pages a dataset. Build a separate versioned corpus.

Start with 25 carefully labeled papers across:

- CNNs and residual networks;
- transformers and language models;
- vision transformers;
- segmentation and detection;
- generative and diffusion architectures.

For each paper, store:

- legally usable PDF/source;
- canonical metadata;
- implementation-relevant chunks;
- gold architecture specification;
- trusted reference implementation link/version;
- expected constructor;
- synthetic input;
- expected output shape;
- required architectural invariants;
- smoke tests;
- human review notes.

Split papers by architecture lineage to prevent near-duplicate leakage:

- development set;
- validation set;
- locked test set.

After the pipeline meets quality gates on 25 papers, expand in batches of 25 until reaching approximately 200.

Use the corpus for retrieval and evaluation first. Do not fine-tune a model until benchmark results show that retrieval and prompting have plateaued.

**Exit gate:** at least 80% of supported locked-test papers pass execution, and at least 70% pass architecture-fidelity review without manual repair.

### Phase 6 — Deliver transparent results in the workspace

Display:

- current pipeline stage;
- evidence extracted from each page/section;
- architecture specification;
- generated files;
- compiler output;
- repair attempts;
- verification checks;
- confidence level;
- unsupported or inferred details;
- citations to source chunks.

Result states:

- **Verified:** all required checks passed.
- **Partially verified:** executable, but fidelity checks are incomplete.
- **Needs review:** generation completed but important checks failed.
- **Unsupported:** the paper does not map to the current PyTorch architecture scope.

Never label code "correct" merely because it compiles.

**Exit gate:** users can understand what was generated, what was tested, and what remains uncertain.

### Phase 7 — Hardening, rollout, and continuous evaluation

- Run the locked corpus in CI/nightly evaluation.
- Track:
  - extraction accuracy;
  - retrieval recall;
  - first-pass execution rate;
  - final execution rate;
  - shape/fidelity pass rate;
  - average repair attempts;
  - latency;
  - token and sandbox cost;
  - failure category.
- Add prompt, schema, embedding, model, and dataset versions to every run.
- Roll out by architecture family behind feature flags.
- Preserve failed artifacts for diagnosis without exposing private paper content.
- Add Sentry events and structured logs for each stage.
- Introduce human review for low-confidence outputs and feed approved corrections back into the corpus.

**Exit gate:** regressions are detected automatically and unsupported papers fail transparently instead of returning misleading code.

## §8 Next Phases (supersedes original Phases 5–7)

The original plan sequenced corpus-building next. The benchmark evidence
says otherwise: labeling 200 papers against an extractor with 0.2 recall
would mostly produce an expensive, precise measurement of a known problem.
Fix extraction first, then scale the corpus against something worth
measuring.

### Phase 5 — Extraction quality (next; the bottleneck)

**Goal:** move layer-type recall from ~0.2–0.75 to a defensible number on
the 10-paper set, with the movement *attributable* rather than incidental.

- **5.1 Diagnose the collapse cases first.** ResNet-50 (2 layers, 0 params)
  and EfficientNet-B0 (1 layer, 0 params) are the highest-signal failures:
  same pipeline, same prompt, and U-Net extracted 45 layers. Whatever
  differs between them is the bug. For each, determine which stage dropped
  the content — (a) retrieval never put the relevant text in front of the
  LLM, (b) the LLM saw it and didn't emit it, or (c) it was emitted and
  `_normalize_type` / the prompt's allowed-type enum discarded it. These
  need different fixes. Instrument and attribute; do not guess. Start by
  dumping the focused text actually sent to the LLM for ResNet-50 versus
  U-Net — if ResNet's focused text is missing the architecture section,
  it is a retrieval bug and 5.2 is the fix.
- **5.1b Parameter extraction is a separate failure.** `hyperparam_accuracy`
  is 0.000 across every paper that had labeled hyperparameters. Even
  richly-extracted papers carry few params (densenet121: 9 layers, 1
  parameterized). `_PARAM_PATTERNS` and the prompt's params instruction
  should be treated as their own workstream, not as a side effect of fixing
  layer-type recall.
- **5.2 Make the benchmark measure the real pipeline.** `_live_extractor`
  calls `ConfigExtractor().extract_from_text(text)` with no `source_chunks`
  and no `chunk_retriever`, so `_focus_text` skips the hybrid/KAG branch
  entirely and falls back to plain BM25. **Phase 3's retrieval work and
  Phase 4's Prompts 6–7 are currently unmeasured.** Wire the benchmark the
  way production does it (`chunk_pages_with_provenance` → `source_chunks`
  + `hybrid_rank_texts`), then re-baseline. Expect the number to move on
  its own — that alone is a real result.
- **5.3 Widen the extraction vocabulary.** The prompt's allowed-`type`
  enum lists 14 types while `CANONICAL_TYPES` has 39. Types the model
  cannot name, it cannot extract. Reconcile these deliberately (not every
  canonical type belongs in the prompt) and extend `_LAYER_PATTERNS` /
  few-shot examples for the families that benchmark worst.
- **5.4 Close the loop.** Every change in 5.1–5.3 re-runs the benchmark.
  Record before/after in the results directory. A change that does not
  move a metric gets reverted, not rationalized.

- **5.5 Fix the fidelity reporting gap.** Either generate code in the
  benchmark path so `score_fidelity` can run, or remove `fidelity_score`
  from the live aggregate. A permanently-null metric in a results file is
  worse than an absent one — it reads as "measured, scored zero."

**Exit gate:** from the 2026-08-31 baseline (recall 0.502, precision 0.710,
hyperparam 0.000, family 0.700):
- layer-type recall ≥ 0.70 mean, **and no paper below 0.40** — the mean
  alone would let ResNet-50 stay broken;
- hyperparam_accuracy ≥ 0.50 (from 0.0);
- no paper extracting fewer than 5 layers without a written explanation;
- precision not below 0.65 (guard against the opposite failure — dumping
  layer types to inflate recall);
- a written attribution for every remaining miss.

### Phase 5 — RESULT: 3 of 5 criteria met (2026-09-04)

Measured on a fully clean run: all 10 papers on the LLM path,
`rule_based_fallback_count: 0`, `hard_failures: 0`
(`benchmarks/results/phase5-clean.json`).

| Exit criterion | Target | Actual | |
|---|---|---|---|
| layer-type recall (mean) | >= 0.70 | **0.713** | MET |
| precision | >= 0.65 | **0.727** | MET |
| `--check` guarding a baseline | exists | `benchmarks/baseline.json` | MET |
| hyperparam_accuracy | >= 0.50 | **0.400** | missed |
| no single paper below | 0.40 | dcgan at **0.25** | missed |

Per paper: mobilenet_v1 1.00, unet 1.00, densenet121 0.83, vit_base 0.80,
transformer_base 0.75, ddpm 0.67, efficientnet_b0 0.67, resnet50 0.67,
bert_base 0.50, dcgan 0.25. family_correct 0.90.

**The earlier 0.465 figure was never a measurement of the pipeline.** It
was an unlabeled blend of LLM output and silently-degraded rule-based
output. Three separate defects were causing that, all found and fixed:

1. **Empty completions were treated as success** (`core/llm_client.py`).
   `content or ""` returned an empty string as a successful result; it
   surfaced downstream as "LLM did not return valid JSON" and
   `extract_from_text`'s broad `except` swapped in the rule-based
   extractor. A **production** bug, not a benchmark one: any ingestion
   hitting an empty completion was quietly degrading. Now retried like a
   rate limit, then raised.
2. **The JSON parser rejected recoverable output**
   (`core/rag/config_extractor.py`). Models emit `// comments` and trailing
   commas inside JSON; densenet121 failed on `// bottleneck 1x1`. The
   parser now repairs comments and trailing commas, tolerates preamble
   text and truncated fences, and preserves slashes inside strings.
3. **No `max_tokens` ceiling.** The provider default truncated long specs
   mid-JSON — U-Net was cut at 1084 chars. Reasoning models spend much of
   the budget before emitting output, so 4096 was still not enough (3219
   chars, still truncated). At 16384 U-Net completes: **recall 1.00**,
   up from a rule-based 0.75 with a garbage spec.

Also settled during the phase: the noise floor is **zero** (`temperature=0`
is pinned, and three live re-extractions were bit-identical), and
**production retrieval beats legacy** — controlled comparison gave recall
0.583 vs 0.417 and family 0.75 vs 0.50, with the production focused text
containing far more architecture vocabulary (EfficientNet: `conv` x25 vs
x9, `mbconv` x7 vs x0). An earlier n=1 reading that suggested the opposite
was wrong.

**Carried into Phase 6:** `hyperparam_accuracy` at 0.400 and dcgan at 0.25
recall. Both are extraction-correctness problems, which is exactly Phase 6's
scope — parameters and dimensions. They do not need a separate remediation
pass.

### Phase 6 — RESULT: 3 of 4 criteria met (2026-09-07)

Best measurement: `benchmarks/results/20260907T065724Z.json` (25s pacing,
**0 rule-based fallbacks**, 4 provider fallbacks). The aggregate is therefore
**not** baseline-comparable; the 6 Groq-served papers are.

| Exit criterion | Target | Actual | |
|---|---|---|---|
| hyperparam_accuracy | >= 0.50 | **0.533** | MET |
| layer_type_recall | >= 0.70 | **0.702** | met, barely |
| precision | >= 0.65 | **0.654** | met, barely |
| no single paper below 0.40 | — | ddpm at **0.33** | missed |

Verified on Groq-served papers (valid against `phase5-clean`):

| paper | recall | precision | hyperparam |
|---|---|---|---|
| bert_base | 0.50 -> **0.75** | 0.33 -> **0.60** | 0.67 -> 0.67 |
| dcgan | 0.25 -> **0.50** | 1.00 -> 0.50 | — |
| resnet50 | 0.67 -> **0.83** | 1.00 -> 0.71 | 0.33 -> **0.67** |
| transformer_base | 0.75 -> **1.00** | 1.00 -> 0.57 | 1.00 -> 0.67 |
| densenet121 | 0.83 -> 0.50 | 0.56 -> 0.75 | 0.00 |
| ddpm | 0.67 -> 0.33 | 0.50 -> 0.50 | — |

`benchmarks/baseline.json` was **not** refreshed. A provider-mixed run must
never become the reference.

### The root cause took five attempts to find

BERT's `hidden_size: 768` was unreachable throughout Phase 6. Four
hypotheses were tested and refuted or found insufficient before the real
cause surfaced. All four are recorded because each was reasonable, and
because the pattern — *the cause moving upstream at every step* — is the
most useful thing this phase produced.

1. **Prompt text.** Rolling back the params instruction did not restore
   BERT's hyperparameters. Refuted by measurement on a Groq-served paper.
2. **Numeric-only reservation filter.** Reserving structured chunks that
   merely contained a digit protected BERT's SQuAD and CoNLL *results*
   tables — the most numeric objects in any ML paper — while evicting the
   prose stating the dimensions. Real bug, fixed, insufficient.
3. **Reservation evicting prose.** Making the reservation require
   architectural vocabulary stopped results tables consuming slots
   (`selected_structured: []`). Real, necessary, still insufficient.
4. **Expansion iterating by index, not rank.** `ranked_indices` was a
   `set`, discarding retrieval order, then iterated with `sorted()`. Chunk 0
   (title/abstract) expanded first purely for being early and consumed the
   budget before the architecture heading got a turn. Real bug, fixed,
   still insufficient.
5. **Query parity — the actual cause.** `hybrid_rank_texts` searched
   `_ARCHITECTURE_QUERY`, which had **no numeric tokens**, while the
   standalone `retrieve_top_chunks` searched `_ARCH_QUERY_TERMS`, which
   includes `64, 128, 256, 512, 768, 1024, 2048`. Plain BM25 ranked BERT's
   spec chunk **first**; the production hybrid retriever did not rank it at
   all. The hybrid's BM25 half had been given a strictly weaker query than
   the fallback it was meant to improve on — which is how production could
   lose to its own baseline. Adding the numerics moved that chunk from
   **rank 6, evicted by MMR** to **rank 2, selected**, and BERT's
   `hyperparam_accuracy` from 0.00 to **1.00** on an isolated probe.

Also fixed in this phase: symbolic parameter values (`channels: "2k"`,
`compression: "theta"`, `out_features: "num_classes"`) are now rejected at
normalization rather than reaching codegen; the `convtranspose2d` pattern
matches `transposed convolution` and `fractional-strided convolutions`
including plurals (it previously matched only `deconvolution`, and only in
the singular); and provider provenance is recorded per paper, with the
litellm routing-prefix false positive fixed.

### Post-Phase 6: PDF text extraction was silently degraded (2026-09-08)

This sits **upstream of every fix in Phases 5 and 6**. `pdfplumber`'s
`extract_text()` defaults to `x_tolerance=3`, which merges adjacent words.
All ten benchmark papers were affected, and nothing anywhere in the pipeline
reported a problem — the text arrived, it was simply wrong.

| paper | space% before -> after | avg letter-run before -> after |
|---|---|---|
| transformer_base | 3.3% -> **13.4%** | 12.4 -> 5.1 |
| efficientnet_b0 | 5.6% -> 14.3% | 8.3 -> 4.7 |
| dcgan | 5.9% -> 13.6% | 10.2 -> 5.4 |
| ddpm | 5.9% -> 15.1% | 10.1 -> 4.9 |
| vit_base | 6.6% -> 14.0% | 8.7 -> 5.0 |
| densenet121 | 7.9% -> 14.5% | 7.8 -> 5.0 |
| mobilenet_v1 | 8.0% -> 15.0% | 7.9 -> 5.1 |
| resnet50 | 8.3% -> 15.3% | 7.6 -> 4.8 |
| bert_base | 10.1% -> 14.2% | 6.4 -> 4.8 |
| unet | 11.1% -> 14.1% | 6.0 -> 4.9 |

Ordinary English prose runs ~16% spaces. Before the fix the corpus ran
3.3-11.1%; after, 13.4-15.3%, with letter runs falling from 6.0-12.4 chars
to 4.7-5.4. Method: `[A-Za-z]+` runs over the first 30 pages. (An earlier
note in this project cited 14.4 chars for `transformer_base`; that figure is
not reproducible by either letter-run or whitespace tokenisation and has been
corrected in the code comments to 12.4.)

**Why it degraded everything at once.** Run-together text breaks regex word
boundaries, BM25 tokenisation, and embedding quality simultaneously — so the
layer-pattern matcher, the lexical half of hybrid retrieval, and the dense
half all failed together, for one shared reason, while each looked like an
independent tuning problem.

**Evidence at the retrieval level** (ddpm, the only paper below the Phase 6
floor). Its focused text before and after:

| expected type | before | after |
|---|---|---|
| `multiheadattention` | **absent** | `self-attention` |
| `groupnorm` | `groupnorm` | `group normalization`, `group norm` |
| `positionalembedding` | `sinusoidal` only | `position embedding`, `sinusoidal` |

The second row is the instructive one: `groupnorm` "matched" before only
because the words had been *fused* into a token resembling the canonical
name. An accidental hit, not working extraction.

**Correlation with scores is suggestive, not established.** Spearman rho
between pre-fix space% and Phase 6 recall is **0.75 on n = 6**, against a
critical value of ~0.83 — it does not clear significance, and
`transformer_base` is a visible outlier (worst spacing, middling recall).
The two worst-spaced papers were the two lowest scorers (dcgan 0.25,
ddpm 0.33) and the best-spaced scored 1.00 (unet), which is consistent with
the theory without confirming it.

**Fixed at all three extraction sites**, including the production ingestion
path, so this improves real uploads and not only the benchmark:
`benchmarks/harness.py`, `core/paper_to_code_generator.py`,
`backend/services/paper_ingestion_service.py`.

**Status: unproven at the score level.** This demonstrates better text
reaching the model and no test regressions. It does *not* show recall or
hyperparam accuracy moving — no test asserts on extraction quality. The
clean live run remains the deciding measurement.

### Measurement constraint: daily token quota, not rate limiting

Seven consecutive full-run attempts were blocked or contaminated. Slowing
pacing from 25s to 45s made results **worse** (provider fallbacks 4 -> 6,
rule-based 0 -> 3), which establishes the limit as a **daily token budget**
rather than a request-rate ceiling. Pacing cannot help with an exhausted
budget; it only consumes more of what remains.

Practical consequence: spend a fresh daily budget on **one** ten-paper run,
and diagnose everything else deterministically. `_focus_text` calls no LLM,
so retrieval questions are answerable at zero quota cost — that is how the
five-step chain above was actually solved.

### Carried into Phase 7

- **ddpm regressed 0.67 -> 0.33** and is the only paper below the floor.
  Partly explained by the PDF finding above: two of its three expected layer
  types had no readable evidence in the focused text at all. Re-measure
  before pursuing other hypotheses.
- **Precision is below its 0.727 baseline (0.654) and the cause is still
  unattributed.** It first dropped with Prompts 1+2 and never fully
  recovered. Two live candidates: numeric query tokens pulling results
  tables into *prose* slots (the reservation filter guards only the reserved
  slots), and degraded text inflating spurious layer matches. The PDF fix
  may resolve it without further work — measure before changing anything.
- **ViT's recovery is unverified** — it was Gemini-served in the best run.
- One clean ten-paper run, then refresh `benchmarks/baseline.json`.
  **`--check` now fails until that rebuild happens**, by design:
  `check_against_baseline` compares `primary_model`, and the current
  baseline predates the field. Its source run (`phase5-clean.json`) recorded
  no provider information, so the field cannot be honestly backfilled — the
  baseline is genuinely unverifiable for provider purity, and now says so
  instead of passing blind.

### Phase 7 — Corpus scale-out (10 -> 25 -> 200)

Was Phase 6; deferred again, behind extraction correctness. Unchanged in
substance.

- Scale labels 10 → 25 across the families in `core/classification.py`,
  then 25 → 200 once per-paper labeling cost is understood.
- Report first-pass and repaired pass rates **separately**.
- Track metric drift across runs; a corpus is only useful as a time series.
- Labeling discipline stands: label only what the paper explicitly states.
- **Precondition:** Phase 6's gate met. Do not scale the corpus against a
  known-broken extractor.

**Exit gate:** 25 papers labeled and green in CI; per-family breakdown
showing which architectures the system is actually reliable on.

### Phase 8 — Hardening, rollout, continuous evaluation

Mostly the original Phase 7, plus what execution has since surfaced.

- **Commit and deploy the Phase 3–4 backlog.** ~1,600 lines are sitting
  uncommitted in the working tree. This is the largest standing risk in
  the project and is not a code problem.
- **Benchmark in CI** on a small subset, so extraction quality regressions
  fail a build instead of being discovered a phase later.
- **Revisit OCR.** Deliberately skipped in Phase 4 — `rapidocr-onnxruntime`
  pulls `onnxruntime` + `opencv` into a Docker build that has already hit
  pip OOM, for an edge case with no observed demand. Revisit only on real
  user evidence. The `ocr_pdf_pages()` seam already exists. PyMuPDF (already
  a dependency) can supply rasterization if it is ever added.
- **Private-paper leak audit** through Qdrant and graph retrieval —
  specified in the original test plan, never verified. Chunk-level indexing
  makes this materially more important than when it was written: chunks now
  carry paper text into a second collection.
- **Cost and latency budget** per paper, now that the pipeline makes
  multiple LLM calls per extraction plus embedding calls.

### Explicitly not doing yet

- **Agent-layer orchestration.** Deferred since the original assessment and
  still correct to defer. A bounded coordinator for hard papers is only
  worth building once extraction quality is measured and improved —
  otherwise it adds an expensive layer over an unmeasured problem.
- **Training-result reproduction.** Out of scope by planning default;
  the target remains verified minimal model reproduction.

---

## Interfaces and Persistence

Introduce durable records equivalent to:

- `PaperPipelineRun`
  - paper ID, status, stage, versions, timestamps, cost, final confidence.
- `PaperPipelineAttempt`
  - attempt number, prompt/evidence references, generated artifact, diagnostic, validation result.
- `PaperChunk`
  - paper, page, section, text, metadata, embedding ID.
- `GeneratedArtifact`
  - files, language/framework, source type, verification status.
- `VerificationReport`
  - syntax, import, instantiate, forward, shape, backward, safety, runtime, logs.

Keep the existing upload endpoint asynchronous. Enrich task polling with normalized progress and return the final `paper_id`, `run_id`, and verification summary.

## Test and Acceptance Plan

- Upload fails clearly when Terms are not accepted and succeeds when accepted.
- Frontend correctly handles the asynchronous task response.
- Scanned/image-only PDFs return an OCR-required state rather than "no architecture."
- Papers with missing architecture details produce documented defaults.
- Known-family code is self-contained and executable.
- Unknown-family generation is tested through the same compiler.
- Runtime, tensor-shape, dependency, timeout, memory, and unsafe-code failures each produce structured diagnostics.
- Repair stops after three attempts and preserves every attempt.
- Retrieval results include page/section provenance.
- Private papers cannot leak through Qdrant or graph retrieval.
- Corpus evaluation reports first-pass and repaired pass rates separately.
- Existing focused tests remain green, supplemented by genuine PDF-to-sandbox integration tests.

## Work Estimate and Defaults

For one experienced engineer, a reliable initial release is approximately **8–12 engineering weeks**, with corpus curation requiring another **4–8 person-weeks** that can run in parallel. Two engineers plus a technical content reviewer could reach the first 25-paper quality gate in roughly **5–7 weeks**.

Planning defaults:

- PyTorch architecture papers only.
- Verified minimal model reproduction, not complete training-result reproduction.
- Asynchronous 5–10 minute jobs.
- Maximum three repair attempts.
- Existing Qdrant, Postgres, Celery, LiteLLM, and E2B infrastructure.
- Retrieval and evaluation before fine-tuning.
- No autonomous git commits or pushes.
