# Phase 5 — Closeout Prompt (single, final)

One prompt. Everything remaining in Phase 5. Give this to Codex as-is.

---

## GROUND RULES

1. **REALITY CHECK FIRST.** This prompt states exact paths, line numbers,
   and flags, all verified 2026-09-04. Confirm each before editing. If
   something genuinely contradicts the repo, STOP and report it. But note
   the distinction: *"the repo contradicts this prompt"* is worth stopping
   for; *"my prepared patch no longer applies"* is not — in that case
   re-read the file and edit fresh. Never patch against a remembered
   snapshot; this repo changes between your turns.
2. **DO NOT INVENT APIs.** Only call what you have read here.
3. **NO NEW DEPENDENCIES.**
4. **REPORTED IS NOT LANDED.** Phase 5 has twice had work reported complete
   that was never applied. Before claiming a task is done, re-read the file
   and paste the actual changed lines. The diff is the evidence.
5. **THE EMBEDDER IS NOT HUNG.** `_get_embedder()` takes ~24s to load from
   a 129M local cache on a cold process. Wait it out. Three separate
   diagnostics have been abandoned by killing it early.
6. **A RUN WITH FALLBACKS IS NOT A MEASUREMENT.** If
   `rule_based_fallback_count > 0`, the aggregate is a blend of LLM output
   and rate-limit-degraded rule-based output. Do not record it as a result.
7. **DO NOT COMMIT.** No git commit/push/reset/checkout --.
8. **REPORT HONESTLY, INCLUDING NEGATIVES.**

**ENVIRONMENT**
```
Python: .venv/Scripts/python.exe    Root: C:\papper2code
Suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
CURRENT: 1859 passed, 2 skipped, 9 deselected, 0 failed  <- must not regress
temperature=0 is pinned (core/llm_client.py:95): repeat runs are
bit-identical, so the noise floor is ~zero and any difference IS signal.
```

---

## VERIFIED STATE (2026-09-04)

**Already landed — do not redo:** expanded type vocabulary (28 types),
label audit with `schema_version`, `--retrieval`, parameter extraction
(`num_classes` + reworded params instruction), `benchmarks/diagnose.py`,
`benchmarks/audit_params.py`, `extraction_method` + WARNING logging
(`core/rag/config_extractor.py:355-373`), `rule_based_fallback_count`,
`--strict`, rate-limit pacing.

**Existing CLI flags** (`benchmarks/harness.py:457-464`):
`--live`, `--strict` (BooleanOptionalAction, defaults on for `--live`),
`--pace-seconds` (default 10.0), `--retrieval`, `--timestamp`, positional
label paths.

**Existing aggregate keys** (`harness.py:331-343`):
`layer_type_recall`, `layer_type_precision`, `hyperparam_accuracy`,
`family_correct`, `fidelity_score`, `rule_based_fallback_count`,
`hard_failures`, `papers`.

**Still missing:** `--check`, `benchmarks/baseline.json`, and the Phase 5
documentation updates.

**Why no verdict exists yet.** The last full 10-paper run predates the
fallback instrumentation, so its `rule_based_fallback_count` is `None` and
its recall of 0.465 is known to be contaminated: several papers silently
fell back to the rule-based extractor after hitting Groq rate limits. The
only clean data is a 2-paper run (`efficientnet_b0` 0.67, `unet` 0.75,
0 fallbacks, aggregate 0.708).

`.gitignore` was already fixed to un-ignore `benchmarks/labels/*.json`,
`benchmarks/baseline.json`, and `benchmarks/results/*.json`. **Do not edit
`.gitignore`.**

---

## TASK 1 — Get one clean full run (blocking; everything else depends on it)

```
.venv/Scripts/python.exe -m benchmarks.harness --live --retrieval production
```

Report the full per-paper table, the aggregate, and
`rule_based_fallback_count`.

- If `rule_based_fallback_count == 0`: this is the first trustworthy Phase 5
  number. Proceed to Task 2.
- If it is **> 0**: `--strict` will name the affected papers. Raise
  `--pace-seconds` (try 20, then 30) and retry, up to three attempts. If it
  still will not come back clean, **stop and report** which papers keep
  degrading and at what pacing. Do not proceed to Task 2 with a
  contaminated run, and do not average the fallback papers in.

This is the only step that needs LLM quota. Budget for it.

---

## TASK 2 — Create `benchmarks/baseline.json`

Only from a run with `rule_based_fallback_count == 0`.

Record, alongside the metrics: the retrieval mode, the label
`schema_version`, the source results filename, and the paper count. A
baseline compared against different labels or a different retrieval mode is
meaningless — that confound has already cost this project one round of false
conclusions.

Verify it is git-visible:
```
git status --porcelain benchmarks/baseline.json
```
A `??` line means git sees it. (Do **not** use `git check-ignore -v` — it
exits 0 for both ignored and negated paths and cannot tell them apart.)

---

## TASK 3 — Implement `--check`

Offline regression guard. Scores cached extractions, compares to
`benchmarks/baseline.json`, exits non-zero if any metric drops by more than
**0.05 absolute**.

Requirements:
- Zero LLM calls.
- An **improved** metric never fails the check.
- A metric present in the baseline but **missing** from the run is an
  error, not a silent pass — this is the failure mode that lets a metric
  quietly disappear.
- A baseline whose `schema_version` or `retrieval` mode differs from the
  current run is an error naming both values.

**CI:** unit-test the `--check` logic against synthetic fixtures. Do **not**
wire real `--check` into CI — `.cache/` is intentionally untracked, so a
clean checkout has no inputs, and a CI run scoring a frozen cache could not
detect extraction regressions anyway (the cache does not change when the
extractor does). Document in the harness docstring that real `--check` runs
locally after `--live`, before a phase gate. Say plainly in your report that
CI cannot run the real check, and why.

---

## TASK 4 — Update both documents

**`docs/PAPER_TO_CODE_EXECUTION_MEMORY.md`** — append a dated section
covering: what the remediation prompts changed, the silent-fallback root
cause (LLM rate limiting swallowed by `except Exception`, producing
rule-based specs with tells like `name="UnknownModel"` and
`relu{stride: 2}`), the clean benchmark numbers, and what remains open.
Match the existing style.

**`docs/PAPER_TO_CODE_MASTER_PLAN.md`** — update the
`### Phase 5 — RESULT` section (currently dated 2026-09-02 and carrying the
contaminated 0.465). Replace the numbers with the clean run's, and state
each exit criterion explicitly:

| Criterion | Target | Actual | Met? |
|---|---|---|---|
| layer-type recall (mean) | >= 0.70 | | |
| no single paper below | 0.40 | | |
| hyperparam_accuracy | >= 0.50 | | |
| precision | >= 0.65 | | |
| `--check` guarding a baseline | exists | | |

Add a line recording that the earlier 0.465 figure was contaminated by
rate-limit-induced fallbacks and is not comparable.

---

## VERIFY

```
.venv/Scripts/python.exe -m pytest tests/test_benchmark_harness.py -q
.venv/Scripts/python.exe -m benchmarks.harness --check
.venv/Scripts/python.exe -m pytest -q -m "not live"
git status --porcelain benchmarks/baseline.json
```

1. `--check` against an unchanged baseline exits 0.
2. `--check` against a synthetically degraded result exits non-zero and
   names the dropped metric.
3. An improved metric does not fail.
4. A missing metric is an error.
5. A `schema_version` / retrieval-mode mismatch is an error naming both.
6. `--check` makes zero LLM calls.
7. Full suite >= 1859 passed, 0 failed.

---

## EXIT GATE

State plainly whether Phase 5's gate is met, with the numbers behind it.

**Partial success stated honestly is the expected outcome.** Recall may well
land short of 0.70 even on a clean run — if so, say so and name what
remains. Phase 5 has already been declared complete once while two prompts
sat unapplied; a second false all-clear is worse than an honest miss.

If the gate is not met, end your report with the specific remaining gap and
your recommendation for whether it belongs in Phase 5 or should carry into
Phase 6 (extraction correctness: dimensions).
