# Phase 5 — Remediation Prompts

Phase 5 was reported complete. It was not. An audit of the working tree on
2026-09-02 found **two of six prompts never landed**, and one of them is
precisely the prompt targeting the metric that did not move.

Run these before Phase 6. One at a time, §0.5 preamble pasted above each.

---

## §0 — WHAT ACTUALLY LANDED (audited, not reported)

| Phase 5 prompt | Landed | Evidence in the tree |
|---|---|---|
| 1 — Unblock type vocabulary | YES | Expanded allowed-type list present in `_LLM_EXTRACTION_PROMPT` |
| 2 — Re-baseline + label audit | YES | Labels 50 → 45 types, `schema_version` + validation in `load_label` |
| 3 — Production retrieval | YES | `--retrieval` flag, mode-keyed cache (`<id>.<mode>.json`) |
| 4 — Parameter extraction | **NEVER LANDED** | Params instruction byte-identical to the original; `num_classes` absent from `_PARAM_PATTERNS` |
| 5 — Diagnose collapse cases | partial | `benchmarks/diagnose.py` and `.diag.json` files exist |
| 6 — Fidelity + CI guard | **NEVER LANDED** | No `--check`, no `benchmarks/baseline.json` |

**This explains the gate failure.** `hyperparam_accuracy` went 0.000 → 0.000
across the whole phase because the prompt meant to fix it was never applied.
That is a different problem from "we tried and it is hard," and it needs a
different response: do the work, then measure.

### The gate, as it actually stands

| Criterion | Target | Actual | |
|---|---|---|---|
| layer-type recall (mean) | >= 0.70 | 0.465 | fail |
| no single paper below | 0.40 | 5 of 10 below | fail |
| hyperparam_accuracy | >= 0.50 | **0.000** | fail |
| precision | >= 0.65 | 0.593 | fail |
| `--check` guarding a baseline | exists | missing | fail |

### Hard evidence to work from (verified 2026-09-02)

ResNet-50, current production cache — 33 layers, and the params are:

```
conv2d           {"kernel_size": 3}    x22
residualblock    {}                    x7
linear           {}                    x2
globalavgpool2d  {}                    x1
residualblock    {"kernel_size": 3}    x1
```

Read that carefully: **22 convolutions, every one asserting a 3x3 kernel,
and not a single channel count in the entire spec.** The paper explicitly
states a 7x7 stem, 64 channels, and 1000 classes. The model captured none
of them and invented `3` everywhere instead.

There is **no default-injection bug in the code** — `normalize_config` does
not add params, and the table path at `config_extractor.py:~750` did not run
here. The fabrication comes from the LLM itself, guessing the most
conventional value while omitting the stated ones. The prompt already
forbids this and is being ignored. This single line is the entire current
parameter instruction:

```
- "params": ONLY extract values EXPLICITLY stated in the text. Do NOT guess.
```

---

## §0.5 — UNIVERSAL PREAMBLE (paste above EVERY prompt)

```
GROUND RULES.

1. REALITY CHECK FIRST. This prompt states exact paths, signatures, and
   line numbers. Confirm each before editing. If ANY claim does not match
   the repo, STOP and report it. Do not substitute a plausible guess.

2. THIS PHASE HAS A HISTORY OF WORK BEING REPORTED BUT NOT APPLIED.
   Before claiming a task is done, re-read the file you edited and paste
   the actual changed lines into your report. "I updated the prompt" is
   not evidence; the diff is.

3. DO NOT INVENT APIs. Only call what you have read in this repo.

4. NO NEW DEPENDENCIES. None, in any prompt here.

5. MEASURE, DO NOT ASSERT. Every prompt ends with a benchmark run and a
   pasted before/after aggregate. A change that moves nothing is REVERTED,
   not rationalized.

6. NEVER GAME THE METRIC. No paper-specific special cases, no tuning
   thresholds against the 10 labeled papers, no emitting extra types or
   params to inflate a score. Overfitting the test set destroys the
   benchmark permanently.

7. ABSENT BEATS FABRICATED. An unstated value must be OMITTED, never
   defaulted and never None. A fabricated number is worse than a missing
   one: it looks like evidence, survives normalization, and the fidelity
   scorer accepts it.

8. NULL SAFETY. Guard with `x is not None`, never `hasattr`, never bare
   truthiness (0 is falsy).

9. DO NOT COMMIT. No git commit/push/reset/checkout --.

10. REPORT HONESTLY, INCLUDING NEGATIVES. If a number gets worse, show it.

ENVIRONMENT
  Python: .venv/Scripts/python.exe   Root: C:\papper2code
  Suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
  MUST NOT REGRESS: 1803 passed, 2 skipped, 9 deselected, 0 failed.
  Benchmark: .venv/Scripts/python.exe -m benchmarks.harness [--live]
  CURRENT: recall 0.465 | precision 0.593 | hyperparam 0.000 | family 0.500
  temperature=0 is pinned (core/llm_client.py:95), so repeat runs are
  bit-identical. The noise floor is ~zero: any difference IS signal.
```

---

## PROMPT R1 — Apply Prompt 4 (parameter extraction) for real

> Copy from here down

**CONTEXT.** Phase 5 Prompt 4 was reported done but never landed. Verify
this yourself first: `_LLM_EXTRACTION_PROMPT`'s params line should still
read exactly `- "params": ONLY extract values EXPLICITLY stated in the
text. Do NOT guess.` and `_PARAM_PATTERNS` should have no `num_classes`.
If either has changed, STOP — someone else has been editing and this
prompt's premise is stale.

`hyperparam_accuracy` is **0.000** and has never moved.

**ALLOWED FILES**
- `core/rag/config_extractor.py`
- `tests/test_phase2_config_extractor.py` (extend)

**TASK**

1. **Reword the params instruction.** Keep the anti-fabrication intent — it
   is protecting against a real and worse failure — but make it permissive
   about *stated* values and explicit about omission. The current one line
   is doing three jobs badly. Replace it with guidance that says, in your
   own words: extract every value the text explicitly states; when a value
   is not stated, omit the key entirely rather than guessing; never
   substitute a conventional default. Name the specific keys worth looking
   for (`channels`, `kernel_size`, `stride`, `padding`, `hidden_size`,
   `num_heads`, `num_layers`) so the model knows what to hunt for.

2. **Address the observed fabrication directly.** The model currently emits
   `kernel_size: 3` on all 22 ResNet convolutions while omitting channels
   entirely. Add an explicit instruction that a layer with no stated
   parameters must have `"params": {}` — and that guessing a common value
   like 3x3 is a failure, not a helpful default.

3. **Strengthen the few-shot examples.** Example 1 (ResNet) already shows
   `kernel_size`/`channels`/`stride`. Check Examples 2 and 3 — if they show
   sparse params, enrich at least one to demonstrate a densely
   parameterized layer including `channels`. The model imitates the
   examples more reliably than it obeys the rules text.

4. **Add `num_classes` to `_PARAM_PATTERNS`.** Labels expect it and the
   pipeline cannot produce it. Patterns in the spirit of
   `(\d+)[\s-]*way classification`, `(\d+) classes`,
   `num_classes\s*[:=]\s*(\d+)`. Match the existing list's structure
   exactly. Also add it to the key list in the reworded prompt from step 1.

**VERIFY**

1. Rule-based extraction on "7x7 convolution with 64 filters, stride 2"
   yields `kernel_size=7`, `channels=64`, `stride=2`.
2. "1000-way classification" yields `num_classes=1000`.
3. Text with no class count yields **no `num_classes` key at all** —
   absent, not `None`, not 0. Assert key absence explicitly.
4. **Anti-fabrication regression test:** text naming only a layer type
   ("the network uses a convolutional layer") yields `params: {}`. This is
   the guard that the reword did not trade one failure for a worse one.
5. Paste the actual changed prompt text in your report.
6. Live benchmark, before/after pasted.

```
.venv/Scripts/python.exe -m pytest tests/test_phase2_config_extractor.py -q -m "not live"
.venv/Scripts/python.exe -m benchmarks.harness --live
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — `hyperparam_accuracy` > 0.000 with the real number, recall
and precision not regressed, and the changed prompt text quoted verbatim.

> Copy to here up

---

## PROMPT R2 — Verify the fabrication actually stopped

> Copy from here down

**PREREQ:** R1 landed.

**CONTEXT.** R1 changes what the model is *told*. This prompt checks what it
actually *did*. A metric can improve while the underlying pathology
persists: `hyperparam_accuracy` could rise because `num_classes` now
matches, while all 22 convolutions still carry a fabricated `kernel_size: 3`.

This is a measurement prompt. It may end with no code change at all.

**ALLOWED FILES**
- create `benchmarks/audit_params.py`
- `core/rag/config_extractor.py` (only if the audit proves a fix is needed)
- tests as needed

**TASK**

1. Write `benchmarks/audit_params.py` — offline, no LLM calls, no new
   dependencies. For every cached spec it reports:
   - total layers, and layers with non-empty `params`;
   - each distinct `(param_key, value)` and its repeat count;
   - a **fabrication flag**: any `(key, value)` appearing on more than half
     of same-typed layers, which is the signature of a stamped default
     rather than an extracted value;
   - a **completeness flag**: any `conv2d` / `linear` layer with no
     `channels` or output size.
2. Run it on all cached specs, before and after R1. Paste both.
3. For ResNet-50 specifically, answer with evidence: does the spec now
   contain `channels=64` and `kernel_size=7`? Do the 22 convolutions still
   all claim 3? Quote the actual params.
4. **If fabrication persists**, do not patch around it. Use
   `benchmarks/diagnose.py` and the `.diag.json` files to check whether the
   focused text sent to the LLM even contains the sentence stating the 7x7
   stem and 64 channels. Report which it is:
   - text absent -> retrieval problem, fix belongs in Phase 6;
   - text present but ignored -> prompt/model problem, propose a fix and
     stop for approval.

**VERIFY**
1. `audit_params.py` runs offline with zero LLM calls.
2. It flags the known pre-R1 ResNet case (`kernel_size=3` on 22 of 22
   convolutions) as fabrication — validate the detector against the
   known-bad case before trusting it on new data.
3. It flags 22 convolutions with no `channels` as incomplete.

**EXIT GATE** — an evidence-backed statement of whether fabrication stopped,
with real param dumps quoted. No speculative fixes.

> Copy to here up

---

## PROMPT R3 — Apply Prompt 6 (baseline + regression guard) for real

> Copy from here down

**PREREQ:** R1–R2 landed.

**CONTEXT.** Phase 5 Prompt 6 never landed: there is no `--check`, no
`benchmarks/baseline.json`, and `fidelity_score` is still permanently null
on the live path. Until this exists, every gain from R1 is one refactor away
from silently disappearing — which is exactly how Phase 5 lost two prompts'
worth of work without anyone noticing.

`.gitignore` was already fixed (2026-09-02) to un-ignore
`benchmarks/labels/*.json`, `benchmarks/baseline.json`, and
`benchmarks/results/*.json`. Do not edit `.gitignore`.

**ALLOWED FILES**
- `benchmarks/harness.py`
- `tests/test_benchmark_harness.py` (extend)
- create `benchmarks/baseline.json`
- CI config, if one exists

**TASK**

1. **Fidelity — choose (b) unless you can argue otherwise.** Drop
   `fidelity_score` from the live aggregate and report it only for offline
   runs where cached `code` exists. The benchmark measures *extraction*;
   folding in generation makes a regression harder to localize. If you
   choose (a) instead, it must sit behind a `--fidelity` flag, off by
   default. Justify the choice in your report.
2. Add `--check`: run offline against cached extractions, compare to
   `benchmarks/baseline.json`, exit non-zero if any metric drops by more
   than 0.05 absolute.
3. Create `benchmarks/baseline.json` from the current post-R1 aggregate.
   Record in it the retrieval mode and the label `schema_version` — a
   baseline compared against different labels is meaningless, which is the
   confound that already bit this phase once.
4. Wire `--check` into CI if a config exists; otherwise document the command
   in the harness docstring and say plainly that CI wiring was not possible.

**VERIFY**
1. `--check` against an unchanged baseline exits 0.
2. `--check` against a synthetically degraded result exits non-zero and
   names the metric that dropped.
3. An improved metric does not fail the check.
4. A metric in the baseline but missing from the run is an **error**, not a
   silent pass — this is the failure mode that lets a metric quietly vanish.
5. A baseline whose `schema_version` differs from the current labels is an
   error naming both versions.
6. `--check` makes zero LLM calls.

```
.venv/Scripts/python.exe -m pytest tests/test_benchmark_harness.py -q
.venv/Scripts/python.exe -m benchmarks.harness --check
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — `--check` works both directions, `benchmarks/baseline.json`
exists and is git-visible (`git check-ignore -v benchmarks/baseline.json`
returns nothing), full suite >= 1803.

> Copy to here up

---

## PROMPT R4 — Structural misses: CNNs with no convolution

> Copy from here down

**PREREQ:** R1–R3 landed.

**CONTEXT.** Recall is 0.465, and the misses are not exotic. From the latest
10-paper run:

| Paper | recall | Missing |
|---|---:|---|
| bert_base | 0.25 | `transformerblock`, `gelu`, `positionalembedding` |
| dcgan | 0.25 | `convtranspose2d`, `batchnorm2d`, `relu` |
| densenet121 | 0.33 | `concat`, `batchnorm2d`, `relu`, `globalavgpool2d` |
| efficientnet_b0 | 0.33 | **`conv2d`**, `silu` |
| vit_base | 0.40 | `patchembedding`, `layernorm`, `positionalembedding` |
| mobilenet_v1 | 0.50 | **`conv2d`**, `linear`, `avgpool2d` |
| unet | 0.75 | **`conv2d`** |

**Three CNN papers produced specs containing no `conv2d` at all**, and BERT
produced one with no `transformerblock`. These are not vocabulary gaps — all
four types were always allowed. Meanwhile DCGAN gained `multiheadattention`
and `maxpool2d`, and MobileNet/EfficientNet gained `residualblock`, none of
which belong.

So the extractor is simultaneously **missing the obvious and inventing the
absent**. That combination usually means the model is not seeing the
architecture description at all and is pattern-matching from the abstract.

**ALLOWED FILES**
- `benchmarks/diagnose.py` (extend if needed)
- `core/rag/config_extractor.py` (only if the diagnosis proves the fix is there)
- tests as needed

**TASK**

1. **Diagnose before touching anything.** For `efficientnet_b0` (missing
   `conv2d`) and `unet` (missing `conv2d`), dump the focused text actually
   sent to the LLM. Answer with quoted evidence: does it contain the
   architecture description, or is it abstract / intro / references text?
2. Attribute to exactly one stage and say which:
   - **retrieval** — the focused text does not contain the architecture
     section;
   - **prompt/model** — the text is there and the model ignored it;
   - **normalization** — the type was emitted then dropped.
3. Compare against a paper that scores well (`unet` at 0.75, or `ddpm` at
   0.67) at the same stage. The difference between a good and a bad case is
   the signal; a single bad case in isolation is not.
4. **Report the attribution and a proposed fix, then stop for approval.**
   Retrieval, prompt, and normalization fixes are entirely different pieces
   of work and guessing wrong wastes the remediation.

**VERIFY**
1. Diagnosis is evidence-backed: quote actual focused text, not summaries.
2. The comparison covers at least one good and one bad paper.
3. No code change without an approved attribution.

**EXIT GATE** — written attribution for the missing-`conv2d` cases with
quoted evidence, and a proposed fix awaiting approval.

> Copy to here up

---

## PROMPT R5 — Final Phase 5 verdict

> Copy from here down

**PREREQ:** R1–R4 landed (R4 may have ended at diagnosis without a fix —
that is acceptable).

**TASK**

1. Run the full suite and the live benchmark. Paste both.
2. Report each Phase 5 exit criterion explicitly:

| Criterion | Target | Actual | Met? |
|---|---|---|---|
| layer-type recall (mean) | >= 0.70 | | |
| no single paper below | 0.40 | | |
| hyperparam_accuracy | >= 0.50 | | |
| precision | >= 0.65 | | |
| `--check` guarding a baseline | exists | | |

3. Append a dated section to `docs/PAPER_TO_CODE_EXECUTION_MEMORY.md`
   covering what R1–R4 changed, what moved, and what did not.
4. Update the Phase 5 RESULT table in `docs/PAPER_TO_CODE_MASTER_PLAN.md`
   §8 with the final numbers.
5. State plainly whether Phase 5's gate is met. **If it is not, say so and
   list precisely what remains** — a partial result stated honestly is the
   expected outcome and is worth more than a claim of success. Phase 5 has
   already been reported complete once while two prompts sat unapplied; do
   not repeat that.

**EXIT GATE** — an honest verdict with numbers, both docs updated.

> Copy to here up
