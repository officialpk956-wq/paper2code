# Phase 6 — Execution Prompts (extraction correctness: dimensions)

One prompt at a time, in the run order in §2. Paste **§0.5 (Universal
Preamble)** above each. Do not chain prompts. Do not skip a VERIFY gate.

Phase 6 targets the two criteria Phase 5 missed:
**`hyperparam_accuracy` 0.400 → ≥ 0.50**, and **no paper below 0.40 recall**
(dcgan is at 0.25). Everything here either moves one of those or makes it
honestly measurable.

---

## §0 — WHERE PHASE 5 LEFT THIS

Clean baseline, all 10 papers on the LLM path, `rule_based_fallback_count: 0`
(`benchmarks/results/phase5-clean.json`, frozen as `benchmarks/baseline.json`):

| Metric | Value | Phase 5 target | |
|---|---:|---|---|
| layer_type_recall | 0.713 | >= 0.70 | met |
| layer_type_precision | 0.727 | >= 0.65 | met |
| family_correct | 0.900 | — | |
| **hyperparam_accuracy** | **0.400** | >= 0.50 | **missed** |
| **dcgan recall** | **0.25** | no paper < 0.40 | **missed** |

Per-paper recall: mobilenet_v1 1.00, unet 1.00, densenet121 0.83,
vit_base 0.80, transformer_base 0.75, ddpm 0.67, efficientnet_b0 0.67,
resnet50 0.67, bert_base 0.50, **dcgan 0.25**.

### Parameter extraction, as it actually is today

Measured from the clean cache 2026-09-04:

| Paper | layers | with params | conv2d | conv2d carrying `channels` |
|---|---:|---:|---:|---:|
| unet | 53 | 27 | 23 | **0** |
| resnet50 | 22 | 21 | 4 | 4 |
| efficientnet_b0 | 20 | 18 | 18 | 18 |
| densenet121 | 15 | 10 | 1 | 1 |
| mobilenet_v1 | 13 | 4 | 2 | **0** |
| bert_base | 9 | 4 | 0 | — |
| transformer_base | 8 | 5 | 0 | — |
| vit_base | 7 | **0** | 0 | — |
| ddpm | 6 | **0** | 0 | — |
| dcgan | 3 | **0** | 3 | **0** |

**The `kernel_size=3`-stamped-everywhere fabrication from Phase 5 is gone.**
ResNet-50 now carries `channels` on 21 layers. What remains is four
*distinct* problems, and they need different fixes. Do not treat them as one.

**(a) Symbolic values copied instead of resolved — densenet121.**
The spec literally contains:
```
channels: "2k"        growth_rate: "k"      compression: "\u03b8"
num_layers: ["L1","L2","L3"]                out_features: "num_classes"
```
The model faithfully transcribed the paper's algebra. `nn.Conv2d("2k", ...)`
is not buildable, and `out_features: "num_classes"` is the *name* of a
parameter stored as its *value*.

**(b) No parameters extracted at all — vit_base, ddpm, dcgan.**
ViT's paper states hidden size 768, 12 heads, 12 layers plainly in a table.
The spec has `params: {}` on every layer.

**(c) Non-canonical key names — bert_base.**
Spec has `input_size` / `output_size`; the canonical vocabulary has neither.
`core/rag/normalizer.py`'s param synonym map covers `kernel`, `filter_size`,
`size`, `hidden`, `hidden_dim`, `units`, `d_model`, `heads` and friends —
but **not** `out_features`, `output_size`, `input_size`, `growth_rate`, or
`num_classes`. The harness matches keys exactly, so a correctly-extracted
value under an unmapped name scores zero.

**(d) Wrong values — resnet50.** `kernel_size: [3, 3, 3]`; the paper's 7x7
stem is still not captured. The label wants 7.

---

## §0.5 — UNIVERSAL PREAMBLE (paste above EVERY prompt)

```
GROUND RULES.

1. REALITY CHECK FIRST. This prompt states exact paths, values, and
   measurements, all verified 2026-09-04. Confirm before editing.
   Distinguish two cases: "the repo contradicts this prompt" -> STOP and
   report. "My prepared patch no longer applies" -> re-read the file and
   edit fresh, do not stop. This repo changes between turns.

2. DO NOT INVENT APIs. Only call what you have read in this repo.

3. NO NEW DEPENDENCIES.

4. REPORTED IS NOT LANDED. Before claiming a task done, re-read the file
   and paste the changed lines. The diff is the evidence.

5. MEASURE, DO NOT ASSERT. Every prompt ends with a benchmark run and a
   pasted before/after. A change that moves nothing is REVERTED.

6. NEVER GAME THE METRIC. No paper-specific special cases, no tuning
   thresholds against the 10 labeled papers, no emitting params to
   inflate a score. Overfitting the test set destroys the benchmark.

7. ABSENT BEATS FABRICATED, AND ABSENT BEATS SYMBOLIC. An unstated value
   is OMITTED -- never defaulted, never None, and never a placeholder
   string like "k" or "num_classes". A symbolic value is worse than a
   missing one: it passes schema checks, reaches codegen, and produces
   `nn.Conv2d("2k", ...)`.

8. NULL SAFETY. Guard with `x is not None`, never `hasattr`, never bare
   truthiness (0 is falsy and is a legitimate padding value).

9. A RUN WITH FALLBACKS IS NOT A MEASUREMENT. If
   rule_based_fallback_count > 0, --strict rejects it. Do not record it.

10. DO NOT COMMIT. No git commit/push/reset/checkout --.

11. REPORT HONESTLY, INCLUDING NEGATIVES.

ENVIRONMENT
  Python: .venv/Scripts/python.exe    Root: C:\papper2code
  Suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
  MUST NOT REGRESS: 1878 passed, 2 skipped, 9 deselected, 0 failed.
  Benchmark live:    -m benchmarks.harness --live --retrieval production
  Benchmark offline: -m benchmarks.harness
  Regression guard:  -m benchmarks.harness --check   (vs benchmarks/baseline.json)
  BASELINE TO BEAT: recall 0.713 | precision 0.727 | hyperparam 0.400
                    | family 0.900 | fallbacks 0
  temperature=0 is pinned; the noise floor is ~zero. Any difference IS signal.
  The embedder takes ~24s to load on a cold process. It is NOT hung.
  LLM_MAX_COMPLETION_TOKENS defaults to 16384; long specs need it.
```

---

## §1 — GROUND TRUTH

**`core/rag/config_extractor.py`**
```python
_LAYER_PATTERNS      # regex -> canonical type
_PARAM_PATTERNS      # keys produced: kernel_size, channels, stride, padding,
                     #   hidden_size, num_heads, num_layers, num_classes
_LLM_EXTRACTION_PROMPT   # slots: {few_shot} {graph_rules} {operation_context} {text}
                         # allowed "type" list: 28 canonical types
_operation_context(text, limit=8) -> str
_repair_json(text) -> str            # strips // and /* */ comments, trailing commas
ConfigExtractor._parse_json_response(response) -> dict
ConfigExtractor.extract_from_text(text, source_chunks=None) -> ConfigDict
   # records extraction_method: "llm" | "llm_verified" | "rule_based_fallback"
   # and extraction_reason on failure
```

**`core/rag/normalizer.py`** — `CANONICAL_TYPES` (39), `_SYNONYM_MAP` (layer
types), a **separate param-key map** (~line 205-233) containing `in_channels`,
`kernel`, `filter_size`, `size`, `pool_size`, `stride`, `padding`, `hidden`,
`hidden_size`, `hidden_dim`, `units`, `num_units`, `d_model`->`embed_dim`,
`heads`->`num_heads`. `normalize_config()`, `_normalize_params()`,
`_normalize_type()` (never raises on unknown types).

**`core/llm_client.py`** — `MAX_COMPLETION_TOKENS` (env
`LLM_MAX_COMPLETION_TOKENS`, default 16384), `temperature=0`, retry-then-
fallback on rate limits, empty completions retried then raised,
circuit breaker (`FAILURE_THRESHOLD=5`, `CIRCUIT_OPEN_DURATION=60`).

**`benchmarks/harness.py`** — `--live`, `--strict`, `--pace-seconds`,
`--retrieval`, `--check`, `--write-baseline`, `--baseline`, `--tolerance`,
`--timestamp`, positional labels. `_hyperparameters()` does **exact key
matching** on a recursive walk of the spec. `build_baseline()` refuses runs
containing fallbacks. `check_against_baseline()` errors on schema/retrieval
mismatch and on a metric missing from a run.

**Instruments already built:** `benchmarks/diagnose.py` (six-stage dumps from
`<id>.<mode>.diag.json`), `benchmarks/audit_params.py`.

---

## §2 — RUN ORDER

| # | Prompt | Targets |
|---|---|---|
| 1 | Reject symbolic and placeholder parameter values | correctness of what is already extracted |
| 2 | Parameter key vocabulary contract | hyperparam_accuracy (key mismatches) |
| 3 | Extract dimensions where none are extracted | hyperparam_accuracy (vit/ddpm/dcgan) |
| 4 | dcgan structural collapse | "no paper below 0.40" |
| 5 | Final gate, baseline refresh, docs | closeout |

**After every prompt:** run the suite (>= 1878), run `--live`, paste
before/after, append to `docs/PAPER_TO_CODE_EXECUTION_MEMORY.md`, STOP.

---

## PROMPT 1 — Reject symbolic and placeholder parameter values

> Copy from here down

**CONTEXT.** densenet121's spec contains parameters copied verbatim from the
paper's algebra rather than resolved to numbers:

```
channels: "2k"          growth_rate: "k"       compression: "\u03b8"
num_layers: ["L1","L2","L3"]                   out_features: "num_classes"
```

`out_features: "num_classes"` stores a parameter *name* as a *value*. None of
this is buildable — `nn.Conv2d("2k", ...)` fails — and none of it is caught,
because nothing validates that a numeric parameter is actually numeric.

This is the Phase 5 fabrication problem in a new costume: a value that looks
like data and is not. **Absent beats symbolic.**

**ALLOWED FILES**
- `core/rag/normalizer.py`
- `core/rag/config_extractor.py` (prompt text only)
- `tests/test_phase2_null_safety.py` or `tests/test_phase2_config_extractor.py`

**TASK**

1. In `_normalize_params`, drop any value for a **numeric** parameter that is
   not coercible to a number. Numeric keys are at minimum: `channels`,
   `in_channels`, `kernel_size`, `stride`, `padding`, `hidden_size`,
   `embed_dim`, `num_heads`, `num_layers`, `num_classes`, `pool_size`,
   `growth_rate`. Accept ints, floats, and numeric strings (`"64"` -> `64`).
   Reject `"2k"`, `"k"`, Greek letters, `"num_classes"`, and lists of
   symbols. **Drop the key entirely — do not substitute a default or None.**
2. Log dropped values once per layer at DEBUG with the key and the rejected
   value, so the loss is visible without flooding output.
3. Keep genuinely non-numeric parameters that are legitimately non-numeric
   (e.g. `bottleneck: true`). Only coerce/reject the numeric set above.
4. In `_LLM_EXTRACTION_PROMPT`, add one instruction: parameter values must be
   concrete numbers as stated in the paper; if the paper expresses a value
   symbolically (a growth rate `k`, a compression factor), omit the key
   rather than copying the symbol. Keep it to a sentence or two — the rules
   block is already long and every token is spent on every extraction.

**VERIFY**
1. `_normalize_params({"channels": "2k"})` drops `channels`; the key is
   absent, not None.
2. `_normalize_params({"channels": "64"})` yields `64` as an int.
3. `_normalize_params({"channels": 0})` **keeps** 0 — a legitimate value that
   a truthiness check would silently eat.
4. `_normalize_params({"bottleneck": True})` keeps it.
5. `_normalize_params({"out_features": "num_classes"})` drops it.
6. A full `normalize_config` on a densenet-shaped spec containing all the
   symbolic values above produces no symbolic values and does not raise.
7. Live benchmark; paste before/after. **Expect `hyperparam_accuracy` to stay
   flat or dip slightly** — you are removing bad data, not adding good data.
   That is a correct outcome; report it plainly rather than treating it as a
   regression.

```
.venv/Scripts/python.exe -m pytest tests/test_phase2_null_safety.py tests/test_phase2_config_extractor.py -q -m "not live"
.venv/Scripts/python.exe -m benchmarks.harness --live --retrieval production
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

**EXIT GATE** — no symbolic values survive normalization; suite >= 1878;
benchmark run reported even if a metric dipped.

> Copy to here up

---

## PROMPT 2 — Parameter key vocabulary contract

> Copy from here down

**PREREQ:** Prompt 1 landed.

**CONTEXT.** bert_base's spec carries `input_size` and `output_size`.
Neither is in the canonical parameter vocabulary, and
`benchmarks/harness.py:_hyperparameters` matches keys **exactly** — so a
correctly extracted value under an unmapped name scores zero. densenet121
similarly produced `out_features` where the label wants `num_classes`.

The param-key synonym map in `core/rag/normalizer.py` (~line 205-233) covers
`kernel`, `filter_size`, `size`, `hidden`, `hidden_dim`, `units`, `d_model`,
`heads` and friends — but **not** `out_features`, `output_size`,
`input_size`, `growth_rate`, or `num_classes`.

This is the same class of bug as Phase 5's type-enum drift: two vocabularies
that must agree, with nothing asserting that they do.

**ALLOWED FILES**
- `core/rag/normalizer.py`
- `benchmarks/labels/*.json` (only if the audit finds a label at fault)
- `tests/test_phase2_config_extractor.py` (extend)

**TASK**

1. Extend the param-key synonym map for the names actually observed:
   `out_features` / `output_size` / `output_dim` -> decide deliberately
   whether these mean `num_classes` (final classifier) or `hidden_size`
   (intermediate). **They are not the same thing** — a blanket mapping will
   corrupt transformer specs where `output_size: 3072` is a feed-forward
   width, not a class count. If context cannot disambiguate, map to
   `hidden_size` and say why in your report.
   Also map `input_size` / `input_dim` -> `in_channels` or `hidden_size` by
   the same reasoning, and `num_output_classes` / `n_classes` ->
   `num_classes`.
2. **Add a drift guard test.** Every key any label uses in
   `expected.key_hyperparams` must be a key the pipeline can actually
   produce — i.e. present in `_PARAM_PATTERNS` or as a value in the param
   synonym map. Parametrize over `benchmarks/labels/*.json`. This is the
   most important item in this prompt: it is what stops the two vocabularies
   silently diverging again.
3. If the guard fails for a label key that the pipeline genuinely should not
   produce, fix the **label** and justify it. Do not widen the vocabulary
   just to make a test pass.

**VERIFY**
1. `_normalize_params({"out_features": 1000})` yields the canonical key you
   chose, with 1000 intact.
2. `_normalize_params({"num_heads": 12})` is unchanged — existing mappings
   are not disturbed.
3. The drift guard passes for all 10 labels, and **fails** when given a
   synthetic label using an invented key like `wibble_size`.
4. Live benchmark; paste before/after with attention to
   `hyperparam_accuracy` on bert_base and densenet121 specifically.

**EXIT GATE** — drift guard in place and passing; `hyperparam_accuracy`
reported. If it did not move, say so — that would mean key mismatch was not
the binding constraint and Prompt 3 is where the gain is.

> Copy to here up

---

## PROMPT 3 — Extract dimensions where none are extracted

> Copy from here down

**PREREQ:** Prompts 1-2 landed.

**CONTEXT.** Three papers produce specs with **zero** parameters on every
layer: `vit_base` (7 layers), `ddpm` (6), `dcgan` (3). ViT's paper states
hidden size 768, 12 heads, and 12 layers plainly in a table. U-Net has 23
convolutions and **not one** carries `channels`; mobilenet_v1 has 2
convolutions and neither does.

So the model is producing correct *structure* with no *dimensions* — the
central Phase 6 problem. A spec of 23 unparameterized convolutions cannot
generate correct code, and `score_fidelity` cannot tell that it is wrong.

**Diagnose before changing anything.** The cause is not established, and
three candidates need three different fixes:
- the parameter-bearing text never reaches the LLM (retrieval — tables and
  captions score low on an architecture query, and MMR evicts near-duplicate
  chunks, which is exactly where repeated hyperparameters live);
- the text reaches it and the model omits parameters anyway (prompt);
- values are emitted then dropped (normalization — Prompt 1 now drops
  non-numerics, so check this explicitly).

**ALLOWED FILES**
- `benchmarks/diagnose.py` (extend if needed)
- `core/rag/config_extractor.py`
- tests as needed

**TASK**

1. Using `benchmarks/diagnose.py` and the `.diag.json` files, for `vit_base`
   and `unet` answer with **quoted evidence**: does the focused text sent to
   the LLM contain the sentences or tables stating hidden size 768 / 12 heads
   / 12 layers (ViT) and the channel progression 64-128-256-512 (U-Net)?
2. Compare against a paper that parameterizes well — `efficientnet_b0` (18 of
   18 convolutions carry `channels`) or `resnet50` (21 layers with params).
   The difference between a good and a bad case is the signal.
3. **Report the attribution and stop for approval before implementing a
   fix.** State which of the three causes it is, with the evidence.
4. Once approved, implement only the fix the evidence supports. If it is a
   prompt fix, the likely shape is a per-layer instruction — the model
   currently emits one JSON blob covering structure and parameters together,
   and may be treating parameters as optional decoration. If it is
   retrieval, note that tables are already extracted as `chunk_type="table"`
   by `core/utils.py:extract_table_chunks` and may simply be losing the
   ranking.

**VERIFY**
1. Attribution is evidence-backed: quote actual focused text, not summaries.
2. After the approved fix, `vit_base` carries `hidden_size` and `num_heads`,
   or you state precisely why not.
3. Anti-fabrication holds: a paper stating no dimensions still yields
   `params: {}`, not invented numbers. Assert this — it is the failure the
   whole phase is guarding against.
4. Live benchmark; paste before/after.

**EXIT GATE** — evidence-backed attribution, approved fix implemented,
`hyperparam_accuracy` reported against the 0.400 baseline.

> Copy to here up

---

## PROMPT 4 — dcgan structural collapse

> Copy from here down

**PREREQ:** Prompts 1-3 landed.

**CONTEXT.** dcgan is the only paper below the 0.40 floor: **recall 0.25**,
3 layers total, 0 parameters. Its 3 layers are all `conv2d`. The label
expects `convtranspose2d`, `batchnorm2d`, and `relu` among others — DCGAN's
generator is built from transposed convolutions.

`convtranspose2d` was added to the allowed-type list in Phase 5, so this is
no longer a vocabulary gap. Three layers for a paper describing a full
generator and discriminator is a near-collapse of the same shape that
efficientnet_b0 showed before Phase 5 fixed it.

**ALLOWED FILES**
- `benchmarks/diagnose.py` (extend if needed)
- `core/rag/config_extractor.py` (only if the diagnosis supports it)
- `benchmarks/labels/dcgan.json` (only if the label is wrong)
- tests as needed

**TASK**

1. Diagnose as in Prompt 3: dump dcgan's focused text and raw LLM response.
   Does the text describe the generator architecture, or is it abstract and
   results prose? Quote it.
2. **Check the label too.** DCGAN's paper is unusually light on explicit
   layer enumeration — much of the architecture is in a figure. If the label
   demands types the paper never states in text, the label is wrong and the
   honest fix is to correct it, with justification. Do not assume the
   pipeline is at fault.
3. A paper whose architecture lives only in a figure is a **legitimate
   limitation**, not a bug. If that is the finding, say so plainly and
   propose either a label correction or documenting dcgan as a known
   out-of-scope case. Do not manufacture a fix for a paper the text-only
   pipeline cannot serve.
4. Report the attribution and stop for approval before changing extraction
   code.

**VERIFY**
1. Evidence quoted, not summarized.
2. If the label changed, every removed expectation has a one-line
   justification tied to the paper text.
3. Live benchmark; dcgan's recall reported.

**EXIT GATE** — dcgan either clears 0.40, or is documented as a known
limitation with evidence. Both are acceptable outcomes; an unexplained 0.25
is not.

> Copy to here up

---

## PROMPT 5 — Final gate, baseline refresh, documentation

> Copy from here down

**PREREQ:** Prompts 1-4 landed.

**TASK**

1. Run the full suite and a clean live benchmark
   (`rule_based_fallback_count` must be 0; `--strict` enforces it). Paste both.
2. Report each Phase 6 exit criterion:

| Criterion | Target | Actual | Met? |
|---|---|---|---|
| hyperparam_accuracy | >= 0.50 | | |
| layer-type recall | >= 0.70 (no regression from 0.713) | | |
| no single paper below | 0.40 | | |
| precision | >= 0.65 (no regression from 0.727) | | |
| zero fabricated or symbolic parameters | verified on 3 papers | | |

3. Spot-check the "zero fabricated parameters" criterion by hand on three
   papers: every parameter value in the spec must trace to text in the
   paper. Quote three examples. This cannot be automated and is the
   criterion most likely to be quietly skipped.
4. Refresh `benchmarks/baseline.json` from the new clean run
   (`--write-baseline`). Confirm `--check` passes against it and that
   `git status --porcelain benchmarks/baseline.json` shows the file.
5. Append a dated section to `docs/PAPER_TO_CODE_EXECUTION_MEMORY.md`.
6. Update `docs/PAPER_TO_CODE_MASTER_PLAN.md` §8: add a
   `### Phase 6 — RESULT` section in the same shape as Phase 5's, and state
   whether Phase 7 (corpus scale-out) is unblocked.

**EXIT GATE** — honest verdict with numbers. Partial success stated plainly
is the expected outcome. Phase 5 was declared complete twice while work sat
unapplied; do not repeat that. If Phase 6's gate is not met, say which
criteria failed and recommend whether they block Phase 7's corpus scale-out
or can run alongside it.

> Copy to here up
