# Phase 6 — Reservation Fix: architectural, not merely numeric

One prompt. Implementable and fully unit-testable **today**; only the
confirming benchmark waits on Groq's daily quota reset.

---

## GROUND RULES

1. **REALITY CHECK.** Verified 2026-09-07. "The repo contradicts this
   prompt" -> STOP and report. "My prepared patch no longer applies" ->
   re-read and edit fresh, do not stop.
2. **DO NOT INVENT APIs.** Only call what you have read.
3. **NO NEW DEPENDENCIES.**
4. **REPORTED IS NOT LANDED.** Paste the actual changed lines.
5. **TEST AGAINST THE REAL TEXT, NOT A PARAPHRASE.** The fixtures below are
   copied from actual paper chunks in `benchmarks/.cache/*.diag.json`. Use
   them verbatim. A previous fix in this phase passed its tests while being
   broken on the only phrasing that mattered, because the tests were written
   from a paraphrase in the prompt rather than from the paper.
6. **NULL SAFETY.** `x is not None`, never `hasattr`, never bare truthiness.
7. **QUOTA IS EXHAUSTED TODAY.** Do not attempt a live benchmark until Groq
   resets. Everything in Tasks 1-2 is verifiable without one.
8. **DO NOT COMMIT.**
9. **REPORT HONESTLY, INCLUDING NEGATIVES.**

```
Python: .venv/Scripts/python.exe    Root: C:\papper2code
Suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
MUST NOT REGRESS: 1919 passed, 2 skipped, 9 deselected, 0 failed
Baseline (frozen, Groq-only): recall 0.713 | precision 0.727 | hyperparam 0.400
```

---

## WHY THIS EXISTS — the evidence

The last run was provider-mixed (Groq daily quota exhausted, 4 papers served
by Gemini) so its aggregate is not baseline-comparable. **But 6 papers were
Groq-served and therefore are.** Among them:

```
bert_base   hyperparam  0.67 -> 0.00     (Groq-served, key-list sentence LANDED)
```

So restoring the parameter-key sentence **did not** recover BERT's
hyperparameters. That hypothesis is refuted: naming the keys cannot help
when the values are not in the context.

Inspecting BERT's focused text directly (`bert_base.production.diag.json`,
4935 chars) confirms it:

```
'768'          present=False
'12 heads'     present=False
'hidden size'  present=False
```

What occupied the reserved structured slots instead:

```
Table 2: SQuAD 1.1 results. The BERT ensemble answer when s^ > s +tau, ...
Table 7: CoNLL-2003 Named Entity Recognition re- sults. Hyperparameters were se...
```

**Results tables.** The reservation faithfully protected two slots for them
while evicting the prose stating H=768, A=12, L=12.

`_has_numeric_structured_content` (line 53, used at line 504) tests whether a
structured chunk contains a **digit** after its `Table N:` label. A SQuAD
results table is the most numeric object in the entire paper. The filter is
measuring the wrong property: what matters is whether the table is
**architectural**, not whether it is numeric.

This also explains the one case the reservation helped — resnet50's reserved
table was architectural, and its hyperparam went 0.33 -> 0.67.

---

## TASK 1 — Reserve on architectural content (no LLM needed)

**File:** `core/rag/config_extractor.py`

Replace the numeric test with one that asks whether a structured chunk
describes **architecture**. Keep it deterministic, no new dependencies, and
no parser — the existing `_ARCHITECTURE_QUERY` already names the vocabulary
that matters (`channels`, `kernel`, `stride`, `layers`, `hidden`, `heads`,
`conv`, `block`, `encoder`, `decoder`, ...).

Requirements:
- A structured chunk earns a reserved slot only if it carries architectural
  vocabulary **and** a numeric value. Both, not either.
- Keep the reservation bounded at `_RESERVED_STRUCTURED_SLOTS = 2`; unused
  reserved slots return to prose.
- Reading order preserved in the returned selection.
- The existing reservation tests must pass unchanged.
- Do **not** implement this as a blocklist of result-table words
  (`accuracy`, `F1`, `SQuAD`, ...). That is unbounded and will rot. Decide on
  presence of architecture vocabulary, not absence of everything else.

Beware one trap in the fixtures below: BERT's CoNLL caption contains the word
*"Hyperparameters"*. If your vocabulary check matches on `param`, that table
will be wrongly reserved. Match on structural and dimensional terms only.

**VERIFY — use these exact strings, copied from the real diagnostics:**

Must be **REJECTED** (results tables, BERT's actual reserved chunks):
```
"Table 2: SQuAD 1.1 results. The BERT ensemble answer when s > s +tau, i,j null"
"Table 7: CoNLL-2003 Named Entity Recognition re- sults. Hyperparameters were se-"
```

Must be **ACCEPTED** (architecture tables, EfficientNet's actual chunk):
```
"Conv3x3 32 MBConv1 16 MBConv6 24 MBConv6 40 MBConv6 80 MBConv6 112 MBConv6 192 MBConv6 320 Conv1x1&Pooling&FC 1280"
"Table 1: EfficientNet-B0 baseline network. Stage 1 Conv3x3 resolution 224x224 channels 32 layers 1"
```

Also assert:
- A caption with architecture words but **no numbers** is rejected
  (`"Figure 1: The U-Net architecture with contracting and expanding paths"`).
- A chunk with numbers but **no architecture words** is rejected
  (`"Table 3: Results on ImageNet. 76.3 77.1 78.8"`).
- Total selected never exceeds 6; reserved structured never exceeds 2.
- With no qualifying structured chunk, all 6 slots go to prose — identical
  to pre-reservation behaviour.

```
.venv/Scripts/python.exe -m pytest tests/test_phase2_config_extractor.py -q -m "not live"
.venv/Scripts/python.exe -m pytest -q -m "not live"
```

---

## TASK 2 — Prove it on BERT without spending quota (no LLM needed)

`_focus_text` is deterministic: it calls no LLM. So the fix can be verified
end to end against the real paper before any benchmark run.

1. Rebuild BERT's `source_chunks` the way the benchmark does — see
   `benchmarks/harness.py:_live_extractor` (download the PDF, page-aware
   `chunk_pages_with_provenance`, plus `extract_table_chunks` and
   `extract_caption_chunks`).
2. Call `_focus_text` with the production `chunk_retriever` and print whether
   the focused text now contains `768`, `12`, `hidden`, and `heads`.
3. Report the before/after presence table, and which structured chunks (if
   any) took reserved slots.

**This is the acceptance test for the whole fix.** If BERT's dimensions still
do not appear, the reservation is not the remaining cause and you should stop
and report rather than proceeding to Task 3.

Note: the embedder takes ~24s to load on a cold process and is **not** hung.

---

## TASK 3 — Confirming benchmark (ONLY after Groq quota resets)

Do not start this today.

```
.venv/Scripts/python.exe -m benchmarks.harness --live --retrieval production --pace-seconds 25
```

Report the aggregate, `rule_based_fallback_count`, and
`provider_fallback_count`. **A run with `provider_fallback_count > 0` is not
baseline-comparable** — but do not discard it: the Groq-served subset is
still valid per-paper, and comparing those against `phase5-clean.json`
extracted real signal from the last "invalid" run. Report both the full
aggregate and the Groq-only per-paper comparison.

Then `--check`. Refresh `benchmarks/baseline.json` **only** if the run is
100% Groq-served and no metric regressed.

---

## EXIT

Report:
1. Suite count before/after (must not regress from 1919).
2. Task 1 test results, including the four real-text fixtures.
3. Task 2's before/after presence table for BERT's dimensions — this is the
   headline result.
4. Task 3 only if quota allowed it.

If BERT's dimensions still do not reach the model after this fix, say so
plainly. That would mean the reservation was not the cause and the diagnosis
must reopen — which is a legitimate and useful outcome, not a failure.
