# Phase 6 — Parameter Recovery (single prompt)

Everything remaining, in one prompt. Two code changes that need no LLM, then
**one** clean measurement, then a decision made from a table rather than a
round trip.

---

## GROUND RULES

1. **REALITY CHECK.** Paths and values below were verified 2026-09-06/07.
   "The repo contradicts this prompt" -> STOP and report. "My prepared patch
   no longer applies" -> re-read and edit fresh, do not stop.
2. **DO NOT INVENT APIs.** Only call what you have read here.
3. **NO NEW DEPENDENCIES.**
4. **REPORTED IS NOT LANDED.** Paste the actual changed lines.
5. **ABSENT BEATS FABRICATED.** Unstated values are omitted, never defaulted,
   never `None`, never a placeholder string.
6. **NULL SAFETY.** `x is not None`, never `hasattr`, never bare truthiness
   (0 is a legitimate padding value).
7. **QUOTA IS THE SCARCE RESOURCE.** Groq rate limits have blocked five
   measurement attempts. This prompt is built to need **one** full live run.
   Do not spend quota on exploratory runs.
8. **DO NOT COMMIT.**
9. **REPORT HONESTLY, INCLUDING NEGATIVES.**

```
Python: .venv/Scripts/python.exe    Root: C:\papper2code
Suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
Live:   -m benchmarks.harness --live --retrieval production --pace-seconds 25
Guard:  -m benchmarks.harness --check
MUST NOT REGRESS: 1916 passed, 2 skipped, 9 deselected, 0 failed
        (verified 2026-09-07, after the reservation and pattern work).
temperature=0 is pinned. The embedder takes ~24s to load and is NOT hung.
```

---

## WHY THIS EXISTS

`benchmarks/baseline.json` (frozen, clean, all-LLM):
`recall 0.713 | precision 0.727 | hyperparam 0.400 | family 0.900`

Current cache — **incoherent**, spanning three code states, so it settles
nothing on its own:
`recall 0.725 | precision 0.566 | hyperparam 0.133`

**Recall improved** (vit 0.80->1.00, densenet 0.83->1.00, bert 0.50->0.75,
mobilenet precision 0.75->0.86). **Parameters collapsed.** The dimension
values were not mis-mapped — they were replaced by prose metadata:

| Paper | label wants | spec now has |
|---|---|---|
| bert_base | `hidden_size: 768, num_heads: 12, num_layers: 12` | `description: "residual connection after attention"`, `activation: "none"`, `name: "intermediate"` |
| transformer_base | `hidden_size: 512, num_heads: 8, num_layers: 6` | `channels: [512]`, `subtype: ["encoder", ...]` |
| vit_base | `hidden_size: 768, num_heads: 12, num_layers: 12` | `name: ["embedding_norm", ...]` |

In `phase5-clean`, bert carried `hidden_size: [768, 768]` and
`num_heads: [12]`. Those are gone.

**Two suspected causes, both plausible, both addressed below:**

**(a) The prompt rollback threw out a good half.** The reverted P1 text
included *"Look for channels, kernel_size, stride, padding, hidden_size,
num_heads, num_layers, and num_classes."* That was the only place the model
was told which keys to use. Without it, it invents `description`, `subtype`,
`name`. The precision damage came from P1's *second* rule (the symbolic-value
paragraph), not from this sentence.

**(b) The reservation backfires on prose-dimension papers.** Transformer-
family papers state dimensions in prose (*"d_model = 512, 8 heads"*), not in
tables. Reserving 2 of 6 slots for `table`/`caption` chunks evicts exactly
that prose. Consistent with the split: resnet50 (table-bearing) went
hyperparam **0.33 -> 0.67**, while bert/vit/transformer lost theirs entirely.

---

## TASK 1 — Restore the key-list sentence only (no LLM needed)

**File:** `core/rag/config_extractor.py`

The params rule currently reads, in full:
```
- "params": ONLY extract values EXPLICITLY stated in the text. Do NOT guess.
```

Add **one** sentence naming the expected keys. Something in the spirit of:
*"Use these parameter names where the paper states them: channels,
kernel_size, stride, padding, hidden_size, num_heads, num_layers,
num_classes."*

**Do NOT re-add P1's second paragraph** (the symbolic-value rule). Symbolic
rejection is now enforced in `core/rag/normalizer.py`, so it does not need
prompt budget, and that paragraph is the prime suspect for the precision
regression (0.727 -> 0.588 when both landed together).

Keep the addition to one sentence. Prompt weight has measurable structural
side effects — that is the lesson of the last two runs.

**VERIFY (no LLM):** `_LLM_EXTRACTION_PROMPT` contains the key list; the
symbolic-value paragraph is absent; `.format(...)` still renders with all
four slots.

---

## TASK 2 — Make the structured reservation conditional (no LLM needed)

**File:** `core/rag/config_extractor.py`

Currently `_select_focus_chunks` reserves `_RESERVED_STRUCTURED_SLOTS = 2`
unconditionally. On a paper whose dimensions live in prose, that is two
evicted prose chunks for no gain.

Make the reservation earn its slot: reserve for a structured chunk **only
when that chunk actually carries numeric content** — a table of layer names
with no numbers is not worth displacing prose for. A simple digit-density or
digit-count test over the chunk text is sufficient; do not build a parser.

Requirements:
- A structured chunk with no digits never takes a reserved slot.
- The reservation stays bounded at 2; unused reserved slots go back to prose,
  so a paper with no numeric tables gets the pre-reservation behaviour
  exactly.
- Keep reading order in the returned selection.
- All six existing reservation tests must still pass unchanged.

**VERIFY (no LLM):**
1. A pool with a numeric table (`"Table 1: stage 1 conv3x3 channels 64 128"`)
   still reserves it.
2. A pool whose only "table" is `"Table 2: ablation study of components"`
   (no digits beyond the label) reserves nothing, and all 6 slots go to prose.
3. Total selected never exceeds 6; reserved structured never exceeds 2.
4. Existing tests in `tests/test_phase2_config_extractor.py` pass unchanged.

---

## TASK 3 — One clean full run, then decide from the table

Run **once**:
```
.venv/Scripts/python.exe -m benchmarks.harness --live --retrieval production --pace-seconds 25
```

`--strict` rejects the run if `rule_based_fallback_count > 0`. If it is
rejected, raise `--pace-seconds` to 40 and retry **once**. If it is rejected
again, stop and report — do not keep burning quota.

Also report `provider_fallback_count`. It should now be **0** for a
Groq-served run; the routing-prefix false positive was fixed by
`_bare_model_id`. If it is non-zero, the run really did cross providers and
is not comparable to `benchmarks/baseline.json`.

Then run `--check` and report both outputs.

### Decision table — apply this yourself, do not come back for a ruling

| Outcome | Action |
|---|---|
| hyperparam >= 0.40 **and** precision >= 0.68 | Both fixes worked. Refresh `baseline.json` with `--write-baseline`, update both docs, report Phase 6 status. |
| hyperparam recovers, precision still < 0.68 | Task 1 worked, precision damage is elsewhere. **Do not** refresh the baseline. Report, and name the papers whose precision is furthest below their `phase5-clean` value. |
| hyperparam still < 0.25 | Task 1 was not the cause. **Do not** refresh the baseline. Run the two-paper disambiguation below. |
| recall drops below 0.65 | Something regressed badly. Report immediately with the per-paper table; change nothing further. |

### Two-paper disambiguation (only if hyperparam stays < 0.25)

```
-m benchmarks.harness --live --retrieval production benchmarks/labels/bert_base.json benchmarks/labels/resnet50.json
```
These two separate the hypotheses: **bert_base** states its dimensions in
prose (hurt by the reservation), **resnet50** states them in a table (helped
by it). If bert recovers and resnet does not, the reservation is the problem;
if neither recovers, the cause is neither Task 1 nor the reservation and the
diagnosis must reopen.

---

## KNOWN CONFOUND — state it in your report

Tasks 1 and 2 land together, so a single run cannot attribute the result to
one of them. That is a deliberate trade: quota has blocked five consecutive
measurements, and one clean run is worth more than two blocked ones. The
disambiguation probe above exists precisely to resolve it if the headline
number is ambiguous. **Say this in your report** rather than implying the run
attributes cleanly.

---

## EXIT

Report:
1. Suite count before and after (must not regress).
2. The full per-paper table and aggregate from the clean run.
3. `--check` output.
4. Which decision-table row applied, and what you did.
5. `provider_fallback_count` and `rule_based_fallback_count`.

If Phase 6's gate (hyperparam >= 0.50, no paper below 0.40 recall, recall
>= 0.70, precision >= 0.65) is not met, say so plainly and list what remains.
Phase 5 was declared complete twice while work sat unapplied; a partial
result stated honestly is the expected outcome and is worth more than a
claim of success.
