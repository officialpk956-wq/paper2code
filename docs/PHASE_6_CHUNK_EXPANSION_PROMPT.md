# Phase 6 — Chunk-boundary expansion (single prompt)

Implementable and fully verifiable **without any LLM quota**. The acceptance
test is a string check on deterministic output, not a benchmark.

---

## GROUND RULES

1. **REALITY CHECK.** Verified 2026-09-07. "The repo contradicts this prompt"
   -> STOP and report. "My prepared patch no longer applies" -> re-read and
   edit fresh, do not stop.
2. **DO NOT INVENT APIs.** Only call what you have read.
3. **NO NEW DEPENDENCIES.**
4. **REPORTED IS NOT LANDED.** Paste the actual changed lines.
5. **NO SUBSTRING PROBES THAT CAN PASS BY COINCIDENCE.** The previous
   acceptance check reported "768: present -> yes" when the only match was
   `"two-layer 768-dimensional BiLSTM"` — an unrelated ablation, not BERT's
   hidden size. Assert on the **spec sentence**, and print the surrounding
   context so a false positive is visible.
6. **NO PAPER-SPECIFIC LOGIC.** No branching on paper id, title, or any
   identifying string. The fix must be general; BERT is the test case, not
   the target.
7. **NULL SAFETY.** `x is not None`, never `hasattr`, never bare truthiness.
8. **DO NOT COMMIT.**
9. **REPORT HONESTLY, INCLUDING NEGATIVES.**

```
Python: .venv/Scripts/python.exe    Root: C:\papper2code
Suite:  .venv/Scripts/python.exe -m pytest -q -m "not live"
MUST NOT REGRESS: 1925 passed, 2 skipped, 9 deselected, 0 failed
No live benchmark in this prompt. Groq quota is exhausted.
The embedder takes ~24s to load on a cold process and is NOT hung.
```

---

## WHY THIS EXISTS

BERT's architecture paragraph is split across three consecutive chunks:

| index | chars | content |
|---:|---:|---|
| 8 | 1139 | ends: `Model Architecture BERT's model architec-` |
| 9 | 1136 | defines `number of layers ... L`, `hidden size ... H`, `self-attention heads ... A` |
| 10 | 1172 | starts mid-sentence `tion in this section...`, contains `BERT (L=12, H=768, A=12...)` |

Retrieval selects **chunk 8** — it carries the "Model Architecture" heading,
so it ranks — but not 9 or 10, which hold the definitions and the values.
Chunk 10 in isolation reads as unrelated prose and cannot rank on an
architecture query.

Verified absent from the focused text while present in the raw paper:
`L=12`, `H=768`, `A=12`, `self-attention heads`.

**The economics are favourable.** BERT's focused text used **4,935 of 10,000**
`max_context_chars`. Roughly 5,000 chars — about four chunks — are simply
unspent. Pulling in neighbours costs budget nobody is using; it does not
evict anything currently selected.

---

## TASK 1 — Expand selected chunks into their neighbours

**File:** `core/rag/config_extractor.py` (selection only)

After the existing ranked selection, spend **leftover character budget** on
chunks adjacent to those already chosen. A chunk that ranks well because it
holds a section heading should bring its body with it.

Requirements:

- Expansion consumes only budget left under `max_context_chars` after the
  ranked selection. It must **never evict** a ranked chunk.
- Expand **forward** from a selected chunk (heading -> body is the pattern
  that matters), up to a small bounded depth. BERT needs depth 2 to reach
  chunk 10 from chunk 8; do not exceed what the budget allows.
- Backward expansion by one is permitted if budget remains — a chunk opening
  mid-sentence (`tion in this section...`) is evidence its predecessor is
  needed.
- Adjacency means **consecutive index in `source_chunks`**. Do not attempt to
  reason about pages or sections; keep it simple and deterministic.
- Preserve reading order in the returned selection.
- Never exceed `max_context_chars` in the final merged text.
- **Do not modify `core/utils.py:chunk_pages_with_provenance`.** Chunk
  boundaries and offsets are persisted on `PaperChunk` rows and are what
  `core/evidence_tracking.py` verifies quotes against. Changing them would
  invalidate stored citations. This is a *selection* fix, not a chunking fix.
- All existing reservation and selection tests must pass unchanged.

**VERIFY (no LLM):**
1. A selected chunk with an unselected successor pulls it in when budget
   allows.
2. Expansion stops at `max_context_chars`; with no headroom the selection is
   byte-identical to today's.
3. Expansion never displaces a ranked chunk.
4. Reading order preserved.
5. No duplicates when two selected chunks are adjacent.
6. Empty and single-chunk inputs do not raise.

---

## TASK 2 — Acceptance test on the real paper (no LLM needed)

`_focus_text` calls no LLM, so this is the real gate.

Rebuild BERT's `source_chunks` as `benchmarks/harness.py:_live_extractor`
does (download the PDF, `chunk_pages_with_provenance`, plus
`extract_table_chunks` and `extract_caption_chunks`), call `_focus_text` with
the production `chunk_retriever`, then assert on the **spec sentence**:

```
required: "L=12"      (or the paper's exact rendering, e.g. "L=12, H=768, A=12")
required: "H=768"
required: "A=12"
required: "self-attention heads"
```

**Print the surrounding context for each match.** A bare boolean is what
produced the previous false positive. If `768` matches only inside
`"two-layer 768-dimensional BiLSTM"`, that is a **failure**, not a pass.

Report which chunk indices were ranked, which were added by expansion, and
the final focused-text length against the 10,000 budget.

**If the spec sentence still does not appear, stop and report.** Do not
attempt a further fix in the same pass — that would be the fourth untested
hypothesis in a row on this problem.

---

## TASK 3 — Regression check on the other papers (no LLM needed)

Expansion changes what every paper sees, not just BERT. For **efficientnet_b0**
and **resnet50** — the two papers whose parameters currently work — run the
same focus-only path and confirm their architecture tables are still present:

```
efficientnet_b0: "MBConv" and a channel progression (16, 24, 40, 80, ...)
resnet50:        the 7x7 stem / channel figures currently being extracted
```

If expansion pushes those out by consuming budget, report it — that is a
real trade-off and it needs to be visible before any benchmark spend.

---

## EXIT

Report:
1. Suite count before/after (must not regress from 1925).
2. Task 1 unit-test results.
3. **Task 2: the BERT spec-sentence check with quoted context** — the
   headline result.
4. Task 3: efficientnet_b0 and resnet50 still intact, or not.

Do **not** run a live benchmark, refresh `benchmarks/baseline.json`, or
update the Phase 6 docs in this pass. The confirming benchmark waits for
Groq's daily reset; this prompt's job is to make that single run worth
spending.

If BERT's dimensions still do not reach the model, say so plainly. Three
hypotheses on this problem have already been refuted by measurement
(prompt rollback, numeric filter, reservation alone). A fourth honest
negative is more useful than a claimed fix.
