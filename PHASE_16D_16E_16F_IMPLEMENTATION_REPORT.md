# Phase 16D + 16E + 16F Implementation Report
**Date:** 2026-06-20 | **Branch:** phase-12b5-backup

---

## Goal

Complete the Research → Learn → Practice → Research learning loop by connecting existing content surfaces with CTAs.

---

## Files Changed

| File | Change |
|---|---|
| `src/components/paper-upload/PaperKnowledgeGraph.tsx` | Enhanced node selection panel with difficulty badge, explanation, and "Learn This Concept" CTA |
| `src/components/topic/RecommendedProblems.tsx` | **New** — Recommended Problems section for topic pages |
| `src/app/learn/[domain]/[topic]/page.tsx` | Added `<RecommendedProblems topicSlug={topic} />` after SummarySection |
| `src/components/dojo/DojoProblemPage.tsx` | Added `RelatedResearchPanel` shown on solve in history tab |

---

## Phase 16D — Research → Learn

**File:** `src/components/paper-upload/PaperKnowledgeGraph.tsx`

### What was added

- `CONCEPT_META` record (38 entries) mapping lowercase concept names to `{difficulty, explanation, learnUrl}`
- `DIFFICULTY_COLORS` record for badge styling
- `getConceptMeta(name)` lookup function
- `selectedMeta` computed in component body from selected node name
- Updated info panel to show:
  1. **Difficulty badge** — beginner (green) / intermediate (amber) / advanced (red) — when concept is in CONCEPT_META
  2. **Short explanation** — 1-sentence description of what the concept means
  3. **"Learn This Concept →" CTA** — cyan button linking to verified topic/architecture page

### Concept-to-URL mappings (verified)

| Concept(s) | URL |
|---|---|
| Attention, Multi Head Attention, Self Attention | `/learn/deep-learning/attention` |
| Transformer | `/architectures/transformer` |
| Residual Connection, ResNet | `/architectures/resnet` |
| BERT | `/architectures/bert` |
| GPT | `/architectures/gpt` |
| Vision Transformer, ViT | `/architectures/vit` |
| Layer Normalization, Feed-Forward Network, Positional Encoding, Softmax | explanation only (learnUrl: null) |

All URLs verified to exist in `src/content/architectures/` and via the live topic registry.

---

## Phase 16E — Learn → Practice

**Files:**
- `src/components/topic/RecommendedProblems.tsx` (new)
- `src/app/learn/[domain]/[topic]/page.tsx` (updated)

### What was added

New `RecommendedProblems` component that:
- Takes `topicSlug` as prop
- Looks up problems from `TOPIC_TO_PROBLEM_SLUGS` mapping
- Reads problem data from `PROBLEMS` array in `@/data/problems`
- Renders each problem as a card with: difficulty badge (Easy/Medium/Hard), estimated time, title, and "Solve →" link to `/dojo/[slug]`
- Returns `null` if no mapping for the given topic slug

### Topic-to-problems mapping

| Topic slug | Problems shown |
|---|---|
| `attention` | scaled-dot-product-attention, positional-encoding, multi-head-attention, layer-normalization |
| `multi-head-attention` | scaled-dot-product-attention, multi-head-attention, masked-attention, layer-normalization |

All problem slugs verified to exist in `PROBLEMS` array.

### Placement

Added after `<SummarySection>` in the topic page content flow, so it appears at the bottom of the chapter after the summary. The section id is `recommended-problems` for scroll-spy compatibility.

---

## Phase 16F — Practice → Research

**File:** `src/components/dojo/DojoProblemPage.tsx`

### What was added

Inline `RelatedResearchPanel` component that:
- Takes `problem: Problem` as prop
- Filters `problem.relatedPapers` against `PAPER_LABELS` (known verified slugs)
- Filters `problem.relatedArchitectures` against `ARCH_LABELS` (known verified slugs)
- Shows paper links (indigo) → `/papers/[slug]`
- Shows architecture links (cyan) → `/architectures/[slug]`
- Returns `null` if no verified content found

### Placement

Shown in the left panel's history tab when `isSolved === true`. Auto-appears when problem is solved in session (tab switches to history automatically). Also visible when returning to a previously-solved problem and clicking History tab.

### Verified paper slugs

| Slug | Used by |
|---|---|
| `attention-is-all-you-need` | All transformer + linear algebra problems |
| `deep-residual-learning` | dl-1 through dl-5, cnn-1 through cnn-3 |
| `gpt-3` | llm-1 (top-k-sampling), llm-2 (kv-cache) |

### Verified architecture slugs

| Slug | Verified in `src/content/architectures/` |
|---|---|
| `transformer` | ✅ |
| `resnet` | ✅ |
| `bert` | ✅ |
| `gpt` | ✅ |
| `vit` | ✅ |
| `llama` | ✅ |

---

## The Complete Learning Loop

```
/papers/[id] (KG tab)                     /dojo/[slug]
  └─ Click "Attention" node           └─ Solve problem
     └─ Panel shows explanation          └─ History tab shows:
        └─ "Learn This Concept →"           📄 Attention Is All You Need
           └─ /learn/deep-learning/attention   🏗 Transformer Architecture
              └─ Bottom of page shows:         └─ back to /papers/[slug]
                 Recommended Problems
                 └─ "Solve →" → /dojo/[slug]
```

---

## Test Results

```
Test Files  55 passed (55)
Tests       578 passed (578)
Duration    ~24s
```

No regressions. All 578 tests pass.

---

## Build Result

```
✓ Compiled successfully
✓ Generating static pages (156/156)
```

No TypeScript errors. No new dead links introduced — all URLs verified against `src/content/` directory before use.

---

## Constraints Respected

- ❌ No new pages built
- ❌ No new backend services
- ✅ Reused existing content: `PROBLEMS` array, `src/content/architectures/`, `src/content/papers/`
- ✅ Only links to verified content (checked `src/content/` directory)
- ✅ No mock/placeholder content
