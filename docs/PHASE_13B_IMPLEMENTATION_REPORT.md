# Phase 13B — Architecture Extraction & Knowledge Graph Generation

## Summary

Phase 13B extends the paper ingestion pipeline with deterministic knowledge extraction and an interactive SVG knowledge graph embedded in the generated paper workspace.

---

## Deliverables

### Backend

| File | Role |
|------|------|
| `backend/services/knowledge_extraction_service.py` | Deterministic extraction + graph construction |
| `backend/services/paper_ingestion_service.py` | Integration: calls `build_knowledge_graph` during ingestion |
| `backend/server.py` | New endpoint: `GET /api/papers/{id}/knowledge-graph` |

### Frontend

| File | Role |
|------|------|
| `src/components/paper-upload/PaperKnowledgeGraph.tsx` | SVG graph component with pan, zoom, selection |
| `src/components/paper-upload/PaperWorkspaceTabs.tsx` | Tabbed center panel (Overview / Figures / Equations / Knowledge Graph) |
| `src/app/papers/upload/[paperId]/page.tsx` | Updated server page: delegates center to `PaperWorkspaceTabs` |
| `src/app/api/papers/generated/[id]/knowledge-graph/route.ts` | Next.js proxy to the FastAPI endpoint |

### Tests

| File | Count |
|------|-------|
| `tests/test_knowledge_extraction.py` | 48 Python tests |
| `src/__tests__/components/paper-upload/PaperKnowledgeGraph.test.tsx` | 21 frontend tests |

---

## Architecture

### Knowledge Extraction (no LLM)

`build_knowledge_graph()` runs four deterministic registry scans over the paper's title, abstract, and extracted sections:

1. **Architecture detection** (`ARCHITECTURE_REGISTRY`) — ~50 regex entries mapping patterns to canonical names and optional content slugs. Example: `\bvit\b` → `ViT` → `/architectures/vit`.
2. **Concept detection** (`CONCEPT_REGISTRY`) — 20 entries for ML concepts like "Multi-Head Attention", "Residual Connection", "Dropout".
3. **Dataset detection** (`DATASET_REGISTRY`) — 17 entries: ImageNet, COCO, SQuAD, LibriSpeech, etc.
4. **Metric detection** (`METRIC_REGISTRY`) — 16 entries: Accuracy, BLEU, FID, PSNR, mAP, etc.
5. **Equation nodes** — First 10 equation spans from the ingestion payload.

### Graph Node Types

| Type | Color | Radius |
|------|-------|--------|
| paper | `--accent-primary` | 14 |
| architecture | `--accent-cyan` | 10 |
| concept | `--accent-transformer` | 8 |
| dataset | `--color-warning` | 8 |
| metric | `--color-success` | 8 |
| equation | `--color-text-tertiary` | 6 |

### Edge Relations

| Source → Target | Relation | Condition |
|-----------------|----------|-----------|
| paper → architecture | `introduces` | Architecture pattern found in the paper's title |
| paper → architecture | `uses` | Architecture found in body only |
| paper → dataset | `evaluates_on` | Dataset mentioned in sections |
| paper → metric | `reports` | Metric mentioned in sections |
| paper → equation | `derives_from` | All extracted equations (up to 10) |
| architecture → concept | `uses` | Static `ARCH_CONCEPT_MAP` lookup |
| architecture → architecture | `derives_from` | Static `ARCH_LINEAGE` lookup |

### Persistence

The graph is stored inside the existing `Paper.architecture_graph` JSON column at the path:

```
paper.architecture_graph.ingestion.knowledge_graph = { nodes: [...], edges: [...] }
```

No schema migration required.

---

## Component: PaperKnowledgeGraph

- SVG `viewBox="0 0 800 480"` with grid background pattern
- Row-based layout: paper → architecture → concept → dataset+metric (split L/R) → equation
- Pan: `onMouseDown/Move/Up` drag handler on the SVG element
- Zoom: buttons ±0.2 per click, range 0.4–3.0, shown as percentage
- Node selection: click to select/deselect, `Enter`/`Space` keyboard support
- Info panel: shows type badge, name, and "Open →" link if an architecture slug exists
- Accessibility: `role="img"` on SVG, `role="button"` on node groups, `aria-pressed`, `aria-label`, `role="region"` on info panel

---

## Component: PaperWorkspaceTabs

Client component (`'use client'`) that manages tab state for the center column of the paper workspace. Tabs:

1. **Overview** — Architecture stats cards + Ingestion Timeline card
2. **Figures** — Paginated figure cards with caption
3. **Equations** — Monospace equation spans (up to 16)
4. **Knowledge Graph** — Full `PaperKnowledgeGraph` canvas

---

## Test Results

```
Frontend:  214 / 214  (22 test files)
Python:    670 / 670
Build:     ✓  Next.js production build passes
```

---

## Constraints Satisfied

- No LLM APIs — all extraction is regex-based against registries
- No mock graph data — graph is built from real ingestion payload at upload time
- Deterministic extraction — same input always produces the same graph
- No new database schema migrations — graph stored in existing JSON column
- Build passes with zero TypeScript errors
- All 214 frontend + 670 Python tests pass
- Accessibility preserved — ARIA roles, keyboard navigation, screen-reader labels
