# Performance Audit Report — Phase 11B

**Date:** 2026-06-18  
**Status:** Key optimizations applied; further gains available as future work

---

## 1. Optimizations Applied (Phase 11B)

### KnowledgeGraph — filteredNodes Memoization
**File:** `src/components/knowledge/knowledge-graph.tsx`  
**Issue:** `filteredNodes = NODES.filter(...)` ran on every render, including every mousemove during drag.  
**Fix:** Wrapped in `useMemo([filter, searchQuery])`. Now only recomputes when search or filter changes.

### KnowledgeGraph — Edge Filter O(E×N) → O(E)
**File:** `src/components/knowledge/knowledge-graph.tsx`  
**Issue:** Edge rendering called `filteredNodes.find()` twice per edge on every render (O(E×N) per frame).  
**Fix:** Pre-compute `filteredNodeMap = useMemo(() => new Map(...), [filteredNodes])`. Edge lookup is now O(1). Also pre-compute `visibleEdges = useMemo(...)` so the filter runs only when nodes change, not on every pan frame.

### KnowledgeGraph — dragStart State → Ref
**File:** `src/components/knowledge/knowledge-graph.tsx`  
**Issue:** `dragStart` stored in state caused two `setState` calls per mousemove pixel (one for `setPan`, one for `setDragStart`), doubling renders during drag.  
**Fix:** Replaced `useState({ x, y })` with `useRef({ x, y })`. Only `setPan` triggers re-renders during drag; `dragStart` mutation is side-effect-only.

### LabsPage — Handler Memoization
**File:** `src/app/labs/page.tsx`  
**Issue:** `handleParamChange` and `handleLabSelect` were inline arrow functions, creating new references every render. Defeats `React.memo` on child components.  
**Fix:** Wrapped both in `useCallback` with correct dependency arrays. `handleLabSelect` now has stable identity; `handleParamChange` stabilizes when `activeLabId` and params are unchanged.

### LabsPage — Derived Value Memoization
**File:** `src/app/labs/page.tsx`  
**Issue:** `activeLab = labs.find(...)` and `currentParams = paramValues[activeLabId] ?? {}` evaluated on every render.  
**Fix:** Both wrapped in `useMemo`.

---

## 2. Remaining Opportunities (Not Applied in Phase 11B)

### Missing React.memo on Leaf Components

| Component | Impact | Recommendation |
|-----------|--------|----------------|
| `MetricsPanel` | Medium — re-renders on every `loading` toggle | Wrap with `React.memo` |
| `ParameterControls` | Medium — re-renders when parent loading state flips | Wrap with `React.memo` |
| `BlockVizPage` sub-components (`StatRow`, `NodeInspector`) | Low | Wrap with `React.memo` |

### Missing useMemo

| Location | Computation | Recommendation |
|----------|-------------|----------------|
| `MetricsPanel.tsx:106-165` | `rows` array rebuild on every render | `useMemo([metrics, labId])` |
| `DojoProblemPage.tsx:55-56` | `localStorage` read on every render | `useState` initialized once |
| `learning-analytics.tsx:56-59` | `weeks` slicing loop | Move to module scope |

### Candidates for dynamic import

| Component | Why | Recommendation |
|-----------|-----|----------------|
| `ForwardPassPlayer` | Conditionally rendered (only when user clicks "Animate") | `dynamic(() => import(...), { ssr: false })` |
| `SubmissionHistory` | Only shown on `leftTab === 'history'` | `dynamic()` import |
| `KnowledgeGraph` | Large SVG-heavy component (~460 lines) | `dynamic()` on dashboard |

### Inline `<style>` keyframe tags

**File:** `src/app/labs/page.tsx:190,246`  
Two `@keyframes` blocks (`spin`, `pulse`) rendered inside conditional JSX. They are injected/removed on loading state changes. Should be moved to the global CSS file (`static/design.css` or `app/globals.css`).

---

## 3. Bundle Size Notes

No bundle analysis tool was run in this sprint. For production, run:
```bash
ANALYZE=true npm run build
```
and inspect the bundle report to identify large lazy-load candidates.

---

## 4. Summary

| Category | Issues Found | Fixed in 11B | Remaining |
|----------|-------------|--------------|-----------|
| Unnecessary re-renders (state → ref) | 1 | 1 | 0 |
| O(n²) render-path computations | 2 | 2 | 0 |
| Missing useMemo | 7 | 2 | 5 |
| Missing useCallback | 5 | 2 | 3 |
| Missing React.memo | 5 | 0 | 5 |
| Dynamic import candidates | 3 | 0 | 3 |
| Inline style keyframes | 2 | 0 | 2 |

Drag performance on KnowledgeGraph is the highest-impact fix applied — mousemove re-renders halved.
