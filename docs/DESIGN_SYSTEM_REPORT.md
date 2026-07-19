# Design System Audit Report — Phase 11B

**Date:** 2026-06-18  
**Status:** Issues documented; CSS variable tokens identified; fixes deferred to Phase 12

---

## 1. Hardcoded Colors Found

All of the following should use CSS variables instead of raw hex/rgba values. The design system's CSS variables are defined in `static/design.css`.

### Severity Colors (repeated 3+ times across components)

| Current Value | Recommended Variable | Files |
|---------------|---------------------|-------|
| `#22c55e` / `rgba(16,185,129,...)` | `var(--color-success)` | ArchitecturePreview.tsx, BlockBox.tsx, DojoProblemPage.tsx |
| `#f59e0b` / `rgba(245,158,11,...)` | `var(--color-warning)` | ArchitecturePreview.tsx, BlockGraph.tsx |
| `#ef4444` / `rgba(239,68,68,...)` | `var(--color-error)` | ArchitecturePreview.tsx, MetricsPanel.tsx, BlockBox.tsx |

### Accent Colors

| Current Value | Recommended Variable | Files |
|---------------|---------------------|-------|
| `rgba(59,130,246,...)` | `var(--accent-primary)` | ForwardPassPlayer.tsx (7 occurrences), BlockGraph.tsx |
| `#93c5fd` | `var(--accent-primary-light)` | ForwardPassPlayer.tsx |
| `rgba(99,102,241,...)` | `var(--accent-secondary)` | BlockBox.tsx |
| `rgba(139,92,246,...)` | `var(--accent-secondary)` | DojoEditor.tsx |

### Surface Colors

| Current Value | Recommended Variable | Files |
|---------------|---------------------|-------|
| `#1e1e2e` | `var(--bg-editor)` | DojoEditor.tsx |
| `rgba(255,255,255,0.03)` | `var(--bg-card)` | ForwardPassPlayer.tsx |
| `rgba(255,255,255,0.08)` | `var(--color-border)` | ForwardPassPlayer.tsx |
| `#e2e8f0` | `var(--color-text-primary)` | BlockBox.tsx, ForwardPassPlayer.tsx |

### Total hardcoded color occurrences: ~35

---

## 2. Repeated Design Patterns Without Shared Component

The following pattern appears identically in 4 components:
```typescript
{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.12em',
  textTransform: 'uppercase', color: 'var(--color-text-muted)' }
```

**Files:** LabSelector.tsx, ParameterControls.tsx, MetricsPanel.tsx, ExperimentHistory.tsx

**Recommendation:** Extract to a `<SectionLabel>` component in `src/components/ui/SectionLabel.tsx`.

---

## 3. Magic Numbers

| File | Value | Context | Recommendation |
|------|-------|---------|----------------|
| `ForwardPassPlayer.tsx:162` | `outline: 'none'` | Focus removal — **fixed** in a11y pass | Done |
| `DojoEditor.tsx:101` | `opacity: 0.7` | Disabled button | `var(--opacity-disabled)` token |
| `ForwardPassPlayer.tsx:39` | `speedIdx = 1` | Default speed index | `const DEFAULT_SPEED_IDX = SPEED_OPTIONS.findIndex(o => o.label === '×1')` |
| `labs/page.tsx:186` | `width: '32px'` | Spinner | Shared spinner component |
| `labs/page.tsx:240` | `width: '8px'` | Dot indicator | Shared dot component |

---

## 4. Inconsistent Styling Approach

| Approach | Components Using It |
|----------|-------------------|
| Inline `style={{}}` objects exclusively | Labs components (LabSelector, ParameterControls, MetricsPanel, ArchitecturePreview, ExperimentHistory) |
| Tailwind `className` exclusively | Dojo components (DojoProblemPage, DojoEditor), KnowledgeGraph |
| Mixed (both) | BlockBox, BlockGraph, ForwardPassPlayer |

**Recommendation:** Establish a single convention. The majority of the codebase uses Tailwind. New components should use Tailwind with CSS variables for theming. The labs components (written inline-style) should be migrated in a future sprint.

---

## 5. Missing CSS Variable Definitions

The following variables are referenced in components but may not be defined in `static/design.css`:

| Variable | Used In |
|----------|---------|
| `--color-success` | Would replace all `#22c55e` / `rgba(16,185,129,...)` |
| `--color-warning` | Would replace all `#f59e0b` |
| `--color-error` | Would replace all `#ef4444` |
| `--color-severity-low/medium/high` | ArchitecturePreview, BlockBox |
| `--bg-editor` | DojoEditor |
| `--opacity-disabled` | DojoEditor button states |
| `--accent-primary-light` | ForwardPassPlayer icon buttons |

These tokens should be added to `static/design.css` before the migration sprint.

---

## 6. Summary

| Category | Issues Found | Fixed in 11B |
|----------|-------------|--------------|
| Hardcoded hex/rgba colors | 35+ | 0 (documented for Phase 12) |
| Repeated patterns needing extraction | 1 (SectionLabel) | 0 |
| Magic numbers | 5 | 1 (outline:none — fixed in a11y pass) |
| Inconsistent styling approach | 3 groups | 0 (documented) |

Phase 11B focus was security/API/a11y. Design system token migration is planned for Phase 12.
