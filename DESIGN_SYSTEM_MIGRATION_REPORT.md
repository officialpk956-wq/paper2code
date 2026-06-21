# Design System Migration Report — Phase 12B

**Date:** 2026-06-18  
**Branch:** phase-12b5-backup  
**Build:** ✅ passes  
**Tests:** ✅ 182/182 pass  
**Coverage:** ✅ all thresholds ≥70% (overall 88.83% stmts)

---

## 1. Design Tokens Added (`static/design.css`)

New token block inserted before `color-scheme: dark`:

| Token | Value | Purpose |
|---|---|---|
| `--color-success` | `var(--success)` | Semantic success alias |
| `--color-warning` | `var(--warn)` | Semantic warning alias |
| `--color-error` | `var(--danger)` | Semantic error alias |
| `--color-success-rgb` | `16, 185, 129` | RGB triple for rgba() |
| `--color-warning-rgb` | `245, 158, 11` | RGB triple for rgba() |
| `--color-error-rgb` | `239, 68, 68` | RGB triple for rgba() |
| `--color-severity-low` | `#22c55e` | Severity green (distinct from success) |
| `--color-severity-medium` | `var(--warn)` | Severity medium alias |
| `--color-severity-high` | `var(--danger)` | Severity high alias |
| `--color-severity-low-rgb` | `34, 197, 94` | RGB triple for severity-low |
| `--accent-primary` | `var(--page-accent)` | Page-accent-aware primary |
| `--accent-primary-rgb` | `var(--page-accent-rgb)` | RGB triple for accent-primary |
| `--accent-primary-light` | `#93c5fd` | Light tint of primary accent |
| `--accent-secondary` | `#6366f1` | Indigo secondary accent |
| `--accent-secondary-rgb` | `99, 102, 241` | RGB triple |
| `--accent-transformer-light` | `#a78bfa` | Light transformer purple |
| `--bg-body` | `var(--bg-base)` | Body background alias |
| `--bg-card` | `var(--bg-panel)` | Card background alias |
| `--bg-active` | `rgba(var(--page-accent-rgb), 0.10)` | Active state background |
| `--bg-hover` | `var(--bg-panel)` | Hover state background |
| `--bg-editor` | `#1e1e1e` | Monaco editor dark background |
| `--color-text-muted` | `var(--color-text-tertiary)` | Muted text alias |
| `--color-divider` | `var(--color-border)` | Divider alias |
| `--accent-cyan` | `var(--accent-unet)` | Cyan accent alias |
| `--opacity-disabled` | `0.4` | Disabled state opacity |
| `--motion-fast` | `var(--transition-fast)` | Fast motion alias |
| `--motion-normal` | `var(--transition-base)` | Normal motion alias |
| `--motion-slow` | `400ms cubic-bezier(0.4, 0, 0.2, 1)` | Slow motion |

---

## 2. UI Primitives Created

| File | Description |
|---|---|
| `src/components/ui/Button.tsx` | Primary / secondary / ghost / danger variants; disabled state; motion tokens |
| `src/components/ui/Input.tsx` | Focus ring via onFocus/onBlur; disabled; error states |
| `src/components/ui/SectionLabel.tsx` | 10px 700-weight uppercase tracking-wide label |
| `src/components/ui/Spinner.tsx` | Configurable size/color; uses Tailwind `animate-spin` |

---

## 3. Hardcoded Color Replacements

### 3.1 TypeScript Components

| File | Replacements |
|---|---|
| `src/components/labs/ArchitecturePreview.tsx` | `SEVERITY_COLOR` map: `#22c55e` → `--color-severity-low`, `#f59e0b` → `--color-severity-medium`, `#ef4444` → `--color-severity-high`; FlopsBar severity logic |
| `src/components/labs/MetricsPanel.tsx` | Error state rgba vars; "Live Metrics" → `<SectionLabel>` |
| `src/components/labs/LabSelector.tsx` | "Labs" → `<SectionLabel>`; `transition-all` → explicit |
| `src/components/labs/ExperimentHistory.tsx` | "History" → `<SectionLabel>` |
| `src/components/labs/ParameterControls.tsx` | "Parameters" → `<SectionLabel>` |
| `src/components/block-viz/BlockBox.tsx` | `SEVERITY_COLORS` map; border, background, text colors → tokens |
| `src/components/block-viz/ForwardPassPlayer.tsx` | Step card, progress bar, speed buttons, icon buttons → tokens |
| `src/components/dojo/TestResultPanel.tsx` | `STATUS_CONFIG` map; all test row pass/fail colors → tokens |
| `src/components/dojo/DojoEditor.tsx` | Run button (success tokens), Submit button (transformer tokens); spinner → `<Spinner>`; `#1e1e2e` → `var(--bg-editor)`; `transition-all` → `transition-colors` |

### 3.2 `static/design.css` — Replaced Rules

**Tensor visualization:**
- `.tensor-expand` / `.tensor-expand:hover` — `#f59e0b` → `var(--color-warning)` + RGB
- `.tensor-spatial-change` / `:hover` — `#ef4444` → `var(--color-error)` + RGB
- `.editor-pane` — `#1e1e1e` → `var(--bg-editor)`

**Dojo difficulty / state:**
- `.difficulty-pill.l1/l3/l4.active` — success/warning/error
- `.failure-card.explode/.stagnate` — error/warning
- `.gpu-table tr.fits/.no-fit` — success/error
- `.dojo-tick.done` — success
- `.dojo-banner.success/.partial` — success/warning + RGB
- `.dojo-test-row.pass/.fail` — success/error + RGB
- `.hp-callout.ok/.warn/.danger` — success/warning/error + RGB
- `.dojo-badge-difficulty.easy/.medium/.hard` — success/warning/error + RGB
- `.dojo-tag-badge` — success + RGB
- `.dojo-test-case-v3.pass/.fail` — success/error + RGB
- `.dojo-test-status-v3.pass/.fail` — success/error
- `.dojo-badge-easy/.medium/.hard` — success/warning/error + RGB
- `.dojo-test-case-v4.pass/.fail` — success/error + RGB
- `.dojo-objective-badge` — success + RGB
- `.dojo-test-pass/.dojo-test-fail` — success/error
- `.dojo-diff-chip.d-easy/.d-medium/.d-hard.active` — success/warning/error
- `.dojo-submission-status.accepted/.wrong/.error` — success/error/warning + RGB
- `.dojo-notes-info.saved` — success

**Assess solution:**
- `.assess-solution-toggle` / `:hover` / `.assess-solution-body` — success RGB

---

## 4. Inline Keyframe Removal

| File | Before | After |
|---|---|---|
| `src/app/labs/page.tsx` | `<style>{\`@keyframes spin\`}</style>` + `<div style={{animation:'spin...'}}>` | `<Spinner size={32} />` |
| `src/app/labs/page.tsx` | `<style>{\`@keyframes pulse\`}</style>` + hardcoded animation div | `<div className="animate-pulse" />` |

### Naming conflict resolved
`static/design.css` had `@keyframes pulse` (tensor arrows: opacity 0.4→1 + translateX). Renamed to `@keyframes tensor-arrow-pulse` and updated `.tensor-arrow` usage to prevent conflict with Tailwind's `animate-pulse`.

---

## 5. `transition: all` → Explicit Transitions

| Selector | Old | New |
|---|---|---|
| `.card` | `transition: all var(--transition-base)` | `transition: border-color, box-shadow, transform, background` |
| `.flashcard` | `transition: all var(--transition-base)` | `transition: border-color, box-shadow` |
| `.walkthrough-step` | `transition: all var(--transition-base)` | `transition: opacity, border-color` |
| `.tensor-step` | `transition: all var(--transition-base)` | `transition: border-color, box-shadow` |
| `.dojo-diff-chip` | `transition: all var(--transition-fast)` | `transition: background, border-color, color` |

---

## 6. Global CSS Additions

```css
/* Semantic typography classes */
.text-display  { font-size: var(--text-2xl); font-weight: 700; letter-spacing: -0.03em; }
.text-heading  { font-size: var(--text-xl);  font-weight: 700; letter-spacing: -0.02em; }
.text-subheading { font-size: var(--text-lg); font-weight: 600; }
.text-body     { font-size: var(--text-base); line-height: var(--lh-body); }
.text-caption  { font-size: var(--text-xs); color: var(--color-text-tertiary); }

/* Global spin keyframe (supplements Tailwind's animate-spin) */
@keyframes spin { to { transform: rotate(360deg); } }
```

---

## 7. Critical Constraints Verification

| Constraint | Status |
|---|---|
| Zero hardcoded severity colors in components | ✅ All `#22c55e`, `#f59e0b`, `#ef4444`, `#10b981`, `#34d399`, `#f87171` replaced in target files |
| Shared Button/Input/Spinner/SectionLabel components exist | ✅ All 4 in `src/components/ui/` |
| No inline keyframe definitions remain | ✅ Both `<style>` tags removed from labs/page.tsx |
| All tests pass | ✅ 182/182 |
| Design tokens are single source of truth | ✅ All values reference CSS custom properties |
