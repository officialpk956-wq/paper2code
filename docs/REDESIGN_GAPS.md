# Redesign Gaps — Paper2Code Premium OS

**Date:** June 16, 2026  
**Scope:** UI/UX implementation gaps between the spec and reality  
**Focus:** Design system coherence, layout correctness, visual consistency

---

## 1. Design System Gaps

### 1.1 Hardcoded Colors (Critical)

The design system defines CSS custom properties for all colors. Components must use `var(--token-name)` or Tailwind aliases. Direct hex values are forbidden.

**Reality:** 177 hardcoded hex/rgb values found across 34 files.

Selected examples:
- `knowledge-graph.tsx` — `fill="#7C3AED"`, `fill="#06B6D4"`, `stroke="#333"`, `stroke="#555"` (8 occurrences)
- `learning-analytics.tsx` — `backgroundColor: '#7C3AED'`, `backgroundColor: '#374151'` (6 occurrences)
- `architecture-evolution.tsx` — `border-[#7C3AED]` inline Tailwind arbitrary (2 occurrences)
- `comparison-timeline.tsx` — `bg-[#16213e]` (1 occurrence)

**Impact:** Theme changes break consistency. Dark/light toggle (if ever implemented) will leave hardcoded values stranded.

### 1.2 Self-Referential CSS Variable

`src/app/layout.tsx:93`:
```css
--font-heading: var(--font-heading);
```
This resolves to `undefined`. Any component using `font-[--font-heading]` gets no font applied. The heading font stack falls through to the browser default.

### 1.3 Missing Design Tokens

The following tokens are referenced in components but not defined in `layout.tsx`:
- `--bg-surface` — used in mastery-tracker, ai-learning-coach, and others (~15 uses). Falls back to `transparent`.
- `--color-text-tertiary` — used widely for muted text. Defined but value not verified against WCAG contrast on `--bg-body`.
- `--accent-cyan` — defined as `#06B6D4` but no semantic alias (e.g., `--accent-secondary`) exists.

### 1.4 Two Font Families, One Applied

`layout.tsx` loads `Plus_Jakarta_Sans` and `Inter` via `next/font`. The CSS maps both to `--font-sans`. There is no functional distinction — both point to the same variable. `JetBrains Mono` is referenced in component comments but not loaded anywhere.

Code blocks and monospace elements inherit the browser default mono font stack, not JetBrains Mono.

---

## 2. Layout Gaps

### 2.1 ThreeColumnLayout API Mismatch (Three Pages Broken)

`ThreeColumnLayout` accepts named props: `left`, `center`, `right`, `leftWidth`, `rightWidth`, `showRight`.

Pages added in Phases 8–10 pass content as JSX children instead:

```tsx
// What these pages do (WRONG):
<ThreeColumnLayout>
  <div>left content</div>
  <div>center content</div>
  <div>right content</div>
</ThreeColumnLayout>

// What ThreeColumnLayout expects (CORRECT):
<ThreeColumnLayout
  left={<LeftPanel />}
  center={<MainContent />}
  right={<RightPanel />}
/>
```

**Affected pages:**
- `src/app/real-time-collaboration/page.tsx`
- `src/app/advanced-versioning/page.tsx`  
- `src/app/knowledge-intelligence/page.tsx`

**Impact:** All three pages render as single-column. The premium three-column workspace aesthetic is absent on 30% of the app's pages.

### 2.2 Left Rail Height Issue

`left-rail.tsx` has `h-full` but no specified overflow behavior. On content-dense screens or with many nav items, the rail clips without a scrollbar. Current 4-item nav hides this — adding the missing 12 routes will trigger it.

### 2.3 Right Panel Width Inconsistency

`ThreeColumnLayout` defaults `rightWidth` to `320px`. Phase 10's knowledge-intelligence page passes no width override. Phase 8 and 9 pages don't use the right panel at all (wrong API). Design spec called for `240px` left / flex center / `320px` right — this is only achieved on Phase 1–7 pages.

### 2.4 Mobile Breakpoints Not Validated

`ThreeColumnLayout` hides left panel at `< md` (768px) and right panel at `< lg` (1024px). But:
- No hamburger menu or drawer exists to access the left panel on mobile
- No tap target for opening collapsed left rail
- Left rail nav is unreachable on any screen under 768px wide

---

## 3. Navigation Gaps

### 3.1 Active State Indicator

`left-rail.tsx:97` applies `border-l-3` for the active nav item. `border-l-3` is not a valid Tailwind utility — Tailwind ships `border-l-2` and `border-l-4`, not `border-l-3`.

Result: the active nav accent border (the purple left edge) never renders. Current nav is visually identical whether a route is active or not.

**Fix:** Change `border-l-3` to `border-l-2`.

### 3.2 Pages Not Reachable from Nav

The left rail has 4 items: Dashboard, Academy, Dojo, Search. The following routes exist but have no nav entry:

| Route | Phase |
|-------|-------|
| `/explorer` | Phase 2 |
| `/papers` | Phase 3 |
| `/workspace` | Phase 4 |
| `/architecture` | Phase 5 |
| `/implementations` | Phase 6 |
| `/real-time-collaboration` | Phase 8 |
| `/advanced-versioning` | Phase 9 |
| `/knowledge-intelligence` | Phase 10 |

Users can only reach these pages by typing the URL directly.

### 3.3 Spec Route Mismatch

Phase 10 spec explicitly requested route `/knowledge-graph`. Implementation created `/knowledge-intelligence`. The spec-required route 404s.

### 3.4 Dead Routes in Nav

Left rail links to `/search` and `/settings`. Neither route exists — both 404.

---

## 4. Typography Gaps

### 4.1 No Heading Scale Applied

The design spec called for a 4-tier typography system. Commit `eeb71ae` mentions "4-tier typography system." In the actual components:

- Most headings are `text-sm font-semibold` or `text-2xl font-bold` — ad-hoc sizing, not semantic tokens
- No `--text-xs`, `--text-sm`, `--text-base`, `--text-lg` tokens defined
- Font sizes are Tailwind utilities, not design system references

### 4.2 Line Height and Letter Spacing

No `tracking-*` or `leading-*` tokens exist. Components use bare Tailwind defaults. The "premium feel" requires tight letter-spacing on headings (`tracking-tight`) and proper line heights on body copy — neither is systematically applied.

---

## 5. Component-Level Gaps

### 5.1 Button Variants Not Standardized

Buttons across the app have 5+ different visual treatments with no shared component:
- `bg-[--accent-primary] text-white hover:opacity-90`
- `bg-[--bg-surface] hover:bg-[--bg-panel] text-[--color-text-secondary]`
- `border border-[--color-border] hover:border-[--accent-primary]/50`
- Inline `style={{ backgroundColor: '#7C3AED' }}`
- Various ad-hoc combinations

No `<Button>` component with `variant` prop exists.

### 5.2 No Loading States

Zero loading skeletons, spinners, or placeholder UI exists anywhere. Every component renders immediately with hardcoded data. When real data is eventually wired in, every component will flash empty before populating.

### 5.3 No Empty States

No empty state illustrations or messages. When lists are empty (after filtering, after dismissing all recommendations, etc.), components render empty containers with no feedback.

### 5.4 Form Elements Unstyled

The one search input in `knowledge-graph.tsx` uses bare Tailwind without a consistent input style. There is no shared `<Input>` component. If more inputs are added, visual consistency will diverge immediately.

---

## 6. Animation Gaps

### 6.1 Framer Motion Not Installed

Phase 10 spec required "Framer Motion animations." `package.json` has no `framer-motion` entry. The specification deliverable was not met.

All animations are CSS transitions (`transition-all`, `transition-colors`, `transition-opacity`). These are not wrong, but they are not what was specified and they are not as capable (no layout animations, no shared-element transitions, no gesture-driven physics).

### 6.2 Inconsistent Transition Durations

Components use `transition-all` (unspecified duration, defaults to 150ms), `transition-colors`, and `transition-opacity` without consistent duration tokens. Some items feel snappy, others feel sluggish.

---

## 7. Phase-by-Phase Gap Summary

| Phase | Spec | Delivered | Gap |
|-------|------|-----------|-----|
| Phase 1 — Design System | Design tokens, fonts, app shell | ✅ Mostly complete | Self-referential token, missing JetBrains Mono |
| Phase 2 — Explorer | Paper explorer with SVG canvas | ✅ Rendered | Static SVG, no interactions |
| Phase 3 — Papers | Paper reader, code extraction | ✅ UI scaffolded | No extraction logic |
| Phase 4 — Workspace | IDE-style workspace | ✅ UI scaffolded | File tree is static |
| Phase 5 — Architecture | Architecture canvas | ✅ SVG rendered | Canvas is decorative only |
| Phase 6 — Implementations | Code runner | ✅ UI scaffolded | No execution engine |
| Phase 7 — Dojo | LeetCode-style problems | ✅ Complete | Best-implemented phase |
| Phase 8 — Collaboration | Real-time multi-user | 🔲 Stubbed | No network layer, ThreeColumnLayout broken |
| Phase 9 — Versioning | Git-like branching | 🔲 Stubbed | No diff algorithm, ThreeColumnLayout broken |
| Phase 10 — Knowledge | Knowledge graph, analytics | ⚠️ Partial | 3 render bugs, wrong route, Framer Motion missing |
