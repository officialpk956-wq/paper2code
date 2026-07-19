# Paper2Code — UI/UX Compliance Audit Report

**Auditor Role:** Senior Staff Product Designer + Frontend Architect + QA Engineer  
**Audit Date:** 2026-06-17  
**Method:** Direct codebase inspection — zero trust of prior implementation summaries  
**Scope:** All 37 pages, 85+ components, design system, navigation, accessibility, performance

---

## EXECUTIVE SUMMARY

The product exists and is structurally coherent. The design system is well-defined. Core pages (Dashboard, Architectures, Papers, Paper-to-Code, System Design, Model Architecture) implement the three-column workspace correctly. Phases 1–6 are meaningfully implemented with real UI. Phases 7–10 are **UI demonstrations using 100% hardcoded static mock data** — no state persistence, no backend, no real interactivity beyond toggling tabs.

**The product is a sophisticated static mockup, not a functional application.**

---

## PART 1: DESIGN SYSTEM AUDIT

### Color Tokens — Score: 90/100

**Defined in:** `src/app/layout.tsx` (inline `<style>` in `<head>`)

```
--bg-body:             #09090F   ✓
--bg-surface:          #0D0D14   ✓
--bg-panel:            #111827   ✓
--bg-hover:            #1A1F2E   ✓
--bg-active:           #232D3D   ✓
--accent-primary:      #7C3AED   ✓
--accent-cyan:         #06B6D4   ✓
--color-text-primary:  #E2E8F0   ✓
--color-text-secondary:#94A3B8   ✓
--color-border:        #1E293B   ✓
```

**Issues Found:**
- 177 occurrences of hardcoded hex/rgb values across 34 component files (grep confirmed). Top offenders: `architecture-viz.tsx` (49 hits), `model-architecture/attention-viz.tsx`, `knowledge-graph.tsx`, `roadmap-graph.tsx`.
- `architecture-canvas.tsx` uses hardcoded `#7C3AED`, `#06B6D4`, `#1E293B`, `#94A3B8` directly in SVG attributes — these bypass the variable system entirely.
- Token defined twice in wrong location: `--font-heading: var(--font-heading)` on line 93 of layout.tsx is a self-referential no-op.
- `--bg-hover` and `--bg-active` defined but `--bg-surface` inconsistently used in place of `--bg-hover` throughout components.

### Typography System — Score: 75/100

- `Plus_Jakarta_Sans` loaded via Next.js font, mapped to `--font-heading`. ✓
- `Inter` loaded for `--font-sans`. ✓
- `JetBrains Mono` referenced in CSS variable but NOT loaded as a Next.js font — falls back to system monospace. **Bug.**
- Type scale tokens (`--text-h1` through `--text-tiny`) defined but inconsistently applied — most components use raw Tailwind `text-xs`, `text-sm`, `text-lg` classes directly, bypassing the token scale.
- No heading component enforces the token scale automatically.

### Spacing Scale — Score: 70/100

- 9 spacing tokens defined (`--space-1` through `--space-12`). ✓
- Usage: near zero. All components use raw Tailwind spacing (`p-4`, `gap-3`, `px-6`). The tokens exist but are never consumed.

### Motion System — Score: 85/100

- 4 transition tokens defined (`--transition-fast: 100ms`, `--transition-base: 150ms`, etc.). ✓
- Keyframes defined: `fadeIn`, `slideInFromTop/Left/Right`, `popScale`, `shimmer`, `aurora`. ✓
- Missing: No `Framer Motion` anywhere in the codebase despite it being specified in Phase 10 requirements. All animation is pure CSS — functional but less controllable.
- `live-cursor-tracker.tsx` uses `Date.now()` to drive animation — causes hydration mismatches on SSR.

### Shadow / Radius / Border Systems — Score: 80/100

- Shadow tokens defined (`--shadow-hover`, `--shadow-raised`, `--shadow-modal`, `--shadow-glow`). ✓
- Radius tokens defined (`--radius-sm` through `--radius-2xl`). ✓
- Both systems suffer same problem as spacing: tokens defined, raw Tailwind classes used in practice (`rounded-lg`, `shadow-xl`).

### Tailwind Config — Score: 80/100

- CSS variables correctly plumbed into Tailwind `theme.extend.colors`. ✓
- `tailwindcss-animate` plugin present. ✓
- `darkMode: 'class'` set — but no dark mode toggle exists in the app. Mode is hardcoded permanently dark. Not a bug per the spec, but limits future flexibility.

---

## PART 2: LAYOUT AUDIT

### ThreeColumnLayout Component

**File:** `src/components/layout/three-column-layout.tsx`

```
Left:   w-60 (240px) — hidden on < md (768px)
Center: flex-1
Right:  w-80 (320px) — hidden on < lg (1024px)
Height: calc(100vh - 56px)
```

**Assessment:** Structurally correct. The expected 240px/flexible/320px spec is implemented.

**Bug found:** `ThreeColumnLayout` children API uses **positional slots** (`left`, `center`, `right` named props), but pages from Phase 7 onward (`real-time-collaboration`, `advanced-versioning`, `knowledge-intelligence`) call it with **unnamed JSX children** — the component accepts `children: ReactNode` fallback but this ignores the left/right panel props entirely. Those pages have no properly wired three-column layout; all content renders as a single center column.

### Page-by-Page Layout Audit

| Page | Route | Three-Column | Left Panel | Right Panel | Status |
|------|-------|-------------|-----------|------------|--------|
| Home | `/` | No (intentional) | — | — | ✓ Intentional |
| Dashboard | `/dashboard` | ✓ | Left nav stats | Session + milestones | ✓ Implemented |
| Architectures | `/architectures` | ✓ | Sidebar list | Inspector tabs | ✓ Implemented |
| Papers | `/papers` | ✓ | Section timeline | Notes panel | ✓ Implemented |
| Paper-to-Code | `/paper-to-code` | ✓ | Paper excerpts | Implementation map | ✓ Implemented |
| Tensor Trace | `/tensor-trace` | ✗ | — | — | Single column |
| System Design | `/system-design` | ✓ | Pattern library | Properties panel | ✓ Implemented |
| Model Architecture | `/model-architecture` | ✓ | Layers panel | Model stats | ✓ Implemented |
| Workspace Settings | `/workspace-settings` | ✗ | — | — | Single column (tabs only) |
| Real-time Collab | `/real-time-collaboration` | ✗ (broken) | Renders as child | Renders as child | **Broken — wrong API** |
| Advanced Versioning | `/advanced-versioning` | ✗ (broken) | Renders as child | Renders as child | **Broken — wrong API** |
| Knowledge Intelligence | `/knowledge-intelligence` | ✗ (broken) | Renders as child | Renders as child | **Broken — wrong API** |
| Learn | `/learn` | ✗ | — | — | Skeleton placeholder only |
| Schema Design | `/schema-design` | ✗ | — | — | Single-component page |
| API Docs | `/api-docs` | ✗ | — | — | Single-component page |
| Attention Viz | `/attention-viz` | ✗ | — | — | Single-component page |

### Left Rail Navigation

**File:** `src/components/layout/left-rail.tsx`

- Collapse/expand: ✓ (64px ↔ 240px)  
- Active state detection with `usePathname()`: ✓  
- Active left border: **Bug** — uses `border-l-3` which is not a valid Tailwind class (valid classes are `border-l-2`, `border-l-4`). The accent border on the active nav item does not render.
- Nav items: Only 4 (Dashboard, Academy, Search, Settings). Missing: Architectures, Papers, Paper-to-Code, System Design, Knowledge Intelligence, and all Phase 7–10 routes.
- `/search` and `/settings` routes do not exist — clicking them 404s.
- Favorites/Recent sections are empty stubs with no functionality.

---

## PART 3: NAVIGATION AUDIT

### Command Palette — Score: 85/100

**File:** `src/components/layout/command-palette.tsx`

- Cmd+K / Ctrl+K keyboard shortcut: ✓ (works)
- Arrow key navigation: ✓
- Enter to navigate: ✓ (uses `window.location.href`)
- Escape to close: ✓
- Search filtering: ✓ (basic text match)

**Issues:**
- Only 6 hardcoded commands. Does not index actual content (papers, problems, architectures).
- Uses `window.location.href` instead of Next.js `router.push()` — causes full page reload on navigation, losing any in-memory state.
- Not connected to any real search index.
- Two `CommandPalette` components exist (`src/components/command-palette.tsx` and `src/components/layout/command-palette.tsx`) — the layout one is used, the root-level one is orphaned dead code.

### Breadcrumbs

- `src/components/breadcrumb.tsx` exists but is not rendered on any page. Dead code.

### Search Experience

- No global search beyond the command palette's 6 hardcoded items.
- `/search` route referenced in LeftRail nav but the page does not exist (would 404).

### Context Preservation

- No context is preserved between page navigations. All state is local component state — navigating away resets everything.
- No URL state synchronization (selected layers, active tabs, etc.).

---

## PART 4: FEATURE REALITY AUDIT

### Phase 1 — Foundation

| Feature | Status | Evidence |
|---------|--------|---------|
| CSS Design Tokens | ✓ Real | `layout.tsx` inline style block |
| App Shell (Left Rail + Top Bar) | ✓ Real | `app-shell.tsx` renders on every page |
| Command Palette | ✓ Real | `layout/command-palette.tsx` — Cmd+K works |
| ThreeColumnLayout | ✓ Real | Used in 9 pages correctly |
| Dark theme | ✓ Real | Hardcoded permanently dark |

**Phase 1 Score: 90% — Minor: 2 nav links 404, self-referential font token**

---

### Phase 2 — Home Page

| Feature | Status | Evidence |
|---------|--------|---------|
| Hero Section | ✓ Real UI | `home/hero-section.tsx` |
| Features Grid | ✓ Real UI | `home/features-grid.tsx` |
| Tracks Carousel | ✓ Real UI | `home/tracks-carousel.tsx` |
| Aurora animation | ✓ Real | CSS keyframes in globals.css |
| CTA buttons | ✓ Real | Links to `/dashboard`, `/problems` |

**Phase 2 Score: 95% — All intended UI implemented**

---

### Phase 3 — Workspaces (Architecture & Paper)

| Feature | Status | Evidence |
|---------|--------|---------|
| Architecture sidebar with search | ✓ Real UI | `architecture/architecture-sidebar.tsx` |
| Architecture canvas SVG | ✓ Real UI | `architecture/architecture-canvas.tsx` — static SVG |
| Architecture inspector tabs | ✓ Real UI | `architecture/architecture-inspector.tsx` — 3 tabs |
| Paper sidebar timeline | ✓ Real UI | `paper/paper-sidebar.tsx` |
| Paper content viewer | ✓ Real UI | `paper/paper-content.tsx` |
| Paper notes/highlights | ✓ Real UI | `paper/paper-notes.tsx` — saves to local state only |
| Dynamic architecture rendering | ✗ Mock | Canvas is a static SVG, ignores `selectedComponent` prop |

**Phase 3 Score: 85% — UI real, data/interactivity is static**

---

### Phase 4 — Paper-to-Code

| Feature | Status | Evidence |
|---------|--------|---------|
| Paper excerpt viewer | ✓ Real UI | `paper-to-code/paper-excerpt.tsx` |
| Code editor with syntax highlight | ✓ Real UI | `paper-to-code/code-editor.tsx` |
| Implementation map | ✓ Real UI | `paper-to-code/implementation-map.tsx` |
| Line highlighting sync | ✓ Partial | State passed via `selectedExcerpt` prop |
| Tensor Trace | ✓ Real UI | `visualization/tensor-trace.tsx` — static ops |
| Actual code execution | ✗ Missing | No runtime, no eval, no backend |

**Phase 4 Score: 80%**

---

### Phase 5 — System Design

| Feature | Status | Evidence |
|---------|--------|---------|
| Pattern library with search | ✓ Real UI | `system-design/pattern-library.tsx` |
| Design canvas with SVG | ✓ Real UI | `system-design/design-canvas.tsx` — pre-positioned static components |
| Design properties tabs | ✓ Real UI | `system-design/design-properties.tsx` |
| Schema designer | ✓ Real UI | `system-design/schema-designer.tsx` |
| API docs component | ✓ Real UI | `system-design/api-docs.tsx` |
| Drag canvas components | ✗ Missing | Components are pre-positioned SVG, not draggable |
| Add pattern to canvas | ✗ Stub | `console.log('Add pattern:', id)` — not implemented |

**Phase 5 Score: 75%**

---

### Phase 6 — Model Architecture

| Feature | Status | Evidence |
|---------|--------|---------|
| Layers panel | ✓ Real UI | `model-architecture/layers-panel.tsx` |
| Architecture SVG viz | ✓ Real UI | `model-architecture/architecture-viz.tsx` — static SVG |
| Model stats 3-tab | ✓ Real UI | `model-architecture/model-stats.tsx` |
| Attention viz heatmap | ✓ Real UI | `model-architecture/attention-viz.tsx` |
| Layer selection interactivity | ✓ Partial | State propagated via props |
| Dynamic model rendering | ✗ Mock | SVG is hardcoded, ignores actual layer config |

**Phase 6 Score: 80%**

---

### Phase 7 — Collaboration

| Feature | Status | Evidence |
|---------|--------|---------|
| Workspace manager | ✓ Real UI | `collaboration/workspace-manager.tsx` |
| Collaboration panel | ✓ Real UI | `collaboration/collaboration-panel.tsx` |
| Version history timeline | ✓ Real UI | `collaboration/version-history.tsx` |
| Templates library | ✓ Real UI | `collaboration/templates-library.tsx` |
| Page layout (3-col) | ✗ Single tab | `workspace-settings/page.tsx` — tabs only, no 3-col |
| Share link copy | ✓ Real | `navigator.clipboard.writeText()` works |
| Actual data persistence | ✗ None | All mock data, no state persists |
| Template selection action | ✗ Stub | `console.log('Use template:', id)` |

**Phase 7 Score: 50% — UI only, zero backend integration, page layout missing 3-col**

---

### Phase 8 — Real-time Collaboration

| Feature | Status | Evidence |
|---------|--------|---------|
| Presence indicator UI | ✓ Real UI | `collaboration/presence-indicator.tsx` |
| Typing indicator UI | ✓ Real UI | `collaboration/typing-indicator.tsx` |
| Live cursor tracker UI | ✓ Partial UI | `collaboration/live-cursor-tracker.tsx` — uses `Date.now()` SSR bug |
| Activity feed UI | ✓ Real UI | `collaboration/activity-feed.tsx` |
| Toast notifications UI | ✓ Real UI | `collaboration/notification-toast.tsx` |
| Three-column page layout | ✗ Broken | Page passes children not named props to ThreeColumnLayout |
| Real-time WebSocket | ✗ None | No WebSocket, no Pusher, no Supabase realtime |
| Actual presence sync | ✗ None | Hardcoded mock users |
| Cursor position sync | ✗ None | `Date.now()` driven local simulation |
| JSX syntax error | ✗ Bug | Line 168: `<li>✓ Avoid duplicate work</` — unclosed tag |

**Phase 8 Score: 30% — UI scaffolding only, no real-time functionality, page layout broken, JSX bug**

---

### Phase 9 — Advanced Versioning

| Feature | Status | Evidence |
|---------|--------|---------|
| Version diff viewer UI | ✓ Real UI | `versioning/version-diff-viewer.tsx` |
| Branch manager UI | ✓ Real UI | `versioning/branch-manager.tsx` |
| Merge interface UI | ✓ Real UI | `versioning/merge-interface.tsx` |
| Conflict resolver UI | ✓ Real UI | `versioning/conflict-resolver.tsx` |
| Version tags UI | ✓ Real UI | `versioning/version-tags.tsx` |
| Comparison timeline UI | ✓ Real UI | `versioning/comparison-timeline.tsx` |
| Three-column page layout | ✗ Broken | Same ThreeColumnLayout API mismatch as Phase 8 |
| Actual git/version backend | ✗ None | All hardcoded mock data |
| Diff against real data | ✗ None | Static 14-line demo diff |
| Merge conflict resolution | ✗ Mock | Clicking "Use Ours" only updates local React state |

**Phase 9 Score: 30% — UI only, broken layout, zero real version control**

---

### Phase 10 — Knowledge Intelligence

| Feature | Status | Evidence |
|---------|--------|---------|
| Knowledge graph SVG | ✓ Real UI | `knowledge/knowledge-graph.tsx` — pan/zoom work |
| Learning analytics heatmap | ⚠ Bug | `Math.random()` on every render = rerenders show different data |
| AI learning coach UI | ✓ Real UI | `knowledge/ai-learning-coach.tsx` |
| Research journey drag/drop | ✓ Partial | HTML5 drag events wired, works in browser |
| Architecture evolution tree | ✓ Real UI | `knowledge/architecture-evolution.tsx` |
| Mastery tracker UI | ✓ Real UI | `knowledge/mastery-tracker.tsx` |
| Three-column page layout | ✗ Broken | Same API mismatch as Phases 8–9 |
| Real recommendation engine | ✗ None | Static hardcoded recommendations |
| Actual mastery persistence | ✗ None | Local React state only |
| Framer Motion animations | ✗ Missing | Specified in requirements, not implemented |
| `Math.random()` in render | ✗ Bug | `learning-analytics.tsx` lines 17, 45 — non-deterministic SSR |

**Phase 10 Score: 35% — UI scaffolding, three bugs, broken layout, no intelligence**

---

## PART 5: RESPONSIVENESS AUDIT

### Desktop (1280px+)
- Three-column layout renders correctly on Dashboard, Architectures, Papers, Paper-to-Code, System Design, Model Architecture. ✓

### Tablet (768–1023px)
- Left panel hidden (`hidden md:block`) — center panel takes full width. ✓
- Right inspector hidden (`hidden lg:block`). ✓
- Home page hero responsive with `clamp()`. ✓
- Problem: Workspace Settings, Real-time Collab, Versioning pages have no responsive behavior — they were built as single-column anyway.

### Mobile (< 768px)
- Left rail collapses to 64px but still takes horizontal space — on a 375px screen this is 17% of viewport permanently wasted.
- No mobile navigation menu (hamburger).
- Home page CTA buttons wrap correctly. ✓
- Most content pages are effectively broken on mobile — tables overflow, three-column becomes one-column center only.

### Ultrawide (2560px+)
- `max-w-4xl` on home page constrains content — looks correct.
- Three-column layout expands properly as `flex-1` takes available space. ✓
- No max-width cap on the workspace — panels can stretch extremely wide.

### Overflow Issues Found
- `architecture-canvas.tsx`: SVG `viewBox="0 0 800 600"` overflows on screens < 900px.
- `knowledge-graph.tsx`: SVG canvas has no min-height, collapses on small viewports.
- `live-cursor-tracker.tsx`: Cursor positions calculated as percentages of 800×600 — renders incorrectly on any other container size.

---

## PART 6: ACCESSIBILITY AUDIT

**Score: 28/100**

### Findings

| Issue | Severity | Location |
|-------|----------|---------|
| `aria-label` only on 2 left-rail buttons (expand/collapse) | High | `left-rail.tsx:79,128` |
| No `aria-label` on any icon-only buttons | High | All phases 3–10 |
| No `role="dialog"` on command palette modal | High | `command-palette.tsx` |
| No focus trap in command palette | High | `command-palette.tsx` |
| No `aria-current="page"` on active nav items | Medium | `left-rail.tsx` |
| Hardcoded color values bypass CSS variables — no guarantee of WCAG AA contrast | Medium | 34 files |
| No `alt` attributes on SVG visualizations | High | `architecture-canvas.tsx`, `knowledge-graph.tsx` |
| Keyboard navigation impossible in knowledge graph (SVG `<g>` tags) | High | `knowledge-graph.tsx` |
| `placeholder` used as label substitute | Medium | Multiple inputs |
| No skip-to-content link | Medium | `app-shell.tsx` |
| Interactive drag handles have no keyboard equivalent | High | `research-journey.tsx` |
| `tabIndex` not set on draggable elements | High | `research-journey.tsx` |

---

## PART 7: PERFORMANCE AUDIT

**Estimated Score: 55/100**

### Issues Found

| Issue | Impact | Location |
|-------|--------|---------|
| `Math.random()` called during render | High | `learning-analytics.tsx:17,45` — different data on every render/hydration |
| `Date.now()` in render path + `setInterval(50ms)` | High | `live-cursor-tracker.tsx` — 20 state updates/second triggers cascade rerenders |
| Typing indicator `setInterval(300ms)` — always running | Medium | `typing-indicator.tsx` |
| No `useMemo`/`useCallback` anywhere in codebase | Medium | All components |
| No `React.memo` on any component | Medium | All components |
| All pages are `'use client'` — no RSC optimization | High | Every page file |
| SVG with 19 nodes + edges re-renders on every pan/zoom | Medium | `knowledge-graph.tsx` |
| `next/font` for Inter/Jakarta but not JetBrains Mono — fallback causes FOUT | Low | `layout.tsx` |
| Attention viz: 12×12 = 144 SVG cells all re-render on head selection | Medium | `attention-viz.tsx` |
| No lazy loading for phase 7–10 components | Medium | All routes |
| Large inline SVG in architecture canvas (600 lines) blocks paint | Low | `architecture-canvas.tsx` |

---

## PART 8: PREMIUM FEEL AUDIT

Compared against: Linear, Cursor, Vercel, Raycast, TensorTonic

| Dimension | Score | Notes |
|-----------|-------|-------|
| Visual Quality | 7/10 | Dark theme, accent colors, spacing are good. Inconsistent font sizing undermines polish. |
| UX Quality | 5/10 | Three-column layout is correct. Broken pages (Phase 7–10), dead nav links, no state persistence hurt this. |
| Information Density | 7/10 | Panels are dense and well-structured. |
| Professionalism | 6/10 | JSX syntax errors, `console.log` stubs, `Math.random()` in render are not production-ready. |
| Discoverability | 4/10 | Left rail only has 4 items. 12+ pages have no navigation entry point. Command palette has 6 items. |
| Polish | 5/10 | CSS animations work. Active nav border doesn't render (`border-l-3`). Font falls back silently. Cursor tracking creates visible jitter. |

**Overall Premium Feel: 5.7/10**

The Phases 1–6 workspace pages feel close to the target. The Phase 7–10 pages feel like scaffolding.

---

## PART 9: FAKE vs REAL FEATURE MATRIX

| Feature | Real | Partial | Mock Only | Missing |
|---------|------|---------|----------|---------|
| Design Token System | ✓ | | | |
| App Shell / Left Rail | ✓ | | | |
| Command Palette (Cmd+K) | ✓ | | | |
| Three-Column Layout Component | ✓ | | | |
| Dashboard (3-col) | ✓ | | | |
| Home Page Hero/Features | ✓ | | | |
| Architecture Canvas | | ✓ | | |
| Architecture Inspector | | ✓ | | |
| Paper Reader + Notes | | ✓ | | |
| Paper-to-Code Sync | | ✓ | | |
| Tensor Trace | | ✓ | | |
| System Design Canvas | | | ✓ | |
| Drag-and-Drop Canvas | | | | ✗ |
| Model Architecture Viz | | ✓ | | |
| Attention Heatmap | | ✓ | | |
| Workspace Manager | | | ✓ | |
| Collaboration Panel | | | ✓ | |
| Version History | | | ✓ | |
| Templates Library | | | ✓ | |
| Real-Time Presence | | | ✓ | |
| Live Cursor Tracking | | | ✓ | |
| Activity Feed | | | ✓ | |
| Toast Notifications | ✓ | | | |
| Version Diff Viewer | | | ✓ | |
| Branch Manager | | | ✓ | |
| Merge / Conflict Resolution | | | ✓ | |
| Knowledge Graph | | ✓ | | |
| Learning Analytics Heatmap | | | ✓ | (buggy Math.random) |
| AI Learning Coach | | | ✓ | |
| Research Journey Builder | | ✓ | | |
| Architecture Evolution Tree | | ✓ | | |
| Mastery Tracker | | ✓ | | |
| WebSocket / Realtime Backend | | | | ✗ |
| Code Execution / Runtime | | | | ✗ |
| Authentication / User Accounts | | | | ✗ |
| Data Persistence | | | | ✗ |
| Search Index | | | | ✗ |
| Settings Page | | | | ✗ |
| Breadcrumb Navigation | | | | ✗ (dead code) |

---

## PART 10: FINAL SCORECARD

| Dimension | Score | Grade |
|-----------|-------|-------|
| Design System | 80/100 | B |
| UI Consistency | 72/100 | C+ |
| Feature Completeness | 45/100 | F |
| Accessibility | 28/100 | F |
| Performance | 55/100 | D |
| Premium Product Feel | 57/100 | D+ |
| **OVERALL** | **56/100** | **D+** |

### Grade: **C−** (generous for visual quality of core pages; strict for completeness and correctness)

The core six pages (Phases 1–6) would score a **B**. Phases 7–10 drag the total down significantly.

---

## CRITICAL BUGS (Must Fix Before Any Demo)

1. **JSX syntax error** — `src/app/real-time-collaboration/page.tsx:168` — `<li>✓ Avoid duplicate work</` — unclosed tag breaks React parser.
2. **`border-l-3` invalid Tailwind class** — `src/components/layout/left-rail.tsx:97` — active nav item has no accent border.
3. **`Math.random()` in render** — `src/components/knowledge/learning-analytics.tsx:17,45` — different data on every render, SSR/CSR mismatch.
4. **Phases 8/9/10 ThreeColumnLayout API mismatch** — these pages pass content as children, not as `left`/`center`/`right` props — the three-column layout is not active on these pages.
5. **`/search` and `/settings` routes 404** — linked from primary navigation.
6. **Self-referential CSS variable** — `--font-heading: var(--font-heading)` in `layout.tsx:93` — no-op.
