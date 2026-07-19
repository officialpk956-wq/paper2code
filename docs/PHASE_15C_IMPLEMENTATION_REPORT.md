# Phase 15C Implementation Report — Topic Learning Experience

## Overview

Built the `/learn/[domain]/[topic]` route: a premium 3-column AI textbook chapter experience.

- **Route**: `/learn/[domain]/[topic]` (e.g. `/learn/deep-learning/attention`)
- **Components**: 14 new components in `src/components/topic/`
- **Data layer**: `src/data/topics/` with full TypeScript types + `attention` topic data
- **API**: `GET /api/learn/topic/[domain]/[topic]`
- **Tests**: 149 new tests (all passing), total suite: **578/578**

---

## Architecture

### 3-Column Layout

```
┌─────────────┬──────────────────────────────┬──────────────┐
│ Left: Nav   │  Center: Scrollable Content  │ Right: Study │
│ TopicSidebar│  11 content sections         │ StudyAssist. │
│ 208px       │  flex-1 overflow-y-auto      │ 256px        │
└─────────────┴──────────────────────────────┴──────────────┘
```

- Left sidebar: hidden on mobile/tablet (lg:flex), sticky chapter nav
- Center: scrollable via `overflow-y-auto`, scroll spy via IntersectionObserver
- Right sidebar: hidden below xl (xl:flex), sticky study tools

### Scroll Spy

IntersectionObserver with `root: scrollContainer`, `rootMargin: '-10% 0px -65% 0px'` — highlights the section currently near the top of the viewport.

---

## Components (14)

| Component | ID | Description |
|---|---|---|
| `TopicHero` | — | Breadcrumb, title, difficulty, meta row, back button |
| `TopicSidebar` | — | 11-section chapter nav, progress bar, scroll spy |
| `StudyAssistant` | — | Prerequisites, related topics, quick revision, action buttons |
| `MotivationSection` | `motivation` | Problem/limitation/solution cards with icons |
| `IntuitionSection` | `intuition` | Interactive token attention visualization, animated flow |
| `MathSection` | `mathematics` | KaTeX formula cards with expand/variable/derivation |
| `ArchitectureSection` | `architecture` | SVG flowchart with hover-to-inspect nodes |
| `CodeWalkthrough` | `code` | Tabbed syntax-highlighted editor, copy button, line annotations |
| `ApplicationsSection` | `applications` | Where-used / why-useful cards |
| `ResearchTimeline` | `timeline` | Click-to-reveal vertical timeline |
| `RelatedPapers` | `papers` | Paper cards with impact badges, citation counts |
| `InterviewNotes` | `interview` | Accordion Q&A with tags |
| `PracticePreview` | `practice` | MCQ + open question preview, links to `/dojo` |
| `SummarySection` | `summary` | Key takeaway cards with gradient banner |

---

## Data Layer

- `src/data/topics/types.ts` — 20 TypeScript interfaces (TopicData, TopicMeta, Formula, ArchNode, CodeSnippet, etc.)
- `src/data/topics/attention.ts` — Full `attention` topic (45 min, intermediate): 3 motivation cards, 11 tokens, 3 formulas w/ KaTeX, 6 arch nodes, 3 code snippets, 6 apps, 6 timeline events, 4 papers, 6 interview Q&As, 4 practice questions, 4 summary points
- `src/data/topics/index.ts` — `getTopicData(domain, topic)` registry

---

## API

`GET /api/learn/topic/[domain]/[topic]`
- Returns `{ topic: { meta, formulas, code, papers, roadmap, practice } }` or `{ error }` + 404

---

## Tests

- **149 new tests** across 14 component files + 1 API file
- All 578 total tests pass (no regressions)
- Test patterns: framer-motion mocked per-file, next/link + next/navigation mocked globally

### Notable test patterns

- AnimatePresence mock renders children synchronously (exit animations not testable)
- State toggle tests use re-query after click for freshest DOM reference
- SVG nodes queried via `getByRole('button', { name: /label/i })`
- Timeline buttons have `role="listitem"` — queried via `getByRole('listitem', { name: /2014/i })`

---

## Design Rules Followed

- No hardcoded colors — all CSS variables (`var(--accent-primary)`, etc.)
- No `any` TypeScript
- No `Math.random()` at module level
- No modifications to Landing Page or Dashboard
- All Tailwind + CSS tokens

---

## Verification Path

Navigate to `/learn/deep-learning/attention` to see:
- Hero with breadcrumb, difficulty badge, 45 min read
- Left sidebar with 11 chapter sections
- All 11 content sections from Motivation → Summary
- Right sidebar with prerequisites, related topics, quick revision, action buttons
