# Phase 15B — Learn Domain Experience

## Route

`/learn/[domain]` — dynamic domain page (e.g. `/learn/deep-learning`, `/learn/machine-learning`, `/learn/llms`)

## Architecture

### Data Layer (`src/data/domains/`)
| File | Purpose |
|---|---|
| `types.ts` | All TypeScript interfaces: `DomainData`, `DomainStats`, `ProgressionLevel`, `DomainProgress`, `RoadmapStage`, `TopicCluster`, `ClusterTopic`, `DomainLesson`, `DomainKGNode`, `DomainKGEdge`, `DomainProject`, `DomainPaper` |
| `deep-learning.ts` | Full `DomainData` for Deep Learning (8 roadmap stages, 4 clusters, 6 lessons, 9 KG nodes, 10 edges, 4 projects, 5 papers) |
| `machine-learning.ts` | Full `DomainData` for Machine Learning |
| `llms.ts` | Full `DomainData` for LLMs |
| `index.ts` | `DOMAIN_REGISTRY` + `generateFallback()` + `getDomainData(slug)` export |

### Components (`src/components/domain/`)
| Component | Description |
|---|---|
| `DomainHero` | Breadcrumb, domain badge, h1, tagline, 4 stats (Lessons/Tracks/Problems/Projects), progression ladder (Beginner→Intermediate→Advanced) with topic samples |
| `ProgressOverview` | "Your Progress" heading, animated progress bar, 4 metric cards with `requestAnimationFrame` counter animation (cubic ease-out) |
| `LearningRoadmap` | Vertical timeline, `STATUS_CONFIG` map, expandable stages via `AnimatePresence`, auto-expands `in_progress` stage, topic chips link to `/learn/[domain]/[topic-slug]` |
| `TopicClusters` | 2-col grid, `ClusterCard` with animated progress bar, `TopicRow` with status icons, locked topics get `aria-disabled` + `tabIndex=-1` |
| `FeaturedLessons` | 2-col grid, index/completion indicator, "New" badge, progress bar for in-progress lessons, CTA: "Start"/"Continue"/"Review" |
| `DomainKnowledgeGraph` | Interactive SVG (nodes + edges), hover dimming/highlighting, pulse ring animation, edge labels on hover, keyboard nav (tabIndex, Enter/Space) |
| `ProjectShowcase` | 3-col grid, 3D tilt hover via `useMotionValue`+`useSpring`, skill tags, difficulty badges, hours estimate |
| `ResearchConnections` | 3-col grid, impact badges (Landmark/High Impact/Notable), citation count formatting (85k), "Study Paper →" CTA |

### Page (`src/app/learn/[domain]/page.tsx`)
- `'use client'`, `useParams<{ domain: string }>()`
- Calls `getDomainData(slug)` — returns fallback for unknown domains
- 404 state rendered inline if fallback returns null
- AppShell layout: `h-full flex overflow-hidden` → `flex-1 overflow-y-auto min-w-0` with thin purple scrollbar

### API Route (`src/app/api/learn/domain/[slug]/route.ts`)
- `GET /api/learn/domain/[slug]` → `{ domain: DomainData }` or `{ error: "..." }` (404)

## Tests

### Summary
| File | Tests | Status |
|---|---|---|
| `DomainHero.test.tsx` | 10 | ✓ |
| `ProgressOverview.test.tsx` | 10 | ✓ |
| `LearningRoadmap.test.tsx` | 10 | ✓ |
| `TopicClusters.test.tsx` | 10 | ✓ |
| `FeaturedLessons.test.tsx` | 10 | ✓ |
| `DomainKnowledgeGraph.test.tsx` | 10 | ✓ |
| `ProjectShowcase.test.tsx` | 10 | ✓ |
| `ResearchConnections.test.tsx` | 10 | ✓ |
| **Total Phase 15B** | **80** | **✓** |

**Full suite: 429/429 tests pass** (includes all prior phases)

### Test Infrastructure
- Framer Motion mocked per-file via Proxy (motion.* → DOM element, AnimatePresence → passthrough)
- `useMotionValue`/`useSpring` mocked in ProjectShowcase tests
- `next/link` and `next/navigation` mocked globally in `src/__tests__/setup.ts`

## Design Constraints Honoured
- No `any` TypeScript — all props fully typed
- No hardcoded colors — CSS custom properties + Tailwind only
- No `Math.random()` at module level
- Existing pages untouched (Landing Page, Dashboard)

## Vitest Config Update
Added to `coverage.include` in `vitest.config.ts`:
```
src/components/domain/**
src/app/api/learn/domain/**
```

## Next: Phase 15C
The `/learn/[domain]/[topic]` page — individual topic lessons with content, code examples, and interactive exercises.
