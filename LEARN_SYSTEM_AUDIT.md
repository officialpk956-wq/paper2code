# Learn System Audit

**Scope:** `src/app/learn/`, `src/data/domains/`, `src/data/topics/`, `src/data/learn/`, `src/components/domain/`, `src/components/topic/`, `src/components/learn/`  
**Date:** 2026-06-20

---

## 1. Architecture Overview

The Learn system has three routes, each pulling data from a different layer:

```
/learn                         →  LearnPage (hub)          →  src/data/learn/*.ts (direct import)
/learn/[domain]                →  DomainPage               →  getDomainData(slug) from src/data/domains/index.ts
/learn/[domain]/[topic]        →  TopicPage                →  getTopicData(domain, topic) from src/data/topics/index.ts
```

All three pages are `'use client'` components. Zero server-side rendering, zero API calls. Every piece of data is either a TypeScript constant imported at module load time or computed synchronously from those constants.

---

## 2. Feature Status Table

| Feature | Real / Backend | Mock / Hardcoded | Dynamic | Status |
|---|---|---|---|---|
| Domain list (12 domains) | ❌ | ✅ hardcoded in `src/data/learn/domains.ts` | ❌ | Mock |
| Domain detail — authored (3 domains) | ❌ | ✅ hand-written `deep-learning.ts`, `machine-learning.ts`, `llms.ts` | ❌ | Mock (rich) |
| Domain detail — fallback (9 domains) | ❌ | ✅ `generateFallback()` — algorithmic placeholder data | ❌ | Generated placeholder |
| Learning paths (6 paths) | ❌ | ✅ hardcoded in `src/data/learn/paths.ts` | ❌ | Mock |
| Trending topics (6 topics) | ❌ | ✅ hardcoded in `src/data/learn/topics.ts` — fake `learnerCount`, `weeklyGrowth` | ❌ | Mock |
| "Continue Learning" card | ❌ | ✅ hardcoded `progress: 62`, `lastAccessedAt: '2026-06-19T14:32:00Z'` | ❌ | Mock (fake state) |
| Recommendations (4 items) | ❌ | ✅ hardcoded with fake confidence scores (96%, 91%, 88%, 84%) | ❌ | Mock |
| Recently Added (5 items) | ❌ | ✅ hardcoded with absolute timestamps — content does not exist at those URLs | ❌ | Mock |
| KnowledgeGraphPreview on /learn | ❌ | ✅ 7 hardcoded SVG nodes, 7 hardcoded edges | ❌ | Mock |
| Domain progress bar (`masteryPercent`) | ❌ | ✅ hardcoded number in each domain file (34, 45, 18, …) | ❌ | Mock (never changes) |
| Domain streak counter | ❌ | ✅ hardcoded `currentStreak: 5` in authored domains, `0` in fallback | ❌ | Mock (never changes) |
| Domain KG (nodes + edges) | ❌ | ✅ hardcoded per-domain SVG coordinates | ❌ | Mock |
| Roadmap stages + lock status | ❌ | ✅ hardcoded `status: 'locked'/'completed'/'in_progress'` in each domain | ❌ | Mock (static layout) |
| Topic clusters + completion % | ❌ | ✅ hardcoded per-domain, completion derived from static status field | ❌ | Mock |
| Featured lessons | ❌ | ✅ hardcoded per-domain with static `completionPercent` | ❌ | Mock |
| Projects + papers per domain | ❌ | ✅ hardcoded per-domain; fallback domains use `"Foundational Paper in {name}"` | ❌ | Mock |
| Topic page — attention (1 topic) | ❌ | ✅ hand-written `attention.ts` (3 formulas, 3 code snippets, 6 interview Q&As, …) | ❌ | Mock (rich) |
| Topic page — all other topics | ❌ | ❌ | ❌ | Missing (404 screen) |
| "Mark Complete" button | ❌ | ❌ | ✅ local useState only — discarded on navigation | Fake interactivity |
| "Add Bookmark" button | ❌ | ❌ | ✅ local useState only — discarded on navigation | Fake interactivity |
| "Save Notes" button | ❌ | ❌ | ✅ local useState only — discarded on navigation | Fake interactivity |
| MCQ answer check in PracticePreview | ❌ | ✅ answers hardcoded in topic TS file | ✅ local toggle | Mock (correct answer hardcoded) |
| Adaptive recommendations | ❌ — FastAPI `GET /api/adaptive/recommendations` exists but is never called | ❌ | ❌ | Missing |
| Learner progress tracking | ❌ — FastAPI `POST /api/progress/update` exists but is never called | ❌ | ❌ | Missing |
| AI tutor | ❌ — FastAPI `POST /api/tutor/ask` exists but is never called | ❌ | ❌ | Missing |
| Assessment / quiz per topic | ❌ — FastAPI `GET /api/assessment/challenge` exists but is never called | ❌ | ❌ | Missing |
| Learner identity (X-Learner-ID) | ❌ — never set anywhere | ❌ | ❌ | Missing |

---

## 3. Data Layer Inventory

### 3A. `src/data/learn/` — Hub-Level Static Data

| File | Content | Usage |
|---|---|---|
| `domains.ts` | `DOMAINS: Domain[]` — 12 domains with hardcoded progress/topicCount/lessonCount | `/learn` page, `generateFallback()` in `src/data/domains/index.ts` |
| `paths.ts` | `LEARNING_PATHS: LearningPath[]` — 6 paths with hardcoded `completionPercent` | `/learn` page |
| `topics.ts` | `TRENDING_TOPICS: TrendingTopic[]` — 6 topics with fake `learnerCount`/`weeklyGrowth` | `/learn` page |
| `recommendations.ts` | `CONTINUE_LEARNING`, `RECOMMENDATIONS`, `RECENTLY_ADDED` — all hardcoded | `/learn` page |

The hub page imports all four files directly — no API call.

### 3B. `src/data/domains/` — Domain-Level Static Data

| File | Authoring Quality | Topics Covered |
|---|---|---|
| `deep-learning.ts` | Rich: 8 roadmap stages, 4 clusters (21 topics), 6 lessons, 9 KG nodes, 4 projects, 5 papers | Perceptron → Diffusion Models |
| `machine-learning.ts` | Rich: 7 stages, 4 clusters (16 topics), 5 lessons, 8 KG nodes, 3 projects, 3 papers | Regression → Gaussian Processes |
| `llms.ts` | Rich: 6 stages, 4 clusters (16 topics), 5 lessons, 8 KG nodes, 3 projects, 4 papers | Tokenization → Inference Optimization |
| `index.ts` | `generateFallback()` — programmatic stub for all other slugs | Returns "Core Topic 1..4", "Applied Method 1..4", generic papers/projects |

**9 domains that render fallback content:** `mathematics`, `statistics`, `computer-vision`, `nlp`, `ai-agents`, `rag-systems`, `reinforcement-learning`, `mlops`, `research-methodology`

These domains render visually but all topic names, paper titles, and project names are templates like `"Core Topic 1"`, `"Foundational Paper in Statistics"`, `"Statistics Starter Project"`.

**Critical:** `getDomainData()` uses a static registry and falls back synchronously. It never fetches from FastAPI.

### 3C. `src/data/topics/` — Topic-Level Static Data

| File | Content |
|---|---|
| `types.ts` | 20 TypeScript interfaces (`TopicData`, `TopicMeta`, `Formula`, `ArchNode`, `CodeSnippet`, …) |
| `attention.ts` | One fully authored topic: `attention` — 45 min, intermediate, all 11 sections populated |
| `index.ts` | Registry with 3 entries: `deep-learning/attention`, `deep-learning/multi-head-attention` (alias), `llms/attention` |

**All other topic routes return `null` from `getTopicData()` and render a "Topic Not Found" screen.** This covers hundreds of topics linked from domain pages (perceptron, backpropagation, batch-normalization, svm, xgboost, rlhf, lora, kv-cache, etc.).

### 3D. Duplicate Learn Data Files (Old vs. New)

There are two parallel data layers:

| Old (component-level) | New (data-level) | Relationship |
|---|---|---|
| `src/components/learn/learn-data.ts` | `src/data/learn/domains.ts` | Duplicate domain lists, slightly different shapes |
| `src/components/learn/learn-types.ts` | `src/data/domains/types.ts` | Duplicate type definitions |
| `src/components/learn/learning-domains.tsx` | `src/components/learn/DomainGrid.tsx` | Old and new domain grid components |
| `src/components/learn/hero-section.tsx` | `src/components/learn/LearnHero.tsx` | Old and new hero components |

The `/learn` page imports from the new layer (`src/data/learn/*.ts`). The old files remain in the codebase but are not imported by any live page route.

---

## 4. API Routes vs. Reality

Five Next.js API routes exist for the Learn system. Zero of them are ever called by any page.

| Route | Proxies To | Who Should Call It | Who Actually Calls It |
|---|---|---|---|
| `GET /api/learn/domains` | Static `DOMAINS` constant | `/learn` page | Nobody — page imports directly |
| `GET /api/learn/paths` | Static `LEARNING_PATHS` | `/learn` page | Nobody — page imports directly |
| `GET /api/learn/recommendations` | Static recommendations | `/learn` page | Nobody — page imports directly |
| `GET /api/learn/domain/[slug]` | `getDomainData(slug)` | `/learn/[domain]` page | Nobody — page calls function directly |
| `GET /api/learn/topic/[domain]/[topic]` | `getTopicData(domain, topic)` | `/learn/[domain]/[topic]` page | Nobody — page calls function directly |

These five routes are dead code. They expose the same static data as the direct imports, but add nothing.

**FastAPI routes that exist but the Learn section never touches:**

| FastAPI Route | What It Does | Gap |
|---|---|---|
| `GET /api/adaptive/recommendations` | Personalized content order (X-Learner-ID) | `/learn` shows static RECOMMENDATIONS instead |
| `GET /api/adaptive/review-plan` | Spaced repetition plan | No UI for this |
| `GET /api/adaptive/concept-graph` | Learner-specific concept graph | KnowledgeGraphPreview uses hardcoded 7 nodes |
| `POST /api/progress/update` | Mark lesson/topic complete server-side | Mark Complete is localStorage + useState only |
| `GET /api/analytics/dashboard` | Per-learner analytics | No analytics page |
| `POST /api/tutor/ask` | AI tutor chat | No tutor UI |
| `GET /api/assessment/challenge` | Quiz per paper/topic | No assessment UI |
| `POST /api/assessment/validate` | Grade quiz answer | No assessment UI |

---

## 5. Content Realism Analysis

### What looks real but isn't

| UI Element | What it Shows | What's Actually Happening |
|---|---|---|
| "Continue Learning" card — 62% progress | Looks like personal progress | Hardcoded `progress: 62` in `recommendations.ts` — same for every user |
| "18 minutes remaining" on Continue card | Time-to-complete estimate | Derived from hardcoded `minutesRemaining: 18` |
| "Last accessed 2 days ago" | Activity indicator | Hardcoded `lastAccessedAt: '2026-06-19T14:32:00Z'` — stale after any real date |
| Domain masteryPercent (34%, 45%, 18%) | Personal mastery | Hardcoded per domain file |
| "5 day streak" in deep-learning | Streak counter | `currentStreak: 5` in `deep-learning.ts` |
| Recommendation confidence (96%, 91%) | AI confidence score | Arbitrary numbers in `recommendations.ts` |
| Trending topic learner counts (8,420; 11,250) | Live activity | Hardcoded static numbers |
| Topic cluster completion bars | Learning progress | Derived by counting `status === 'completed'` topics — static fields |
| "Mark Complete" button state | Persisted progress | Local `useState` — cleared on navigation/refresh |
| "Add Bookmark" button state | Saved bookmark | Local `useState` — cleared on navigation/refresh |

### What is genuinely well-authored content

The following content represents real editorial investment and would be worth preserving as seed data in a backend migration:

- **3 authored domains**: `deep-learning`, `machine-learning`, `llms` — roadmaps, clusters, KG, papers, projects are thoughtfully written
- **1 authored topic**: `attention` — 11 complete content sections, production-quality explanations, real LaTeX formulas, working code snippets
- **TypeScript type system** (`src/data/domains/types.ts`, `src/data/topics/types.ts`) — 30+ well-designed interfaces that model the full content schema

### What is missing entirely

1. **Topic content for ~200 topics** — only `attention` exists; every other topic slug 404s
2. **9 domain pages** use generic placeholder data (mathematics, CV, NLP, RL, MLOps, etc.)
3. **Any learner state** — no identity, no progress, no bookmarks that survive navigation
4. **Adaptive/personalized features** — all backend routes are built, none are wired
5. **"Recently Added" links** point to URLs that don't resolve (e.g. `/papers/deepseek-r1`, `/learn/llms/sparse-moe`) — content not in `src/content/`

---

## 6. Component-Level Findings

### `src/components/domain/` (8 components)

All receive data as props from the domain page. None make API calls. All are pure renderers of static data.

| Component | Input | Notable Behaviour |
|---|---|---|
| `DomainHero` | `DomainData` | Renders progression ladder with 3 levels — always static |
| `ProgressOverview` | `DomainProgress` | Animated counters over static numbers — looks live, isn't |
| `LearningRoadmap` | `RoadmapStage[]` | Expandable stages with locked/completed state — static |
| `TopicClusters` | `TopicCluster[]` | Links to `/learn/[domain]/[slug]` — most slugs are stub or missing |
| `FeaturedLessons` | `DomainLesson[]` | Links to `/learn/[domain]/[topicSlug]` — most 404 |
| `DomainKnowledgeGraph` | `DomainKGNode[], DomainKGEdge[]` | Interactive SVG — data hardcoded per domain |
| `ProjectShowcase` | `DomainProject[]` | Links to `/paper-to-code/[slug]` — some content exists |
| `ResearchConnections` | `DomainPaper[]` | Links to `/papers/[slug]` — ~18 papers exist in `src/content/papers/` |

### `src/components/topic/` (14 components)

All receive data as props from the topic page. None make API calls.

| Component | Input | Notable Behaviour |
|---|---|---|
| `TopicHero` | `TopicMeta` | Renders title, difficulty, duration — all static fields |
| `TopicSidebar` | active section + meta | Scroll spy works; progress % is static `completionPercent` field |
| `StudyAssistant` | `TopicData` | Mark Complete / Bookmark / Notes — local useState, never persisted |
| `MotivationSection` | `MotivationCard[]` | Pure render |
| `IntuitionSection` | `IntuitionData` | Interactive token attention viz — hardcoded connections |
| `MathSection` | `Formula[]` | KaTeX rendering works; formulas hardcoded |
| `ArchitectureSection` | `ArchNode[], ArchEdge[]` | SVG flowchart with hover-inspect; data hardcoded |
| `CodeWalkthrough` | `CodeSnippet[]` | Tabbed syntax highlighter; code hardcoded |
| `ApplicationsSection` | `ApplicationCard[]` | Pure render |
| `ResearchTimeline` | `TimelineEvent[]` | Click-to-reveal timeline; events hardcoded |
| `RelatedPapers` | `TopicPaper[]` | Links to `/papers/[slug]` — most exist in `src/content/papers/` |
| `InterviewNotes` | `InterviewQuestion[]` | Accordion; questions/answers hardcoded |
| `PracticePreview` | `PracticeQuestion[]` | MCQ with client-side answer reveal; "Full Practice" links to `/dojo?domain=…&topic=…` |
| `SummarySection` | `SummaryPoint[]` | Pure render |

---

## 7. Summary Gap List

| Gap | Impact | Root Cause |
|---|---|---|
| 9 domains show placeholder content | High — CV, NLP, RL, MLOps all render generic stubs | No authored `DomainData` files for these slugs |
| ~200 topic routes return "Topic Not Found" | Critical — every topic link in domain clusters/roadmaps is broken | Only `attention` exists in `src/data/topics/` |
| Progress is fake and resets on reload | Critical — the core learning loop is broken | All state is module-level constants or local useState |
| No learner identity | Critical — backend personalization is impossible | No auth, no X-Learner-ID assignment |
| Recommendations are hardcoded | High — "Based on your progress" is a lie | Static array, not tied to learner model |
| "Recently Added" links don't resolve | Medium — items point to missing content | Content not in `src/content/`, items hardcoded |
| "Mark Complete" / bookmark / notes reset | High — creates frustration, not learning | useState without any persistence layer |
| 5 Learn API routes are dead code | Low (waste) | Direct imports bypass the API routes |
| Duplicate `learn-data.ts` + `learn-types.ts` | Low | Phase migration left stale files |
