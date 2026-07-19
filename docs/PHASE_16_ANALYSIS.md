# Phase 16 — Pre-Implementation Analysis

**Based on:** BACKEND_REALITY_AUDIT.md · FRONTEND_REALITY_AUDIT.md · USER_JOURNEY_AUDIT.md · LEARN_SYSTEM_AUDIT.md · API_CONNECTIVITY_REPORT.md  
**Date:** 2026-06-20  
**Status:** Analysis only — no code modified

> **Note on FRONTEND_REALITY_AUDIT discrepancy:** The audit claims `src/content/` does not exist. This is stale. A verified `Get-ChildItem` run confirmed `src/content/` has 100+ files: 27 architectures, 18 papers, 9 implementations, 12 system-design, 8 problems, 1 roadmap, 2 interview, 2 math. The 💀 Broken rows in FRONTEND_REALITY_AUDIT apply to index/list pages (stubs), not to individual `[slug]` content pages. All other findings in that audit remain accurate.

---

## 1 — Current Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PAPER2CODE TODAY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  FRONTEND (Next.js 15, App Router)                                       │
│  ─────────────────────────────────                                       │
│  Layout Layer                                                            │
│    AppShell → LeftRail (BROKEN hrefs) + TopNav (hardcoded XP/streak)    │
│    LandingLayout (separate, no LeftRail)                                 │
│                                                                           │
│  ✅ WORKING PAGES (connected to real backend or real data)               │
│    /papers/upload         POST /api/papers/upload → FastAPI              │
│    /papers/upload/[id]    GET  FastAPI directly (server component)       │
│    /dojo                  Static PROBLEMS (110 items, good data)         │
│    /dojo/[slug]           POST /api/dojo/run  (unsandboxed subprocess)   │
│    /labs                  POST /api/labs/transformer (subprocess)        │
│    /block-viz             GET  /api/papers/[id]/block-hierarchy          │
│    /search                Client-side in-memory index                    │
│    /architectures/[slug]  Content loader → src/content/architectures/   │
│    /papers/[slug]         Content loader → src/content/papers/           │
│    /system-design/[slug]  Content loader → src/content/system-design/   │
│    /paper-to-code/[slug]  Content loader → src/content/implementations/ │
│                                                                           │
│  🟡 PARTIAL PAGES (render but limited/1 item)                           │
│    /learn/[domain]        3 domains authored, 9 use placeholder data     │
│    /learn/deep-learning/attention  1 topic — only one in entire system  │
│    /labs                  transformer only; cnn/vit/diffusion partial    │
│                                                                           │
│  🟠 UI-ONLY PAGES (real UI, all hardcoded data)                         │
│    /                      Landing — static, no Upload CTA               │
│    /dashboard             localStorage only, hardcoded streak/XP        │
│    /learn                 Static DOMAINS/PATHS/RECS imports             │
│    /problems              Static PROBLEMS list                           │
│    /roadmaps, /evolution, /compare, /knowledge-intelligence              │
│                                                                           │
│  🔴 STUB PAGES (empty shells)                                            │
│    /papers                ThreeColumnLayout with zero content            │
│    /architectures         ThreeColumnLayout, no architecture listed      │
│    /playground, /settings, /math, /interview                             │
│                                                                           │
│  DATA LAYER                                                              │
│    src/data/learn/        Hub-level static TS constants                  │
│    src/data/domains/      3 authored DomainData files + generateFallback │
│    src/data/topics/       attention.ts only (1 topic)                    │
│    src/data/problems.ts   110 dojo problems (good)                       │
│    src/content/           100+ MDX files (27 arch, 18 papers, 12 SD…)   │
│                                                                           │
│  BACKEND (FastAPI, SQLite/Postgres)                                      │
│  ─────────────────────────────────                                       │
│  DATABASE (6 tables)                                                     │
│    papers             Core. All pipeline output in architecture_graph    │
│    paper_modules      One-to-many with papers                            │
│    learner_progress   Upserted by /api/progress/update (never called)   │
│    assessment_attempts Written by /api/assessment/validate               │
│    tutor_analytics    Written by /api/tutor/ask                          │
│    users              Implemented, no routes, dead code                  │
│                                                                           │
│  IMPLEMENTED BACKEND SERVICES (~35 routes)                               │
│    Paper pipeline     upload → KG → blueprint → exec graph → DB ✅      │
│    Phase 8-9          assessment, adaptive engine, progress — TESTED ✅  │
│    Phase 10           implementation views, training config — TESTED ✅  │
│    Phase 11           lab experiments — TESTED ✅                        │
│    Phase 12           dojo exercises, solutions — TESTED ✅              │
│    Tutor              POST /api/tutor/ask — TESTED ✅                   │
│    Analytics          GET /api/analytics/dashboard — TESTED ✅          │
│    Progress           POST /api/progress/update — TESTED ✅             │
│                                                                           │
│  DEAD BACKEND CODE                                                       │
│    UserService / UserRepository — implemented, zero routes               │
│    paper_document.py dataclasses — implemented, not imported anywhere    │
│    Phase 12 dojo routes — exist but frontend uses subprocess instead     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2 — User Journey Map

### Target Journey
```
Landing → Dashboard → Learn Topic → Practice Problem → Research Paper
       → Knowledge Graph → Architecture Blueprint → Executable Graph
       → Back to Learn
```

### Current Journey (where each step actually goes)

```
LANDING
  CTA "Start Learning" ──────────────────────────→ /dashboard ✅
  CTA "Research" in nav ─────────────────────────→ /papers ❌ (blank stub)
  Upload paper feature ──────────────────────────→ INVISIBLE (no link anywhere)

DASHBOARD
  "Continue Learning" ────────────────────────────→ /learn (hub, not the topic)
  "Read a Paper" ─────────────────────────────────→ /papers ❌ (blank stub)
  "Upload Paper" ─────────────────────────────────→ DOES NOT EXIST
  Streak / XP / heatmap ──────────────────────────→ hardcoded static values
  Recommendations ─────────────────────────────────→ generic top-level routes

LEFT RAIL (persistent navigation)
  "Deep Learning" ────────────────────────────────→ /architectures ❌ WRONG
  "LLMs" ─────────────────────────────────────────→ /system-design ❌ WRONG
  "NLP" ──────────────────────────────────────────→ /papers ❌ WRONG
  "Statistics" ────────────────────────────────────→ /learn (hub) ❌ WRONG
  "Machine Learning" ──────────────────────────────→ /learn (hub) ❌ WRONG
  "Computer Vision" ───────────────────────────────→ /architectures ❌ WRONG
  "Upload Paper" ──────────────────────────────────→ DOES NOT EXIST
  "Research Hub" ──────────────────────────────────→ DOES NOT EXIST

LEARN HUB (/learn)
  Domain cards (12) ──────────────────────────────→ /learn/[domain] (3 work fully)
  "Continue Learning" card ────────────────────────→ /learn/deep-learning/attention ✅
  "Recently Added" (5 items) ─────────────────────→ ALL 5 are dead links
  Recommendations ─────────────────────────────────→ 2/4 are dead links

DOMAIN PAGE (/learn/deep-learning)
  Topic cluster items (17+ listed) ───────────────→ NEARLY ALL 404 (only attention works)
  Featured lessons ─────────────────────────────────→ NEARLY ALL 404
  Roadmap stage topics ────────────────────────────→ plain text, no hrefs
  Papers listed ───────────────────────────────────→ links to /papers/[slug] (most exist ✅)

TOPIC PAGE (/learn/deep-learning/attention) — the ONLY topic that works
  "Mark Complete" ─────────────────────────────────→ local useState ❌ (resets on nav)
  "Bookmark" ──────────────────────────────────────→ local useState ❌
  "Practice" ─────────────────────────────────────→ /dojo?domain=…&topic=… (params ignored)
  End of topic ────────────────────────────────────→ DEAD END (no "Next Topic" CTA)

/papers (list) ─────────────────────────────────────→ BLANK STUB (no papers shown)
  Upload CTA ─────────────────────────────────────→ DOES NOT EXIST

/papers/upload ─────────────────────────────────────→ ✅ Form works
  On success → /papers/upload/[id] ───────────────→ ✅ Full workspace

PAPER WORKSPACE (/papers/upload/[id])
  KG tab ─────────────────────────────────────────→ ✅ SVG rendered
    Node click ─────────────────────────────────────→ NOTHING (no handler)
  Blueprint tab ───────────────────────────────────→ ✅ rendered
  Executable Graph tab ────────────────────────────→ ✅ rendered + export works
  "Back to papers" ────────────────────────────────→ DOES NOT EXIST
  "Learn about this" ──────────────────────────────→ DOES NOT EXIST
  "Practice coding this" ──────────────────────────→ DOES NOT EXIST
  Page ends ───────────────────────────────────────→ DEAD END

DOJO (/dojo, /dojo/[slug])
  Problems list ───────────────────────────────────→ ✅ 110 problems
  Problem editor ──────────────────────────────────→ ✅ Monaco, run/submit works
  After solving ───────────────────────────────────→ DEAD END (no "Read the paper")
  "Read related paper" ────────────────────────────→ DOES NOT EXIST
```

### The 7 Broken Connectors

| # | Broken Link | From | To |
|---|---|---|---|
| C1 | Upload entry point | Everywhere | /papers/upload |
| C2 | KG node → Learn | Paper workspace | /learn/[domain]/[topic] |
| C3 | Topic → Practice | End of topic page | /dojo?topic=attention (filtered) |
| C4 | Practice → Research | After dojo solve | /papers/[slug] |
| C5 | Papers list → workspace | /papers hub | /papers/upload/[id] |
| C6 | Workspace → Back | Paper workspace | /papers hub |
| C7 | Progress persistence | Every action | POST /api/progress/update |

---

## 3 — Existing Backend Capability Map

**What FastAPI can do right now (verified, tested):**

```
PAPER PIPELINE  ──────────────────────────────────────── FULLY WORKING
  POST /api/papers/upload       PDF → KG + blueprint + exec graph + DB ✅
  GET  /api/papers              List all papers with stats ✅
  GET  /api/papers/{id}         Full paper detail ✅
  GET  /api/papers/{id}/knowledge-graph   KG JSON ✅
  GET  /api/papers/{id}/blueprint         Blueprint JSON ✅
  GET  /api/papers/{id}/executable-graph  Executable graph JSON ✅
  GET  /api/papers/{id}/graph-export      Export as json/mermaid/dot ✅
  GET  /api/papers/{id}/modules           Module list ✅

LEARNER PROGRESS  ────────────────────────────────────── FULLY WORKING, NEVER CALLED
  POST /api/progress/update     Upsert learner_progress row ✅ (needs X-Learner-ID)
  GET  /api/analytics/dashboard Full learner analytics ✅ (needs X-Learner-ID)

ADAPTIVE ENGINE  ─────────────────────────────────────── FULLY WORKING, NEVER CALLED
  GET  /api/adaptive/recommendations    Personalized recs ✅
  GET  /api/adaptive/review-plan        Spaced repetition plan ✅
  GET  /api/adaptive/concept-graph      Learner concept graph ✅

ASSESSMENT  ──────────────────────────────────────────── FULLY WORKING, NEVER CALLED
  GET  /api/assessment/challenge        Generate quiz ✅
  POST /api/assessment/validate         Validate + persist attempt ✅

TUTOR  ───────────────────────────────────────────────── FULLY WORKING, NEVER CALLED
  POST /api/tutor/ask           AI tutor chat ✅
  POST /api/tutor/quiz          Targeted quiz generation ✅
  GET  /api/tutor/learning-path Adaptive path ✅

DOJO  ────────────────────────────────────────────────── FULLY WORKING, BYPASSED
  GET  /api/dojo/exercises              Problem catalog ✅
  GET  /api/dojo/exercises/{id}         Problem detail ✅
  GET  /api/dojo/exercises/{id}/solution Reference solution ✅
  POST /api/dojo/submit                 Record attempt ✅

RESEARCH ENGINEER  ───────────────────────────────────── FULLY WORKING, NEVER CALLED
  GET  /api/implementation/{paper_id}   PyTorch mappings ✅
  GET  /api/training/{paper_id}         Training config ✅
  GET  /api/reproduction/{paper_id}     Reproduction guide ✅
```

**Backend capabilities NOT needed for Phase 16 (defer):**

- UserService / UserRepository (no routes, dead code — defer to auth phase)
- Phase 11 lab experiments
- Phase 10 training estimator
- GET /api/hyperparameters

---

## 4 — Existing Frontend Capability Map

**What frontend components can render right now without changes:**

```
RESEARCH PIPELINE  ──────────────────────────────────── COMPLETE
  PaperUploadWorkspace          Upload form + redirect ✅
  PaperWorkspaceTabs            6-tab workspace ✅
  PaperKnowledgeGraph           SVG graph (needs click handler added) ✅
  ArchitectureBlueprintViewer   Blueprint SVG ✅
  ExecutableGraphViewer         Graph + export ✅

DOJO  ───────────────────────────────────────────────── COMPLETE
  DojoProblemPage               Monaco editor, test runner, submit ✅
  DojoPage                      110 problem list ✅

LEARNING  ───────────────────────────────────────────── PARTIAL
  DomainPage + 8 components     Works for 3 domains ✅
  TopicPage + 14 components     Works for attention topic ✅
  LearnPage                     Hub renders, static data ✅

NAVIGATION  ─────────────────────────────────────────── BROKEN HREFS
  LeftRail                      Structure fine, hrefs wrong
  AppShell TopNav               Renders, hardcoded XP/streak

DESIGN SYSTEM  ──────────────────────────────────────── COMPLETE
  Color tokens, typography, spacing all consistent ✅
  Card, button, badge components available ✅
  ThreeColumnLayout, TwoColumnLayout available ✅
  LoadingSpinner, ErrorBoundary available ✅

DATA ACCESS  ────────────────────────────────────────── PARTIAL
  src/data/problems.ts          110 problems with tags ✅
  src/data/learn/               Hub data (static but good structure) ✅
  src/data/domains/             3 authored domains ✅
  src/data/topics/attention.ts  Full topic data ✅
  src/content/                  100+ MDX files ✅ (available for slug pages)
```

**Frontend reusable patterns for Phase 16:**

| Pattern | Location | Reuse For |
|---|---|---|
| `useSWR` + fetch + error state | `/block-viz` page | Research Hub, Progress |
| Server Component + `notFound()` | `/papers/upload/[id]/page.tsx` | Any new server-fetched page |
| Card layout with loading/empty/error | DojoProblemPage | Research Hub paper cards |
| Tag filtering | DojoPage filter bar | Research Hub filter |
| `ThreeColumnLayout` | Multiple pages | Research Hub layout |

---

## 5 — Gap Analysis

### 5A — Navigation Gaps (Phase 16A)

| Item | Current href | Correct href | Risk |
|---|---|---|---|
| LeftRail "Deep Learning" | `/architectures` | `/learn/deep-learning` | Low (fix string) |
| LeftRail "LLMs" | `/system-design` | `/learn/llms` | Low |
| LeftRail "NLP" | `/papers` | `/learn/nlp` | Low |
| LeftRail "Statistics" | `/learn` | `/learn/statistics` | Low |
| LeftRail "Machine Learning" | `/learn` | `/learn/machine-learning` | Low |
| LeftRail "Computer Vision" | `/architectures` | `/learn/computer-vision` | Low |
| LeftRail missing | — | `/papers/upload` | Low (add item) |
| LeftRail missing | — | `/papers` (Research Hub) | Low (exists, add item) |
| Dashboard "Read a Paper" | `/papers` | still `/papers` (after 16B fix) | Low |
| Dashboard missing | — | `/papers/upload` quick action | Low (add card) |
| Dashboard "Complete Multi-Head Attention" | `/learn` | `/learn/deep-learning/attention` | Low |
| Dashboard "Read Attention is All You Need" | `/papers` | `/papers/attention-is-all-you-need` | Low |
| Learn hub "Recently Added" | 5 dead links | Remove or redirect to real content | Low |
| Learn hub "Recommendations" | 2 dead links | Fix to content that exists | Low |

**File to change:** `src/components/layout/left-rail.tsx`, `src/components/dashboard/quick-actions.tsx`, `src/components/dashboard/recommended-steps.tsx`, `src/data/learn/recommendations.ts`

---

### 5B — Research Hub Gap (Phase 16B)

| Gap | Root Cause | Solution |
|---|---|---|
| `/papers` shows blank layout | `page.tsx` renders ThreeColumnLayout with no data | Replace entire file with ResearchHub component |
| No paper list | No `GET /api/papers` proxy route | Add `src/app/api/papers/route.ts` → proxies to FastAPI `GET /api/papers` |
| No uploaded papers visible | No connection to FastAPI paper list | Fetch from new proxy route |
| No empty/loading/error state | No state management | Standard SWR pattern |
| No upload CTA | Page was never rebuilt | Add prominently above paper list |

**Files to change:** `src/app/papers/page.tsx` (full rewrite), new `src/app/api/papers/route.ts`, new `src/components/research/ResearchHub.tsx`

---

### 5C — Upload Discoverability Gap (Phase 16C)

| Gap | Current | Fix | File |
|---|---|---|---|
| Landing hero has no Upload CTA | Two CTAs both → /dashboard | Add "Upload a Paper" secondary CTA | `src/components/landing/hero-section.tsx` |
| Dashboard has no Upload action | 8 quick actions, none for upload | Add "Upload Paper" as 9th action | `src/components/dashboard/quick-actions.tsx` |
| Sidebar has no Upload link | No nav item | Add to RESEARCH section of LeftRail | `src/components/layout/left-rail.tsx` |

---

### 5D — Research → Learn Connection Gap (Phase 16D)

| Gap | Root Cause | Solution |
|---|---|---|
| KG nodes are not clickable | `PaperKnowledgeGraph` SVG has no click handlers on nodes | Add `onClick` to SVG node groups |
| No concept-to-topic mapping | Doesn't exist | New file `src/lib/concept-mapping.ts` |
| No context panel for KG nodes | Doesn't exist | New `ConceptContextPanel` component |
| Panel has no "Learn" link | Doesn't exist | Panel shows topic meta + "Learn This Concept →" |

**Concept mapping needed:**

```
KG entity name  →  learn route
─────────────────────────────────────────────────────
"Transformer"   →  /learn/deep-learning/attention       (exists ✅)
"Attention"     →  /learn/deep-learning/attention       (exists ✅)
"BERT"          →  /learn/llms/bert                     (404 — fallback)
"GPT"           →  /learn/llms/gpt                      (404 — fallback)
"ResNet"        →  /learn/deep-learning/residual-networks (404 — fallback)
"CNN"           →  /learn/deep-learning/convolution      (404 — fallback)
"RAG"           →  /learn/rag-systems/overview          (404 — fallback)
"Diffusion"     →  /learn/deep-learning/diffusion-models (404 — fallback)
```

Fallback behavior: when mapped route doesn't exist, show "Concept Overview" text panel without a broken link. Never show a link that leads to a 404.

---

### 5E — Learn → Practice Connection Gap (Phase 16E)

| Gap | Root Cause | Solution |
|---|---|---|
| Topic page ends with no action | No component after SummarySection | Add `PracticeSection` as last section |
| No topic → problem mapping | Doesn't exist | New `src/lib/topic-practice-mapping.ts` |
| Dojo query params ignored | `DojoProblemPage` doesn't read them | Scoped problem list in `PracticeSection` (no dojo changes needed) |

**Topic → problems mapping (attention topic):**

`src/data/problems.ts` has 110 problems tagged with `topics`. Check which ones map to `attention`:
- `attention-calculation` — tags include `attention`
- `matrix-multiplication` — base for attention
- Problems tagged `transformer`, `attention` in the problem data

The `PracticeSection` should filter `PROBLEMS` from `src/data/problems.ts` by matching `problem.topics.includes(topicSlug)` or the mapping table.

---

### 5F — Practice → Research Connection Gap (Phase 16F)

| Gap | Root Cause | Solution |
|---|---|---|
| No "read related paper" after solving | No post-solve UI | Add `AfterSolveRecommendations` component |
| No problem → paper mapping | Doesn't exist | New `src/lib/problem-paper-mapping.ts` |
| Link to paper leads where? | `/papers/attention-is-all-you-need` exists in `src/content/papers/` | Link to MDX paper page (already works) |

---

### 5G — Real Progress Gap (Phase 16G)

| Gap | Root Cause | Solution |
|---|---|---|
| "Mark Complete" resets on nav | Local `useState` in `StudyAssistant` | Replace with `useLearnerProgress` hook |
| No `X-Learner-ID` ever sent | No learner ID generated | `src/lib/learner.ts` — localStorage UUID |
| Progress backend never called | No hook, no ID | `useLearnerProgress` hook using SWR + POST /api/progress/update |
| Dashboard shows hardcoded stats | No backend fetch | SWR fetch of `GET /api/analytics/dashboard` |
| Streak/XP in TopNav hardcoded | Static JSX strings | Wire to learner analytics response |

**New proxy routes needed:**
- `POST /api/learn/progress` → FastAPI `POST /api/progress/update`
- `GET  /api/analytics/dashboard` → FastAPI `GET /api/analytics/dashboard`

---

### 5H — What NOT to Fix (Scope Guard)

Per mission rules, do NOT touch:

| Item | Reason |
|---|---|
| `/roadmaps` content | Out of scope |
| `/system-design` section | Out of scope |
| `/interview` section | Out of scope |
| Auth / login flow | No user model exposed, scope for later phase |
| Dojo sandboxing | Security, separate infrastructure concern |
| 9 fallback domain pages | No authored content, not fixable without content |
| Topic pages beyond `attention` | No topic data files exist |
| New animations | Explicitly excluded |

---

## 6 — Execution Plan

### Phase sequence

```
16A — Navigation Repair          ~1 day    Unblocks all other phases
16C — Upload Discoverability     ~0.5 day  Pairs with 16A (same files)
16B — Research Hub               ~2 days   Needs the proxy route; depends on nothing else
16D — Research → Learn           ~2 days   Needs concept-mapping.ts
16E — Learn → Practice           ~1.5 days Needs topic-practice-mapping.ts
16F — Practice → Research        ~1 day    Needs problem-paper-mapping.ts
16G — Real Progress              ~3 days   Can start after 16A
```

---

### Phase 16A — Navigation Repair

**What to change:**

**`src/components/layout/left-rail.tsx`**
```
LEARN section — fix all items:
  "Statistics"       href: '/learn/statistics'
  "Machine Learning" href: '/learn/machine-learning'
  "Deep Learning"    href: '/learn/deep-learning'
  "NLP"              href: '/learn/nlp'
  "LLMs"             href: '/learn/llms'
  "Computer Vision"  href: '/learn/computer-vision'

RESEARCH section — add items:
  "Research Hub"     href: '/papers'
  "Upload Paper"     href: '/papers/upload'

PRACTICE section — verify:
  "Coding Dojo"      href: '/dojo'           ✅ (leave as-is)
  "Problems"         href: '/problems'       ✅ (leave as-is)
```

**`src/components/dashboard/quick-actions.tsx`**
```
Add "Upload Paper"  href: '/papers/upload'
Add "My Papers"     href: '/papers'
Fix "Read a Paper"  keep href: '/papers'  (will be fixed by 16B)
```

**`src/components/dashboard/recommended-steps.tsx`**
```
"Complete Multi-Head Attention" → href: '/learn/deep-learning/attention'
"Read Attention is All You Need" → href: '/papers/attention-is-all-you-need'
"Implement Transformer" → href: '/dojo/attention-calculation' (or closest match in problems.ts)
```

**`src/data/learn/recommendations.ts`**
```
RECENTLY_ADDED — remove items that point to non-existent URLs,
  replace with items from src/content/ that actually exist:
  - /papers/attention-is-all-you-need ✅ exists
  - /papers/bert ✅ exists
  - /architectures/transformer ✅ exists
  - /architectures/resnet ✅ exists
  - /learn/deep-learning/attention ✅ exists

RECOMMENDATIONS — fix dead links similarly
```

**Deliverable:** All navigation items resolve to real pages. Verified by clicking every LeftRail item.

---

### Phase 16C — Upload Discoverability

**`src/components/landing/hero-section.tsx`**
```
Add secondary CTA button: "Upload a Paper" → href: '/papers/upload'
Position: beside or below the primary "Start Learning" button
```

**`src/components/dashboard/quick-actions.tsx`** *(same file as 16A)*
```
Add "Upload Paper" card with Upload icon, href: '/papers/upload'
```

**Deliverable:** A brand-new user can find the upload feature within 5 seconds from landing or dashboard.

---

### Phase 16B — Research Hub

**New file: `src/app/api/papers/route.ts`**
```typescript
// GET → proxy to FastAPI GET /api/papers
// Returns: array of paper objects from backend
// No auth required (papers are public for now)
```

**Rewrite: `src/app/papers/page.tsx`**
```
Convert from ThreeColumnLayout stub to ResearchHub component
Data: fetched via SWR from /api/papers
```

**New file: `src/components/research/ResearchHub.tsx`**
```
Sections:
  1. Page header + "Upload Paper" CTA (always visible)
  2. Loading state: skeleton cards
  3. Empty state: "No papers yet. Upload your first paper." + Upload CTA
  4. Error state: error message + retry button
  5. Search input (client-side filter over fetched papers)
  6. Sort: by upload date (default), by title
  7. Paper cards grid: title, date, architecture classification, status badge, "Open" CTA
     Each card links to /papers/upload/[id]
```

**Paper card data shape** (from `GET /api/papers` response):
```typescript
{
  id: number
  title: string
  created_at: string
  architecture_graph: {
    classification: string
    status: "Draft" | "Published"
  }
}
```

**Deliverable:** `/papers` shows all uploaded papers. Empty state guides to upload. Cards link to workspaces.

---

### Phase 16D — Research → Learn Connection

**New file: `src/lib/concept-mapping.ts`**
```typescript
// Maps KG entity names (as returned by knowledge_extraction_service)
// to learn routes. Only maps to routes that actually exist.
const CONCEPT_MAP: Record<string, {
  topicRoute: string | null,  // null if topic doesn't exist yet
  label: string,
  description: string,
  difficulty: 'beginner' | 'intermediate' | 'advanced',
  estimatedMinutes: number,
}> = {
  'Attention': { topicRoute: '/learn/deep-learning/attention', ... },
  'Transformer': { topicRoute: '/learn/deep-learning/attention', ... },
  'Multi-Head Attention': { topicRoute: '/learn/deep-learning/attention', ... },
  // All others: topicRoute: null (smart fallback)
}
```

**Modify: `src/components/paper-upload/PaperKnowledgeGraph.tsx`**
```
Add onClick handler to each SVG node group
Pass selected concept to parent via callback or use local state
Show ConceptContextPanel when node is clicked
```

**New file: `src/components/paper-upload/ConceptContextPanel.tsx`**
```
Panel that appears when KG node is clicked
Shows:
  - Concept name
  - Brief description (from concept-mapping.ts)
  - Difficulty badge + estimated time (if topic exists)
  - "Learn [Concept] →" button → topicRoute (if exists)
  - "Explore Related Papers" → /papers search
  - If no topic: "Coming soon — check back later" (no broken link)
Close: X button or clicking outside
```

**Deliverable:** Clicking any KG node opens a panel. For "Attention"/"Transformer": Learn button works. For others: informative fallback panel with no broken links.

---

### Phase 16E — Learn → Practice Connection

**New file: `src/lib/topic-practice-mapping.ts`**
```typescript
// Maps topic slugs to arrays of problem slugs from src/data/problems.ts
const TOPIC_PROBLEMS: Record<string, string[]> = {
  'attention': [
    'attention-calculation',
    'matrix-multiplication',
    // ... other problems whose topics[] include 'attention' or 'transformer'
  ],
}
```

**New file: `src/components/topic/PracticeSection.tsx`**
```
Rendered after SummarySection in TopicPage
Title: "Practice What You've Learned"
Subtitle: "Reinforce your understanding with hands-on coding problems"
Shows:
  - 3 problems (easy/medium/hard if available, else top 3)
  - Each problem: title, difficulty badge, tags, "Solve →" → /dojo/[slug]
  - "See all [N] related problems →" → /dojo (pre-filtered? Or just link to dojo)
```

**Modify: `src/app/learn/[domain]/[topic]/page.tsx`**
```
Import and render PracticeSection after SummarySection
Pass topicSlug as prop
```

**Deliverable:** Attention topic page ends with 3 practice problem cards linking directly to Dojo problems.

---

### Phase 16F — Practice → Research Connection

**New file: `src/lib/problem-paper-mapping.ts`**
```typescript
const PROBLEM_PAPERS: Record<string, {
  paperSlug: string, // from src/content/papers/
  architectureSlug?: string, // from src/content/architectures/
  label: string,
}[]> = {
  'attention-calculation': [
    { paperSlug: 'attention-is-all-you-need', label: 'Attention Is All You Need' },
  ],
  'matrix-multiplication': [
    { paperSlug: 'attention-is-all-you-need', label: 'Attention Is All You Need' },
  ],
  // etc.
}
```

**New file: `src/components/dojo/AfterSolveRecommendations.tsx`**
```
Rendered below the test results panel when all tests pass
Title: "Great work! Ready to go deeper?"
Shows up to 3 cards:
  1. "Read the Paper" → /papers/[paperSlug]
  2. "Explore the Architecture" → /architectures/[slug] (if exists)
  3. "Upload a related paper" → /papers/upload
```

**Modify: `src/components/dojo/DojoProblemPage.tsx`**
```
Import AfterSolveRecommendations
Render when submitResult?.passed === true
Pass problemSlug to component for mapping lookup
```

**Deliverable:** After solving attention-calculation, user sees cards for "Attention Is All You Need" paper and "Transformer" architecture. Loop closes.

---

### Phase 16G — Real Progress

**New file: `src/lib/learner.ts`**
```typescript
export function getLearnerId(): string {
  if (typeof window === 'undefined') return ''
  let id = localStorage.getItem('p2c_learner_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('p2c_learner_id', id)
  }
  return id
}
```

**New file: `src/hooks/useLearnerProgress.ts`**
```typescript
// SWR hook that reads + writes topic/module completion
// Sends X-Learner-ID header
// POST /api/learn/progress for markComplete/toggleBookmark
```

**New file: `src/app/api/learn/progress/route.ts`**
```typescript
// POST → proxies to FastAPI POST /api/progress/update
// Adds X-Learner-ID from request header
```

**New file: `src/app/api/analytics/dashboard/route.ts`**
```typescript
// GET → proxies to FastAPI GET /api/analytics/dashboard
// Forwards X-Learner-ID header
```

**Modify: `src/components/topic/StudyAssistant.tsx`**
```
Replace useState ActionState with useLearnerProgress hook
markComplete: calls POST /api/learn/progress
toggleBookmark: calls POST /api/learn/progress
Notes: localStorage (acceptable for v1, full persistence is Phase 17)
```

**Modify: `src/app/layout.tsx`**
```
Add LearnerProvider wrapping entire app
(generates + stores ID on mount, provides via context)
```

**Deliverable:** "Mark Complete" on the attention topic persists across navigation and browser refreshes. Dashboard can show real completion count.

---

## Implementation Order Summary

```
Priority  Phase  Description                        Files Changed
────────────────────────────────────────────────────────────────
1         16A    Fix all nav hrefs                  left-rail.tsx, quick-actions.tsx,
                                                     recommended-steps.tsx,
                                                     recommendations.ts
2         16C    Add upload entry points            hero-section.tsx, quick-actions.tsx
                                                     (same file as 16A, same PR)
3         16B    Research Hub                       papers/page.tsx (rewrite),
                                                     api/papers/route.ts (new),
                                                     research/ResearchHub.tsx (new)
4         16G    Real progress (learner ID + hook)  learner.ts (new),
                                                     useLearnerProgress.ts (new),
                                                     StudyAssistant.tsx (modify),
                                                     api/learn/progress/route.ts (new),
                                                     layout.tsx (modify)
5         16D    KG node → Learn                    concept-mapping.ts (new),
                                                     PaperKnowledgeGraph.tsx (modify),
                                                     ConceptContextPanel.tsx (new)
6         16E    Topic → Practice                   topic-practice-mapping.ts (new),
                                                     PracticeSection.tsx (new),
                                                     [topic]/page.tsx (modify)
7         16F    Practice → Research                problem-paper-mapping.ts (new),
                                                     AfterSolveRecommendations.tsx (new),
                                                     DojoProblemPage.tsx (modify)
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `GET /api/papers` returns shape different from assumed | Medium | High | Read actual FastAPI route response shape before writing ResearchHub component |
| KG SVG structure makes node click handlers hard to add | Low | Medium | Read PaperKnowledgeGraph.tsx before writing handlers |
| `POST /api/progress/update` shape changed from backend audit | Low | Medium | Read backend/server.py endpoint before writing proxy |
| Topic-practice mapping: no attention problems tagged in problems.ts | Medium | Low | Verify tags in src/data/problems.ts before writing mapping |
| `src/content/papers/attention-is-all-you-need/` doesn't exist | Low | Medium | Verify before writing AfterSolveRecommendations hrefs |
| Design system: new components don't match visual style | Low | Low | Copy CSS class patterns from existing cards (e.g., DojoProblemCard) |

---

## Success Criteria Verification (Post-Implementation)

| Criterion | Verified By |
|---|---|
| New user discovers upload within 5 seconds | Landing hero shows "Upload a Paper" CTA |
| Upload → workspace works | Already works (no change needed) |
| KG node → Learn concept | Clicking "Attention" node opens panel with "Learn Attention →" |
| Learn topic → Practice | Attention topic page shows 3 practice problem cards |
| Practice → Research | Solving attention-calculation shows "Read: Attention Is All You Need" |
| "Mark Complete" persists | Marking complete, navigating away, returning shows still-complete state |
| No dead ends in core journey | Every page has a visible next step |
| No broken nav items | Every LeftRail item resolves to a real page |
| Research Hub shows real papers | Empty state or uploaded papers (no hardcoded list) |

---

## Files Created/Modified — Complete List

```
MODIFIED (7 files)
  src/components/layout/left-rail.tsx              (16A+16C — fix hrefs + add items)
  src/components/dashboard/quick-actions.tsx       (16A+16C — fix + add Upload action)
  src/components/dashboard/recommended-steps.tsx   (16A — fix hrefs)
  src/data/learn/recommendations.ts               (16A — fix dead links)
  src/components/landing/hero-section.tsx         (16C — add Upload CTA)
  src/components/topic/StudyAssistant.tsx          (16G — replace useState with hook)
  src/app/layout.tsx                               (16G — add LearnerProvider)
  src/app/learn/[domain]/[topic]/page.tsx          (16E — add PracticeSection)
  src/components/dojo/DojoProblemPage.tsx          (16F — add AfterSolveRecommendations)
  src/components/paper-upload/PaperKnowledgeGraph.tsx (16D — add click handlers)

REWRITTEN (1 file)
  src/app/papers/page.tsx                         (16B — full rewrite to ResearchHub)

NEW FILES (11 files)
  src/app/api/papers/route.ts                     (16B — proxy to GET /api/papers)
  src/app/api/learn/progress/route.ts             (16G — proxy to POST /api/progress/update)
  src/app/api/analytics/dashboard/route.ts        (16G — proxy to GET /api/analytics/dashboard)
  src/lib/learner.ts                               (16G — localStorage UUID)
  src/hooks/useLearnerProgress.ts                  (16G — SWR hook)
  src/contexts/LearnerContext.tsx                  (16G — context provider)
  src/components/research/ResearchHub.tsx          (16B — Research Hub component)
  src/lib/concept-mapping.ts                       (16D — entity → topic route map)
  src/components/paper-upload/ConceptContextPanel.tsx (16D — KG click panel)
  src/lib/topic-practice-mapping.ts               (16E — topic → problems map)
  src/components/topic/PracticeSection.tsx         (16E — end-of-topic practice CTA)
  src/lib/problem-paper-mapping.ts                (16F — problem → paper map)
  src/components/dojo/AfterSolveRecommendations.tsx (16F — post-solve panel)

TOTAL: ~20 files affected
```

---

## Pre-Coding Checklist

Before writing the first line of code, verify these facts by reading the actual files:

- [ ] Read `src/components/layout/left-rail.tsx` — confirm current NAV structure
- [ ] Read `src/components/dashboard/quick-actions.tsx` — confirm current action list
- [ ] Read `src/data/learn/recommendations.ts` — confirm RECENTLY_ADDED structure
- [ ] Read `src/components/landing/hero-section.tsx` — confirm CTA button structure
- [ ] Read `backend/server.py` around `GET /api/papers` — confirm response shape
- [ ] Read `backend/server.py` around `POST /api/progress/update` — confirm request shape
- [ ] Read `src/components/paper-upload/PaperKnowledgeGraph.tsx` — confirm SVG structure
- [ ] Read `src/components/topic/StudyAssistant.tsx` — confirm current useState pattern
- [ ] Read `src/data/problems.ts` — confirm problem tags for attention-related problems
- [ ] Verify `src/content/papers/attention-is-all-you-need/` exists
- [ ] Read `src/components/dojo/DojoProblemPage.tsx` — confirm submit result state shape
