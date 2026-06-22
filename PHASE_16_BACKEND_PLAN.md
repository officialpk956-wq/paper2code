# Phase 16 — Learn Backend Plan

**Goal:** Convert the Learn system from a fully static TypeScript constant store into a backend-driven platform with real learner state, adaptive content, and persistent progress tracking.

**Constraint:** Do not break the 3 authored domains or the 1 authored topic. Their content is seed data, not mock data — migrate it, don't discard it.

---

## Overview

Phase 16 has four independent sub-phases. They can be parallelized but Phase 16A is a prerequisite for 16C and 16D.

```
16A — Learner Identity + Session          (prerequisite for C, D)
16B — Content Data API                    (independent)
16C — Progress Tracking                   (needs 16A)
16D — Adaptive Engine                     (needs 16A + 16C)
```

---

## Phase 16A — Learner Identity & Session

**What's missing today:** Every progress, bookmark, and recommendation feature requires an `X-Learner-ID` header. The frontend never sets it. The backend has the full `UserService` and `LearnerProgress` table, but they're dead.

### Backend changes needed

None required for basic identity — FastAPI already has the routes. The issue is purely frontend.

### Frontend changes

**1. Generate a durable learner ID on first visit**

```typescript
// src/lib/learner.ts
export function getLearnerId(): string {
  let id = localStorage.getItem('paper2code_learner_id');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('paper2code_learner_id', id);
  }
  return id;
}
```

A `crypto.randomUUID()` approach is sufficient for Phase 16 — it avoids adding an auth system while still enabling per-learner server state. Replace with a real auth system in Phase 17.

**2. Create a hook that attaches the header to all Learn API calls**

```typescript
// src/hooks/useLearner.ts
export function useLearnerHeaders(): HeadersInit {
  return { 'X-Learner-ID': getLearnerId() };
}
```

**3. Create a LearnerContext** that makes the ID available without prop-drilling:

```tsx
// src/contexts/LearnerContext.tsx
export const LearnerContext = createContext<{ learnerId: string }>({ learnerId: '' });
export function LearnerProvider({ children }: { children: ReactNode }) {
  const learnerId = getLearnerId(); // called once on mount
  return <LearnerContext.Provider value={{ learnerId }}>{children}</LearnerContext.Provider>;
}
```

Mount `LearnerProvider` in the root layout (`src/app/layout.tsx`).

### Deliverables
- `src/lib/learner.ts` — ID generation
- `src/hooks/useLearner.ts` — header factory
- `src/contexts/LearnerContext.tsx` — context provider
- Update `src/app/layout.tsx` to wrap with `LearnerProvider`
- Delete `CONTINUE_LEARNING.lastAccessedAt` hardcoded value — will be replaced by real backend data

---

## Phase 16B — Content Data API

**What's missing today:** All domain and topic data is compiled into the JavaScript bundle as static TypeScript constants. Adding a new domain requires a code deployment. Adding a new topic requires a code deployment.

### Strategy: Backend-first for domains, keep TypeScript types

The `DomainData` and `TopicData` TypeScript interfaces are well-designed and should be preserved as the canonical content schema. The migration moves the *data* (not the schema) into FastAPI's database.

### Backend changes needed

**New FastAPI models (database tables):**

```python
# Domain catalogue
class DomainRecord(Base):
    __tablename__ = "domains"
    id          = Column(Integer, primary_key=True)
    slug        = Column(String, unique=True, nullable=False)
    name        = Column(String)
    tagline     = Column(String)
    description = Column(String)
    color       = Column(String)
    glow_color  = Column(String)
    icon        = Column(String)
    stats_json  = Column(JSON)       # {lessons, tracks, problems, projects}
    progression_json = Column(JSON)  # ProgressionLevel[]
    roadmap_json     = Column(JSON)  # RoadmapStage[]
    topic_clusters_json = Column(JSON)  # TopicCluster[]
    featured_lessons_json = Column(JSON)
    kg_nodes_json   = Column(JSON)
    kg_edges_json   = Column(JSON)
    projects_json   = Column(JSON)
    papers_json     = Column(JSON)
    order_index     = Column(Integer, default=0)
    is_published    = Column(Boolean, default=True)

# Topic catalogue
class TopicRecord(Base):
    __tablename__ = "topics"
    id           = Column(Integer, primary_key=True)
    domain_slug  = Column(String, ForeignKey("domains.slug"))
    slug         = Column(String, nullable=False)
    title        = Column(String)
    # ... all TopicData fields stored as JSON columns
    content_json = Column(JSON)      # Full TopicData payload
    is_published = Column(Boolean, default=True)
```

**New FastAPI routes:**

```python
# Domain catalogue
GET  /api/learn/domains                      # list all domains (no auth)
GET  /api/learn/domains/{slug}               # full DomainData for one domain
GET  /api/learn/domains/{slug}/topics        # list topics for a domain (slug, title, meta only)
GET  /api/learn/topics/{domain}/{slug}       # full TopicData for one topic

# Hub-level content
GET  /api/learn/paths                        # learning paths
GET  /api/learn/trending                     # trending topics (could be computed or curated)
```

**Seed script:**

Write a one-time migration script that reads the three authored `DomainData` files and the `attention` topic from TypeScript, converts them to JSON, and inserts into the database. The `generateFallback()` function output can be used to pre-populate the 9 stub domains so the UI doesn't regress.

```bash
# scripts/seed_learn_content.py
# Reads: deep-learning.json, machine-learning.json, llms.json, attention.json
# Writes: database rows via SQLAlchemy
```

### Frontend changes

**1. Replace `getDomainData()` direct call with API fetch in DomainPage**

```tsx
// src/app/learn/[domain]/page.tsx
// Change from:
const data = getDomainData(domain);

// To:
const res = await fetch(`/api/learn/domains/${domain}`, { cache: 'force-cache' });
const data: DomainData | null = res.ok ? await res.json() : null;
```

Convert `DomainPage` from `'use client'` to a Server Component — it doesn't need browser APIs, and server rendering improves SEO.

**2. Replace `getTopicData()` with API fetch in TopicPage**

```tsx
const res = await fetch(`/api/learn/topics/${domain}/${topic}`, { cache: 'force-cache' });
const data: TopicData | null = res.ok ? await res.json() : null;
```

The 11-section content remains client-rendered for interactivity. Convert the outer layout to a server component that passes data as props.

**3. Wire the existing Next.js API proxy routes** (currently dead):

`src/app/api/learn/domains/route.ts` → proxy to FastAPI `GET /api/learn/domains`  
`src/app/api/learn/domain/[slug]/route.ts` → proxy to FastAPI `GET /api/learn/domains/${slug}`  
`src/app/api/learn/topic/[domain]/[topic]/route.ts` → proxy to FastAPI `GET /api/learn/topics/${domain}/${topic}`

**4. Delete dead data files** (after migration is complete):

- `src/data/domains/deep-learning.ts`
- `src/data/domains/machine-learning.ts`
- `src/data/domains/llms.ts`
- `src/data/domains/index.ts` (replace with API call)
- `src/data/topics/attention.ts`
- `src/data/topics/index.ts` (replace with API call)
- `src/components/learn/learn-data.ts` (old duplicate)
- `src/components/learn/learn-types.ts` (old duplicate)

Keep: `src/data/domains/types.ts`, `src/data/topics/types.ts` — the type system stays, only the data moves.

### Deliverables
- 2 new database tables: `domains`, `topics`
- 6 new FastAPI GET routes
- `scripts/seed_learn_content.py`
- Convert `DomainPage` to Server Component
- Convert outer `TopicPage` to Server Component (inner sections stay client)
- Wire 5 existing dead Next.js proxy routes
- Delete stale TS data files post-migration

---

## Phase 16C — Progress Tracking

**What's missing today:** "Mark Complete", "Add Bookmark", "Save Notes" are local `useState` and reset on every page load.

### Backend integration

**Existing FastAPI routes to wire up:**

```
POST /api/progress/update   — requires X-Learner-ID header
GET  /api/analytics/dashboard — per-learner dashboard stats
```

**`POST /api/progress/update` request shape** (from `backend/server.py`):

```json
{
  "topic_slug": "attention",
  "domain_slug": "deep-learning",
  "action": "complete" | "bookmark" | "progress",
  "percent": 100
}
```

### Frontend changes

**1. Replace `useState` in `StudyAssistant` with a `useLearnerProgress` hook**

```typescript
// src/hooks/useLearnerProgress.ts
export function useLearnerProgress(domainSlug: string, topicSlug: string) {
  const { learnerId } = useLearner();

  const { data, mutate } = useSWR(
    learnerId ? `/api/learn/progress/${domainSlug}/${topicSlug}` : null,
    fetcher
  );

  const markComplete = async () => {
    await fetch('/api/learn/progress', {
      method: 'POST',
      headers: { 'X-Learner-ID': learnerId, 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain_slug: domainSlug, topic_slug: topicSlug, action: 'complete', percent: 100 }),
    });
    mutate(); // revalidate SWR cache
  };

  const toggleBookmark = async () => { /* similar */ };

  return { completed: data?.completed ?? false, bookmarked: data?.bookmarked ?? false, markComplete, toggleBookmark };
}
```

**2. New Next.js proxy routes needed:**

`POST /api/learn/progress` → proxy to FastAPI `POST /api/progress/update`  
`GET  /api/learn/progress/[domain]/[topic]` → proxy to FastAPI `GET /api/progress?domain=&topic=`

**3. Update `ProgressOverview` in DomainPage**

Replace static `data.progress` with a SWR fetch:

```typescript
const { data: progress } = useSWR(
  learnerId ? `/api/learn/domain/${domainSlug}/progress` : null,
  fetcher
);
```

**4. Replace hardcoded `CONTINUE_LEARNING` in LearnPage**

```typescript
const { data: continueLearning } = useSWR(
  learnerId ? '/api/learn/continue' : null,
  fetcher
);
// Proxy: GET /api/learn/continue → FastAPI GET /api/progress/last-accessed
```

### Deliverables
- `src/hooks/useLearnerProgress.ts`
- 3 new Next.js API proxy routes for progress
- Update `StudyAssistant` to use hook instead of useState
- Update `ProgressOverview` to use SWR
- Replace hardcoded `CONTINUE_LEARNING` with SWR fetch

---

## Phase 16D — Adaptive Engine

**What's missing today:** Recommendations, trending topics, and "Continue Learning" are all hardcoded. The FastAPI adaptive engine exists and is tested but has zero callers.

### Backend routes to wire (all require X-Learner-ID)

```
GET /api/adaptive/recommendations  → replace RECOMMENDATIONS constant
GET /api/adaptive/review-plan      → new "What to Study Next" section
GET /api/adaptive/concept-graph    → replace hardcoded KnowledgeGraphPreview nodes
POST /api/tutor/ask                → new AI tutor panel in topic page
GET /api/assessment/challenge      → replace hardcoded practice questions in PracticePreview
POST /api/assessment/validate      → replace client-side MCQ answer check
```

### Frontend changes

**1. Wire `Recommendations` component to real adaptive API**

```tsx
// src/components/learn/Recommendations.tsx
const { learnerId } = useLearner();
const { data } = useSWR(
  learnerId ? '/api/learn/recommendations' : null,
  (url) => fetch(url, { headers: { 'X-Learner-ID': learnerId } }).then(r => r.json())
);
const items = data?.recommendations ?? RECOMMENDATIONS; // fallback to static while loading
```

**2. Wire `KnowledgeGraphPreview` to adaptive concept graph**

Replace the 7 hardcoded nodes with the learner-specific concept graph from `/api/adaptive/concept-graph`. Fall back to static nodes if the request fails or learnerId is absent.

**3. Add AI Tutor panel in TopicPage**

Add a new `TutorPanel` component to the right sidebar of the topic page (below `StudyAssistant`). Uses `/api/tutor/ask` with the topic slug as context seed.

**4. Upgrade `PracticePreview` to server-validated assessment**

Replace the hardcoded MCQ answer check with a round-trip to `/api/assessment/validate`. This enables spaced repetition scoring on the backend.

**5. Wire trending topics**

```tsx
const { data } = useSWR('/api/learn/trending', fetcher);
const topics = data?.trending ?? TRENDING_TOPICS; // graceful fallback
```

### Deliverables
- `src/components/learn/Recommendations.tsx` — SWR-backed
- `src/components/learn/KnowledgeGraphPreview.tsx` — adaptive concept graph
- `src/components/topic/TutorPanel.tsx` — new component
- Update `PracticePreview` to use `/api/assessment/validate`
- 6 new Next.js proxy routes wiring adaptive/tutor/assessment endpoints

---

## Migration Sequence

```
Phase 16A (identity)         — 1–2 days   — no risk, additive only
Phase 16B (content API)      — 3–5 days   — risk: seed script must not corrupt rich authored data
Phase 16C (progress)         — 2–3 days   — needs 16A
Phase 16D (adaptive)         — 3–4 days   — needs 16A + 16C for personalized results
```

Total estimate: **~2 weeks** if phases run in series. 16B can run concurrently with 16A since it has no dependency.

---

## Rollback Strategy for 16B (Content API)

The riskiest change is Phase 16B (moving authored domain/topic data to the database). Protect against a broken seed or API failure by:

1. **Keep the TypeScript files** until integration tests confirm the API returns correct data
2. **Feature flag the data source** in `getDomainData()`:

```typescript
const USE_BACKEND = process.env.NEXT_PUBLIC_LEARN_BACKEND === 'true';

export async function getDomainData(slug: string): Promise<DomainData | null> {
  if (!USE_BACKEND) return STATIC_REGISTRY[slug] ?? generateFallback(slug);
  const res = await fetch(`/api/learn/domains/${slug}`);
  return res.ok ? res.json() : null;
}
```

3. Delete the static files only after the feature flag has been `true` in production for one week

---

## Files to Create (Phase 16 Total)

| File | Phase | Type |
|---|---|---|
| `src/lib/learner.ts` | 16A | New |
| `src/hooks/useLearner.ts` | 16A | New |
| `src/contexts/LearnerContext.tsx` | 16A | New |
| `scripts/seed_learn_content.py` | 16B | New |
| `src/app/api/learn/domains/route.ts` | 16B | Rewrite (currently returns static TS) |
| `src/app/api/learn/domain/[slug]/route.ts` | 16B | Rewrite |
| `src/app/api/learn/topic/[domain]/[topic]/route.ts` | 16B | Rewrite |
| `src/hooks/useLearnerProgress.ts` | 16C | New |
| `src/app/api/learn/progress/route.ts` | 16C | New |
| `src/app/api/learn/progress/[domain]/[topic]/route.ts` | 16C | New |
| `src/app/api/learn/continue/route.ts` | 16C | New |
| `src/components/topic/TutorPanel.tsx` | 16D | New |
| `src/app/api/learn/recommendations/route.ts` | 16D | Rewrite |
| `src/app/api/learn/trending/route.ts` | 16D | New |
| `src/app/api/tutor/route.ts` | 16D | New |
| `src/app/api/assessment/route.ts` | 16D | New |

## Files to Delete (Post-Migration)

| File | Safe to Delete After |
|---|---|
| `src/data/domains/deep-learning.ts` | 16B in production 1 week |
| `src/data/domains/machine-learning.ts` | 16B in production 1 week |
| `src/data/domains/llms.ts` | 16B in production 1 week |
| `src/data/domains/index.ts` | 16B in production 1 week |
| `src/data/topics/attention.ts` | 16B in production 1 week |
| `src/data/topics/index.ts` | 16B in production 1 week |
| `src/data/learn/recommendations.ts` | 16C + 16D in production |
| `src/components/learn/learn-data.ts` | Immediately (not imported) |
| `src/components/learn/learn-types.ts` | Immediately (not imported) |
| `src/components/learn/hero-section.tsx` | Verify unused, then delete |
| `src/components/learn/learning-domains.tsx` | Verify unused, then delete |
