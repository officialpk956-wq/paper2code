# Frontend Reality Audit

**Scope:** Every page in `src/app/`, every route, every API call, every major component, every phase 13–15 feature.  
**Method:** Full read of all page files, component entry points, data files, and content loader.  
**Date:** 2026-06-20

---

## How to Read This Table

| Status | Meaning |
|--------|---------|
| ✅ Finished | Functional, connected to real data/backend, production-quality |
| 🟡 Partial | Works but missing data, only one data item, or limited functionality |
| 🟠 UI Only | Renders without error but uses 100% mock/hardcoded data, no real integration |
| 🔴 Stub | Renders PageSkeleton or empty shell, no real content |
| 💀 Broken | Route exists, page exists, but always returns 404 due to missing content dir |
| 🚫 Dead | Route registered but not reachable from any nav |

---

## Page Inventory

| Page | Route | Backend Connected | Mock Data | Loading State | Error State | Empty State | Status |
|------|-------|:-----------------:|:---------:|:-------------:|:-----------:|:-----------:|--------|
| Landing | `/` | No | Yes (static components) | N/A | N/A | N/A | 🟠 UI Only |
| Dashboard | `/dashboard` | No | Yes (localStorage summary) | No | No | No | 🟠 UI Only |
| Learn Home | `/learn` | No | Yes (DOMAINS, PATHS, TRENDING, RECS) | No | No | No | 🟠 UI Only |
| Learn Domain | `/learn/[domain]` | No | Yes (3 domains: deep-learning, ml, llms) | No | No | 404 for unknown | 🟡 Partial |
| Learn Topic | `/learn/[domain]/[topic]` | No | Yes (only `attention` exists) | No | No | 404 for unknown | 🟡 Partial |
| Paper Upload | `/papers/upload` | **Yes** (POST `/api/papers/upload` → FastAPI) | No | Yes (Spinner) | Yes | Yes | ✅ Finished |
| Paper Workspace | `/papers/upload/[paperId]` | **Yes** (GET `/api/papers/[id]` → FastAPI) | No | Via `notFound()` | Yes (notFound) | N/A | ✅ Finished |
| Papers List | `/papers` | No | No (empty ThreeColumnLayout stub) | No | No | No | 🔴 Stub |
| Paper Detail | `/papers/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| Dojo List | `/dojo` | No | Yes (static PROBLEMS, 110 problems) | N/A | N/A | N/A | ✅ Finished |
| Dojo Problem | `/dojo/[slug]` | Partial (POST `/api/dojo/run`, `/api/dojo/submit`) | No | Yes | Yes | No | ✅ Finished |
| Block Viz | `/block-viz` | Partial (GET `/api/papers/[id]/block-hierarchy`, `/forward-pass` → Python subprocess) | No | Yes | Yes | No | ✅ Finished |
| AI Labs | `/labs` | **Yes** (GET `/api/labs`, POST `/api/labs/transformer`) | Partial (4 labs, transformer only fully wired) | Yes (Spinner) | Yes | Yes | 🟡 Partial |
| Architectures List | `/architectures` | No | No (ThreeColumnLayout with no-op state) | No | No | No | 🔴 Stub |
| Architecture Detail | `/architectures/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| Attention Explorer | `/architectures/transformer/attention` | No | Yes (MOCK_ATTENTION_DATA, 3 heads) | No | No | No | 🟠 UI Only |
| Transformer Math | `/architectures/transformer/math` | No | Yes (5 hardcoded formulas) | No | No | No | 🟠 UI Only |
| Transformer Simulator | `/architectures/transformer/simulator` | No | Yes (6 pipeline stages) | No | No | No | 🟠 UI Only |
| Problems | `/problems` | No | Yes (static PROBLEMS + LEARNING_TRACKS) | No | No | Yes | 🟠 UI Only |
| Roadmaps | `/roadmaps` | No | Yes (static ROADMAPS, hardcoded progress %) | No | No | No | 🟠 UI Only |
| Roadmap Detail | `/roadmaps/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| Evolution | `/evolution` | No | Yes (EVOLUTION_NODES, RESEARCH_JOURNEYS, FAMILY_TREES) | No | No | No | 🟠 UI Only |
| Compare | `/compare` | No | Yes (EVOLUTION_NODES) | No | No | No | 🟠 UI Only |
| Model Architecture | `/model-architecture` | No | No (stub LayersPanel/ArchitectureViz/ModelStats) | No | No | No | 🔴 Stub |
| Attention Viz | `/attention-viz` | No | Delegates to AttentionViz component | Unknown | Unknown | Unknown | 🟠 UI Only |
| Playground | `/playground` | No | No | No | No | No | 🔴 Stub |
| Real-time Collab | `/real-time-collaboration` | No | Yes (mock presence/activity/cursors) | No | No | No | 🟠 UI Only |
| Advanced Versioning | `/advanced-versioning` | No | Yes (mock branches/diffs/tags) | No | No | No | 🟠 UI Only |
| Knowledge Intelligence | `/knowledge-intelligence` | No | Yes (mock KG/analytics/journeys/evolution) | No | No | No | 🟠 UI Only |
| Schema Design | `/schema-design` | No | Delegates to SchemaDesigner component | Unknown | Unknown | Unknown | 🟠 UI Only |
| Settings | `/settings` | No | No (static list, no forms) | No | No | No | 🔴 Stub |
| Search | `/search` | No | Client-side index (search/engine.ts) | No | No | Yes | ✅ Finished |
| Workspace Settings | `/workspace-settings` | No | Yes (mock workspaces/versions/templates) | No | No | No | 🟠 UI Only |
| Admin Platform Health | `/admin/platform-health` | No | Yes (hardcoded checklist) | N/A | N/A | N/A | 🟠 UI Only |
| API Docs | `/api-docs` | No | Delegates to APIDocs component | Unknown | Unknown | Unknown | 🟠 UI Only |
| Paper-to-Code | `/paper-to-code` | No | Delegates to PaperExcerpt/CodeEditor/ImplementationMap | Unknown | Unknown | Unknown | 🟠 UI Only |
| Implementation Detail | `/paper-to-code/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| Math Home | `/math` | No | No | No | No | No | 🔴 Stub |
| Math Topic | `/math/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| Interview Home | `/interview` | No | No | No | No | No | 🔴 Stub |
| Interview Question | `/interview/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| System Design | `/system-design` | No | Delegates to PatternLibrary/DesignCanvas/DesignProperties | Unknown | Unknown | Unknown | 🟠 UI Only |
| System Design Detail | `/system-design/[slug]` | No (content loader) | N/A | N/A | N/A | N/A | 💀 Broken |
| System Trace | `/system-design/[slug]/trace` | No | Yes (SYSTEM_TRACES static data) | No | Yes (notFound) | N/A | ✅ Finished |
| Tensor Trace Home | `/tensor-trace` | No | Delegates to TensorTrace component | Unknown | Unknown | Unknown | 🟠 UI Only |
| Tensor Trace Model | `/tensor-trace/[model]` | No | Yes (TRACES static data) | No | Yes (notFound) | N/A | ✅ Finished |
| Learning Track | `/learning-tracks/[slug]` | No | Yes (LEARNING_TRACKS static data) | No | Yes (notFound) | N/A | ✅ Finished |
| Error Boundary | `error.tsx` | N/A | N/A | N/A | Yes | N/A | ✅ Finished |
| Not Found | `not-found.tsx` | N/A | N/A | N/A | N/A | N/A | ✅ Finished |

---

## Status Summary

| Status | Count | Pages |
|--------|------:|-------|
| ✅ Finished | 12 | Upload, Workspace, Dojo List, Dojo Problem, Block Viz, Labs, Search, System Trace, Tensor Trace Model, Learning Track, Error, Not Found |
| 🟡 Partial | 3 | Learn Domain (3 domains only), Learn Topic (1 topic only), AI Labs (transformer only) |
| 🟠 UI Only | 22 | Landing, Dashboard, Learn Home, Attention Explorer, Transformer Math, Transformer Simulator, Problems, Roadmaps, Evolution, Compare, Attention Viz, Real-time Collab, Advanced Versioning, Knowledge Intelligence, Schema Design, Workspace Settings, Platform Health, API Docs, Paper-to-Code, System Design, Tensor Trace Home, and more |
| 🔴 Stub | 6 | Papers List, Architectures List, Model Architecture, Playground, Settings, Math Home, Interview Home |
| 💀 Broken | 7 | Papers `[slug]`, Architecture `[slug]`, Roadmap `[slug]`, Math `[slug]`, Interview `[slug]`, System Design `[slug]`, Implementation `[slug]` — **all broken because `src/content/` directory does not exist** |

---

## Backend API Surface Used by Frontend

| Frontend Route | API Endpoint | Method | Backend | Notes |
|----------------|-------------|--------|---------|-------|
| `/papers/upload` | `/api/papers/upload` | POST | FastAPI via Next.js proxy | multipart/form-data, full ingestion pipeline |
| `/papers/upload/[paperId]` | `/api/papers/generated/[id]` | GET | FastAPI via Server Component | direct fetch from server |
| `/papers/upload/[paperId]` (KG tab) | `/api/papers/generated/[id]/knowledge-graph` | GET | FastAPI proxy | returns KG JSON |
| `/papers/upload/[paperId]` (blueprint tab) | `/api/papers/generated/[id]/blueprint` | GET | FastAPI proxy | returns blueprint dict |
| `/papers/upload/[paperId]` (exec graph tab) | `/api/papers/generated/[id]/executable-graph` | GET | FastAPI proxy | returns ExecutableGraph |
| `/papers/upload/[paperId]` (export) | `/api/papers/generated/[id]/graph-export?format=` | GET | FastAPI proxy | json/mermaid/dot |
| `/block-viz` | `/api/papers/[id]/block-hierarchy` | GET | Python subprocess (block_viz_service.py) | 10-min cache |
| `/block-viz` | `/api/papers/[id]/forward-pass` | GET | Python subprocess (block_viz_service.py) | same script |
| `/dojo/[slug]` | `/api/dojo/run` | POST | Python subprocess (unsandboxed) | 8s timeout |
| `/dojo/[slug]` | `/api/dojo/submit` | POST | Python subprocess (unsandboxed) | 10s timeout |
| `/labs` | `/api/labs` | GET | Hardcoded LABS_META (not FastAPI) | metadata only |
| `/labs` | `/api/labs/transformer` | POST | Python subprocess (lab_service.py) | 5-min cache |

**FastAPI endpoints declared in backend but NOT called by any frontend:**
- All Phase 8-9 assessment/adaptive routes
- All Phase 10 implementation routes  
- All Phase 11 lab experiment routes
- All Phase 12 dojo routes (frontend uses Next.js API routes instead, bypassing FastAPI for dojo)
- Tutor, analytics, playground routes
- All `/api/users/*` (UserService dead code — no routes exist)

---

## Content Loader Analysis (Critical)

`src/lib/content/loader.ts` reads from `src/content/<type>/<slug>/meta.json` + `content.mdx`.

**`src/content/` directory does NOT exist.**

This breaks all slug-based content pages:

| Content Type | Route | Affected Pages |
|-------------|-------|----------------|
| `architecture` | `/architectures/[slug]` | 0 architectures → always 404 |
| `paper` | `/papers/[slug]` | 0 papers → always 404 |
| `math` | `/math/[slug]` | 0 math topics → always 404 |
| `system-design` | `/system-design/[slug]` | 0 designs → always 404 |
| `problem` | Problems (content-based) | N/A — uses `src/data/problems.ts` instead |
| `interview` | `/interview/[slug]` | 0 questions → always 404 |
| `roadmap` | `/roadmaps/[slug]` | 0 roadmaps → always 404 |
| `implementation` | `/paper-to-code/[slug]` | 0 implementations → always 404 |
| `tensor-trace` | Not used by any page | N/A |

`generateStaticParams()` returns `[]` for all of the above → no static pages built.  
Runtime requests → `notFound()` immediately.

---

## Data Sources Inventory

| Data File | Type | Content | Used By |
|-----------|------|---------|---------|
| `src/data/problems.ts` | Static TS | 110 LeetCode-style DS problems | `/dojo`, `/dojo/[slug]`, `/problems`, `/search` |
| `src/data/roadmaps.ts` | Static TS | ~10 roadmaps with progress | `/roadmaps` |
| `src/data/architecture-catalog.ts` | Static TS | Architecture metadata | `/architectures`, `/search` |
| `src/data/learning-tracks.ts` | Static TS | Learning tracks | `/problems`, `/learning-tracks/[slug]` |
| `src/data/paper-timeline.ts` | Static TS | Paper timeline data | Unknown |
| `src/data/evolution.ts` | Static TS | Evolution nodes, research journeys, family trees | `/evolution`, `/compare` |
| `src/data/tensor-traces.ts` | Static TS | TRACES map | `/tensor-trace/[model]` |
| `src/data/system-traces.ts` | Static TS | SYSTEM_TRACES map | `/system-design/[slug]/trace` |
| `src/data/learn/domains.ts` | Static TS | DOMAINS array | `/learn` |
| `src/data/learn/paths.ts` | Static TS | LEARNING_PATHS | `/learn` |
| `src/data/learn/topics.ts` | Static TS | TRENDING_TOPICS, CONTINUE, RECENTLY_ADDED | `/learn` |
| `src/data/learn/recommendations.ts` | Static TS | RECOMMENDATIONS array | `/learn` |
| `src/data/domains/deep-learning.ts` | Static TS | getDomainData('deep-learning') | `/learn/deep-learning` |
| `src/data/domains/machine-learning.ts` | Static TS | getDomainData('machine-learning') | `/learn/machine-learning` |
| `src/data/domains/llms.ts` | Static TS | getDomainData('llms') | `/learn/llms` |
| `src/data/topics/attention.ts` | Static TS | Full attention topic data | `/learn/deep-learning/attention` |
| `src/lib/search/engine.ts` | Client-side | In-memory index over problems/architectures/roadmaps/tracks | `/search`, command palette |

---

## Phase 13–15 Feature Verification

| Phase | Feature | Route | Backend | Frontend Finished |
|-------|---------|-------|---------|-------------------|
| 13A | PDF ingestion pipeline | `/papers/upload` | ✅ Full pipeline | ✅ Upload + redirect |
| 13B | Knowledge graph extraction | `/papers/upload/[id]` tab 3 | ✅ KG API | ✅ PaperKnowledgeGraph SVG |
| 14A | Architecture blueprint | `/papers/upload/[id]` tab 4 | ✅ Blueprint API | ✅ ArchitectureBlueprintViewer |
| 14B | Executable graph compiler | `/papers/upload/[id]` tab 5 | ✅ Exec graph API | ✅ ExecutableGraphViewer + export |
| 15B | Learn domain pages | `/learn/[domain]` | ❌ Static only | 🟡 3 domains, no backend |
| 15C | Learn topic textbook | `/learn/[domain]/[topic]` | ❌ Static only | 🟡 1 topic (attention) only |

---

## Component Depth (Key Components)

| Component | Depth | Backend | Real Data | Notes |
|-----------|-------|---------|-----------|-------|
| `PaperUploadWorkspace` | Full | POST `/api/papers/upload` | Yes | Loading + error + redirect on success |
| `PaperWorkspaceTabs` | Full | Multiple GET APIs | Yes | 6 tabs: raw text, KG, blueprint, exec graph, export |
| `PaperKnowledgeGraph` | Full | KG tab API | Yes | SVG graph rendering |
| `ArchitectureBlueprintViewer` | Full | Blueprint API | Yes | SVG template viewer |
| `ExecutableGraphViewer` | Full | Exec graph API + export | Yes | Graph + export buttons |
| `DojoProblemPage` | Full | POST dojo/run + submit | Yes | Monaco editor, test runner |
| `BlockVizPage` | Full | Python subprocess APIs | Yes | 3-level hierarchy + forward pass |
| `LabsPage` | Full | Python subprocess APIs | Yes | 4 labs, params → metrics |
| `PresenceIndicator` | UI Only | None | Mock | Static user list |
| `ActivityFeed` | UI Only | None | Mock | Hardcoded activity items |
| `LiveCursorTracker` | UI Only | None | Mock | Simulated cursor positions |
| `BranchManager` | UI Only | None | Mock | Hardcoded branch list |
| `VersionDiffViewer` | UI Only | None | Mock | Hardcoded diff content |
| `AILearningCoach` | UI Only | None | Mock | Static suggestions |
| `KnowledgeGraph` | UI Only | None | Mock | Static graph nodes |
| `LearningAnalytics` | UI Only | None | Mock | Hardcoded charts |
| `WorkspaceManager` | UI Only | None | Mock | Mock workspace list |
| `SchemaDesigner` | UI Only | None | Mock | Canvas + pattern library |
| `TensorTrace` | UI Only | None | Static data | TRACES static map |
| `TrackPageClient` | Partial | None | Static LEARNING_TRACKS | Real data, no backend |
