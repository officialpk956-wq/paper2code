# API Connectivity Report

**Generated from:** Full frontend + backend trace (2026-06-20)  
**See also:** `FRONTEND_REALITY_AUDIT.md`, `FRONTEND_GAPS.md`

---

## 1. Full API Call Map

| Frontend Page | Frontend Route | Call Type | API Called | Backend Endpoint | Working? |
|---|---|---|---|---|---|
| `/papers` (upload form) | `PaperUploadWorkspace` | `fetch POST` | `/api/papers/upload` | FastAPI `POST /api/papers/upload` | ✅ |
| `/papers/upload/[paperId]` | Server Component | `fetch GET` (direct, no proxy) | FastAPI directly | `GET /api/papers/${paperId}` | ✅ |
| `/block-viz` | `BlockVizPage` (useSWR) | `useSWR GET` | `/api/papers/${archId}/block-hierarchy` | `python block_viz_service.py --action hierarchy` | ✅ |
| `/block-viz` | `BlockVizPage` (conditional useSWR) | `useSWR GET` | `/api/papers/${archId}/forward-pass` | `python block_viz_service.py --action forward-pass` | ✅ |
| `/dojo/[slug]` | `DojoProblemPage` | `fetch POST` | `/api/dojo/run` | `python exec()` subprocess (UNSANDBOXED) | ⚠️ |
| `/dojo/[slug]` | `DojoProblemPage` | `fetch POST` | `/api/dojo/submit` | `python exec()` subprocess (UNSANDBOXED) | ⚠️ |
| `/labs` | `LabsPage` | `fetch GET` | `/api/labs` | Hardcoded LABS_META (no backend call) | ✅ |
| `/labs` | `LabsPage` | `fetch POST` | `/api/labs/transformer` | `python lab_service.py --lab transformer` | ✅ |
| `/labs` | `LabsPage` | `fetch POST` | `/api/labs/cnn` | `python lab_service.py --lab cnn` | ✅ |
| `/labs` | `LabsPage` | `fetch POST` | `/api/labs/vit` | `python lab_service.py --lab vit` | ✅ |
| `/labs` | `LabsPage` | `fetch POST` | `/api/labs/diffusion` | `python lab_service.py --lab diffusion` | ✅ |
| All other pages | — | None | — | None | ❌ static |

**Legend:**
- ✅ Working (or no known breakage)
- ⚠️ Working but insecure (unsandboxed subprocess)
- ❌ No backend connection

### Notes on `/papers/upload/[paperId]`

The server component makes **one** direct FastAPI call to `GET /api/papers/${paperId}`. The response payload includes nested `ingestion.knowledge_graph`, `ingestion.architecture_blueprint`, and `ingestion.executable_graph`. All three are extracted from this single response and passed as props to `PaperWorkspaceTabs` — no additional API calls are made. The five proxy routes at `/api/papers/generated/[id]/*` exist but are never invoked.

---

## 2. Next.js API Routes That Are Never Called

These routes exist in `src/app/api/` but no frontend page ever fetches them.

### 2A. Superseded Paper Proxy Routes

The paper workspace page calls FastAPI directly. These proxy routes exist but are dead code.

| Route | File | Proxies To | Reason Never Called |
|---|---|---|---|
| `GET /api/papers/generated/[id]` | `src/app/api/papers/generated/[id]/route.ts` | FastAPI `GET /api/papers/${id}` | Server component calls FastAPI directly |
| `GET /api/papers/generated/[id]/knowledge-graph` | `.../knowledge-graph/route.ts` | FastAPI `GET /api/papers/${id}/knowledge-graph` | KG embedded in single paper response |
| `GET /api/papers/generated/[id]/blueprint` | `.../blueprint/route.ts` | FastAPI `GET /api/papers/${id}/blueprint` | Blueprint embedded in single paper response |
| `GET /api/papers/generated/[id]/executable-graph` | `.../executable-graph/route.ts` | FastAPI `GET /api/papers/${id}/executable-graph` | ExecutableGraph embedded in single paper response |
| `GET /api/papers/generated/[id]/graph-export` | `.../graph-export/route.ts` | FastAPI `GET /api/papers/${id}/graph-export` | Export is 100% client-side (buildMermaid/buildDot/JSON.stringify) |

### 2B. Learn API Routes Bypassed by Static Imports

The Learn section pages import TypeScript data constants directly. The API routes wrap the same data but nobody calls them.

| Route | File | Data Source | Reason Never Called |
|---|---|---|---|
| `GET /api/learn/domains` | `src/app/api/learn/domains/route.ts` | `src/data/learn/domains.ts` | `/learn` imports `DOMAINS` constant directly |
| `GET /api/learn/paths` | `src/app/api/learn/paths/route.ts` | `src/data/learn/*.ts` | `/learn` imports `LEARNING_PATHS` constant directly |
| `GET /api/learn/recommendations` | `src/app/api/learn/recommendations/route.ts` | `src/data/learn/*.ts` | `/learn` imports `RECOMMENDATIONS` constant directly |
| `GET /api/learn/domain/[slug]` | `src/app/api/learn/domain/[slug]/route.ts` | `getDomainData(slug)` from static TS | `/learn/[domain]` calls `getDomainData()` directly |
| `GET /api/learn/topic/[domain]/[topic]` | `src/app/api/learn/topic/[domain]/[topic]/route.ts` | `getTopicData()` from static TS | `/learn/[domain]/[topic]` calls `getTopicData()` directly |

---

## 3. FastAPI Routes Never Called by Any Frontend Page

The backend has ~40 routes across phases 8–12. Only 4 are connected to the frontend. These 35+ are fully implemented, tested, and completely unreachable from the UI.

### 3A. Legacy Routes (Pre-Phase 13)

| FastAPI Route | Notes |
|---|---|
| `GET /` | Serves `static/index.html` — legacy entry point |
| `POST /api/parse_pdf` | Legacy PDF parser — superseded by `/api/papers/upload` |
| `POST /api/parse_text` | Legacy text parser |
| `POST /api/compare_text` | Legacy compare endpoint |
| `POST /api/analyze_graph` | Legacy graph analysis |
| `POST /api/playground/generate` | ResNet/Transformer/U-Net generator — no `/playground` page implemented |

### 3B. Paper/Module Routes Never Surfaced

| FastAPI Route | Notes |
|---|---|
| `GET /api/papers` | List all uploaded papers — `/papers` page is an empty stub |
| `GET /api/papers/{paper_id}/modules` | Per-paper module list — no module detail page |
| `GET /api/modules/{module_id}` | Module detail — no frontend page |
| `POST /api/papers/{paper_id}/publish` | Publish paper — no publish button or flow in UI |

### 3C. Phase 8 — Adaptive Assessment (Never Connected)

All routes require `X-Learner-ID` header; frontend never sets this header.

| FastAPI Route | Notes |
|---|---|
| `GET /api/assessment/challenge` | Adaptive quiz — no assessment page |
| `POST /api/assessment/validate` | Validate quiz answer — no assessment page |

### 3D. Phase 9 — Tutor + Adaptive + Analytics (Never Connected)

| FastAPI Route | Notes |
|---|---|
| `POST /api/tutor/ask` | AI tutor chat — no tutor UI |
| `POST /api/tutor/quiz` | Tutor quiz — no tutor UI |
| `GET /api/tutor/learning-path` | Learner path from tutor — no tutor UI |
| `GET /api/adaptive/recommendations` | Adaptive recommendations — `/learn` section is static |
| `GET /api/adaptive/review-plan` | Review plan — not surfaced |
| `GET /api/adaptive/concept-graph` | Concept graph — not surfaced |
| `POST /api/progress/update` | Update learner progress — no mark-complete flow anywhere |
| `GET /api/analytics/dashboard` | Analytics — no analytics page |
| `GET /api/analytics/recommendations` | Analytics recommendations — not surfaced |
| `GET /api/health/db` | DB health check — not surfaced in UI |

### 3E. Phase 10 — Implementation + Training (Never Connected)

| FastAPI Route | Notes |
|---|---|
| `GET /api/implementation/{paper_id}` | Implementation detail — no implementation page per paper |
| `GET /api/modules/{module_id}/implementation` | Module implementation — no module implementation page |
| `GET /api/training/{paper_id}` | Training config for paper — not surfaced |
| `GET /api/hyperparameters` | Hyperparameter reference — not surfaced |
| `POST /api/training-estimator` | Training cost estimator — not surfaced |
| `GET /api/reproduction/{paper_id}` | Reproduction guide — not surfaced |

### 3F. Phase 11 — FastAPI Lab Routes (Never Connected)

The `/labs` page uses Python subprocesses via Next.js routes instead of these FastAPI endpoints.

| FastAPI Route | Notes |
|---|---|
| `POST /api/lab/mutate` | Mutation experiment — bypassed by subprocess approach |
| `POST /api/lab/predict` | Lab prediction — bypassed |
| `POST /api/lab/experiment` | Experiment run — bypassed |
| `GET /api/lab/tradeoffs` | Tradeoff analysis — bypassed |
| `GET /api/lab/prediction-prompt` | Prediction prompt — bypassed |
| `GET /api/lab/mutations` | List mutations — bypassed |

### 3G. Phase 12 — FastAPI Dojo Routes (Bypassed)

The `/dojo` feature uses Next.js subprocess routes instead of the FastAPI dojo implementation.

| FastAPI Route | Notes |
|---|---|
| `GET /api/dojo/exercises` | List dojo exercises — frontend uses static `src/data/problems.ts` |
| `GET /api/dojo/exercises/{exercise_id}` | Exercise detail — frontend uses static data |
| `GET /api/dojo/exercises/{exercise_id}/solution` | Exercise solution — frontend uses static data |
| `POST /api/dojo/submit` | Submit solution (FastAPI version) — frontend uses Next.js subprocess route instead |

---

## 4. Frontend Pages With No Backend Connection

These pages render entirely from static TypeScript data, content loader (MDX), or client state. Zero API calls.

| Page | Route | Data Source |
|---|---|---|
| Landing | `/` | Static JSX |
| Dashboard | `/dashboard` | `localStorage` + static data |
| Learn hub | `/learn` | `src/data/learn/*.ts` direct import |
| Domain page | `/learn/[domain]` | `getDomainData()` from `src/data/domains/` |
| Topic page | `/learn/[domain]/[topic]` | `getTopicData()` from `src/data/topics/` |
| Papers browser | `/papers` (list) | Empty stub — no data at all |
| Paper content | `/papers/[slug]` | `src/lib/content/loader.ts` → `src/content/papers/` MDX |
| Architectures | `/architectures` | Stub + `useState('Transformer')` |
| Architecture content | `/architectures/[slug]` | `src/lib/content/loader.ts` → `src/content/architectures/` MDX |
| Roadmaps | `/roadmaps` | `src/data/roadmaps.ts` direct import |
| Roadmap content | `/roadmaps/[slug]` | `src/lib/content/loader.ts` → `src/content/roadmaps/` MDX |
| Problems list | `/problems` | `src/data/problems.ts` direct import |
| Math | `/math` | `PageSkeleton` stub |
| Math content | `/math/[slug]` | `src/lib/content/loader.ts` → `src/content/math/` MDX |
| Interview | `/interview` | `PageSkeleton` stub |
| Interview content | `/interview/[slug]` | `src/lib/content/loader.ts` → `src/content/interview/` MDX |
| System Design | `/system-design` | Static components |
| System Design content | `/system-design/[slug]` | `src/lib/content/loader.ts` → `src/content/system-design/` MDX |
| Paper-to-Code | `/paper-to-code` | Static components |
| Paper-to-Code content | `/paper-to-code/[slug]` | `src/lib/content/loader.ts` → `src/content/implementations/` MDX |
| Playground | `/playground` | `PageSkeleton` stub |
| Evolution | `/evolution` | Static data |
| Compare | `/compare` | Static data |
| Knowledge Intelligence | `/knowledge-intelligence` | Static data / tabs |
| Explorer | `/explorer` | Static + animated components |
| Model Architecture | `/model-architecture` | Stub components |
| Settings | `/settings` | Static list, no persistence |
| Search | `/search` | Client-side search index over static data |
| Dojo problem list | `/dojo` | `src/data/problems.ts` direct import |
| Collaboration | `/real-time-collaboration` | No WebSocket server exists |
| Versioning | `/advanced-versioning` | Static stub |

---

## 5. Duplicate APIs

### 5A. Duplicate Paper Data Routes (Proxy vs Direct)

The proxy routes in `/api/papers/generated/[id]/*` duplicate endpoints that are either called directly or embedded in other responses. Five routes, all dead.

| Next.js Proxy Route | FastAPI Route It Mirrors | Status |
|---|---|---|
| `GET /api/papers/generated/[id]` | `GET /api/papers/{paper_id}` | Dead — server component calls FastAPI directly |
| `GET /api/papers/generated/[id]/knowledge-graph` | `GET /api/papers/{paper_id}/knowledge-graph` | Dead — KG in main paper payload |
| `GET /api/papers/generated/[id]/blueprint` | `GET /api/papers/{paper_id}/blueprint` | Dead — blueprint in main paper payload |
| `GET /api/papers/generated/[id]/executable-graph` | `GET /api/papers/{paper_id}/executable-graph` | Dead — executable graph in main paper payload |
| `GET /api/papers/generated/[id]/graph-export` | `GET /api/papers/{paper_id}/graph-export` | Dead — export is client-side |

### 5B. Duplicate Learn Data Routes (API vs Direct Import)

The Learn API routes duplicate TypeScript data that pages import directly. Five routes, all dead.

| Next.js API Route | Data It Wraps | Who Imports It Directly |
|---|---|---|
| `GET /api/learn/domains` | `src/data/learn/domains.ts` | `/learn` page |
| `GET /api/learn/paths` | `src/data/learn/*.ts` | `/learn` page |
| `GET /api/learn/recommendations` | `src/data/learn/*.ts` | `/learn` page |
| `GET /api/learn/domain/[slug]` | `getDomainData()` | `/learn/[domain]` page |
| `GET /api/learn/topic/[domain]/[topic]` | `getTopicData()` | `/learn/[domain]/[topic]` page |

### 5C. Duplicate Dojo Submission (FastAPI vs Subprocess)

FastAPI has a full dojo implementation. The frontend bypasses it and runs a Python subprocess directly from a Next.js route instead.

| Frontend Route | FastAPI Route | Difference |
|---|---|---|
| `POST /api/dojo/run` (subprocess) | (no FastAPI equivalent) | Run-only route, no FastAPI mirror |
| `POST /api/dojo/submit` (subprocess) | `POST /api/dojo/submit` (FastAPI) | Same purpose, two parallel implementations |

The FastAPI version would enable persistent submission history, leaderboard, and server-side progress tracking. The subprocess version is stateless.

---

## 6. Missing APIs

These are gaps where the frontend needs something that doesn't exist.

| Gap | Frontend Page Affected | What's Needed |
|---|---|---|
| Papers list endpoint | `/papers` | `GET /api/papers` proxy route (FastAPI has it, no Next.js route or frontend call) |
| Learner identity | All progress/learn features | `X-Learner-ID` header generation + session management |
| Mark topic complete | `/learn/[domain]/[topic]` | `POST /api/progress/update` integration + header |
| Mark paper complete / link paper to learn | `/papers/upload/[paperId]` | Link from workspace to domain/topic pages |
| Adaptive recommendations in Learn | `/learn` | Wire `/api/adaptive/recommendations` with learner ID |
| Assessment flow | Any learn/topic page | `GET /api/assessment/challenge` + `POST /api/assessment/validate` |
| Tutor chat | Any learn/topic page | `POST /api/tutor/ask` integration |
| Playground execution | `/playground` | Page is a stub — needs any backend (FastAPI has `POST /api/playground/generate`) |

---

## 7. Summary Statistics

| Category | Count |
|---|---|
| Next.js API routes total | 20 |
| Next.js API routes actively called by frontend | 8 |
| Next.js API routes that are dead code | 12 |
| FastAPI routes total | ~40 |
| FastAPI routes reachable from frontend (direct or via proxy) | 4 |
| FastAPI routes unreachable from any frontend page | ~36 |
| Frontend pages making at least one API call | 5 |
| Frontend pages with zero backend connection | ~35 |
