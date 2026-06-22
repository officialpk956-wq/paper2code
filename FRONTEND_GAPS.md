# Frontend Gaps Analysis

**Generated from:** Full frontend architecture audit (2026-06-20)  
**See also:** `FRONTEND_REALITY_AUDIT.md` for the full page-by-page table, `BACKEND_REALITY_AUDIT.md` for backend gaps.

---

## 1. Missing Pages (Routes That Should Exist But Don't)

### 1A. Backend Has APIs — Frontend Never Calls Them

These FastAPI routes are fully implemented and tested but no frontend page uses them:

| Missing Page | FastAPI Route | What It Should Do |
|-------------|--------------|-------------------|
| Assessment/Quiz page | `POST /api/assess`, `GET /api/assessment/results` | Phase 8 adaptive quiz per paper |
| Adaptive learning feed | `POST /api/adaptive/recommend`, `GET /api/adaptive/path` | Phase 9 personalized content order |
| Tutor chat interface | `POST /api/tutor/message`, `GET /api/tutor/sessions` | Phase 9 AI tutor |
| Analytics dashboard | `GET /api/analytics/summary`, `GET /api/analytics/progress` | Phase 9 learning analytics |
| Module detail page | `GET /api/modules/[id]`, `GET /api/modules/[paperId]/list` | Phase 10 per-paper modules |
| Lab experiment history | `POST /api/experiments/start`, `GET /api/experiments/list` | Phase 11 experiment tracking |
| Dojo stats / leaderboard | `GET /api/dojo/stats`, `GET /api/dojo/leaderboard` | Phase 12 dojo backend (bypassed by frontend) |
| Learner progress page | `GET /api/learner/[id]/progress`, `/dashboard`, `/streaks` | No user-facing learner profile |

### 1B. Content Routes Are Populated

`src/content/` exists and is well-populated. Individual content `[slug]` pages should work for known slugs:

| Content Type | Directory | Count |
|---|---|---|
| Architectures | `src/content/architectures/` | 27 (ae, alexnet, bert, clip, densenet, diffusion, dino, efficientnet, fcn, gan, gpt, gru, googlenet, inceptionv3, lenet, llama, lstm, moe, resnet, rnn, seq2seq, stable-diffusion, swin, t5, transformer, unet, vae, vgg16, vgg19, vit) |
| Papers | `src/content/papers/` | 18 (alexnet, attention-is-all-you-need, batch-normalization, bert, chinchilla, clip, deep-residual-learning, gan, gpt, gpt-2, gpt-3, latent-diffusion-models, llama, palm, segment-anything, stable-diffusion, switch-transformer, vgg, vision-transformer) |
| Implementations | `src/content/implementations/` | 9 (attention-is-all-you-need, bert, clip, gan, gpt, llama, resnet, stable-diffusion, vision-transformer) |
| System Design | `src/content/system-design/` | 12 (advanced-rag, agentic-rag, basic-rag, chatgpt-system-design, github-copilot, multi-agent, netflix-recommendation, perplexity, recommendation-engine, single-agent, tiktok-recommendation, youtube-recommendation) |
| Problems | `src/content/problems/` | 8 (attention-calculation, clip-batch-size, gpt-kv-cache-scaling, llama-rope, matrix-multiplication, moe-routing, stable-diffusion-cfg, vit-patch-size) |
| Roadmaps | `src/content/roadmaps/` | 1 (ai-engineer) |
| Interview | `src/content/interview/` | 2 (explain-attention, gradient-descent) |
| Math | `src/content/math/` | 2 (linear-algebra, softmax) |

**Remaining gap:** the browser/index pages (`/papers`, `/architectures`, `/math`, `/interview`, `/roadmaps`) do not list these entries — they are stubs that need to call the content loader to enumerate available slugs.

---

## 2. Placeholder / Stub Pages (Registered Routes With No Content)

These pages render without error but deliver zero value to the user:

| Route | What's There | What's Missing |
|-------|-------------|----------------|
| `/playground` | `PageSkeleton` only | Entire feature — no editor, no execution, no models |
| `/math` | `PageSkeleton` + heading | Math topic browser — all content in `src/content/math/` missing |
| `/interview` | `PageSkeleton` + heading | Interview question browser — all content in `src/content/interview/` missing |
| `/papers` | Empty `ThreeColumnLayout` | Papers list, search, filter, sort — no backend connection, no data |
| `/architectures` | `ThreeColumnLayout` with `useState('Transformer')` | No architecture selector working, no sidebar items, no canvas content |
| `/model-architecture` | Three stub components | Entire neural network layer visualizer |
| `/settings` | Static list of 5 section buttons | No forms, no persistence, no actual settings — "coming soon" note in UI |

---

## 3. Missing API Integration (Pages Exist, Backend Available, But Frontend Never Connects)

### 3A. Learn Section — Fully Static When Backend Has Data

The entire `/learn` section reads from static TypeScript files. The backend has `X-Learner-ID`-based progress tracking, but it's never used:

| Page | Static Source | Backend Available | Gap |
|------|--------------|-------------------|-----|
| `/learn` | `src/data/learn/*.ts` | `/api/learner/[id]/progress` | Progress bars are hardcoded `0` |
| `/learn/[domain]` | `src/data/domains/[domain].ts` | Adaptive recommendations | Domain progress not fetched |
| `/learn/[domain]/[topic]` | `src/data/topics/attention.ts` | Learner progress, assessment | Mark-complete missing |
| `/problems` | `src/data/problems.ts` | Dojo stats, submissions | Filter shows all as "unsolved" |
| `/roadmaps` | `src/data/roadmaps.ts` | Learner progress | Progress % is hardcoded, not real |
| `/dashboard` | localStorage only | `/api/learner/[id]/dashboard` | Never queries backend learner API |

### 3B. Dojo — Frontend Bypasses FastAPI Entirely

The backend has a full Phase 12 dojo implementation (`/api/dojo/problems`, `/api/dojo/submit`, `/api/dojo/leaderboard`). The frontend instead uses Next.js API routes that run Python subprocesses directly — bypassing FastAPI.

Result: no persistent submission history, no leaderboard, no server-side progress tracking.

### 3C. AI Labs — Only 1 of 4 Labs Wired

`/api/labs/transformer` is functional. Three labs have no API route:
- `cnn` → no `/api/labs/cnn` endpoint
- `vit` → no `/api/labs/vit` endpoint  
- `diffusion` → no `/api/labs/diffusion` endpoint

The UI shows all 4 labs, but only the transformer lab executes real Python.

---

## 4. Missing User Flows

### 4A. Upload → Learn → Practice Flow (Broken)

The intended flow (upload paper → extract concepts → learn via domain/topic → practice in dojo) is disconnected:

1. **Upload** → creates a paper in DB with KG + blueprint ✅
2. **Learn from paper** → no link from workspace to any `/learn/` page ❌
3. **Topic mastery** → only "attention" topic exists; no topic created from uploaded paper ❌
4. **Practice** → dojo problems are static, not generated from uploaded paper ❌

### 4B. Authentication / Identity (Missing Entirely)

- `X-Learner-ID` header is required by all learner routes but frontend never sends it
- No login, registration, or session page exists
- No identity set up in any API call
- `users` table and `UserService` fully implemented but dead — zero routes, zero frontend

### 4C. Progress Tracking (Missing)

- No "mark complete" action on any topic or module
- Progress shown in roadmaps is hardcoded
- Dashboard stat counts come from `localStorage` (dojo submissions only)
- No server-side streak or XP tracking visible to user

### 4D. Paper → Workspace Navigation (Partially Broken)

- After upload, user is redirected to `/papers/upload/[paperId]` ✅
- No way to list previously uploaded papers — `/papers` page is empty stub ❌
- No link from dashboard "Recent Papers" to any real paper ❌

---

## 5. Broken Navigation

| Link | Where It Appears | Target | Problem |
|------|-----------------|--------|---------|
| `/papers/[slug]` | Evolution page, compare page, paper detail links | Paper content page | Works for known slugs (18 papers in `src/content/papers/`); 404 for unknown slugs |
| `/architectures/[slug]` | Architectures page, search results | Architecture content | Works for known slugs (27 architectures in `src/content/architectures/`); index page is still a stub |
| `/roadmaps/[slug]` | Roadmaps page cards | Roadmap content | Works for `ai-engineer` (only 1 roadmap exists); others would 404 |
| `/math/[slug]` | Learn topic math section, search | Math content | Works for `linear-algebra` and `softmax`; index page is a stub |
| `/interview/[slug]` | Dojo, learn section | Interview Q content | Works for `explain-attention` and `gradient-descent`; index page is a stub |
| `/paper-to-code/[slug]` | Implementation map | Implementation | Works for 9 known slugs in `src/content/implementations/` |
| `/problems/[slug]` | Problems list | Problem detail | Works for 8 known slugs in `src/content/problems/` |
| Dashboard "Start →" button | Dashboard quick actions | `/problems/multi-head-attention` | No content file for `multi-head-attention` — would 404 |
| Collaboration features | `/real-time-collaboration`, `/advanced-versioning` | WebSocket/Yjs server | No WebSocket server exists |

---

## 6. Duplicate UI Patterns

### 6A. Three Column Layout Overuse

`ThreeColumnLayout` is used on 10+ pages, many with stub left/right panels:
- `/architectures` — left panel is empty list, right panel is empty inspector
- `/model-architecture` — all three panels are stub components
- `/system-design` — functional, but left/right component depth unknown
- `/paper-to-code` — functional, but depends on unverified components
- `/papers` — left is empty, center is empty, right is empty

### 6B. Two Separate Dojo Entry Points

- `/dojo` — DS Coding Dojo (110 data science problems, Phase 12)
- `/problems` — Problems list (same PROBLEMS data + LEARNING_TRACKS)

Both render the same `PROBLEMS` dataset from `src/data/problems.ts`. No clear user-facing distinction. `/dojo/[slug]` has the actual editor; `/problems` has no editor link per problem.

### 6C. Duplicate Learning Paths

- `/learn` — shows LEARNING_PATHS
- `/roadmaps` — shows ROADMAPS
- `/problems` — shows LEARNING_TRACKS

All three describe "learning paths" using different data structures, different UI patterns, and no cross-linking.

### 6D. Duplicate Architecture Evolution Views

- `/evolution` — research journeys, family trees, architecture replacements
- `/knowledge-intelligence` → Evolution tab — `ArchitectureEvolution` component by family
- `/compare` — side-by-side architecture comparison

All three explore "how architectures relate to each other" with no unified model.

---

## 7. Security Issues (Frontend)

| Issue | Severity | Location |
|-------|----------|---------|
| Unsandboxed Python execution via `exec()` | Critical | `/api/dojo/run`, `/api/dojo/submit` — user code runs directly on server |
| No authentication on any route | High | All pages accessible with no identity |
| No rate limiting on dojo execution | High | Dojo run/submit endpoints have only a timeout guard |
| `dangerouslySetInnerHTML` in search results | Medium | `/search/page.tsx` — mitigated by HTML-escape in `highlight()` but comment says "Safe" |

---

## 8. Priority Ranking for Fixes

### P0 — Breaks Core Functionality
1. **Wire `/papers` list page** to backend `/api/papers` — currently empty stub, users can't see uploaded papers
2. **Send `X-Learner-ID` header** in all API calls — no progress data persists without this
3. **Fix Dashboard "Start →" link** — points to `/problems/multi-head-attention` which has no content file

### P1 — Core Feature Gaps
4. **Connect `/learn` section to real learner progress** via backend APIs instead of all-static data
5. **Add remaining lab endpoints** (`/api/labs/cnn`, `vit`, `diffusion`) or hide labs 2-4 until ready
6. **Wire dojo submission to FastAPI** instead of direct subprocess — enables leaderboard + history

### P2 — Missing Major Pages
7. **Build `/settings` page** — currently just a list of buttons with "coming soon" note
8. **Build a real `/papers` list** — users can upload papers but have no way to see them
9. **Add login / identity flow** — `X-Learner-ID` is a UUID in header but no UI sets it

### P3 — Content and Polish
10. **Populate `src/content/` with seed data** for at least: 5 architectures, 5 papers, 3 roadmaps
11. **Link from `/papers/upload/[id]` workspace back to domain/topic** in Learn section
12. **Deduplicate learning path views** — merge `/roadmaps`, `/problems` tracks, and `/learn` paths
