# Reality Verification Report
**Method:** Direct source code read. No reliance on prior audit documents.  
**Date:** 2026-06-20  
**Files read:** left-rail.tsx, quick-actions.tsx, hero-section.tsx, recommended-steps.tsx, learning-progress.tsx, papers/page.tsx, papers/upload/[paperId]/page.tsx, server.py (full), lib/content/loader.ts, data/domains/index.ts, data/topics/index.ts, data/learn/recommendations.ts, PaperKnowledgeGraph.tsx, StudyAssistant.tsx, api/papers/upload/route.ts, data/problems.ts (grep)

---

## Part 1 — Content Audit

### src/content/ directory — verified with `find`

| Content Type | Directory | Count | Example Slugs |
|---|---|---|---|
| Architectures | `src/content/architectures/` | **31** | ae, alexnet, bert, clip, deeplabv3plus, densenet, diffusion, dino, efficientnet, fcn, gan, googlenet, gpt, gru, inceptionv3, lenet, llama, lstm, moe, resnet, rnn, seq2seq, stable-diffusion, swin, t5, **transformer**, unet, vae, vgg16, vgg19, **vit** |
| Papers | `src/content/papers/` | **19** | alexnet, **attention-is-all-you-need**, batch-normalization, bert, chinchilla, clip, deep-residual-learning, gan, gpt, gpt-2, gpt-3, latent-diffusion-models, llama, palm, segment-anything, stable-diffusion, switch-transformer, vgg, vision-transformer |
| Implementations | `src/content/implementations/` | **9** | attention-is-all-you-need, bert, clip, gan, gpt, llama, resnet, stable-diffusion, vision-transformer |
| System Design | `src/content/system-design/` | **12** | advanced-rag, agentic-rag, basic-rag, chatgpt-system-design, github-copilot, multi-agent, netflix-recommendation, perplexity, recommendation-engine, single-agent, tiktok-recommendation, youtube-recommendation |
| Problems | `src/content/problems/` | **8** | attention-calculation, clip-batch-size, gpt-kv-cache-scaling, llama-rope, matrix-multiplication, moe-routing, stable-diffusion-cfg, vit-patch-size |
| Roadmaps | `src/content/roadmaps/` | **1** | ai-engineer |
| Interview | `src/content/interview/` | **2** | explain-attention, gradient-descent |
| Math | `src/content/math/` | **2** | linear-algebra, softmax |
| **Total** | | **84 slug directories** | |

### Content schema verified (meta.json)

`src/content/papers/attention-is-all-you-need/meta.json` — confirmed valid:
```json
{
  "type": "paper", "slug": "attention-is-all-you-need",
  "title": "Attention Is All You Need",
  "relationships": {
    "architectures": ["transformer","bert","gpt","vit"],
    "problems": ["attention-calculation"],
    "interview": ["explain-attention"]
  }
}
```

`src/content/architectures/transformer/meta.json` — confirmed valid:
```json
{
  "type": "architecture", "slug": "transformer",
  "relationships": {
    "papers": ["attention-is-all-you-need"],
    "problems": ["attention-calculation","matrix-multiplication"]
  }
}
```

### Content loader (src/lib/content/loader.ts)

- Reads `src/content/<type-dir>/<slug>/meta.json` + `content.mdx`
- Zod-validates all metadata at read time
- `getAllSlugs(type)` returns every directory with a `meta.json`
- `getContentItem(type, slug)` returns null if not found (no throw to caller)
- **`src/content/` exists and is populated** — all slug pages that use the content loader CAN render for the 84 known slugs

### What cannot render (despite content existing)

- `/architectures` (index page) — stub `ThreeColumnLayout`, no architecture list rendered
- `/papers` (list page) — stub, not connected to content or backend
- `/math` (index page) — `PageSkeleton` stub
- `/interview` (index page) — `PageSkeleton` stub
- `/roadmaps` (index page) — static `ROADMAPS` data, no content loader connection

---

## Part 2 — Paper Pipeline Verification

### Upload flow (confirmed by reading server.py + route.ts)

```
src/app/api/papers/upload/route.ts
  → POST (proxy) → getBackendUrl('/api/papers/upload')
  → backend/server.py: POST /api/papers/upload
  → ingest_pdf_paper(db, pdf_bytes, source_filename, paper_name)
  → Paper row + PaperModule rows committed to DB
  → Returns: { id, title, ... } (paper record)

Frontend (PaperUploadWorkspace):
  → On success: router.push(`/papers/upload/${result.id}`)
```

### Workspace load flow (confirmed)

```
src/app/papers/upload/[paperId]/page.tsx (Server Component)
  → fetch(getBackendUrl(`/api/papers/${paperId}`), { cache: 'no-store' })
  → Extracts from response:
      ingestion = paper.ingestion ?? paper.architecture_graph.ingestion ?? {}
      knowledgeGraph = ingestion.knowledge_graph ?? { nodes: [], edges: [] }
      architectureBlueprint = ingestion.architecture_blueprint ?? null
      executableGraph = ingestion.executable_graph ?? null
  → Passes all as props to PaperWorkspaceTabs
```

### Backend GET /api/papers/{id} response shape (verified)

```json
{
  "metadata": {
    "id": 1, "title": "...", "full_title": "...",
    "authors": "...", "abstract": "...",
    "architecture_type": "Transformer",
    "status": "Draft",
    "source_filename": "...",
    "figure_count": 5, "equation_count": 12
  },
  "module_summary": [{ "id": 1, "layer_name": "...", "module_type": "...", ... }],
  "architecture_statistics": { "depth": 4, "node_count": 8, "edge_count": 7 },
  "architecture_graph": {
    "nodes": [...], "edges": [...],
    "ingestion": {
      "knowledge_graph": { "nodes": [...], "edges": [...] },
      "architecture_blueprint": { ... },
      "executable_graph": { ... },
      "figures": [...], "equations": [...],
      "raw_text_excerpt": "..."
    }
  },
  "flops": 0, "parameter_count": 0,
  "ingestion": { ... }
}
```

**Frontend/Backend match:** The page code reads `paper.ingestion ?? paper.architecture_graph.ingestion` — this safely handles the double-path. ✅

### CRITICAL FINDING — Progress update schema mismatch

`POST /api/progress/update` backend schema (server.py line 844):
```python
class ProgressUpdateRequest(BaseModel):
    paper_id: int
    module_id: int
    status: str
```

This endpoint tracks **paper module** progress, not topic/lesson progress. It requires `paper_id` (integer FK to papers table) and `module_id` (integer FK to paper_modules table). It **cannot** be used for "mark topic complete" on `/learn/[domain]/[topic]` pages — there is no `topic_slug`, `domain_slug`, or any string-based identifier in the schema.

**Impact on Phase 16G:** Topic completion cannot be wired to this backend endpoint without a schema change. localStorage persistence is the correct Phase 16 approach for topic completion.

### CRITICAL FINDING — GET /api/papers missing created_at

Backend `GET /api/papers` returns per-paper objects (server.py line 339):
```python
results.append({
    "id": p.id, "title": title,
    "architecture_type": arch_type, "module_count": modules_count,
    "parameter_count": params, "flops": flops,
    "status": status, "support_level": support_level
})
```

`created_at` is a column on the `Paper` model but is **not included in the list response**. "Sort by upload date" in the Research Hub requires adding `"created_at": p.created_at.isoformat() if p.created_at else None` to this response.

---

## Part 3 — Learn System Verification

### Domain registry (src/data/domains/index.ts)

```typescript
const DOMAIN_REGISTRY: Record<string, DomainData> = {
  'deep-learning':    deepLearning,    // ✅ fully authored
  'machine-learning': machineLearning, // ✅ fully authored
  'llms':             llms,            // ✅ fully authored
};
// All other slugs → generateFallback(slug)
```

### Fallback domain content (verified)

`generateFallback()` produces generic content for any slug found in `DOMAINS`:
- Topic slugs: `${slug}-core-1`, `${slug}-core-2`, `${slug}-applied-1`, etc.
- These topic slugs DO NOT exist in `src/data/topics/`
- Clicking any topic in a fallback domain → "Topic Not Found" screen

### Topic registry (src/data/topics/index.ts)

```typescript
const TOPIC_REGISTRY = {
  'deep-learning': { attention, 'multi-head-attention': attention },
  'llms':          { attention },
};
```

- **1 unique topic** (`attention.ts`) registered under 3 slugs
- All other topic routes return `null` → Topic Not Found

### Learn system status table

| Component | Authored/Real | Fallback/Fake | Status |
|---|---|---|---|
| DOMAINS list (12 domains) | ❌ | ✅ static array | Static TS import |
| deep-learning domain page | ✅ Rich content | — | Works |
| machine-learning domain page | ✅ Rich content | — | Works |
| llms domain page | ✅ Rich content | — | Works |
| 9 other domain pages (CV, NLP, Math, etc.) | ❌ | ✅ generateFallback | Shows placeholder |
| attention topic | ✅ 45min, 11 sections | — | Works |
| multi-head-attention topic | ✅ (alias) | — | Works (same data) |
| All other topic routes (~200+) | ❌ | ❌ | 404 "Topic Not Found" |
| Domain topic links (e.g., perceptron, backprop) | ❌ | ❌ | All dead links |
| Progress (masteryPercent, streak) | ❌ | ✅ hardcoded | Fake, never changes |
| "Mark Complete" | ❌ | ❌ | `useState` only, resets |
| "Bookmark" | ❌ | ❌ | `useState` only, resets |
| "Save Notes" | ❌ | ❌ | `useState` only, resets |

---

## Part 4 — Navigation Verification

### Left Rail (src/components/layout/left-rail.tsx) — every item

| Section | Label | Actual href | Correct href | Status |
|---|---|---|---|---|
| HOME | Dashboard | `/dashboard` | `/dashboard` | ✅ |
| LEARN | Foundations | `/learn` | `/learn` | ✅ |
| LEARN | Mathematics | `/math` | `/math` | ✅ (but /math is a stub) |
| LEARN | Statistics | `/learn` | `/learn/statistics` | ❌ WRONG |
| LEARN | Machine Learning | `/learn` | `/learn/machine-learning` | ❌ WRONG |
| LEARN | Deep Learning | `/architectures` | `/learn/deep-learning` | ❌ WRONG |
| LEARN | NLP | `/papers` | `/learn/nlp` | ❌ WRONG |
| LEARN | LLMs | `/system-design` | `/learn/llms` | ❌ WRONG |
| LEARN | Computer Vision | `/architectures` | `/learn/computer-vision` | ❌ WRONG |
| EXPLORE | Architecture Explorer | `/architectures` | `/architectures` | ✅ (but stub) |
| EXPLORE | System Design | `/system-design` | `/system-design` | ✅ |
| BUILD | Playground | `/playground` | `/playground` | ✅ (but stub) |
| BUILD | Projects | `/paper-to-code` | `/paper-to-code` | ✅ |
| PRACTICE | Coding Problems | `/problems` | `/problems` | ✅ |
| PRACTICE | Quizzes | `/dojo` | `/dojo` | ✅ |
| PRACTICE | Assessments | `/dojo` | Same as Quizzes | DUPE |
| RESEARCH | Research Lab | `/papers` | `/papers` | ✅ (but /papers is stub) |
| RESEARCH | Paper Implementations | `/paper-to-code` | `/paper-to-code` | ✅ |
| RESEARCH | Reproducibility | `/labs` | `/labs` | ✅ |
| CAREER | Roadmaps | `/roadmaps` | `/roadmaps` | ✅ |
| CAREER | Interview Hub | `/interview` | `/interview` | ✅ (stub) |
| ANALYTICS | Progress | `/dashboard` | `/dashboard` | ✅ |
| ANALYTICS | Achievements | `/dashboard` | Same as Progress | DUPE |
| — | Upload Paper | NOT PRESENT | `/papers/upload` | ❌ MISSING |

**6 broken hrefs. 2 duplicates. 1 critical missing item (Upload Paper).**

### Dashboard Quick Actions (src/components/dashboard/quick-actions.tsx) — every item

| Label | Subtitle | Destination | Status |
|---|---|---|---|
| Continue Learning | "Transformers track" | `/learn` | ✅ (hub not topic) |
| Solve a Problem | "Daily challenge" | `/problems` | ✅ |
| Explore Arch | "ResNet deep-dive" | `/architectures` | ✅ (stub) |
| Read a Paper | "Attention is All You Need" | `/papers` | ❌ /papers is blank stub |
| Quick Lab | "Forward pass lab" | `/labs` | ✅ |
| Take a Quiz | "Backprop basics" | `/dojo` | ✅ |
| Research Lab | "Reproduce a result" | `/paper-to-code` | ✅ |
| Roadmap | "ML Engineer path" | `/roadmaps` | ✅ |
| Upload Paper | — | NOT PRESENT | ❌ MISSING |
| My Papers | — | NOT PRESENT | ❌ MISSING |

**1 broken ("Read a Paper" → blank stub). 2 missing items.**

### Dashboard Recommended Steps (src/components/dashboard/recommended-steps.tsx)

| Title | Destination | Status |
|---|---|---|
| Complete Multi-Head Attention | `/learn` | ❌ should be `/learn/deep-learning/attention` |
| Read "Attention is All You Need" | `/papers` | ❌ should be `/papers/attention-is-all-you-need` |
| Implement Transformer in PyTorch | `/problems` | ✅ (acceptable) |
| Practice System Design: ML Serving | `/system-design` | ✅ |

### Dashboard Learning Progress (src/components/dashboard/learning-progress.tsx)

| Track | Progress | href | Status |
|---|---|---|---|
| Transformers | 34% (hardcoded) | `/learn` | ❌ should be `/learn/deep-learning` |
| Reinforcement Learning | 12% (hardcoded) | `/learn` | ❌ should be `/learn/reinforcement-learning` |
| Computer Vision | 60% (hardcoded) | `/architectures` | ❌ should be `/learn/computer-vision` |
| System Design | 22% (hardcoded) | `/system-design` | ✅ |

### Landing Page Hero (src/components/landing/hero-section.tsx)

| CTA | Destination | Status |
|---|---|---|
| "Start Learning" | `/dashboard` | ✅ |
| "Explore Platform" | `/learn` | ✅ |
| "Upload a Paper" | NOT PRESENT | ❌ MISSING |
| Feature pills (8 items) | Decorative marquee, NO hrefs | ❌ None clickable |

Stats row: "100+ Architectures", "500+ Practice Problems", "200+ System Designs", "1000+ Interview Questions" — all inflated fabricated numbers.

### RECENTLY_ADDED dead links (src/data/learn/recommendations.ts)

| Item | href | Actually exists? |
|---|---|---|
| DeepSeek-R1 | `/papers/deepseek-r1` | ❌ Not in src/content/papers/ |
| Sparse MoE lesson | `/learn/llms/sparse-moe` | ❌ No such topic |
| Mamba-2 Architecture | `/architectures/mamba2` | ❌ Not in src/content/architectures/ |
| Build a ReAct Agent | `/paper-to-code/react-agent` | ❌ Not in src/content/implementations/ |
| RAG Evaluation with RAGAS | `/learn/rag-systems/ragas-evaluation` | ❌ No such topic |

**All 5 RECENTLY_ADDED items are dead links.**

### RECOMMENDATIONS dead links (same file)

| Item | href | Actually exists? |
|---|---|---|
| FlashAttention | `/papers/flash-attention` | ❌ Not in src/content/papers/ |
| RoPE Embeddings | `/learn/llms/rope-embeddings` | ❌ No such topic |
| KV Cache Optimization | `/architectures/kv-cache` | ❌ Not in src/content/architectures/ |
| Build GPT-2 from Scratch | `/paper-to-code/gpt2` | ❌ Not in src/content/implementations/ |

**All 4 RECOMMENDATIONS are dead links.**

---

## Part 5 — Backend Utilization

### FastAPI routes: categorized

| Route | Method | Used By Frontend | Category |
|---|---|---|---|
| `/api/papers/upload` | POST | ✅ via proxy route.ts | USED |
| `/api/papers/{id}` | GET | ✅ direct from server component | USED |
| `/api/papers/{id}/knowledge-graph` | GET | ✅ via proxy | USED |
| `/api/papers/{id}/blueprint` | GET | ✅ via proxy | USED |
| `/api/papers/{id}/executable-graph` | GET | ✅ via proxy | USED |
| `/api/papers/{id}/graph-export` | GET | ✅ via proxy | USED |
| `/api/health/db` | GET | ❌ | UNUSED |
| `/api/parse_pdf` | POST | ❌ legacy | DEAD |
| `/api/parse_text` | POST | ❌ legacy | DEAD |
| `/api/compare_text` | POST | ❌ legacy | DEAD |
| `/api/analyze_graph` | POST | ❌ legacy | DEAD |
| `/api/playground/generate` | POST | ❌ | UNUSED |
| `/api/papers` | GET | ❌ No proxy route | **BLOCKED — proxy missing** |
| `/api/papers/{id}/modules` | GET | ❌ | UNUSED |
| `/api/modules/{id}` | GET | ❌ | UNUSED |
| `/api/papers/{id}/publish` | POST | ❌ | UNUSED |
| `/api/tutor/ask` | POST | ❌ | UNUSED |
| `/api/tutor/quiz` | POST | ❌ | UNUSED |
| `/api/tutor/learning-path` | GET | ❌ | UNUSED |
| `/api/adaptive/recommendations` | GET | ❌ | UNUSED |
| `/api/adaptive/review-plan` | GET | ❌ | UNUSED |
| `/api/adaptive/concept-graph` | GET | ❌ | UNUSED |
| `/api/assessment/challenge` | GET | ❌ | UNUSED |
| `/api/assessment/validate` | POST | ❌ | UNUSED |
| `/api/progress/update` | POST | ❌ | **BLOCKED — schema mismatch for topics** |
| `/api/analytics/dashboard` | GET | ❌ | UNUSED |
| `/api/analytics/recommendations` | GET | ❌ | UNUSED |
| `/api/implementation/{id}` | GET | ❌ | UNUSED |
| `/api/modules/{id}/implementation` | GET | ❌ | UNUSED |
| `/api/training/{id}` | GET | ❌ | UNUSED |
| `/api/hyperparameters` | GET | ❌ | UNUSED |
| `/api/reproduction/{id}` | GET | ❌ | UNUSED |
| `/api/training-estimator` | POST | ❌ | UNUSED |
| `/api/lab/mutate` | POST | ❌ bypassed | BYPASSED |
| `/api/lab/predict` | POST | ❌ bypassed | BYPASSED |
| `/api/lab/experiment` | POST | ❌ bypassed | BYPASSED |
| `/api/lab/tradeoffs` | GET | ❌ bypassed | BYPASSED |
| `/api/lab/mutations` | GET | ❌ bypassed | BYPASSED |
| `/api/dojo/exercises` | GET | ❌ bypassed | BYPASSED |
| `/api/dojo/exercises/{id}` | GET | ❌ bypassed | BYPASSED |
| `/api/dojo/exercises/{id}/solution` | GET | ❌ bypassed | BYPASSED |
| `/api/dojo/submit` | POST | ❌ bypassed | BYPASSED |

**Summary:**

| Category | Count |
|---|---|
| USED (connected, working) | 6 |
| BLOCKED (implementation needed) | 2 |
| UNUSED (implemented, reachable if called) | 18 |
| BYPASSED (frontend uses subprocess instead) | 9 |
| DEAD (legacy, no frontend equivalent) | 4 |
| **Total FastAPI routes** | **~39** |

---

## Part 6 — User Journey Simulation

### Landing page

- Both CTAs: "Start Learning" → /dashboard, "Explore Platform" → /learn
- No upload entry point
- Feature pills not clickable
- Stats bar shows inflated numbers

### Dashboard

- LearningProgress tracks link to /learn or /architectures, not specific domains
- RecommendedSteps: "Complete Multi-Head Attention" → /learn (not the topic)
- QuickActions: "Read a Paper" → /papers (blank stub)
- No upload CTA of any kind
- All progress numbers hardcoded and static

### Learn hub (/learn)

- 12 domain cards link to /learn/[domain] — 3 render real content, 9 render fallback
- CONTINUE_LEARNING card → /learn/deep-learning/multi-head-attention — EXISTS ✅
- RECENTLY_ADDED: all 5 items → dead links ❌
- RECOMMENDATIONS: all 4 items → dead links ❌

### Domain page (/learn/deep-learning)

- Topic cluster items link to /learn/deep-learning/[slug]
- Only `attention` and `multi-head-attention` exist as topics
- Every other topic slug (perceptron, backprop, batch-norm, etc.) → "Topic Not Found"
- Featured lessons link to same non-existent topic slugs
- Roadmap stage topics are plain text (no href)

### Topic page (/learn/deep-learning/attention) — the ONLY topic

- 11 sections render correctly ✅
- "Mark Complete" → `useState`, resets on navigation ❌
- "Full Practice" → /dojo (params ignored) ❌
- Topic ends → no next topic CTA, no "Practice This" section ❌
- Sidebar progress bar always 0% ❌

### Papers list (/papers)

```
Current code (papers/page.tsx):
  return <ThreeColumnLayout
    left={<PaperSidebar />}    // hardcoded section list
    center={<PaperContent />}  // content for selected section
    right={<PaperNotes />}
  />
```

No backend connection. No paper loaded. No upload CTA. A genuine blank reading interface for a paper that doesn't exist.

### Upload (/papers/upload)

- Form renders, upload works ✅
- No "cancel" / back navigation ❌
- On success → /papers/upload/[id] ✅

### Paper workspace (/papers/upload/[id])

- Fetches and renders correctly ✅
- KG tab: nodes clickable, panel appears with node name + type ✅
- Panel shows "Open →" button ONLY if `node.href` is not null — concept nodes likely have null href
- Panel has NO "Learn This Concept" CTA ❌
- No breadcrumb back to /papers ❌
- No "Practice coding this" CTA ❌
- No "Learn about this concept" CTA ❌
- After all 6 tabs → dead end, no next steps ❌

### Dojo (/dojo/[slug])

- Problem editor works ✅
- Run and submit work ✅
- After solving → no "Read the related paper" recommendation ❌
- `relatedPapers: ["attention-is-all-you-need"]` exists in problems.ts data — just never surfaced ❌

---

## Part 7 — Implementation Priorities

### P0 — Breaks the core journey, fixable in < 1 day

| Fix | File | Why |
|---|---|---|
| Fix 6 wrong LeftRail LEARN hrefs | `src/components/layout/left-rail.tsx` | Deep Learning → /architectures is the single most confusing bug. Users land on wrong page. |
| Add Upload Paper to LeftRail | `src/components/layout/left-rail.tsx` | Upload is the platform's primary feature and is invisible in all navigation |
| Add Upload Paper to Dashboard quick actions | `src/components/dashboard/quick-actions.tsx` | Same — dashboard is the user's second stop |
| Fix RECENTLY_ADDED (all 5 dead links) | `src/data/learn/recommendations.ts` | Every "recently added" item is a dead link — breaks trust immediately |
| Fix RECOMMENDATIONS (all 4 dead links) | `src/data/learn/recommendations.ts` | Same — "Based on your progress" links that all 404 |
| Fix RecommendedSteps hrefs | `src/components/dashboard/recommended-steps.tsx` | "Complete Multi-Head Attention" → /learn loses context; should link directly |
| Replace /papers page | `src/app/papers/page.tsx` | Blank reading interface misleads users; needs to become Research Hub |

### P1 — Closes the core loop, fixable in 2–3 days

| Fix | Files | Why |
|---|---|---|
| Research Hub with real paper list | `papers/page.tsx` (rewrite) + `api/papers/route.ts` (new) + `ResearchHub.tsx` (new) | `GET /api/papers` exists and works — just needs a proxy and UI |
| Add Upload CTA to landing page hero | `hero-section.tsx` | New users see the platform before signing in — primary feature must be visible |
| Add "Practice This Topic" section at end of topic page | `PracticeSection.tsx` (new) + `[topic]/page.tsx` | Topic page ends with no action — loop is broken |
| Add "Learn This Concept" to KG node panel | `PaperKnowledgeGraph.tsx` | Panel already exists, click handlers already exist — just needs concept-to-topic mapping |

### P2 — Improves completeness, fixable in 3–5 days

| Fix | Files | Why |
|---|---|---|
| After-solve recommendations in Dojo | `AfterSolveRecommendations.tsx` (new) + `DojoProblemPage.tsx` | `relatedPapers` already exists in problems.ts — just needs surfacing |
| Persist topic completion in localStorage | `StudyAssistant.tsx` | useState resets; localStorage is correct scope for Phase 16 (backend route is wrong schema) |
| Fix LearningProgress track hrefs in dashboard | `learning-progress.tsx` | CV track → /architectures; should be /learn/computer-vision |
| Add "Back to Papers" link in workspace | `papers/upload/[paperId]/page.tsx` | Stranded users with no escape |

---

## Appendix: Verified File States

| File | Status |
|---|---|
| `src/components/layout/left-rail.tsx` | 6 wrong hrefs, 2 duplicate destinations, missing Upload item |
| `src/components/dashboard/quick-actions.tsx` | 8 actions, no Upload, "Read a Paper" goes to blank stub |
| `src/components/dashboard/recommended-steps.tsx` | All 4 items hardcoded, 2 wrong destinations |
| `src/components/dashboard/learning-progress.tsx` | All 4 tracks hardcoded, 3 wrong hrefs |
| `src/components/landing/hero-section.tsx` | 2 CTAs, no Upload, inflated stats |
| `src/data/learn/recommendations.ts` | 9/9 items are dead links (RECENTLY_ADDED + RECOMMENDATIONS) |
| `src/app/papers/page.tsx` | Blank stub — PaperSidebar/PaperContent/PaperNotes with no data |
| `src/app/papers/upload/[paperId]/page.tsx` | Works correctly, no back-navigation |
| `src/components/paper-upload/PaperKnowledgeGraph.tsx` | Click panel exists, no Learn CTA |
| `src/components/topic/StudyAssistant.tsx` | Pure useState, zero persistence |
| `src/data/domains/index.ts` | 3 authored, 9 fallback (generateFallback) |
| `src/data/topics/index.ts` | 1 topic (attention), 3 aliases |
| `backend/server.py` | GET /api/papers missing created_at; POST /api/progress/update wrong schema for topics |
| `src/content/` | EXISTS: 31 arch, 19 papers, 9 impl, 12 sysdesign, 8 problems, 1 roadmap, 2 interview, 2 math |
