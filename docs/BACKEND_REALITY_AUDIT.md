# Backend Reality Audit — Paper2Code
*Audited: 2026-06-20 | Auditor: senior staff engineer review*
*Audit scope: read-only analysis; no code was modified*

---

## Summary

The backend is substantially complete for its core paper-processing features. The FastAPI server has ~35 real endpoints, all key pipelines are implemented without LLMs (deterministic/template-based), and the paper upload → ingest → knowledge graph → blueprint → executable graph chain works end-to-end. However, several components are dead code, there are concrete security risks in the Python execution path, and the entire Learn section (6 routes) serves static TypeScript data with no personalization.

---

## Module Status Table

| Module | Implemented | Tested | Used By Frontend |
|---|---|---|---|
| **FastAPI server** (`backend/server.py`) | ✅ ~35 routes | Partial (smoke + integration) | Yes — via Next.js proxy |
| **Database** (`backend/database.py`) | ✅ SQLite/Postgres, `create_all` | ✅ | Indirectly via routes |
| **ORM models** (`backend/models.py`) | ✅ 6 tables | ✅ | Indirectly |
| **User model + table** | ✅ schema only | ✅ unit | ❌ No API routes expose users |
| **UserService** | ✅ implemented | ✅ | ❌ Dead code — never called |
| **UserRepository** | ✅ implemented | ✅ | ❌ Dead code — never called |
| **Paper pipeline** (`paper_ingestion_service.py`) | ✅ full chain | ✅ `test_paper_ingestion_service.py` | Yes — upload route |
| **PDF text extraction** | ✅ pdfplumber + PyMuPDF fallback | ✅ | Yes |
| **Section extraction** | ✅ regex, 12 canonical headings | ✅ | Yes |
| **Figure extraction** | ✅ metadata only (xref, dims, caption) | ✅ | Metadata only — no image binary served |
| **Equation extraction** | ✅ regex-based | ✅ | Yes |
| **Knowledge graph pipeline** (`knowledge_extraction_service.py`) | ✅ deterministic | ✅ `test_knowledge_extraction.py` | Yes — knowledge-graph route |
| **Architecture blueprint** (`architecture_reconstruction_service.py`) | ✅ 8 templates + generic | ✅ `test_architecture_reconstruction.py` | Yes — blueprint route |
| **Executable graph compiler** (`architecture_graph_compiler.py`) | ✅ shape-inferred, validated | ✅ `test_architecture_graph_compiler.py` | Yes — executable-graph + graph-export routes |
| **Block viz service** (`block_viz_service.py`) | ✅ real PyTorch hooks | ✅ `test_block_viz_service.py` | Yes — subprocess via Next.js |
| **Lab service** (`lab_service.py`) | ✅ real PyTorch forward pass | ✅ `test_lab_service.py` | Yes — subprocess via Next.js |
| **Tutor agent** (`core/agents/tutor_agent.py`) | ✅ | ✅ | Yes — tutor routes |
| **Adaptive engine** (`core/analytics/adaptive_engine.py`) | ✅ | ✅ `test_phase9_adaptive.py` | Yes — adaptive routes |
| **Assessment engine** (`core/assessment/engine.py`) | ✅ | ✅ `test_phase8_assessment.py` | Yes — assessment routes |
| **Recommendation engine** (`core/analytics/recommendation_engine.py`) | ✅ | ✅ | Yes — analytics dashboard |
| **Dojo exercises** (`core/dojo/`) | ✅ catalog + solutions | ✅ `test_dojo.py`, `test_dojo_service.py` | Yes — dojo routes |
| **Dojo code execution** (Next.js `api/dojo/run` + `api/dojo/submit`) | ✅ | ✅ | Yes — direct subprocess, no backend |
| **Implementation views** (`core/implementation/`) | ✅ | ✅ `test_phase10_impl.py` | Yes — implementation routes |
| **Lab experiments** (`core/lab/`) | ✅ | ✅ `test_phase11_lab.py` | Yes — lab routes |
| **`paper_document.py`** dataclasses | ✅ defined | None | ❌ Dead code — not used by ingestion service |
| **Learn domains API** | Static TS data only | ✅ frontend tests | Frontend — returns hardcoded `DOMAINS` array |
| **Learn paths API** | Static TS data only | ✅ frontend tests | Frontend — returns hardcoded `LEARNING_PATHS` |
| **Learn recommendations API** | Static TS data only | ✅ frontend tests | Frontend — returns hardcoded arrays |
| **Learn domain detail API** | Static TS data (`src/data/domains/`) | ✅ frontend tests | Frontend — 3 domains have data files |
| **Learn topic API** | Static TS data (`src/data/topics/`) | ✅ frontend tests | Frontend — only `attention` topic exists |

---

## 1. FastAPI Routes

### 1.1 Route Inventory

All routes are in `backend/server.py`. There is no router module splitting.

**Health**
```
GET  /api/health/db                     — DB ping (SELECT 1)
```

**Legacy Analysis (no DB persistence)**
```
POST /api/parse_pdf                     — PDF → pipeline → graph+code, not stored
POST /api/parse_text                    — text → pipeline → graph+code, not stored
POST /api/compare_text                  — compare two architecture texts
POST /api/analyze_graph                 — layers JSON → pipeline run
POST /api/playground/generate          — ResNet/Transformer/U-Net parametric compare
```

**Paper CRUD (DB-backed)**
```
GET  /api/papers                        — list all papers + aggregate stats
GET  /api/papers/{paper_id}             — full paper detail + modules summary
POST /api/papers/upload                 — ingest PDF → Paper + PaperModule rows
POST /api/papers/{paper_id}/publish     — set status = "Published"
GET  /api/papers/{paper_id}/modules     — all modules for a paper (full detail)
GET  /api/modules/{module_id}           — single module + prev/next nav
```

**Generated Artifact Retrieval**
```
GET  /api/papers/{paper_id}/knowledge-graph    — KG from architecture_graph JSON
GET  /api/papers/{paper_id}/blueprint          — ArchitectureBlueprint from JSON
GET  /api/papers/{paper_id}/executable-graph   — ExecutableGraph from JSON
GET  /api/papers/{paper_id}/graph-export       — export as json|mermaid|dot (query param)
```

**Phase 10 — Research Engineer**
```
GET  /api/implementation/{paper_id}     — per-module PyTorch mappings
GET  /api/modules/{module_id}/implementation — single module impl view
GET  /api/training/{paper_id}           — training pipeline config
GET  /api/hyperparameters               — all hyperparameter explanations
GET  /api/reproduction/{paper_id}       — reproduction card
POST /api/training-estimator            — cost estimation
```

**Phase 8/9 — Assessment & Adaptive**
```
GET  /api/assessment/challenge          — generate challenge (tensor/arch/flops/comparison)
POST /api/assessment/validate           — validate + persist AssessmentAttempt
POST /api/progress/update               — upsert LearnerProgress row
GET  /api/analytics/dashboard           — full learner analytics aggregation
GET  /api/analytics/recommendations     — recommendation engine
GET  /api/adaptive/recommendations      — personalized recommendations
GET  /api/adaptive/review-plan          — daily review plan
GET  /api/adaptive/concept-graph        — concept graph for learner
```

**Tutor**
```
POST /api/tutor/ask                     — ask question, persist TutorAnalytics
POST /api/tutor/quiz                    — generate quiz with weakness targeting
GET  /api/tutor/learning-path           — adaptive learning path
```

**Phase 11 — Research Lab**
```
POST /api/lab/mutate                    — apply mutations, compute diff
POST /api/lab/predict                   — score hypothesis prediction
POST /api/lab/experiment                — full experiment result
GET  /api/lab/tradeoffs                 — scatter plot data for tradeoff explorer
GET  /api/lab/prediction-prompt         — prediction challenge prompt
GET  /api/lab/mutations                 — list all mutation types
```

**Phase 12 — Dojo**
```
GET  /api/dojo/exercises                — exercise catalog (no solutions)
GET  /api/dojo/exercises/{id}           — full exercise (test cases, no solution)
GET  /api/dojo/exercises/{id}/solution  — reveal reference solution
POST /api/dojo/submit                   — record AssessmentAttempt(type='code')
```

**Static**
```
GET  /                                  → static/index.html
```

### 1.2 Missing FastAPI Routes (no `/api/users/*` anywhere)

The `User` model, `UserService`, and `UserRepository` are fully implemented but there are **zero FastAPI endpoints that expose users**. No registration, no login, no leaderboard API, no point-awarding endpoint. User data is siloed.

---

## 2. Database

### 2.1 Tables

| Table | Model | Schema | Notes |
|---|---|---|---|
| `users` | `User` | id, email(unique), name, avatar_url, points, streak, last_active, created_at | **Never queried by any FastAPI route** |
| `papers` | `Paper` | id, title(unique), authors, abstract, architecture_graph(JSON), flops_analysis(JSON), created_at | Core table. `architecture_graph` carries all pipeline outputs nested under `ingestion.*` |
| `paper_modules` | `PaperModule` | id, paper_id(FK), layer_name, module_type, explanation, tensor_flow(JSON), graph_nodes(JSON), flops_context(JSON), order_index | One-to-many with Paper |
| `learner_progress` | `LearnerProgress` | id, learner_id, paper_id(FK), module_id(FK), status, started_at, completed_at, time_spent_seconds | Upserted by `/api/progress/update` |
| `assessment_attempts` | `AssessmentAttempt` | id, learner_id, assessment_type, architecture, difficulty, question_text, user_answer, correct_answer, score, attempt_count, is_correct, created_at | Written by assessment validate + dojo submit |
| `tutor_analytics` | `TutorAnalytics` | id, learner_id, architecture, module, reasoning_type, question_count, created_at | Written by `/api/tutor/ask` |

### 2.2 Migration Strategy

- **No Alembic** — schema is created by `Base.metadata.create_all(bind=engine)` on startup.
- **Dev fallback**: `sqlite:///./tensortonic_dev.db` when `DATABASE_URL` is not set.
- **Production gap**: Deploying to Postgres with a fresh DB works; adding new columns to existing DB requires manual migration.

### 2.3 Learner Identity

All learner routes read `X-Learner-ID` header with `default=""`. There is **no authentication**. Any client can read or write another learner's data by forging the header. The empty-string default means all unauthenticated requests share a single `""` learner bucket.

---

## 3. Paper Pipeline

### 3.1 Upload Flow (`POST /api/papers/upload`)

```
PDF bytes
  → extract_pdf_pages()          pdfplumber, PyMuPDF fallback, max 30 pages
  → extract_raw_text()           join page texts, raises if blank
  → extract_figures()            PyMuPDF image xrefs: metadata only (no binary)
  → extract_equations()          regex [$...$, =, ±, ∑, ∫ …], cap 80
  → extract_sections()           regex 12 canonical headings, content[:4000]
  → PaperToCodeGenerator.from_pdf()   existing pipeline (graph + code)
  → classify_architecture()      classify graph
  → generate_modules()           per-module explanation + tensor flow
  → build_knowledge_graph()      deterministic entity extraction
  → reconstruct_architecture()   template-based blueprint + FLOPs
  → compile_blueprint()          ExecutableGraph + validation
  → Paper() + PaperModule() × N  committed to DB
```

### 3.2 Persistence Layout

All pipeline outputs are stored **inside one JSON column** (`Paper.architecture_graph`) to avoid schema migrations:

```json
{
  "classification": "Transformer",
  "status": "Draft",
  "ingestion": {
    "source_filename": "...",
    "sections": [...],
    "figures": [...],         // metadata only, no binary
    "equations": [...],
    "knowledge_graph": { "nodes": [...], "edges": [...] },
    "architecture_blueprint": { ... },
    "executable_graph": { ... }
  }
}
```

**Consequence**: the JSON column grows large for complex papers. There is no size limit enforced.

### 3.3 Limitations

| Area | Limitation |
|---|---|
| Scanned PDFs | Only text-layer PDFs work; OCR is not implemented |
| Figure binary | Images are not stored or served — only width/height/caption metadata |
| Equation parsing | Regex-only; misses LaTeX display math in many PDFs |
| Section detection | Limited to 12 canonical names; custom headings become uncategorized |
| Text cap | Only first 30 pages processed; `raw_text_excerpt` capped at 4000 chars in DB |

---

## 4. Knowledge Graph Pipeline

**File**: `backend/services/knowledge_extraction_service.py`

- **Method**: fully deterministic regex matching — no LLM, no NLP library
- **Entity types**: architectures (33 patterns), concepts (20), datasets (17), metrics (16)
- **Edge types**: `introduces`, `uses`, `evaluates_on`, `reports`, `derives_from`
- **Architecture lineage**: static hard-coded map (e.g., BERT derives_from Transformer)
- **Equation nodes**: first 10 equations, truncated to 40 chars

**Strengths**: fast, reproducible, no API calls.

**Weaknesses**:
- Pattern sensitivity — "VGG-16" matches "vgg" but "VGGNet" may not
- No deduplication across papers (e.g., ResNet appears independently in every paper that mentions it)
- Equation nodes are raw text fragments, not parsed LaTeX

**Storage**: `paper.architecture_graph.ingestion.knowledge_graph`
**Frontend access**: `GET /api/papers/generated/[id]/knowledge-graph` → proxies to backend ✅

---

## 5. Architecture Blueprint Pipeline

**File**: `backend/services/architecture_reconstruction_service.py`

- **Method**: template-based (8 templates + generic fallback)
- **Template dispatch**: via knowledge graph's primary architecture detection
- **Signal extraction**: regex extracts depth, num_heads, d_model, ffn_dim, patch_size, image_size, channels, num_classes, vocab_size, seq_len from paper text
- **Templates**: ResNet-family, ViT-family, U-Net/FCN, Transformer-family (BERT/GPT/T5/LLaMA), GAN, VAE, Diffusion (DDPM/Stable/Latent), LSTM/GRU/RNN
- **FLOPs simulation**: uses `FLOPsEngine` on tensor flow steps
- **Confidence score**: 0.0–1.0 based on introduces/uses edges (0.35/0.15), method section present (+0.20), signal count (+0.05 each, max 0.25), equation count (+0.05/0.10)

**Confidence interpretation**:
- Papers the system "introduces" a known architecture: max ~0.90
- Papers that only "use" architectures: max ~0.70
- Unknown architectures → generic template, confidence ≤ 0.30

**Storage**: `paper.architecture_graph.ingestion.architecture_blueprint`
**Frontend access**: `GET /api/papers/generated/[id]/blueprint` → proxies to backend ✅

---

## 6. Executable Graph Pipeline

**File**: `backend/services/architecture_graph_compiler.py`

- **Input**: `ArchitectureBlueprint` dict from Phase 14A
- **Shape inference**: `TensorTracker.propagate_shapes()` (existing system)
- **Validation checks**: cycle detection (DFS), dangling edges, disconnected nodes, missing input/output nodes, sequential shape mismatches
- **Validation states**: `valid` | `warnings` | `invalid`
- **Export formats**: JSON, Mermaid flowchart (`flowchart TD`), Graphviz DOT

**Storage**: `paper.architecture_graph.ingestion.executable_graph`
**Frontend access**:
- `GET /api/papers/generated/[id]/executable-graph` → backend proxy ✅
- `GET /api/papers/generated/[id]/graph-export?format=json|mermaid|dot` → backend proxy ✅

---

## 7. Learn APIs — Backend vs Mock Content

All `/api/learn/*` routes serve **static TypeScript data with no backend connection**. No user-specific content, no real-time data.

| Next.js Route | Data Source | Is Dynamic? |
|---|---|---|
| `GET /api/learn/domains` | `src/data/learn/domains.ts` DOMAINS constant | ❌ Static |
| `GET /api/learn/paths` | `src/data/learn/paths.ts` LEARNING_PATHS constant | ❌ Static |
| `GET /api/learn/recommendations` | `src/data/learn/recommendations.ts` 3 constants | ❌ Static |
| `GET /api/learn/domain/[slug]` | `src/data/domains/` TS files (3 domains) | ❌ Static |
| `GET /api/learn/topic/[domain]/[topic]` | `src/data/topics/` TS files | ❌ Static — **only `attention` topic exists** |
| `GET /api/labs` | Hardcoded `LABS_META` array in route file | ❌ Static |

**Static data issues**:
- `DOMAINS` has hardcoded `progress: 72` etc. — no per-user progress tracking in Learn section
- `topicCount: 18` on Mathematics domain but no topic data files exist for it
- Domain slugs referenced in navigation (mathematics, machine-learning, llms) only have `src/data/domains/` files for 3 domains; other slugs 404

---

## 8. Missing Implementations and Dead Code

### 8.1 UserService + UserRepository (dead code)

`backend/services/user_service.py` and `backend/repositories/user_repository.py` are fully implemented with CRUD, point-awarding, streak tracking, and leaderboard. The `users` table is created in DB. **Nothing in `server.py` imports or calls them.** There are no `/api/users/*` routes. Points and streaks are never updated.

### 8.2 Deleted Repository Files

`__pycache__` contains `.pyc` files for `problem_repository.py` and `submission_repository.py`, but the source files are gone. The models.py comment explains: *"We are dropping submissions for now in V1 Pivot."* The cached bytecode is stale artifact.

### 8.3 `paper_document.py` Dataclasses (dead code)

`backend/services/paper_document.py` defines `Section`, `Figure`, `Equation`, `PaperDocumentMetadata` dataclasses. The ingestion service (`paper_ingestion_service.py`) **does not import or use them** — it returns plain dicts throughout. These dataclasses are unused dead code.

### 8.4 Only One Topic Data File

`src/data/topics/` contains `attention.ts` only. The `/learn/[domain]/[topic]` route returns 404 for every topic except `deep-learning/attention`. Domain pages list topics that have no backing data.

### 8.5 Learn Progress is Not Tracked

The `LearnerProgress` table tracks module-level progress in the paper pipeline (module read/completed). The Learn section (domain/topic pages) has no equivalent — `progress: 72` in the DOMAINS array is a hardcoded placeholder.

### 8.6 Dojo Code Execution — Unsandboxed

`POST /api/dojo/run` and `POST /api/dojo/submit` in Next.js write user-supplied Python to a temp file and run `exec('python "..."')` / `execFile('python', [...])`. The only protection is:
- Max code length: 20,000 chars
- Function name validation: `^[a-zA-Z_][a-zA-Z0-9_]{0,99}$`
- Timeout: 8s (run) / 10s (submit)

**There is no sandbox.** User code runs with the same OS privileges as the Next.js server process. It can read files, make network calls, and spawn subprocesses. This is a **high-severity security risk** for any multi-user deployment.

### 8.7 Backend URL Hardcoded

`src/lib/backend.ts` defaults to `http://127.0.0.1:8000`. In production this must be set via `PAPER2CODE_BACKEND_URL` env var; there is no `.env.example` hint for this key.

### 8.8 No Authentication on Learner Routes

All analytics, assessment, progress, and tutor endpoints accept `X-Learner-ID: <string>` with no verification. Any caller can masquerade as any learner ID. The default is `""` which means all anonymous calls share data.

---

## 9. Test Coverage Summary

| Test File | Module Under Test | Approximate Coverage |
|---|---|---|
| `tests/test_paper_ingestion_service.py` | Paper ingestion pipeline | High |
| `tests/test_knowledge_extraction.py` | Knowledge graph | High |
| `tests/test_architecture_reconstruction.py` | Architecture blueprint | High |
| `tests/test_architecture_graph_compiler.py` | Executable graph compiler | High |
| `tests/test_block_viz_service.py` | Block viz (PyTorch hooks) | High |
| `tests/test_lab_service.py` | AI labs (PyTorch forward pass) | High |
| `tests/test_phase8_assessment.py` | Assessment engine | High |
| `tests/test_phase9_adaptive.py` | Adaptive engine | High |
| `tests/test_phase10_impl.py` | Implementation views | High |
| `tests/test_phase11_lab.py` | Lab experiment system | High |
| `tests/test_dojo.py` + `test_dojo_service.py` | Dojo exercises | High |
| `tests/smoke_phase9.py` | Smoke test, Phase 9 integration | Smoke only |
| **No test file** | `UserService` / `UserRepository` | No route tests |
| **No test file** | `paper_document.py` | Untested |
| **No test file** | FastAPI route handlers (integration) | No integration tests against running server |
| `src/__tests__/` (578 tests) | All Next.js components + API routes | ≥70% coverage |

**Python test count**: ~793 at last measure (Phase 14B)
**Frontend test count**: 578 (after Phase 15C)

---

## 10. Issues by Priority

### Critical
1. **Dojo code execution unsandboxed** (`src/app/api/dojo/run/route.ts`, `submit/route.ts`): arbitrary Python runs as the server process. Use `gVisor`, `nsjail`, or a dedicated judge container before any multi-user deployment.

### High
2. **No authentication on learner APIs**: All `X-Learner-ID` routes are unauthenticated. Implement session tokens or JWT before exposing to users.
3. **Only `attention` topic exists**: Every other topic URL 404s. Add topic data files or remove navigation links to non-existent topics.

### Medium
4. **UserService is dead code**: Implement `/api/users/*` routes or delete `user_service.py` / `user_repository.py` to avoid maintenance confusion.
5. **Learn progress not tracked**: Domain `progress` values are hardcoded. Wire `LearnerProgress` to Learn section or add a separate progress store.
6. **No DB migrations**: `create_all` will fail silently on schema drift in existing databases. Add Alembic.
7. **`paper_document.py` is unused**: Delete or integrate the dataclasses into the ingestion service.

### Low
8. **`PAPER2CODE_BACKEND_URL` undocumented**: Add to `.env.example`.
9. **Equation extraction regex-only**: Structured LaTeX parsing would improve coverage for math-heavy papers.
10. **`X-Cache` header is in-process only**: The cache in `block-hierarchy/route.ts` and `forward-pass/route.ts` resets on every Next.js cold start.
11. **`__pycache__` stale .pyc files**: `problem_repository.pyc` / `submission_repository.pyc` should be removed from version control.
