# Product: Paper2Code (TensorTonic)

**Last updated:** 2026-06-11
**Method:** codebase scan + conversation

## Product Identity
- **One-liner:** Learners upload ML/DL research papers, the system extracts architecture graphs, and users learn by exploring layers, solving 110 coding exercises, taking assessments, and running experiments in a virtual research lab — all in the browser.
- **Category:** ai-ml-tool / edtech
- **Product type:** B2C (single-player now, multiplayer/classroom planned)
- **Collaboration:** single-player (multiplayer planned)

## Business Model
- **Monetization:** Free / open-source
- **Pricing tiers:** None — single free tier
- **Billing integration:** None detected

## Tech Stack
- **Primary language:** Python (backend), JavaScript (frontend)
- **Framework:** FastAPI (backend), vanilla JS SPA with hash-based routing (frontend)
- **Database:** SQLite (dev default), PostgreSQL supported via DATABASE_URL
- **Background jobs:** None detected — all processing is synchronous
- **HTTP client patterns:** Browser fetch API (frontend), Python requests (backend LLM calls)
- **Module organization:** Python packages: `core/` (domain logic), `backend/` (API + DB), `static/` (SPA frontend)
- **Key dependencies:** Pyodide (client-side Python execution), Monaco Editor (code editor), KaTeX (math rendering), Chart.js, Cytoscape.js (graph visualization)

## Value Mapping

### Primary Value Action
**Solving a coding exercise in Code Dojo** — the learner writes Python code to implement an ML primitive (ReLU, softmax, attention, etc.), runs tests against it, and gets pass/fail feedback. If exercise completion drops to zero, the product has failed as a learning tool.

### Core Features (directly deliver value)
1. **Code Dojo** — 110 coding exercises across 18 categories (Activation, Loss, Optimizer, Layer, Backprop, NumPy, Statistics, etc.) with client-side Pyodide execution and test validation
2. **Paper Library & Architecture Explorer** — Upload research papers (PDF/text), extract architecture graphs, explore layer-by-layer with FLOPs analysis, tensor flow, and code generation
3. **Research Lab** — Mutate architectures (add layers, change params), predict outcomes, run experiments, analyze tradeoffs
4. **Assessments** — Architecture, tensor, FLOPs, and comparison challenges with difficulty scaling and scoring
5. **Architecture Tutor** — AI chatbot that answers questions about the current architecture context

### Supporting Features (enable core actions)
1. **Dashboard** — Learning analytics: papers started, assessment accuracy, tutor questions asked
2. **Hyperparameters Explorer** — Browse and understand hyperparameter configurations
3. **Cost Estimator (Training Estimator)** — Estimate training costs for architectures
4. **Implementation Guide** — Code mapping and reproduction cards for papers
5. **Playground** — Interactive architecture configuration and visualization

## Entity Model

### Learners (Users)
- **ID format:** Browser-generated UUID stored in `localStorage.learner_id` (anonymous). Database `users` table uses integer auto-increment with email for registered users.
- **Roles:** Learner (single role, no admin distinction in frontend)
- **Multi-account:** No — single anonymous identity per browser

### Papers
- **ID format:** Integer auto-increment
- **Relationship:** A paper has many modules (PaperModule)
- **Created by:** Learner uploads PDF or pastes text

### Paper Modules
- **ID format:** Integer auto-increment
- **Relationship:** Belongs to a Paper, ordered by `order_index`
- **Contains:** Layer explanation, tensor flow, graph nodes, FLOPs context

### Dojo Exercises
- **ID format:** String slug (e.g., `relu`, `sigmoid_backward`, `np_matmul`)
- **Static:** Defined in Python code, served via API — not user-created
- **Categories:** 18 categories, 110 exercises total

### Assessment Attempts
- **ID format:** Integer auto-increment
- **Relationship:** Linked to `learner_id`
- **Contains:** Question, answer, score, assessment type, difficulty

### Learner Progress
- **ID format:** Integer auto-increment
- **Relationship:** Linked to `learner_id`, `paper_id`, `module_id`
- **Contains:** Status (not_started/in_progress/completed), time spent

## Group Hierarchy

No group hierarchy — user-level tracking only. The product is single-player with anonymous browser-based identity. If multiplayer/classroom features are added later, a group hierarchy (Organization → Classroom → Learner) would be appropriate.

## Current State
- **Existing tracking:** Custom localStorage only — `dojo_progress`, `dojo_submissions`, `dojo_notes`, `dojo_split_ratio`, `dojo_sidebar_collapsed` are stored client-side. No external analytics platform.
- **Server-side tracking:** `LearnerProgress`, `AssessmentAttempt`, and `TutorAnalytics` tables in the database track learning activity server-side.
- **Documentation:** No tracking documentation
- **Known issues:** No way to aggregate learner behavior across sessions or devices. localStorage is ephemeral (cleared on browser reset). Server-side tracking exists but no analytics dashboards or export.

## Integration Targets
| Destination | Purpose | Priority |
|-------------|---------|----------|
| localStorage | Client-side progress persistence (current) | Existing |
| SQLite/PostgreSQL | Server-side learning analytics (current) | Existing |
| PostHog (suggested) | Open-source product analytics — self-hostable, privacy-friendly | Future |

## Codebase Observations
- **Feature areas inferred:** 9 main routes — Library (`#/`), Code Dojo (`#/dojo`), Cost Estimator (`#/training-estimator`), Hyperparameters (`#/hyperparameters`), Assessments (`#/assessment`), Research Lab (`#/lab`), Dashboard (`#/dashboard`), Implementation (`#/implementation/:id`), Explorer (`#/explorer/:id`)
- **Entity model inferred:** Users, Papers, PaperModules, LearnerProgress, AssessmentAttempts, TutorAnalytics — all in SQLAlchemy ORM models
- **API surface:** 40+ endpoints covering paper upload/parsing, architecture exploration, dojo exercises, assessments, tutor interactions, lab experiments, and analytics
- **Client-side execution:** Pyodide runs Python code in-browser for dojo exercises — no server round-trip for code execution
- **Identity:** Anonymous `learner_id` (UUID in localStorage) sent via `X-Learner-ID` header on same-origin requests
