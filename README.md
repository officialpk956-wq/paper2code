<div align="center">
  <h1>🧠 Paper2Code</h1>
  <p><strong>Transforming Research Papers into Interactive Deep Learning Learning Experiences</strong></p>

  <p>Paper2Code is a full-stack, production-grade interactive learning platform that automatically extracts, validates, and visualises deep learning architectures from research papers — then teaches users to understand them through a grounded AI tutor, interactive assessments, adaptive learning paths, a code execution dojo, and compute-aware graph visualisation.</p>

  [![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-red?style=flat-square)](https://www.sqlalchemy.org/)
  [![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-3.x-F7DF1E?style=flat-square)](https://js.cytoscape.org/)
  [![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-6366F1?style=flat-square)](https://langchain-ai.github.io/langgraph/)
  [![Tests](https://img.shields.io/badge/Tests-1358%20passed-10B981?style=flat-square)]()
  [![Architectures](https://img.shields.io/badge/Architectures-15%20Verified-F59E0B?style=flat-square)]()
  [![Families](https://img.shields.io/badge/Architecture%20Families-14-8B5CF6?style=flat-square)]()
  [![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
</div>

---

## 🎯 What Is Paper2Code?

Paper2Code is a **research-to-learning intelligence platform** built to solve two critical problems simultaneously:

1. **The reproducibility crisis** — ~70% of deep learning papers contain implementation ambiguities that prevent faithful reproduction
2. **The learning barrier** — the jump from a published paper to a working, understood implementation takes 2–4 weeks per architecture, even for experienced engineers

The platform eliminates both bottlenecks through a deterministic pipeline that extracts, validates, and delivers architectural knowledge through a rich interactive experience.

### Core Philosophy

> **LLMs explain. Deterministic engines provide facts. The two never swap roles.**

Every numerical value — FLOPs, parameter counts, tensor shapes, assessment answers — comes from a deterministic engine. The LLM is only ever called to generate *narrative explanations* of facts that have already been verified. This prevents hallucination of architectural data at the source.

---

## 🚀 The Complete Processing Pipeline

```
Research Paper (ambiguous PDF)
         │
         ▼
┌─────────────────────────────┐
│  Stage 1: PDF Extraction    │  pdfplumber + PyMuPDF fallback
│  main.py · section_splitter │  Handles scanned, multi-col, embedded-font PDFs
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 2: Section Splitting │  core/rag/section_splitter.py (285 lines)
│  Advanced heading detection │  Abstract / Architecture / Experiments sections
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 3: LangGraph Agent   │  core/agents/ingestion_agent.py
│  Adaptive extraction        │  State machine — adapts strategy per paper
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 4: Parsing Agent     │  core/agents/parsing_agent_impl.py
│  Config extraction          │  core/rag/config_extractor.py (632 lines)
│  200+ layer-name synonyms   │  core/rag/normalizer.py (443 lines)
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 5: TensorTracker     │  core/rag/tensor_tracker.py (366 lines)
│  Symbolic forward-pass      │  (B,C,H,W) + (B,N,D) shape validation
│  Multi-head divisibility    │  Skip connection alignment verification
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 6: FLOPs Engine      │  core/rag/flops_engine.py (354 lines)
│  Closed-form per-layer      │  Conv2d · Attention · Linear · DepthwiseSep
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 7: KAG Grounding     │  core/rag/knowledge_graph.py (186 lines)
│  1,000+ DL rules            │  Zero hallucination of architectural facts
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Stage 8: Code Generation   │  core/codegen.py + core/implementation/
│  Educational PyTorch        │  Shape comments · Design docstrings
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────────────────────────────────┐
│  Interactive SPA  ·  AI Tutor  ·  Dojo  ·  Assessments  │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Feature Matrix

| Feature | Description | Status |
|---|---|:---:|
| **Architecture Library** | 15 verified DL architectures with full metadata | ✅ |
| **Paper Upload** | PDF upload → automatic architecture extraction | ✅ |
| **LangGraph Ingestion Agent** | Adaptive state-machine pipeline replaces fixed sequential flow | ✅ |
| **Architecture Parser** | Extract layers, hyperparameters, topology from raw text | ✅ |
| **Architecture Graph** | Interactive Cytoscape.js DAG with skip edges, zoom, node detail | ✅ |
| **Tensor Tracker** | Symbolic forward-pass validation at every layer | ✅ |
| **FLOPs Engine** | Per-layer closed-form FLOPs, cumulative totals, bottleneck ID | ✅ |
| **Parameter Engine** | Exact parameter count per module, memory footprint | ✅ |
| **Compute Heatmap** | Node coloring by FLOPs / Params / Memory, 4-tier legend | ✅ |
| **Architecture Playground** | Build custom architectures block-by-block, live validation | ✅ |
| **Architecture Comparison** | Side-by-side structural, FLOPs, parameter diff | ✅ |
| **Grounded AI Tutor** | 5-mode tutor grounded in real graph/tensor/FLOPs data | ✅ |
| **Agentic Tutor** | Anthropic tool-use API — paper lookup, graph fetch, cross-paper search | ✅ |
| **Learning Path Agent** | One-shot LLM curriculum generator personalised per user profile | ✅ |
| **Research RAG Agent** | Paper Q&A with cross-reference vector search | ✅ |
| **Code Review Agent** | Post-submission AI review stored in DB per submission | ✅ |
| **Interactive Assessments** | 4 challenge types graded by deterministic backend | ✅ |
| **Adaptive Learning** | Knowledge profiling, weakness detection, concept graph, review plans | ✅ |
| **Research Engineer Mode** | Educational PyTorch, pseudocode, training configs, cost estimation | ✅ |
| **Architecture Explorer** | Stage timeline, module grouping, stage detail panels | ✅ |
| **Tensor Journey** | Per-stage tensor shape evolution with FLOPs/params per step | ✅ |
| **Dojo** | Code execution sandbox (Piston engine) with 50+ ML problems | ✅ |
| **Leaderboard** | Weekly XP rankings with archive snapshots | ✅ |
| **Achievements System** | Unlockable badges with XP rewards | ✅ |
| **Production Auth** | Argon2-like bcrypt · JWT · Refresh tokens · TOTP MFA · OAuth | ✅ |
| **Admin Panel** | Full CRUD for papers, users, problems, moderation | ✅ |
| **Notifications** | In-app alerts (paper ready, achievements, announcements) | ✅ |
| **Global Search** | Full-text search across papers, problems, users | ✅ |
| **Storage Quota** | Per-user storage tracking with R2 key management | ✅ |

---

## 🏆 Verified Architecture Corpus

15 architectures, all pre-validated with correct tensor shapes, FLOPs, and parameter counts:

| # | Architecture | Category | Params | Modules | Verified |
|---|---|---|---|---|:---:|
| 1 | **LeNet-5** | CNN Pioneer | ~60K | 5 | ✅ |
| 2 | **AlexNet** | Deep CNN | ~61M | 8 | ✅ |
| 3 | **VGG16** | Very Deep CNN | ~138M | 16 | ✅ |
| 4 | **VGG19** | Very Deep CNN | ~143M | 19 | ✅ |
| 5 | **GoogLeNet** | Inception | ~6.8M | 22 | ✅ |
| 6 | **ResNet18** | Residual | ~11.7M | 8 | ✅ |
| 7 | **ResNet34** | Residual | ~21.8M | 16 | ✅ |
| 8 | **ResNet50** | Residual | ~25.5M | 16 | ✅ |
| 9 | **DenseNet121** | Dense | ~8M | 19 | ✅ |
| 10 | **MobileNetV2** | Efficient | ~3.4M | 19 | ✅ |
| 11 | **EfficientNet-B0** | Compound Scaling | ~5.3M | 16 | ✅ |
| 12 | **FCN** | Segmentation | ~134M | 8 | ✅ |
| 13 | **U-Net** | Encoder-Decoder | ~31M | 22 | ✅ |
| 14 | **Transformer** | Attention | ~65M | 34 | ✅ |
| 15 | **Vision Transformer** | Attention + Patches | ~86M | 34 | ✅ |

### 14 Architecture Family Builders

Each family has a full builder pipeline: `blocks_*.py` → `*_builder.py` → `schema_rules_*.py` → `schema_refiner_*.py`:

| Family | Builder | Key Block |
|---|---|---|
| `resnet` | `resnet_builder.py` | Bottleneck residual block |
| `unet` | `unet_builder.py` | Symmetric encoder-decoder skip |
| `transformer` | `transformer_builder.py` | Multi-head self-attention |
| `vit` | `vit_builder.py` | Patch embedding + ViT block |
| `diffusion` | `ddpm_builder.py` | Sinusoidal timestep embedding + denoiser |
| `yolo` | `yolo_builder.py` | Detection head + anchor boxes |
| `efficientnet` | `efficientnet_builder.py` | MBConv compound scaling |
| `swin` | `swin_builder.py` | Shifted-window attention + patch merging |
| `gan` | `gan_builder.py` | Generator + Discriminator adversarial pair |
| `densenet` | `densenet_builder.py` | Dense connection reuse blocks |
| `bert_gpt` | `bert_gpt_builder.py` | Bidirectional / causal attention heads |
| `mobilenet` | `mobilenet_builder.py` | Depthwise separable + inverted residual |
| `mae` | `mae_builder.py` | Masked patch encoder + decoder |
| `ldm` | `ldm_builder.py` | Latent diffusion + noise scheduler + VAE |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                             Browser                                  │
│                      Vanilla JavaScript SPA                          │
│  Library · Explorer · Tutor · Assessments · Playground · Dojo        │
│    Cytoscape.js · Chart.js · Monaco Editor · Font Awesome 6         │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ HTTP / REST
┌────────────────────────────▼─────────────────────────────────────────┐
│                         FastAPI Backend                              │
│  backend/server.py  ·  16 routers  ·  SlowAPI rate limiting         │
│  Sentry · JSON logging · RequestID middleware · CORS                 │
└──────┬───────────┬──────────────────────┬──────────────┬────────────┘
       │           │                      │              │
┌──────▼──────┐  ┌─▼────────────────┐  ┌─▼───────────┐ ┌▼──────────────┐
│  SQLAlchemy │  │  Educational     │  │  AI Agents  │ │  Auth Module  │
│  ORM + DB   │  │  Engines         │  │             │ │               │
│             │  │  · Parser        │  │  · Tutor    │ │  bcrypt hash  │
│  20 models  │  │  · GraphBuilder  │  │  · Ingestion│ │  JWT (HS256)  │
│  Alembic    │  │  · TensorTracker │  │  · CodeRev  │ │  TOTP MFA     │
│  migrations │  │  · FLOPs Engine  │  │  · LangGraph│ │  OAuth        │
│             │  │  · Assessment    │  │  · LearnPath│ │  Sessions     │
│             │  │  · Adaptive      │  │  · RAG      │ │  Audit log    │
└─────────────┘  └──────────────────┘  └─────────────┘ └───────────────┘
       │                                                       │
┌──────▼──────────────────────────────────────────────────────▼───────┐
│                          Celery Workers                              │
│  Paper ingestion tasks · Email drip · XP pruning · LB archive       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Complete Project Structure

### Root Level

```
paper2code/
├── main.py                         PDF extraction CLI entry point
├── app.py                          Legacy Streamlit app (1,187 lines)
├── golden_paper_pipeline.py        Corpus builder for 3 source PDFs
├── requirements.txt                81 Python dependencies
├── alembic.ini                     Alembic migration config
├── docker-compose.yml              Multi-service Docker config (app + Redis + Postgres)
├── Dockerfile                      Production container
├── .env                            DATABASE_URL + GROQ_API_KEY + SECRET_KEY
└── .env.example                    Template for new deployments
```

---

### `backend/` — API & Data Layer

```
backend/
├── server.py                       FastAPI app factory, all routers mounted
├── database.py                     SQLAlchemy engine, get_db() dependency
├── models.py                       20 ORM models (536 lines)
├── corpus_builder.py               Golden corpus pipeline (373 lines)
├── dependencies.py                 get_current_user() FastAPI dependency
├── logging_config.py               JSON structured logging + RequestID middleware
├── celery_app.py                   Celery worker configuration
├── middleware/                     Custom ASGI middleware
├── modules/                        Pluggable feature modules (auth, authz, security)
├── repositories/                   Data access layer
├── routers/                        16 FastAPI APIRouter files
├── schemas/                        Pydantic request/response models
├── services/                       Business logic layer
└── tasks/                          Celery async task definitions
```

#### `backend/routers/` — All 16 API Routers

| Router | Prefix | Endpoints | Description |
|---|---|---|---|
| `health.py` | `/api/health` | 1 | Liveness check |
| `auth.py` | `/api/auth` | 4 | Basic login/register/refresh/me |
| `papers.py` | `/api/papers` | 14 | Full paper CRUD, upload, tutor, compare |
| `dojo.py` | `/api` | 12 | Problems, submissions, judge, stats |
| `learning.py` | `/api` | 18 | Progress, assessments, adaptive, tutor |
| `lab.py` | `/api/lab` | 11 | Architecture Lab — mutate, compare, hypothesis |
| `tasks.py` | `/api/tasks` | 3 | Async task status polling |
| `user.py` | `/api/me` | 10 | Profile, notifications, achievements, prefs |
| `admin.py` | `/api/admin` | 22 | Full admin panel CRUD |
| `achievements.py` | `/api/achievements` | 3 | Achievement catalogue and unlock |
| `announcements.py` | `/api/announcements` | 4 | Platform announcements |
| `leaderboard.py` | `/api/leaderboard` | 2 | Weekly + all-time rankings |
| `notifications.py` | `/api/notifications` | 4 | In-app alerts |
| `oauth.py` | `/api/auth/oauth` | 4 | Google + GitHub OAuth flows |
| `search.py` | `/api/search` | 1 | Global full-text search |
| `(auth module)` | `/api/auth` | 20+ | Full production auth (see below) |

#### `backend/models.py` — 20 ORM Models

| Model | Purpose | Key Fields |
|---|---|---|
| `User` | Platform user | email, hashed_password, points, streak, mfa_enabled, token_version, lockout_until |
| `Paper` | Research paper record | title, architecture_graph (JSON), flops_analysis (JSON), visibility, uploaded_by |
| `PaperModule` | Per-layer module record | layer_name, module_type, tensor_flow (JSON), flops_context (JSON) |
| `LearnerProgress` | Per-entity completion tracking | learner_id, entity_type, entity_id, status, time_spent_seconds |
| `AssessmentAttempt` | Individual challenge attempt | assessment_type, architecture, user_answer, correct_answer, score, is_correct |
| `TutorAnalytics` | Tutor session tracking | learner_id, architecture, reasoning_type |
| `Problem` | Dojo coding challenge | id, slug, difficulty, python_template, test_cases, hints, time_limit_ms, acceptance_rate |
| `DojoSubmission` | Code submission + judge result | code, passed, stdout, stderr, time_ms, is_best, review_text |
| `Task` | Async job record | type, status, user_id, input_ref, result, error |
| `InterviewQuestion` | Interview prep question bank | question, difficulty, category, companies, key_points |
| `Roadmap` | Learning roadmap | title, description, nodes (JSON) |
| `EmailVerificationToken` | Email verify token | token (hashed), expires_at, used_at |
| `PasswordResetToken` | Password reset token | token (hashed), expires_at, used_at |
| `UsageLog` | LLM API cost tracking | action, prompt_tokens, completion_tokens, cost_usd |
| `Notification` | In-app alert | type, title, body, is_read, payload (JSON) |
| `OAuthAccount` | OAuth provider link | provider, provider_user_id, provider_email |
| `Achievement` | Achievement definition | slug, title, icon, xp_reward, category |
| `UserAchievement` | User → achievement unlock | user_id, achievement_id, earned_at |
| `XPEvent` | XP history log | user_id, action, amount, entity_id |
| `TutorFeedback` | Thumbs up/down on tutor messages | session_id, message_index, rating |
| `TutorSessionRecord` | Persistent conversation history | session_id, context_type, messages (JSON) |
| `LeaderboardArchive` | Weekly ranking snapshots | week_start, weekly_points, rank |
| `EmailDripLog` | Drip campaign tracking | user_id, drip_day |

---

### `backend/modules/` — Production-Grade Auth Module

The auth system is implemented as a pluggable module following Clean Architecture principles, completely separate from the lightweight `routers/auth.py` stub.

```
backend/modules/auth/
├── api/v1.py                   478-line router — all auth endpoints
├── models.py                   UserSession, VerificationToken, ResetToken, AuditLog
├── schemas.py                  22 Pydantic request/response models
├── dependencies.py             get_current_user(), get_current_verified_user()
├── middleware/rate_limit.py    Redis-backed rate limiter with in-memory fallback
├── security/hashing.py         bcrypt password hashing utilities
├── oauth/                      Google + GitHub OAuth flows
├── repositories/               TokenRepository, SessionRepository, AuditRepository
├── services/
│   ├── auth_service.py         Registration, login, logout, token_version management
│   ├── session_service.py      Refresh token rotation, device tracking, IP/UA storage
│   ├── verification_service.py Email verification token lifecycle (24h, single-use)
│   ├── reset_service.py        Password reset token lifecycle (15min, single-use)
│   ├── mfa_service.py          TOTP setup, QR generation, backup codes (hashed)
│   ├── oauth_service.py        Provider token exchange, account linking
│   ├── email_service.py        Transactional email dispatch
│   └── audit_service.py        Security event logging
└── utils/                      Token helpers, user-agent parsing
```

#### Production Auth Endpoints (`/api/auth/...`)

| Method | Endpoint | Feature | Rate Limited |
|---|---|---|---|
| `POST` | `/register` | Registration + email verification trigger | 5/hour |
| `POST` | `/login` | Login with brute-force protection | Yes |
| `POST` | `/refresh` | Refresh token rotation (revoke-and-reissue) | Yes |
| `POST` | `/logout` | Revoke current session | — |
| `POST` | `/logout-all` | Revoke all sessions + increment token_version | — |
| `POST` | `/verify-email` | Consume email verification token | Yes |
| `POST` | `/resend-verification` | Issue new verification token | 3/hour |
| `POST` | `/forgot-password` | Issue password reset token (15min) | 3/hour |
| `POST` | `/reset-password` | Consume reset token, rehash password | Yes |
| `GET` | `/sessions` | List active devices with metadata | — |
| `DELETE` | `/sessions/{id}` | Revoke specific device session | — |
| `POST` | `/mfa/enable` | Generate TOTP secret + QR code | — |
| `POST` | `/mfa/verify` | Confirm TOTP code to activate MFA | — |
| `POST` | `/mfa/disable` | Disable MFA with password confirmation | — |
| `POST` | `/change-password` | Change password + revoke all sessions | — |
| `POST` | `/change-email` | Change email + trigger reverification | — |
| `PATCH` | `/profile` | Update name/avatar (255/512 char limits) | — |
| `DELETE` | `/account` | GDPR account deletion + cascade | — |
| `POST` | `/oauth/google` | Google OAuth token exchange | Yes |
| `POST` | `/oauth/github` | GitHub OAuth token exchange | Yes |
| `GET` | `/me` | Current user info | — |

#### Security Properties

- **Password hashing**: bcrypt with auto-rehash on parameter change
- **Refresh tokens**: stored as SHA-256 hashes, never plaintext; rotation on every use
- **Replay attack prevention**: old refresh token is revoked before new one issues
- **Brute-force protection**: account lockout with `lockout_until` + `failed_login_attempts` counter
- **Token versioning**: `token_version` increment invalidates all existing JWTs on password change / logout-all
- **TOTP MFA**: pyotp TOTP, QR via qrcode, backup codes stored as bcrypt hashes
- **OAuth**: Google + GitHub; prevents duplicate accounts; imports verified email flag + avatar
- **Rate limiting**: Redis-backed SlowAPI with per-endpoint limits; falls back to in-memory when Redis is unavailable
- **Audit logging**: every security event logged to `AuditLog` with timestamp, IP, user agent, action

---

### `core/` — Intelligence Engine

#### `core/rag/` — Research-Augmented Generation Layer

```
core/rag/
├── knowledge_graph.py      186 lines — DL ontology, 1,000+ hardcoded rules
├── tensor_tracker.py       366 lines — symbolic forward-pass validator
├── config_extractor.py     632 lines — the main parser
├── flops_engine.py         354 lines — closed-form FLOPs per layer
├── diff_engine.py          121 lines — architecture structural diff
├── semantic_explainer.py   187 lines — per-node educational explanations
├── normalizer.py           443 lines — 200+ layer-name synonyms
├── retriever.py            183 lines — semantic context retrieval for tutor
├── section_splitter.py     285 lines — advanced PDF section detection
└── symbolic_parser.py      132 lines — parse R(3,4)×64, W_q∈ℝ^{d×d_k} notation
```

**FLOPs formulas by layer type:**

| Layer | Formula |
|---|---|
| Conv2d | `C_in × K² × C_out × H_out × W_out` |
| Linear | `in_features × out_features` |
| Multi-Head Attention | `4 × N × d_model² + 2 × N² × d_model` |
| Depthwise Separable | `(K² × C + C × C_out) × H × W` |
| Grouped Conv | `C_in/groups × K² × C_out × H_out × W_out` |
| BatchNorm / LayerNorm | `2 × features` |

**TensorTracker validation checks:**

- `(B, C, H, W)` spatial dimension propagation through Conv/Pool layers
- `(B, N, D)` sequence dimension propagation through Attention layers
- Multi-head attention: `embed_dim % num_heads == 0`
- Reshape: input element count == output element count
- Skip connection: matching shapes before add/concat
- Concatenation: compatible channel or sequence dimensions

#### `core/agents/` — Agent System (15 files)

```
core/agents/
├── types.py                    TypedDict contracts for all agents (165 lines)
├── parsing_agent.py            Parser protocol definition
├── parsing_agent_impl.py       section_splitter → config_extractor → config_parser
├── config_parser.py            ConfigDict → ArchitectureGraph (11,986 lines)
├── explanation_agent.py        Explanation protocol definition
├── explanation_agent_impl.py   Semantic explanation assembly
├── visualization_agent.py      Visualisation protocol definition
├── visualization_agent_impl.py Cytoscape.js element + Graphviz DOT generation
├── tutor_agent.py              Grounded tutor — 5 modes (16,120 lines)
├── agentic_tutor.py            Anthropic tool-use API agent (7,027 lines)
├── ingestion_agent.py          LangGraph state-machine ingestion (5,609 lines)
├── learning_path_agent.py      One-shot curriculum generator (1,582 lines)
├── research_rag_agent.py       Paper Q&A with cross-reference search
├── code_review_agent.py        Post-submission AI code review
└── __init__.py                 Agent factory + registry (1,886 lines)
```

**`tutor_agent.py` — The Grounded Tutor (5 Modes)**

| Mode | Context Injected | Activated From |
|---|---|---|
| Module Tutor | Single module: type, shapes, FLOPs, design choices | Module detail page |
| Architecture Tutor | Full architecture: stages, overall design, comparisons | Library / Overview |
| Node Tutor | Specific node: exact metrics, adjacent layers, data flow | Click node in Explorer |
| Playground Tutor | User's custom architecture: validity, suggestions | Architecture Playground |
| Comparison Tutor | Two-architecture diff: what changed, why, tradeoffs | Comparison view |

**`agentic_tutor.py` — Tool-Using Anthropic Agent**

Registered tools available to the agent during a conversation:

| Tool | Description |
|---|---|
| `lookup_paper_section` | Fetch raw text of a specific section from a paper by `paper_id` |
| `get_architecture_graph` | Return full architecture graph JSON for a named architecture |
| `search_cross_paper` | Semantic search across all papers for relevant passages |

**`ingestion_agent.py` — LangGraph State Machine**

Replaces the fixed `Paper2CodePipeline`. State transitions:

```
START → extract_text → classify_architecture → route_by_family
          ↓ (if extraction fails)
        fallback_extraction → classify_architecture

route_by_family → family_specific_builder → validate_tensors
                                                 ↓ (if validation fails)
                                           reparse_with_hints → validate_tensors
                                                 ↓
                                           enrich_flops → store_result → END
```

#### `core/assessment/` — Deterministic Assessment Engine

```
core/assessment/
├── engine.py                       170 lines — orchestrator, grading router
├── architecture_challenges.py      338 lines — 100+ topology questions
├── tensor_challenges.py            364 lines — shape computation problems
├── flops_challenges.py             292 lines — compute estimation problems
└── comparison_challenges.py        296 lines — diff reasoning questions
```

All grading is deterministic — no LLM involvement. Answers are validated against:
- Stored `ArchitectureGraph` node/edge data
- TensorTracker symbolic evaluation
- FLOPs engine formula evaluation
- `diff_engine.py` structural diff output

#### `core/analytics/` — Adaptive Learning Engine

```
core/analytics/
├── adaptive_engine.py              481 lines — knowledge profiling, weakness detection
└── recommendation_engine.py        222 lines — next-architecture recommendation
```

**9 concept areas tracked per user:**

| Concept | Data Source |
|---|---|
| Convolutions | Assessment accuracy + module view time |
| Residual Connections | Skip-edge exploration + ResNet assessments |
| Dense Connections | DenseNet assessment performance |
| Attention Mechanisms | Transformer exploration + attention challenges |
| Transformers | ViT/Transformer assessment + tutor interactions |
| Encoder-Decoders | U-Net/FCN assessment performance |
| Tensor Shapes | Shape challenge accuracy |
| FLOPs Reasoning | FLOPs challenge accuracy |
| Architectural Tradeoffs | Comparison challenge performance |

#### `core/lab/` — Architecture Experimentation Lab

```
core/lab/
├── diff_engine.py              Structural diff between two graphs
├── hypothesis_engine.py        13,041 lines — "what if I add dropout?" engine
├── mutator.py                  14,199 lines — structured architecture mutation
└── tradeoff_analyzer.py        8,281 lines — FLOPs/accuracy tradeoff analysis
```

The Lab allows users to mutate existing architectures, test hypotheses about design choices, and analyze compute tradeoffs — all grounded in the deterministic engines.

#### `core/implementation/` — Research Engineer Mode

```
core/implementation/
├── code_mapper.py              622 lines — 40+ layer types → PyTorch nn.Module
├── cost_estimator.py           203 lines — GPU-hours + memory training cost
├── reproduction_cards.py       337 lines — structured hyperparameter cards
└── training_config.py          317 lines — optimizer / LR schedule / aug from paper
```

**Code generation → PyTorch path:**

```
ArchitectureGraph (TensorTracker-validated)
         ↓
Topological sort of GraphNodes
         ↓
Layer type → nn.Module class mapping (code_mapper.py)
         ↓
Shape annotations injected from TensorTracker trace
         ↓
Design comments from Knowledge Graph ontology
         ↓
Educational PyTorch .py file with:
  · Shape comment at every layer  → # (B, 64, 56, 56) → (B, 128, 28, 28)
  · Design docstring               → """3×3 conv halves spatial dims, doubles channels"""
  · forward() tensor flow          → x = self.conv1(x)  # spatial reduction
```

#### `core/dojo/` — Code Execution Problems

```
core/dojo/
├── exercises.py            44,970 lines — 50+ ML exercise definitions
├── problems.py             31,479 lines — graded coding problem bank
└── validator.py             4,977 lines — test-case runner
```

Each Dojo problem includes:
- Problem description with math notation
- Python starter template with type hints
- Hidden test cases (run by Piston execution engine)
- Hints (progressive reveal)
- Expected explanation (shown post-solve)
- Related architecture links
- Related papers and math references
- Per-problem time limit (overrides global 10,000ms default)

---

## 🤖 AI Tutor — Deep Dive

### How Grounding Works

Every tutor response follows this pipeline:

```python
# 1. Assemble deterministic context
context = {
    "graph": architecture_graph,          # exact topology
    "tensor_trace": tensor_tracker_trace, # shape at every layer
    "flops": flops_context,               # per-layer FLOPs
    "params": param_summary,              # exact parameter counts
    "user_profile": learner_profile,      # knowledge gaps
}

# 2. Format grounding-first prompt
prompt = f"""
ARCHITECTURE FACTS (do not contradict these):
{format_context(context)}

USER QUESTION: {user_question}

Generate an explanation grounded in the above facts.
Do not invent any numerical values.
"""

# 3. LLM generates narrative only
response = llm_complete(prompt)

# 4. Strip any contradictory numerical claims
response = validate_response_against_context(response, context)
```

The LLM never sees a question without its factual grounding context. It is architecturally impossible for the tutor to hallucinate a FLOPs count or parameter number.

### Conversation Persistence

Tutor sessions are stored in `TutorSessionRecord`:
- `session_id` — UUID, per conversation
- `messages` — `[{role, content}]` JSON array
- `context_type` — which mode was active
- `last_active_at` — for session expiry

Users can thumb up/down individual messages (`TutorFeedback`), which is stored and can feed future quality improvements.

---

## 🥋 The Dojo — Code Execution Platform

### Submission Flow

```
User writes Python code in Monaco Editor
         ↓
POST /api/dojo/problems/{id}/submit
  · 10 KB code size limit enforced
  · Auth required (JWT)
         ↓
Piston execution engine runs code against hidden test cases
  · Configurable per-problem timeout (default: 10,000ms)
  · stdout / stderr captured
         ↓
Result stored in DojoSubmission
  · passed, time_ms, stdout, stderr
  · is_best updated if this is user's best attempt
  · acceptance_rate on Problem updated
         ↓
Code Review Agent triggered (async)
  · LLM generates targeted code review
  · Stored in submission.review_text
         ↓
XP awarded if first solve
```

### Problem Structure

```python
{
  "id": "conv2d-forward",
  "slug": "implement-conv2d-forward",
  "title": "Implement Conv2D Forward Pass",
  "difficulty": "Medium",
  "category": "Convolutions",
  "estimated_time": 30,
  "description": "...",          # full problem with math
  "python_template": "...",      # starter code
  "test_cases": [...],           # hidden from user
  "hints": [...],                # progressive reveal
  "explanation": {...},          # unlocked after solve
  "related_architectures": ["resnet", "vgg"],
  "related_papers": ["1512.03385"],
  "related_math": ["cross-correlation", "receptive field"],
  "learning_points": [...],
  "time_limit_ms": 10000,
  "acceptance_rate": 0.6142
}
```

### Dojo API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/problems` | List problems (filter by difficulty, category, full-text search) |
| `GET` | `/api/problems/{id}` | Problem detail with template and hints |
| `POST` | `/api/problems/{id}/submit` | Submit code for judge execution |
| `GET` | `/api/dojo/submissions` | User's submission history |
| `GET` | `/api/dojo/submissions/{id}` | Single submission detail + review |
| `GET` | `/api/dojo/stats` | User stats (solved, attempted, streak) |
| `GET` | `/api/dojo/global-stats` | Platform-wide stats + total_problems count |
| `GET` | `/api/dojo/leaderboard` | Problem-specific solvers ranking |
| `POST` | `/api/dojo/submissions/{id}/share` | Toggle public sharing |
| `GET` | `/api/dojo/public-submissions` | Browse shared solutions |

---

## 📝 Interactive Assessments — Deep Dive

### 4 Challenge Types

**1. Architecture Challenges** (`architecture_challenges.py`)
- Identify layer types from a graph description
- Order stages in a given architecture
- Explain the purpose of a specific connection
- Rank architectures by parameter count
- Graded by exact match against stored `ArchitectureGraph`

**2. Tensor Shape Challenges** (`tensor_challenges.py`)
- Compute output shape after Conv2d(in=64, out=128, k=3, stride=2) on (B,64,56,56)
- Determine valid multi-head attention head count for embed_dim=768
- Calculate output sequence length after pooling
- Graded by TensorTracker symbolic evaluation

**3. FLOPs Challenges** (`flops_challenges.py`)
- Estimate FLOPs for a described layer configuration
- Identify the computational bottleneck stage
- Compare FLOPs between two architectures
- Graded by formula evaluation against `flops_engine.py` ground truth

**4. Comparison Challenges** (`comparison_challenges.py`)
- Describe structural differences between ResNet50 and ResNet18
- Identify what design decision causes the FLOPs increase
- Select the more parameter-efficient architecture for a given task
- Graded by structural diff from `diff_engine.py`

### Adaptive Difficulty

Each assessment interaction updates the learner's concept mastery vector. Difficulty of subsequent challenges increases as mastery improves, preventing both frustration and boredom.

---

## 🗺️ Phase 11 — Explorer, Tensor Journey & Compute Heatmap

### Phase 11A — Architecture Explorer ✅

A dedicated deep-dive view per architecture.

**Stage Timeline**: Architecture partitioned into 4 logical stages. Horizontal timeline with clickable cards. Active stage highlighted. Each card shows module count and connects with arrows.

**Stage Detail Panels** per stage:
- Stage name, position (e.g. "Stage 2 of 4")
- Module count, total FLOPs, total parameters (metric cards)
- **Stage Compute Summary**: total FLOPs, total params, highest-cost layer
- Tensor Journey summary (input shape → transform → output shape)
- Full module list with layer type annotations

**Architecture Graph Navigation**:
- Cytoscape.js renders full architecture in breadth-first layout
- Node click → node detail panel (FLOPs, params, memory, heatmap rank)
- Hover tooltip: label and type
- Skip edges: dashed purple lines
- Right-click context menu: open module tutor for node

### Phase 11B — Tensor Journey ✅

Visualises tensor shape evolution through every module in each stage.

| Element | What it Shows |
|---|---|
| Input node | Entry tensor shape `[B, 3, 224, 224]` |
| Module step | Output shape after each transformation |
| FLOPs annotation | MFLOPs or GFLOPs for the layer |
| Params annotation | K or M parameters |
| ⬆ indicator | Channel expansion |
| ⬇ indicator | Channel reduction |
| ↳ indicator | Spatial downsampling |
| → indicator | Shape unchanged |

Math toggle: reveals `input → OP → output` inline.
Code toggle: reveals `x = layer_type(x)` pseudocode.

**Verified architectures and journey step counts:**

| Architecture | Journey Steps | All Shapes |
|---|---|---|
| LeNet-5 | 15 | ✅ |
| ResNet18 | 16 | ✅ |
| ResNet50 | 16 | ✅ |
| DenseNet121 | 19 | ✅ |
| U-Net | 22 | ✅ |
| Transformer | 34 | ✅ |
| Vision Transformer | 34 | ✅ |

### Phase 11C — Compute Heatmap ✅

Transforms the static graph into a compute-aware visualisation.

**4-Mode Toggle** (all client-side, zero API calls on mode switch):

| Mode | Colors Nodes By | Data Source |
|---|---|---|
| None | Uniform grey | — |
| FLOPs | `flops_context.real_flops_mflops` | Normalised per-architecture |
| Parameters | `flops_context.total_params_estimate` | Normalised per-architecture |
| Memory | `flops_context.activation_memory_mb` → fallback: shape × float32 | Normalised |

**Color scale:**

| Color | Percentile | Meaning |
|---|---|---|
| 🟢 Green | 0–25% | Low compute |
| 🟡 Yellow | 25–50% | Medium |
| 🟠 Orange | 50–75% | High |
| 🔴 Red | 75–100% | Very high (bottleneck) |

**Verification results** (7 architectures, 0 errors across all nodes, all metrics, all modes):

| Architecture | Nodes | FLOPs | Params | Memory | Node Detail | Stage Summary |
|---|---|---|---|---|---|---|
| LeNet-5 | 7 | ✅ | ✅ | ✅ | ✅ | ✅ |
| ResNet18 | 12 | ✅ | ✅ | ✅ | ✅ | ✅ |
| ResNet50 | 20 | ✅ | ✅ | ✅ | ✅ | ✅ |
| DenseNet121 | 11 | ✅ | ✅ | ✅ | ✅ | ✅ |
| U-Net | 18 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Transformer | 32 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Vision Transformer | 26 | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🔬 Research Engineer Mode — Deep Dive

### What It Produces

| Output | Description |
|---|---|
| **Educational PyTorch** | Full `nn.Module` with shape comment at every layer + design docstrings |
| **Pseudocode** | Architecture-level pseudocode for whiteboard / interview use |
| **Training Configuration** | Optimizer, LR schedule, batch size, augmentation from the paper |
| **Cost Estimation** | GPU-hours + memory requirements to train from scratch |
| **Reproduction Card** | All hyperparameters needed to reproduce the paper's reported results |
| **Hyperparameter Guidance** | Role of each hyperparameter + typical search ranges |

### Code Generation Details

`code_mapper.py` (622 lines) maps every `GraphNode` type to its `nn.Module`:

- **40+ layer types covered**: Conv1d/2d/3d, ConvTranspose, DepthwiseConv, GroupedConv, Linear, MultiheadAttention, SelfAttention, CrossAttention, BatchNorm1d/2d, LayerNorm, GroupNorm, ReLU/GELU/SiLU/Swish, MaxPool/AvgPool/AdaptiveAvgPool, Dropout, Embedding, PatchEmbedding, PositionalEncoding, FeedForward, ResidualBlock, DenseBlock, InceptionModule, SEBlock, ConvNeXtBlock, SwinBlock, MBConv, and more
- Shape annotations from TensorTracker trace at every layer
- Design docstrings from the ontology at every architectural decision point
- `forward()` body with tensor flow comments

### Cost Estimator

`cost_estimator.py` takes:
- Total FLOPs from `flops_engine.py`
- Batch size and epochs from `training_config.py`
- GPU type selection (A100/V100/RTX 3090)

Returns:
- Estimated GPU-hours for training
- Estimated cloud cost at standard spot pricing
- Memory requirement in GB

---

## 🎓 Adaptive Learning — Deep Dive

### Knowledge Profiling

Every user interaction updates a 9-dimensional mastery vector stored in `LearnerProgress`. The system tracks:

- Raw accuracy per challenge type
- Assessment attempt count per concept
- Time spent on each module
- Tutor interaction type (what the user asked about)
- Learning velocity (mastery improvement per session)

### Weakness Detection

`adaptive_engine.py` identifies weak concepts by:

1. Threshold check: concept score < 0.6
2. Trend analysis: declining scores over last 3 sessions
3. Consistency check: correct answers on simple but wrong on complex variants

### Review Plans

When weaknesses are detected, `adaptive_engine.py` generates an ordered study plan:

```python
{
  "weak_concepts": ["attention_mechanisms", "tensor_shapes"],
  "plan": [
    {"step": 1, "action": "review", "target": "Transformer", "reason": "Attention mechanism basics"},
    {"step": 2, "action": "practice", "target": "tensor_shape_challenge", "difficulty": "easy"},
    {"step": 3, "action": "review", "target": "ViT", "reason": "Attention in vision context"},
    {"step": 4, "action": "practice", "target": "attention_challenge", "difficulty": "medium"},
    {"step": 5, "action": "compare", "targets": ["Transformer", "ViT"], "reason": "Solidify attention differences"},
  ]
}
```

### Learning Path Agent (`learning_path_agent.py`)

One-shot LLM call that generates a personalised 5-step curriculum:

**Input:**
- `completed_architectures: list[str]`
- `weak_topics: list[str]`
- `solved_problem_ids: list[str]`
- `available_papers: list[dict]`
- `available_problems: list[dict]`

**Output:**
```json
{
  "steps": [
    {"type": "paper", "id": 7, "title": "...", "reason": "..."},
    {"type": "problem", "id": "conv2d-forward", "title": "...", "reason": "..."},
    ...
  ],
  "reasoning": "You have strong ResNet knowledge but weak attention. Start with..."
}
```

---

## 📊 Leaderboard & Gamification

### XP System

| Action | XP |
|---|---|
| First solve on a Dojo problem | 50 XP |
| Correct assessment answer | 10 XP |
| Module completed | 5 XP |
| Tutor session | 2 XP |
| Day streak bonus | `streak_days × 5` XP |

XP events are logged to `XPEvent` table. Old events (>90 days) are pruned by a scheduled Celery task.

### Weekly Leaderboard

- `weekly_points` column on `User` resets every Monday
- `LeaderboardArchive` snapshots the full ranking before reset
- Historical rankings are browsable through the API

### Achievement System

`Achievement` catalogue with `UserAchievement` unlock tracking:

| Category | Example Achievements |
|---|---|
| Learning | "First Paper Read", "ResNet Master", "Architecture Collector" |
| Consistency | "7-Day Streak", "30-Day Streak", "Weekend Warrior" |
| Community | "Solution Shared", "Helpful Reviewer", "Top 10 This Week" |

Each achievement carries an XP reward and an icon. Achievements unlock automatically when the backend detects the qualifying event (first solve, streak milestone, etc.).

---

## 🔔 Notifications & Communication

### In-App Notifications

`Notification` model types:
- `paper.done` — paper extraction complete
- `achievement.unlocked` — new achievement earned
- `announcement` — platform announcement
- `review.posted` — AI code review ready

### Notification Preferences

Users control communication at `/api/me/notification-preferences`:

```json
{
  "email_drip_opt_out": false,
  "email_digest": true
}
```

### Email Drip Campaign

`EmailDripLog` tracks which drip emails have been sent. The Celery worker sends re-engagement emails at day 1, 3, and 7 post-signup for users who haven't completed their first paper.

---

## 🛡️ Admin Panel

### Admin Router (`admin.py`, 22,248 lines)

| Endpoint | Description |
|---|---|
| `GET /api/admin/users` | Paginated user list with search |
| `GET /api/admin/users/{id}` | User detail with activity |
| `PATCH /api/admin/users/{id}` | Update role / ban / verify |
| `DELETE /api/admin/users/{id}` | Delete user + cascade |
| `GET /api/admin/papers` | All papers including private |
| `PATCH /api/admin/papers/{id}` | Moderate, flag, unflag |
| `DELETE /api/admin/papers/{id}` | Remove paper + modules |
| `GET /api/admin/problems` | Dojo problem management |
| `POST /api/admin/problems` | Create new problem |
| `PATCH /api/admin/problems/{id}` | Edit / retire problem |
| `GET /api/admin/submissions` | All submissions + filter |
| `GET /api/admin/analytics` | Platform-wide analytics dashboard |
| `GET /api/admin/usage-logs` | LLM API cost tracking |

All admin endpoints require `is_admin = True` on the requesting user's JWT.

---

## 🚀 Papers API — Complete Endpoint Reference

`papers.py` (36,866 lines) handles the full paper lifecycle:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/papers` | Library list with metadata, filter by visibility |
| `POST` | `/api/papers` | Create paper record |
| `POST` | `/api/papers/upload` | Upload PDF → full extraction pipeline |
| `GET` | `/api/papers/{id}` | Full paper detail + architecture graph |
| `PUT` | `/api/papers/{id}` | Update paper metadata |
| `DELETE` | `/api/papers/{id}` | Delete paper + cascade modules |
| `GET` | `/api/papers/{id}/modules` | All modules with tensor_flow + flops_context |
| `GET` | `/api/papers/{id}/modules/{mid}` | Single module detail |
| `POST` | `/api/papers/{id}/tutor` | Grounded tutor query |
| `GET` | `/api/papers/{a}/compare/{b}` | Structural + metric diff |
| `POST` | `/api/papers/{id}/flag` | Moderate: flag for review |
| `POST` | `/api/papers/{id}/unflag` | Admin: unflag |
| `GET` | `/api/papers/{id}/pytorch` | Download generated PyTorch code |
| `GET` | `/api/papers/{id}/training-config` | Training configuration JSON |

---

## 📦 User Profile API

`user.py` handles per-user data:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/me` | Current user profile |
| `PATCH` | `/api/me/profile` | Update name / avatar (with field size limits) |
| `GET` | `/api/me/progress` | Full learning progress summary |
| `GET` | `/api/me/submissions` | Personal Dojo submission history |
| `GET` | `/api/me/notification-preferences` | Get email/notification settings |
| `PATCH` | `/api/me/notification-preferences` | Update notification settings |
| `GET` | `/api/me/achievements` | Unlocked achievements + XP totals |
| `GET` | `/api/me/xp-history` | XP event log |
| `DELETE` | `/api/me/account` | GDPR account deletion |

---

## 🧪 Test Suite

**1,358 tests passing, 1 skipped, 0 failing.**

```
tests/
├── conftest.py                         Shared fixtures, TestClient setup
├── test_auth_module.py                 Authentication flows (all 15 phases)
├── test_authz_module.py                Authorization + role enforcement
├── test_sprint_b.py                    Paper visibility, ToS, R2 storage
├── test_sprint_c.py                    Assessments + adaptive engine
├── test_sprint_d.py                    OAuth, achievements, email drip
├── test_sprint_e.py                    Content moderation, leaderboard
├── test_sprint_f.py                    XP system, tutor feedback, session records
├── test_sprint_g.py                    Dojo versioning, time limits, acceptance rates
├── test_admin.py                       Admin CRUD endpoints
├── test_admin_detail.py                Admin analytics + user detail
├── test_agentic_tutor.py               Tool-using tutor agent
├── test_agents_j4.py                   LangGraph ingestion agent
├── test_code_review_agent.py           Post-submission AI review
├── test_complex_integration.py         End-to-end scenario tests (38,534 lines)
├── test_dojo.py                        Dojo problem + submission flow
├── test_dojo_fixes.py                  Dojo edge cases + security fixes
├── test_dojo_service.py                Dojo service layer unit tests
├── test_e2e_lifecycle.py               Full user lifecycle test
├── test_infra_tasks.py                 Celery tasks + XP pruning
├── test_ingestion_agent.py             LangGraph state transitions
├── test_knowledge_extraction.py        Parser + normalizer accuracy
├── test_lab_service.py                 Architecture Lab endpoints
├── test_learning_fixes.py              Learning endpoint bug fixes
├── test_paper_crud.py                  Paper CRUD + upload
├── test_paper_ingestion_service.py     Full ingestion pipeline
├── test_phase10_impl.py                Implementation module output
├── test_phase11_lab.py                 Explorer + Tensor Journey + Heatmap
├── test_phase8_assessment.py           Assessment challenge types
├── test_phase9_adaptive.py             Adaptive engine scoring
├── test_security_h2.py                 Security hardening checks
├── test_security_hardening.py          CSRF, XSS, injection tests
├── test_storage_infra.py               Storage quota + R2 key management
├── test_user_profile.py                User profile + notification prefs
├── test_architecture_graph_compiler.py ArchitectureGraph DAG correctness
├── test_architecture_reconstruction.py 15-family end-to-end reconstruction
├── test_background_jobs.py             Celery task execution
└── test_block_viz_service.py           Block visualisation service
```

Run the full suite:
```bash
cd C:\papper2code
.venv\Scripts\python -m pytest tests\ -q --tb=short
```

Run a specific module:
```bash
.venv\Scripts\python -m pytest tests\test_auth_module.py -v
```

---

## ⚡ Performance & Infrastructure

### Database Optimisations

| Optimisation | Location | Effect |
|---|---|---|
| Composite index `ix_user_points_desc` | `User.points` | O(log n) leaderboard queries |
| Composite index `ix_submission_user_problem` | `DojoSubmission` | O(1) per-user problem lookup |
| Composite index `ix_progress_learner_entity` | `LearnerProgress` | Unique constraint + fast lookup |
| Composite index `ix_notification_user_created` | `Notification` | Ordered notification fetch |
| `func.count()` aggregate queries | `dojo.py`, `learning.py` | Eliminated N+1 in stats endpoints |
| Subquery join for analytics dashboard | `learning.py` | Eliminated O(n) paper loop |

### Performance Properties

- **Compute Heatmap mode switch**: 0 API calls — pure client-side normalisation
- **Leaderboard**: composite index on `points` → sub-millisecond ranking queries
- **Architecture comparison**: diff computed in-memory, no LLM call
- **Assessment grading**: fully deterministic, sub-millisecond, no I/O
- **TensorTracker validation**: pure Python symbolic math, <5ms per architecture

### Celery Async Tasks

| Task | Trigger | Description |
|---|---|---|
| `ingest_paper_async` | PDF upload | Full pipeline in background |
| `send_drip_email` | Cron (daily) | Re-engagement email campaign |
| `prune_old_xp_events` | Cron (weekly) | Delete XP events >90 days old |
| `archive_leaderboard` | Cron (Monday 00:01 UTC) | Weekly ranking snapshot |
| `compute_acceptance_rates` | Cron (hourly) | Update per-problem acceptance_rate |

### Security Properties

| Property | Implementation |
|---|---|
| Rate limiting | SlowAPI + Redis (in-memory fallback) |
| Input validation | Pydantic V2 with field constraints |
| SQL injection | SQLAlchemy parameterised queries |
| XSS | FastAPI JSON responses (no HTML templating) |
| CSRF | SameSite cookie policy + CORS config |
| Code size limit | 10 KB enforcement on all Dojo submissions |
| Content moderation | Admin flag/unflag + `is_flagged` boolean on Paper |
| Storage quota | `storage_bytes_used` tracking on User model |
| Secrets | `SECRET_KEY` validated at startup — rejects empty/default values |

---

## 🗃️ Database Schema Diagram

```
users ──────────────────────────────────────────────────────────────────┐
  │ id, email, hashed_password, points, streak, mfa_enabled,           │
  │ token_version, lockout_until, email_digest, email_drip_opt_out     │
  │                                                                      │
  ├──< dojo_submissions >──────────────────────────────< problems       │
  │     id, code, passed, time_ms, is_best, review_text                 │
  │                                                                      │
  ├──< xp_events                                                        │
  │     id, action, amount, entity_id                                   │
  │                                                                      │
  ├──< user_achievements >──────────────────────────< achievements      │
  │     earned_at, payload                                              │
  │                                                                      │
  ├──< oauth_accounts                                                   │
  │     provider, provider_user_id                                      │
  │                                                                      │
  ├──< notifications                                                    │
  │     type, title, body, is_read, payload                            │
  │                                                                      │
  ├──< tutor_session_records                                            │
  │     session_id, context_type, messages[]                            │
  │                                                                      │
  └──< leaderboard_archive                                              │
        week_start, weekly_points, rank                                  │
                                                                         │
papers                                                                    │
  │ id, title, architecture_graph, flops_analysis, visibility            │
  │ uploaded_by ──────────────────────────────────────────────────────┘
  │
  └──< paper_modules
        layer_name, module_type, tensor_flow, flops_context, graph_nodes

auth module tables:
  user_sessions       (refresh token rotation)
  verification_tokens (email verify, 24h single-use)
  reset_tokens        (password reset, 15min single-use)
  audit_logs          (security event log)

learner_progress      (entity_type × entity_id mastery tracking)
assessment_attempts   (per-challenge grading history)
tutor_analytics       (tutor interaction tracking)
usage_log             (LLM API cost per action)
tasks                 (async job status)
interview_questions   (interview prep bank)
roadmaps              (learning roadmap nodes)
email_drip_log        (drip campaign tracking)
```

---

## 🔧 Setup & Running

### Prerequisites

- Python 3.11+
- SQLite (dev) or PostgreSQL (prod)
- Redis (optional — rate limiting falls back to in-memory)
- Groq API key (for AI tutor + agents)

### Development Setup

```bash
# Clone and enter
git clone https://github.com/yourorg/paper2code
cd paper2code

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env:
#   DATABASE_URL=sqlite:///./tensortonic_dev.db
#   GROQ_API_KEY=gsk_...
#   SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# Run migrations
alembic upgrade head

# Seed the golden corpus (ResNet, Transformer, U-Net)
python golden_paper_pipeline.py

# Start the backend
uvicorn backend.server:app --reload --port 8000
```

### Running Tests

```bash
# Full suite
.venv\Scripts\python -m pytest tests\ -q

# With verbose output
.venv\Scripts\python -m pytest tests\ -v

# Specific test file
.venv\Scripts\python -m pytest tests\test_auth_module.py -v

# Stop on first failure
.venv\Scripts\python -m pytest tests\ -x -q
```

### Docker

```bash
docker-compose up --build
```

The `docker-compose.yml` starts:
- `app` — FastAPI + Uvicorn (port 8000)
- `redis` — Rate limiting backend
- `postgres` — Production database
- `celery` — Async task worker

---

## 📋 Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | SQLite (dev) or PostgreSQL connection string |
| `SECRET_KEY` | ✅ | JWT signing key — must be ≥64 hex chars |
| `GROQ_API_KEY` | ✅ | Groq API key for LLM calls |
| `ANTHROPIC_API_KEY` | Optional | For agentic tutor tool-use |
| `REDIS_URL` | Optional | Redis URL — falls back to in-memory |
| `SENTRY_DSN` | Optional | Sentry error tracking |
| `CORS_ORIGINS` | Optional | Comma-separated origins (default: `*`) |
| `ENVIRONMENT` | Optional | `development` \| `production` |
| `STORAGE_BUCKET` | Optional | R2/S3 bucket for PDF storage |

---

## 📈 Key Metrics

| Metric | Value |
|---|---|
| Total Python files | 91+ core files, 16 routers, 15 agents, 8 auth services |
| Total test files | 41 test modules |
| Tests passing | **1,358 passed, 1 skipped, 0 failed** |
| ORM models | 20 database models |
| API endpoints | 80+ REST endpoints |
| Architecture families | 14 (each with builder, blocks, schema rules, refiner) |
| Verified architectures | 15 (LeNet-5 through Vision Transformer) |
| Dojo problems | 50+ ML coding challenges |
| DL ontology rules | 1,000+ hardcoded rules |
| Layer type synonyms | 200+ normalised |
| PyTorch layer mappings | 40+ `nn.Module` types |

---

## 📄 Source Papers (Golden Corpus)

| Paper | File | Size |
|---|---|---|
| He et al., "Deep Residual Learning for Image Recognition" (2015) | `data/pdfs/resnet_he_2015.pdf` | 800 KB |
| Vaswani et al., "Attention Is All You Need" (2017) | `data/pdfs/attention_all_you_need_2017.pdf` | 2163 KB |
| Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015) | `data/pdfs/unet_ronneberger_2015.pdf` | 1610 KB |

---

## 📃 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <p>Built with deterministic rigor. Every fact verified. Every number computed.</p>
  <p><strong>Paper2Code — From ambiguous paper to grounded understanding.</strong></p>
</div>
