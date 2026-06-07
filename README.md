<div align="center">
  <h1>🧠 Paper2Code</h1>
  <p><strong>Transforming Research Papers into Interactive Deep Learning Learning Experiences</strong></p>

  <p>Paper2Code is a full-stack interactive learning platform that automatically extracts, validates, and visualises deep learning architectures from research papers — then teaches users to understand them through a grounded AI tutor, assessments, adaptive learning paths, and compute-aware graph visualisation.</p>

  [![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
  [![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-red?style=flat-square)](https://www.sqlalchemy.org/)
  [![Cytoscape.js](https://img.shields.io/badge/Cytoscape.js-3.x-F7DF1E?style=flat-square)](https://js.cytoscape.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
  [![Status](https://img.shields.io/badge/Status-Phase%2011C%20Complete-6366F1?style=flat-square)]()
  [![Architectures](https://img.shields.io/badge/Architectures-15%20Verified-F59E0B?style=flat-square)]()
  [![Phases](https://img.shields.io/badge/Phases%20Complete-1--11C-10B981?style=flat-square)]()
</div>

---

## 🎯 What is Paper2Code?

Paper2Code is a **research-to-learning intelligence platform** that solves two problems at once: the reproducibility crisis in deep learning, and the learning barrier between papers and implementation.

It automatically extracts architectural specifications from research papers, validates them mathematically, and delivers them through an interactive web experience that teaches engineers *why* architectures are designed the way they are — not just *what* they look like.

### The Core Problem

Researchers and students face a critical bottleneck:
- **Ambiguity**: Papers describe architectures using inconsistent terminology and implicit assumptions
- **The gap**: The jump from "what we built" to "here's how to build it" leaves students guessing
- **Validation**: How do you know if your re-implementation matches the original?
- **Understanding**: Why did the authors make these design choices?

### Our Solution

Paper2Code eliminates this bottleneck through a deterministic pipeline:

```
Research Paper (ambiguous PDF)
         ↓
  Text Extraction + Section Splitting
         ↓
  Parsing Agent (grounded in DL Ontology)
         ↓
  TensorTracker Validation (symbolic forward pass)
         ↓
  Validated ArchitectureGraph (mathematically sound)
         ↓
  Interactive Explorer · AI Tutor · Assessments · Compute Heatmaps
         ↓
  Research Engineer: Educational PyTorch Code
```

**The core philosophy:** LLMs explain. Deterministic engines provide facts. The two never swap roles.

---

## 🚀 Why Paper2Code Was Built

### The Crisis

- **Reproducibility failure rate**: ~70% of deep learning papers have implementation ambiguities
- **Time cost**: Translating a single complex paper → working PyTorch code takes 2–4 weeks
- **Skill barrier**: Requires deep understanding of both ML theory AND implementation details
- **Learning gap**: Papers are written for researchers, not for learners

### Key Achievements

| Achievement | Detail | Status |
|:---|:---|:---:|
| **Deterministic KAG** | All architectural facts grounded in a hardcoded DL Ontology — zero hallucination of numerical data | ✅ |
| **TensorTracker** | Symbolic forward-pass validates `(B, C, H, W)` and `(B, N, D)` shapes across every layer before code is generated | ✅ |
| **Grounded AI Tutor** | 5 tutor modes that assemble context from the graph, tensor data, and FLOPs before calling the LLM | ✅ |
| **Deterministic Grading** | Assessments are graded by backend engines — the LLM never grades | ✅ |
| **Compute Heatmap** | Client-side FLOPs/Params/Memory node coloring, verified on 7 architectures, zero API calls on mode switch | ✅ |
| **15 Verified Architectures** | LeNet-5 through Vision Transformer, all with pre-validated graphs and tensor data | ✅ |

---

## ✨ Platform Features at a Glance

| Feature | Description | Status |
|---|---|:---:|
| **Architecture Library** | Browse 15 verified deep learning architectures with metadata | ✅ |
| **Paper Upload** | Upload any PDF and extract its architecture automatically | ✅ |
| **Architecture Parser** | Extract layer specs, hyperparameters, and connections from raw text | ✅ |
| **Architecture Graph** | Interactive Cytoscape.js graph with zoom, pan, node details, skip edges | ✅ |
| **Tensor Tracker** | Symbolic forward-pass validation across every layer | ✅ |
| **FLOPs Engine** | Per-layer FLOPs via closed-form formulas, cumulative totals, bottleneck identification | ✅ |
| **Parameter Engine** | Exact parameter count per module, memory footprint | ✅ |
| **Architecture Playground** | Build custom architectures block-by-block, validate them live | ✅ |
| **Architecture Comparison** | Side-by-side structural, FLOPs, and parameter diff between any two architectures | ✅ |
| **Grounded AI Tutor** | 5-mode tutor grounded in real graph/tensor/FLOPs data — not generic chat | ✅ |
| **Interactive Assessments** | 4 challenge types graded by deterministic backend logic | ✅ |
| **Adaptive Learning** | Knowledge profiling, weakness detection, concept graph, review plans | ✅ |
| **Research Engineer Mode** | Educational PyTorch code, pseudocode, training configs, cost estimation, reproduction cards | ✅ |
| **Architecture Explorer** | Stage timeline, module grouping, stage detail panels, graph navigation | ✅ |
| **Tensor Journey** | Per-stage tensor shape evolution with FLOPs/params per step, math/code toggles | ✅ |
| **Compute Heatmap** | Node coloring by FLOPs / Parameters / Memory, legend, node detail panel, stage compute summary | ✅ |

---

## ⚙️ How It Works — The Complete Pipeline

### Stage 1 — PDF Text Extraction (`main.py`)

- **Tool**: `pdfplumber` with fallback to `PyMuPDF/fitz`
- **Output**: Raw text sections from the paper
- **Why**: Papers come in different formats (scanned, embedded fonts, complex layouts)
- **Resilience**: Fallback strategy handles the vast majority of PDF types

### Stage 2 — Section Splitting (`core/rag/section_splitter.py`)

- **Input**: Raw paper text
- **Process**: Identifies and separates Abstract, Introduction, Architecture, Implementation, Experiments, Conclusion
- **Output**: Dictionary of named sections
- **Why**: Downstream parsers need to focus on the architecture section, not the introduction

### Stage 3 — Parsing Agent (`core/agents/`)

- **Input**: Architecture section text
- **Process**:
  - Extract layer specifications via `config_extractor.py`
  - Parse hyperparameters (kernel size, stride, channels, attention heads)
  - Build preliminary `ArchitectureGraph`
- **Output**: Initial graph with nodes and edges
- **Grounded in**: `core/rag/knowledge_graph.py` — 1,000+ hardcoded DL rules prevent invalid architectures

### Stage 4 — TensorTracker Validation (`core/rag/tensor_tracker.py`)

- **Purpose**: Mathematical validation engine — symbolic forward pass
- **Checks**:
  - Tensor shape compatibility `(B, C, H, W)` at every conv/pool
  - Multi-head attention divisibility (`embed_dim % num_heads == 0`)
  - Reshape operation element preservation
  - Skip connection dimension alignment
  - Concatenation compatibility
- **Output**: Validated or error-flagged graph

### Stage 5 — FLOPs & Parameter Analysis (`core/rag/flops_engine.py`)

- **Calculates**:

| Layer | Formula |
|---|---|
| Conv2d | `C_in × K × K × C_out × H_out × W_out` |
| Linear | `in_features × out_features` |
| Self-Attention | `4 × SeqLen × d_model² + 2 × SeqLen² × d_model` |
| DepthwiseSep | `(K² × C + C × C_out) × H × W` |

- **Output**: Per-layer FLOPs, cumulative totals, bottleneck identification

### Stage 6 — Explanation Generation (`core/rag/semantic_explainer.py`)

- **Input**: Validated ArchitectureGraph
- **Process**: Map nodes to educational explanations using the DL ontology
- **Output**: Why each layer was chosen, design trade-offs, design pattern explanations
- **Audience**: Students through to researchers

### Stage 7 — Knowledge Graph Grounding (`core/rag/knowledge_graph.py`)

- **Contains**: Hardcoded Deep Learning Ontology (1,000+ rules)
- **Prevents**: Invalid architectures from being generated or displayed
- **Handles**: Layer families, advanced blocks, architecture constraints, semantic roles
- **Result**: Semantically sound, grounded graph that cannot hallucinate facts

### Stage 8 — Code Generation (`core/codegen.py`)

- **Output**: Educational PyTorch `nn.Module` implementation
- **Includes**: Shape comments at every layer, design-decision docstrings, layer explanations
- **Ready**: For study, understanding, and adaptation

### Stage 9 — Interactive SPA (`static/index.html`)

- **Framework**: Vanilla JavaScript SPA, no build step
- **Features**: Library, Explorer, Tutor, Assessments, Playground, Comparison, Research Engineer Mode
- **Rendering**: Cytoscape.js graphs, Chart.js charts, Monaco Editor for code

---

## 🏆 Verified Architecture Corpus

15 architectures, all pre-validated with correct tensor shapes, FLOPs, and parameter counts:

| # | Architecture | Category | Params | Modules | Educational Focus |
|---|---|---|---|---|---|
| 1 | **LeNet-5** | CNN Pioneer | ~60K | 5 | Convolution fundamentals, average pooling |
| 2 | **AlexNet** | Deep CNN | ~61M | 8 | ReLU, dropout, GPU parallelism |
| 3 | **VGG16** | Very Deep CNN | ~138M | 16 | Depth through 3×3 uniformity |
| 4 | **VGG19** | Very Deep CNN | ~143M | 19 | Depth scaling limits |
| 5 | **GoogLeNet** | Inception | ~6.8M | 22 | Parallel paths, 1×1 bottleneck convolutions |
| 6 | **ResNet18** | Residual | ~11.7M | 8 | Skip connections, shallow residuals |
| 7 | **ResNet34** | Residual | ~21.8M | 16 | Scaling residual depth |
| 8 | **ResNet50** | Residual | ~25.5M | 16 | Bottleneck blocks, channel growth |
| 9 | **DenseNet121** | Dense | ~8M | 19 | Dense connections, feature reuse |
| 10 | **MobileNetV2** | Efficient | ~3.4M | 19 | Depthwise separable, inverted residuals |
| 11 | **EfficientNet-B0** | Compound Scaling | ~5.3M | 16 | Balanced depth/width/resolution |
| 12 | **FCN** | Segmentation | ~134M | 8 | Fully convolutional, semantic output |
| 13 | **U-Net** | Encoder-Decoder | ~31M | 22 | Symmetric decoder, skip connections |
| 14 | **Transformer** | Attention | ~65M | 34 | Self-attention, positional encoding |
| 15 | **Vision Transformer** | Attention + Patches | ~86M | 34 | Patch embedding, ViT scaling laws |

### Golden Paper Set — Source Papers

Three foundational papers are stored as PDFs and used as ground-truth validation sources:

**ResNet** (`data/pdfs/resnet_he_2015.pdf`, 800KB)
```
ResNet50 Structure:
├── Stem: Conv 7×7 stride-2, MaxPool 3×3 stride-2
├── Stage 1: 3× Bottleneck blocks  (64 → 256 channels)
├── Stage 2: 4× Bottleneck blocks  (128 → 512 channels)
├── Stage 3: 6× Bottleneck blocks  (256 → 1024 channels)
├── Stage 4: 3× Bottleneck blocks  (512 → 2048 channels)
└── Head: GlobalAvgPool → FC(1000)
Educational purpose: Skip connections, bottleneck design, channel growth, gradient flow
```

**Transformer** (`data/pdfs/attention_all_you_need_2017.pdf`, 2163KB)
```
Encoder structure:
├── Token Embedding (d_model=512)
├── Positional Encoding
└── 6× Encoder Layers:
    ├── Multi-Head Self-Attention (8 heads, d_k=64)
    ├── Add & LayerNorm
    ├── Feedforward (Linear 512→2048 → ReLU → Linear 2048→512)
    └── Add & LayerNorm
Educational purpose: Self-attention mechanism, positional encoding, scaling
```

**U-Net** (`data/pdfs/unet_ronneberger_2015.pdf`, 1610KB)
```
Encoder-Decoder structure:
├── Encoder (Contracting): 4× [Conv 3×3 → Conv 3×3 → MaxPool]
├── Bottleneck: Conv 3×3 → Conv 3×3
└── Decoder (Expanding): 4× [UpConv → Skip Concat → Conv 3×3 → Conv 3×3]
Educational purpose: Symmetric architecture, spatial preservation, skip connections
```

---

## 🤖 AI Tutor

The tutor is **not a generic chatbot**. Every response is grounded in the current architecture's actual extracted data before the LLM is called.

### Grounding Sources

| Source | Data Injected into Tutor Context |
|---|---|
| Architecture Graph | Layer types, connection topology, skip edges |
| Tensor Tracker | Input/output shapes at every layer |
| FLOPs Engine | Per-layer compute cost, stage totals, bottlenecks |
| Module Metadata | Design intent, hyperparameter rationale |
| Learning Profile | User's knowledge gaps, completed concepts |

The LLM generates explanations and narratives. The deterministic engines supply every numerical fact. If a user asks "How many FLOPs does Conv2 use?" — the FLOPs engine answers, not the LLM.

### Tutor Modes

| Mode | What the Tutor Knows | Activated From |
|---|---|---|
| **Module Tutor** | Single module: type, tensor shapes, FLOPs, design choices | Module detail page |
| **Architecture Tutor** | Full architecture: stages, overall design, comparisons | Library / Overview page |
| **Node Tutor** | Specific graph node: exact metrics, adjacent layers, data flow | Click node in Explorer |
| **Playground Tutor** | User's custom architecture: validity, suggested improvements | Architecture Playground |
| **Comparison Tutor** | Two-architecture diff: what changed, why it matters, tradeoffs | Comparison view |

---

## 📝 Interactive Assessments

All answers are validated by **deterministic backend logic**. The LLM is never used for grading.

### Challenge Types

| Type | What is Tested | How Graded |
|---|---|---|
| **Architecture Challenges** | Identify layers, explain connections, order stages, rank architectures | Exact match against stored graph data |
| **Tensor Shape Challenges** | Compute output shape after a given conv/pool/attention operation | Symbolic TensorTracker evaluation |
| **FLOPs Challenges** | Estimate or compare computational cost of layers or stages | Formula evaluation against ground truth |
| **Architecture Comparison Challenges** | Describe structural differences between two architectures | Structural diff against stored comparison data |

Challenge difficulty adapts to the user's knowledge profile. Incorrect answers contribute to concept weakness scoring, which drives the Adaptive Learning system.

---

## 🎓 Adaptive Learning

### Knowledge Profiling

Every interaction contributes to a persistent knowledge profile tracking mastery across 9 concept areas:

| Concept | Tracked Via |
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

### Adaptive Features

- **Weakness Detection** — Identifies concept areas with below-threshold mastery
- **Concept Graph** — Shows prerequisite relationships between concepts
- **Review Plans** — Generates ordered study plans targeting identified weaknesses
- **Adaptive Learning Paths** — Reorders architecture exposure based on current profile
- **Progress Dashboard** — Visual mastery indicators per concept, historical trends

---

## 🔬 Research Engineer Mode

Research Engineer Mode bridges education and implementation.

> **Important**: Generated implementations are educational references designed to be read and understood. They are not guaranteed production reproductions and should be reviewed before use in training pipelines.

### Features

| Feature | What It Produces |
|---|---|
| **Educational PyTorch** | Full `nn.Module` with shape comments at every layer and design-decision docstrings |
| **Pseudocode** | Architecture-level pseudocode for whiteboard and interview use |
| **Training Configuration** | Optimizer, learning rate schedule, batch size, augmentation from the paper |
| **Cost Estimation** | Estimated GPU-hours and memory requirements for training from scratch |
| **Reproduction Card** | Structured card of all hyperparameters needed to reproduce the paper's reported results |
| **Hyperparameter Guidance** | Explanation of each hyperparameter's role and typical search ranges |

### Architecture → Code Path

```
ArchitectureGraph (TensorTracker-validated)
          ↓
   Topological sort of GraphNodes
          ↓
   Layer type → nn.Module class mapping
          ↓
   Shape annotations injected from TensorTracker
          ↓
   Design comments from Knowledge Graph ontology
          ↓
   Educational PyTorch .py file
```

---

## 🗺️ Phase 11 — Explorer, Tensor Journey & Compute Heatmap

### Phase 11A — Architecture Explorer ✅

A dedicated deep-dive view for each architecture.

**Stage Timeline**: Architecture partitioned into 4 logical stages. Horizontal timeline with clickable cards; active stage highlighted, each card shows module count and connects with arrows.

**Stage Detail Panels** per stage show:
- Stage name, position (e.g. "Stage 2 of 4")
- Module count, total FLOPs, total parameters (metric cards)
- **Stage Compute Summary** card: total FLOPs, total parameters, highest-cost layer (by FLOPs)
- Tensor Journey summary (input shape → transform → output shape)
- Full module list with layer type annotations

**Architecture Graph Navigation**:
- Cytoscape.js renders full architecture in breadth-first order
- Node click → node detail panel opens below graph (FLOPs, params, memory, heatmap rank)
- Hover tooltip: label and type
- Skip edges: dashed purple lines

**Verified on**: LeNet-5, ResNet18, ResNet50, DenseNet121, U-Net, Transformer, Vision Transformer ✅

---

### Phase 11B — Tensor Journey ✅

Visualises how tensor shapes evolve through every module in each stage.

**Per-stage visualisation**:

| Element | What it Shows |
|---|---|
| **Input node** | Entry tensor shape (e.g. `[B, 3, 224, 224]`) |
| **Module step** | Output shape after each transformation |
| **FLOPs annotation** | MFLOPs or GFLOPs for the layer |
| **Params annotation** | K or M parameters |
| **⬆ indicator** | Channel expansion |
| **⬇ indicator** | Channel reduction |
| **↳ indicator** | Spatial downsampling |
| **→ indicator** | Shape unchanged |

**Math toggle**: Reveals `input → OP → output` notation inline for each step.

**Code toggle**: Reveals `x = layer_type(x)` pseudocode for each step.

**Shape data source**: `tensor_summary.trace[i].input_shape` / `trace[-1].output_shape` with fallback to top-level `input_shape` / `output_shape` — shapes always render, even when top-level fields are null.

| Architecture | Journey Steps | Shapes Verified |
|---|---|---|
| LeNet-5 | 15 | ✅ All populated |
| ResNet18 | 16 | ✅ All populated |
| ResNet50 | 16 | ✅ All populated |
| DenseNet121 | 19 | ✅ All populated |
| U-Net | 22 | ✅ All populated |
| Transformer | 34 | ✅ All populated |
| Vision Transformer | 34 | ✅ All populated |

---

### Phase 11C — Compute Heatmap ✅

Transforms the static architecture graph into a **compute-aware visualisation** so users can instantly identify expensive layers, parameter-heavy regions, and memory bottlenecks without reading tables.

**Heatmap Toggle** (four buttons above graph, all switches are client-side — zero API calls):

| Mode | Colors nodes by | Data Source |
|---|---|---|
| **None** | Uniform grey | — |
| **FLOPs** | `flops_context.real_flops_mflops` | Normalised per-architecture |
| **Parameters** | `flops_context.total_params_estimate` | Normalised per-architecture |
| **Memory** | `flops_context.activation_memory_mb` → fallback: output tensor shape × float32 | Normalised per-architecture |

**Color Scale**:

| Color | Percentile Band | Meaning |
|---|---|---|
| 🟢 Green | 0–25% | Low compute |
| 🟡 Yellow | 25–50% | Medium |
| 🟠 Orange | 50–75% | High |
| 🔴 Red | 75–100% | Very high |

**Legend**: Sidebar panel with four swatches + labels appears when any non-None mode is active.

**Node Detail Panel** (click any node):
- Layer name, FLOPs, Parameters, Memory
- Current heatmap metric value
- Relative rank (e.g., "Top 12%")
- Panel updates automatically when mode changes with node selected

**Stage Compute Summary** (in each stage detail panel):
- Total FLOPs for the stage
- Total parameters for the stage
- Highest-cost layer by FLOPs

**Verification Results:**

| Architecture | Nodes | FLOPs | Params | Memory | Node Detail | Stage Summary | Errors |
|---|---|---|---|---|---|---|---|
| LeNet-5 | 7 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| ResNet18 | 12 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| ResNet50 | 20 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| DenseNet121 | 11 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| U-Net | 18 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| Transformer | 32 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |
| Vision Transformer | 26 | ✅ | ✅ | ✅ | ✅ | ✅ | 0 |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           Browser                                │
│                     Vanilla JavaScript SPA                       │
│  Library · Explorer · Tutor · Assessments · Playground · Compare │
│        Cytoscape.js · Chart.js · Monaco Editor · FA 6           │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP / REST
┌────────────────────────────▼─────────────────────────────────────┐
│                       FastAPI Backend                            │
│   /api/papers · /api/modules · /api/progress · /api/assessments  │
│                /api/playground · /api/tutor                      │
└──────┬─────────────────────┬──────────────────────┬─────────────┘
       │                     │                      │
┌──────▼──────┐   ┌──────────▼───────┐   ┌─────────▼────────┐
│  SQLAlchemy │   │  Educational     │   │   LLM Client     │
│    ORM      │   │  Engines         │   │   (Groq API)     │
│             │   │                  │   │                  │
│  Papers     │   │  · Parser        │   │  Tutor modes     │
│  Modules    │   │  · Graph Builder │   │  grounded in     │
│  Progress   │   │  · TensorTracker │   │  deterministic   │
│  Profiles   │   │  · FLOPs Engine  │   │  engine output   │
│  Assessments│   │  · Assessment    │   │                  │
└─────────────┘   │  · Adaptive      │   └──────────────────┘
                  │  · Tutor         │
                  └──────────────────┘
```

### Educational Engines

| Engine | File | Responsibility |
|---|---|---|
| **Parser** | `core/rag/config_extractor.py` | PDF text → layer specs, hyperparameters, topology |
| **Graph Builder** | `core/architecture_graph.py` | Layer specs → validated ArchitectureGraph DAG |
| **TensorTracker** | `core/rag/tensor_tracker.py` | Symbolic forward-pass validation |
| **FLOPs Engine** | `core/rag/flops_engine.py` | Per-layer closed-form FLOPs computation |
| **Assessment Engine** | `core/assessment/engine.py` | Challenge generation and deterministic grading |
| **Adaptive Engine** | `core/analytics/adaptive_engine.py` | Knowledge profiling, weakness scoring |
| **Tutor Engine** | `core/agents/tutor_agent.py` | Grounding context assembly + LLM orchestration |

---

## 📂 Complete Project Structure & File Reference

### Root Level Files

```
paper2code/
├── main.py                    (47 lines)
├── app.py                     (1187 lines)
├── golden_paper_pipeline.py   (429 lines)
├── requirements.txt           (81 lines)
├── alembic.ini                (149 lines)
└── .env                       (5 lines)
```

**`main.py`** — Entry point for the standalone PDF text extraction CLI. Accepts a PDF path, runs `pdfplumber` extraction with `PyMuPDF` fallback, and outputs the raw extracted text. Used during corpus building and offline extraction workflows.

**`app.py`** — Legacy Streamlit application (1,187 lines). The original frontend before the SPA was built. Contains the Glassmorphism-styled UI, real-time graph exploration, bottleneck highlighting, and side-by-side comparison mode. Still runnable; the primary interface is now `static/index.html` served by FastAPI.

**`golden_paper_pipeline.py`** — Corpus builder pipeline (429 lines). Processes the three source PDFs in `data/pdfs/` through the full extraction pipeline, validates each architecture against the DL ontology and TensorTracker, and seeds them into the database. Run once to populate the golden corpus.

**`requirements.txt`** — All Python dependencies (81 entries). Covers FastAPI, SQLAlchemy, Alembic, pdfplumber, PyMuPDF, transformers, Groq SDK, Playwright, quickjs, and test utilities.

**`alembic.ini`** — Alembic migration configuration. Points at the `migrations/` directory and `DATABASE_URL` from the environment.

**`.env`** — Environment variables: `DATABASE_URL` and `GROQ_API_KEY`.

---

### `backend/` — API & Data Layer

```
backend/
├── server.py          (1514 lines)
├── database.py        (114 lines)
├── models.py          (168 lines)
├── corpus_builder.py  (373 lines)
├── repositories/
│   ├── __init__.py    (4 lines)
│   └── user_repository.py  (112 lines)
└── services/
    ├── __init__.py    (4 lines)
    └── user_service.py  (39 lines)
```

**`backend/server.py`** (1,514 lines) — The heart of the backend. FastAPI application with all REST endpoints. Handles:
- `GET /api/papers` — library listing with metadata
- `GET /api/papers/{id}` — full paper detail with architecture graph
- `GET /api/papers/{id}/modules` — all modules with `tensor_summary` and `flops_context`
- `POST /api/papers/upload` — PDF upload → extraction pipeline
- `POST /api/papers/{id}/tutor` — grounded tutor query (assembles context, calls LLM)
- `GET /api/papers/{a}/compare/{b}` — structural and metric comparison
- `GET /api/progress` — user learning profile
- `POST /api/progress/update` — record learning event
- `POST /api/assessments/{id}/submit` — grade a challenge response
- `POST /api/playground/validate` — TensorTracker validation for custom architectures
- Static file serving for `static/index.html` and assets
- CORS configuration, error handling, health endpoint

**`backend/database.py`** (114 lines) — SQLAlchemy engine and session management. Provides `get_db()` dependency for FastAPI route injection, handles `DATABASE_URL` parsing (SQLite dev / PostgreSQL prod), and creates all tables on startup if they don't exist.

**`backend/models.py`** (168 lines) — SQLAlchemy ORM models:
- `Paper` — title, authors, venue, year, pdf path, extracted text, architecture category, architecture graph (JSON), metadata
- `PaperModule` — per-module record linked to a Paper: layer name, module type, `tensor_summary` (JSON with trace data), `flops_context` (JSON with real FLOPs and param counts), graph nodes, graph edges
- `UserProgress` — concept mastery scores, assessment history, learning profile
- `Assessment` — challenge records with question, user answer, correct answer, graded result

**`backend/corpus_builder.py`** (373 lines) — Orchestrates building the golden corpus. Loads source PDFs, runs extraction for each architecture family, validates with TensorTracker, enriches with FLOPs and parameter data, and writes to the database. Also handles corpus metadata and stats aggregation.

**`backend/repositories/user_repository.py`** (112 lines) — Data access layer for user records. CRUD operations for progress profiles: fetch by user ID, update concept scores, log assessment results, retrieve historical performance.

**`backend/services/user_service.py`** (39 lines) — Business logic for user profile management. Thin service layer between the route handler and the repository. Handles profile creation for new users and score normalisation.

---

### `core/` — Intelligence Engine (Root)

```
core/
├── architecture_graph.py      (64 lines)
├── architecture_extractor.py  (162 lines)
├── codegen.py                 (145 lines)
├── module_generator.py        (548 lines)
├── paper_to_code_generator.py (271 lines)
├── model_builder.py           (60 lines)
├── llm_client.py              (85 lines)
├── metrics_estimator.py       (221 lines)
├── normalizer.py              (53 lines)
├── section_splitter.py        (112 lines)
├── schema.py                  (25 lines)
├── schemas_base.py            (34 lines)
├── utils.py                   (37 lines)
├── verify_model.py            (43 lines)
├── classification.py          (23 lines)
├── param_counter.py           (2 lines)
├── flops_estimator.py         (31 lines)
└── radar_chart.py             (141 lines)
```

**`core/architecture_graph.py`** (64 lines) — Foundational data structure that everything depends on. Defines:
- `GraphNode` — a single layer with id, label, type, hyperparameters, input/output shapes, FLOPs, params, and optional nesting for composite blocks
- `GraphEdge` — a connection between nodes with edge type (sequential, skip, residual, concat) and optional tensor shape annotation
- `ArchitectureGraph` — the full DAG with topological ordering, cycle detection, and serialisation to/from JSON. If this is wrong, everything downstream is wrong.

**`core/architecture_extractor.py`** (162 lines) — Extracts raw layer specifications from paper text. Uses regex patterns and keyword matching calibrated against the DL ontology to identify layer names, types, and hyperparameter values. Handles variable notation (`K` vs `kernel_size`, `C` vs `channels`, `d_model` vs `embed_dim`).

**`core/codegen.py`** (145 lines) — Transforms a validated `ArchitectureGraph` into educational PyTorch code. Iterates nodes in topological order, maps each `GraphNode` to its `nn.Module` class, injects tensor shape comments from TensorTracker output, and adds design-decision docstrings from the ontology.

**`core/module_generator.py`** (548 lines) — Generates detailed educational module records for each layer. For each `GraphNode`, produces: module description, design rationale, alternative choices, common mistakes, tensor shape annotations, FLOPs breakdown, and conceptual connections to related layers. Used to populate the Module Viewer page.

**`core/paper_to_code_generator.py`** (271 lines) — Grand orchestrator pipeline. Accepts a paper path, runs extraction → parsing → validation → explanation → code generation in sequence, handles errors at each stage, and returns the complete result object for storage.

**`core/model_builder.py`** (60 lines) — Selects and delegates to the correct family-specific builder (`resnet_builder`, `vit_builder`, `transformer_builder`, `unet_builder`) based on the detected architecture family. Acts as a router between generic extraction and family-specific construction.

**`core/llm_client.py`** (85 lines) — LLM integration layer. Wraps the Groq SDK with retry logic, token counting, prompt templating, and response parsing. Provides `generate_grounded_response(context, question)` — the only entry point for LLM calls in the tutor path.

**`core/metrics_estimator.py`** (221 lines) — High-level metrics estimation. Aggregates per-layer FLOPs and parameters across the full graph to produce summary statistics: total params, total FLOPs, FLOPs/param breakdown by stage, memory footprint estimate, and bottleneck layer identification.

**`core/normalizer.py`** (53 lines) — Standardises layer naming conventions across different papers. Maps paper-specific names (e.g., `"3x3 conv"`, `"convolutional layer"`, `"MHSA"`) to canonical ontology names (`"Conv2d"`, `"MultiheadAttention"`). Critical for cross-paper consistency.

**`core/section_splitter.py`** (112 lines) — Root-level section splitter (legacy). Splits raw PDF text into sections using heading detection. Superseded by `core/rag/section_splitter.py` for new workflows but retained for backward compatibility.

**`core/schema.py`** (25 lines) — Core Pydantic schema for the extraction pipeline output. Defines the top-level `PaperSchema` containing architecture family, graph, metadata, and validation status.

**`core/schemas_base.py`** (34 lines) — Base Pydantic models shared across all schema definitions. Contains common field types, validators, and the `LayerSpec` base class.

**`core/utils.py`** (37 lines) — Shared utility functions: text normalisation, JSON serialisation helpers, timestamp generation, and logging setup.

**`core/verify_model.py`** (43 lines) — Runs a PyTorch forward pass on generated code with a random input tensor to verify the model executes without errors. Used as a final sanity check in the Research Engineer Mode pipeline.

**`core/classification.py`** (23 lines) — Classifies an architecture into its family (CNN, Residual, Dense, Attention, Segmentation, Diffusion) based on graph structure and layer composition. Used for library categorisation and assessment routing.

**`core/radar_chart.py`** (141 lines) — Generates Plotly/Chart.js radar chart data for architecture comparison. Produces normalised scores across FLOPs, parameters, depth, width, and task performance for visual comparison overlays.

**`core/metrics_estimator.py`** — see above.

**`core/param_counter.py`** (2 lines) — Thin wrapper exposing `count_params(graph)` for backward compatibility. Delegates to `metrics_estimator.py`.

**`core/flops_estimator.py`** (31 lines) — Thin wrapper exposing `estimate_flops(graph)` for backward compatibility. Delegates to `core/rag/flops_engine.py`.

---

### `core/rag/` — Research-Augmented Generation (Intelligence Layer)

```
core/rag/
├── __init__.py          (54 lines)
├── knowledge_graph.py   (186 lines)
├── tensor_tracker.py    (366 lines)
├── config_extractor.py  (632 lines)
├── flops_engine.py      (354 lines)
├── diff_engine.py       (121 lines)
├── semantic_explainer.py (187 lines)
├── normalizer.py        (443 lines)
├── retriever.py         (183 lines)
├── section_splitter.py  (285 lines)
└── symbolic_parser.py   (132 lines)
```

**`core/rag/knowledge_graph.py`** (186 lines) — **The Ontology. The most critical file in the project.** Contains 1,000+ hardcoded Deep Learning rules that ground every extraction and prevent hallucination:
- Layer family definitions with valid hyperparameter ranges
- Architecture family constraints (e.g., ResNet max depth 152)
- Semantic role mappings (`"feature_extraction"`, `"spatial_reduction"`, `"classification"`)
- Forbidden layer combinations and required co-occurrence rules
- Cross-family compatibility rules (attention heads must divide embedding dimension)
If this file is wrong, the entire system will accept invalid architectures. It is the single source of DL truth.

**`core/rag/tensor_tracker.py`** (366 lines) — **Symbolic validation engine.** Simulates a forward pass tracking abstract tensor shapes without running actual PyTorch:
- Tracks `(B, C, H, W)` for convolutional architectures
- Tracks `(B, N, D)` for attention-based architectures
- Validates dimension compatibility at every layer transition
- Checks multi-head attention divisibility
- Verifies reshape element count preservation
- Detects skip connection shape mismatches
- Returns either a validated graph or a detailed error report locating the incompatibility
Runs before any code is generated, any module is displayed, or any assessment answer is accepted.

**`core/rag/config_extractor.py`** (632 lines) — **The parser.** Extracts structured layer configurations from unstructured paper text. Handles:
- Variable notation normalisation
- Regex-based hyperparameter extraction (kernel sizes, strides, channels, heads)
- Table parsing for architecture specifications
- Implicit default inference from the knowledge graph
- Multi-architecture paper disambiguation
This is the largest file in the RAG layer because parsing natural language architecture descriptions is inherently complex.

**`core/rag/flops_engine.py`** (354 lines) — **FLOPs computation engine.** Closed-form per-layer calculation:
- Conv2d, DepthwiseSeparable, Grouped Conv
- Multi-Head Attention (O(N² × d) complexity)
- Linear, BatchNorm, LayerNorm
- Pooling operations
Produces per-layer FLOPs, stage totals, cumulative totals, and bottleneck identification. Results stored in `flops_context.real_flops_mflops` per module.

**`core/rag/diff_engine.py`** (121 lines) — Architecture comparison engine. Accepts two `ArchitectureGraph` objects and produces:
- Structural diff (added/removed/changed nodes)
- Parameter delta per stage
- FLOPs differential with percentage change
- Semantic description of architectural changes
Used by the Comparison view and Comparison Tutor.

**`core/rag/semantic_explainer.py`** (187 lines) — Generates educational explanations for each layer. Maps `GraphNode` types to the knowledge graph, retrieves design rationale, alternative choices, and conceptual connections. Produces the explanations shown in the Module Viewer and used as tutor grounding context.

**`core/rag/normalizer.py`** (443 lines) — **Deep normalisation layer.** The most comprehensive normaliser in the project. Handles:
- 200+ layer name synonyms across different papers
- Hyperparameter unit conversion (pixels vs relative, absolute vs ratio)
- Architecture-specific notation (Transformer uses `d_model`, ViT uses `embed_dim`)
- Historical notation differences across decades of ML papers
Critical for making extractions from different papers comparable.

**`core/rag/retriever.py`** (183 lines) — Retrieves relevant context from the knowledge graph for a given query. Used by the tutor engine to assemble the grounding context before calling the LLM. Supports semantic similarity retrieval over the ontology rules.

**`core/rag/section_splitter.py`** (285 lines) — Advanced section splitter using heuristic heading detection, font-size signals (from pdfplumber metadata), and regex patterns to reliably split papers into named sections. Handles edge cases: unnumbered sections, appendices, multi-column layouts.

**`core/rag/symbolic_parser.py`** (132 lines) — Parses symbolic notation found in architecture papers: `R(3,4)×64` (residual block notation), `[Conv(3,64)] × 3` (repeated block notation), matrix dimension expressions like `W_q ∈ ℝ^{d×d_k}`. Converts to structured hyperparameter dicts.

---

### `core/agents/` — Agent System

```
core/agents/
├── __init__.py                (71 lines)
├── types.py                   (165 lines)
├── config_parser.py           (279 lines)
├── parsing_agent.py           (75 lines)
├── parsing_agent_impl.py      (74 lines)
├── explanation_agent.py       (155 lines)
├── explanation_agent_impl.py  (167 lines)
├── visualization_agent.py     (129 lines)
├── visualization_agent_impl.py (277 lines)
└── tutor_agent.py             (232 lines)
```

**`core/agents/types.py`** (165 lines) — **TypedDict contracts for all agents.** Defines strictly typed input and output dictionaries for every agent, preventing silent failures caused by missing or mistyped keys. Types include: `ParsingInput`, `ParsingOutput`, `ExplanationInput`, `ExplanationOutput`, `VisualizationOptions`, `TutorContext`, `TutorResponse`. These contracts ensure plug-and-play replaceability of any agent implementation.

**`core/agents/config_parser.py`** (279 lines) — Converts `ConfigDict` (structured extraction output) into a validated `ArchitectureGraph`. Handles: multi-stage architecture assembly, skip connection inference from paper text, composite block expansion, and default hyperparameter injection from the ontology.

**`core/agents/parsing_agent.py`** (75 lines) — Agent interface definition (abstract). Defines the `ParsingAgent` protocol: `parse(input: ParsingInput) -> ParsingOutput`. Any implementation that satisfies this interface can be swapped in without changing downstream code.

**`core/agents/parsing_agent_impl.py`** (74 lines) — Default parsing agent implementation. Wires together `section_splitter → config_extractor → config_parser` in the correct order, wraps errors in the typed output dict, and validates the output against the agent contract.

**`core/agents/explanation_agent.py`** (155 lines) — Agent interface for explanation generation. Defines `ExplanationAgent` protocol with `explain(input: ExplanationInput) -> ExplanationOutput`.

**`core/agents/explanation_agent_impl.py`** (167 lines) — Default explanation agent. Uses `semantic_explainer.py` to generate per-node explanations, then assembles them into a coherent narrative. Returns structured explanation data including per-layer rationale, design pattern identification, and connections to related architectures.

**`core/agents/visualization_agent.py`** (129 lines) — Agent interface for graph visualisation. Defines `VisualizationAgent` protocol with `visualise(graph, options) -> VisualizationOutput`.

**`core/agents/visualization_agent_impl.py`** (277 lines) — Default visualisation agent. Applies styling to the ArchitectureGraph for rendering: colour-coding by layer type, bottleneck highlighting, skip edge styling, Graphviz DOT generation for static diagrams, and Cytoscape.js element generation for the interactive frontend.

**`core/agents/tutor_agent.py`** (232 lines) — **Grounded tutor implementation.** The most important agent file for the AI Tutor feature:
- Assembles grounding context from the active architecture's graph, tensor data, FLOPs, and user profile
- Selects the correct tutor mode (Module / Architecture / Node / Playground / Comparison)
- Formats the context + user question into a grounding-first prompt
- Calls `llm_client.generate_grounded_response()`
- Strips any numerically incorrect claims from the LLM response before returning
The LLM never sees a question without its factual grounding context.

---

### `core/assessment/` — Assessment Engine

```
core/assessment/
├── __init__.py                  (8 lines)
├── engine.py                    (170 lines)
├── architecture_challenges.py   (338 lines)
├── tensor_challenges.py         (364 lines)
├── flops_challenges.py          (292 lines)
└── comparison_challenges.py     (296 lines)
```

**`core/assessment/engine.py`** (170 lines) — Assessment orchestrator. Routes incoming challenge submissions to the correct grader, manages difficulty progression, updates the user's knowledge profile after each answer, and returns graded results with explanations. All grading is deterministic — no LLM involved.

**`core/assessment/architecture_challenges.py`** (338 lines) — Architecture challenge bank. 100+ questions about layer identification, connection topology, stage ordering, and architecture ranking. Each challenge includes the correct answer derived from the stored `ArchitectureGraph` and a difficulty level (1–5).

**`core/assessment/tensor_challenges.py`** (364 lines) — Tensor shape challenge bank. Generates shape computation problems using TensorTracker output as ground truth. Questions include: "What is the output shape of Conv2d(in=64, out=128, k=3, stride=2) applied to a (B, 64, 56, 56) tensor?"

**`core/assessment/flops_challenges.py`** (292 lines) — FLOPs challenge bank. Uses `flops_engine.py` formulas to generate and grade estimation questions. Tests understanding of computational complexity for different layer types and architecture comparisons.

**`core/assessment/comparison_challenges.py`** (296 lines) — Comparison challenge bank. Generates questions about structural differences between architecture pairs using `diff_engine.py` output as ground truth. Tests understanding of architectural evolution and tradeoff reasoning.

---

### `core/analytics/` — Adaptive Learning Engine

```
core/analytics/
├── __init__.py               (7 lines)
├── adaptive_engine.py        (481 lines)
└── recommendation_engine.py  (222 lines)
```

**`core/analytics/adaptive_engine.py`** (481 lines) — **Knowledge profiling and weakness detection.** The largest analytics file:
- Maintains a concept mastery graph for each user across 9 concept areas
- Updates mastery scores after every assessment, tutor interaction, and page visit
- Detects weakness patterns using threshold-based scoring and historical trends
- Generates concept prerequisite ordering for adaptive learning paths
- Produces review plans prioritising the weakest concepts
- Tracks learning velocity (how fast mastery improves per session)

**`core/analytics/recommendation_engine.py`** (222 lines) — Recommends which architecture to study next based on the user's current profile. Uses the concept graph to identify the next most educational architecture given what the user already knows. Balances exploration (new concepts) with reinforcement (weak areas).

---

### `core/comparators/` — Architecture Comparison

```
core/comparators/
├── __init__.py               (25 lines)
├── architecture_comparator.py (237 lines)
└── comparison_explainer.py   (250 lines)
```

**`core/comparators/architecture_comparator.py`** (237 lines) — Structural comparison of two `ArchitectureGraph` objects. Produces: node diff (added/removed/changed), edge diff (new/removed connections), parameter delta per stage, FLOPs differential, depth/width comparison, and a normalised similarity score.

**`core/comparators/comparison_explainer.py`** (250 lines) — Generates human-readable narrative from a structural diff. Produces: "ResNet101 adds 17 Bottleneck blocks in Stage 3 compared to ResNet50, increasing FLOPs by 4.2B (+58%) with a potential accuracy gain on ImageNet." Used by the Comparison Tutor as grounding context.

---

### `core/explainers/` — Educational Explanation Generators

```
core/explainers/
├── __init__.py           (5 lines)
├── graph_explainer.py    (147 lines)
└── playground_insights.py (86 lines)
```

**`core/explainers/graph_explainer.py`** (147 lines) — Generates structured educational explanations for a complete architecture. For each stage, produces: stage purpose, key design decisions, computational profile, and connections to related architectures. Used to populate Architecture Tutor grounding context.

**`core/explainers/playground_insights.py`** (86 lines) — Generates real-time insights for custom architectures built in the Playground. Analyses the user's custom graph and produces: validity assessment, bottleneck warnings, suggestions for improvement, and comparisons to similar canonical architectures.

---

### `core/implementation/` — Research Engineer Mode

```
core/implementation/
├── __init__.py           (18 lines)
├── code_mapper.py        (622 lines)
├── cost_estimator.py     (203 lines)
├── reproduction_cards.py (337 lines)
└── training_config.py    (317 lines)
```

**`core/implementation/code_mapper.py`** (622 lines) — **The largest implementation file.** Maps every `GraphNode` type to its PyTorch `nn.Module` implementation. For each layer type, generates:
- The `nn.Module` class instantiation with correct hyperparameters
- Shape comment (input → output)
- Design-decision docstring from the ontology
- `forward()` method body with tensor flow comments
Covers 40+ layer types: Conv variants, attention types, normalisation layers, activation functions, pooling, embedding, etc.

**`core/implementation/cost_estimator.py`** (203 lines) — Estimates training cost for an architecture:
- GPU-hours for standard training runs based on FLOPs × dataset size × epochs
- Peak GPU memory (model weights + optimizer states + activations)
- Mixed-precision vs full-precision comparison
- Multi-GPU scaling estimates
Used to populate the Cost Estimation card in Research Engineer Mode.

**`core/implementation/reproduction_cards.py`** (337 lines) — Generates structured reproduction cards. For each architecture, assembles: all hyperparameters from the paper, training configuration, data augmentation, evaluation protocol, reported metrics, and common mistakes. Produces both JSON (for storage) and Markdown (for display).

**`core/implementation/training_config.py`** (317 lines) — Extracts and structures training configuration from paper text. Identifies: optimizer type and parameters, learning rate schedule, batch size, regularisation (weight decay, dropout), data augmentation pipeline, and evaluation protocol. Falls back to standard defaults from the ontology when not specified in the paper.

---

### `core/lab/` — Architecture Laboratory (Phase 12 Foundation)

```
core/lab/
├── __init__.py           (9 lines)
├── diff_engine.py        (176 lines)
├── hypothesis_engine.py  (283 lines)
├── mutator.py            (371 lines)
└── tradeoff_analyzer.py  (237 lines)
```

**`core/lab/diff_engine.py`** (176 lines) — Lab-specific diff engine with richer output than `core/rag/diff_engine.py`. Tracks: hypothesis-based diffs (what-if a layer was removed?), ablation study results, mutation impact predictions.

**`core/lab/hypothesis_engine.py`** (283 lines) — **Hypothesis testing for architectures.** Allows users to pose hypotheses like "What if ResNet50 had 8 stages instead of 4?" Evaluates the hypothesis using TensorTracker and FLOPs engine, predicts parameter and compute impact, and generates an explanation of the expected effect.

**`core/lab/mutator.py`** (371 lines) — **Architecture mutation engine.** Applies programmatic mutations to `ArchitectureGraph` objects: add/remove layers, change hyperparameters, swap connection types, insert bottleneck blocks. Each mutation is validated by TensorTracker before being returned. Used by the hypothesis engine and ablation simulator.

**`core/lab/tradeoff_analyzer.py`** (237 lines) — Analyses architectural tradeoffs. Given an architecture and a target (e.g., "reduce parameters by 30%"), suggests valid mutations and predicts their impact on FLOPs, accuracy, and memory. Produces a ranked list of tradeoff options.

---

### `core/orchestrator/` — High-Level Pipeline Orchestration

```
core/orchestrator/
├── __init__.py  (11 lines)
└── pipeline.py  (358 lines)
```

**`core/orchestrator/pipeline.py`** (358 lines) — The master pipeline coordinator. Accepts a PDF path or paper ID and orchestrates the complete extraction pipeline:
1. PDF extraction → section splitting
2. Config extraction → graph construction
3. TensorTracker validation
4. FLOPs and parameter analysis
5. Explanation generation
6. Code generation
7. Database storage

Handles errors at each stage, implements retry logic for LLM calls, and produces a structured result object with status and partial outputs if a stage fails.

---

### `core/builders/` — Architecture Family Builders

```
core/
├── transformer_builder.py  (215 lines)
├── unet_builder.py         (57 lines)
├── vit_builder.py          (83 lines)
├── ddpm_builder.py         (108 lines)
├── yolo_builder.py         (72 lines)
├── blocks_resnet.py        (51 lines)
├── blocks_transformer.py   (32 lines)
├── blocks_unet.py          (16 lines)
├── blocks_vit.py           (26 lines)
├── schema_rules_resnet.py      (29 lines)
├── schema_rules_transformer.py (19 lines)
├── schema_rules_unet.py        (10 lines)
├── schema_rules_vit.py         (33 lines)
├── schema_refiner.py           (45 lines)
├── schema_refiner_transformer.py (28 lines)
├── schema_refiner_unet.py      (42 lines)
├── schema_refiner_vit.py       (72 lines)
├── generate_code_ready_schema.py             (33 lines)
├── generate_code_ready_schema_transformer.py (31 lines)
├── generate_code_ready_schema_unet.py        (33 lines)
├── visualizer_resnet.py   (106 lines)
├── visualizer_unet.py     (115 lines)
└── visualizer_vit.py      (66 lines)
```

**`core/transformer_builder.py`** (215 lines) — Builds the `ArchitectureGraph` for Transformer architectures. Handles encoder/decoder variants, multi-head attention with configurable heads and d_k, positional encoding types, and cross-attention in decoder stacks. Validates attention head divisibility during construction.

**`core/unet_builder.py`** (57 lines) — Builds `ArchitectureGraph` for U-Net encoder-decoder architectures. Constructs symmetric encoder and decoder paths, wires skip connections between corresponding encoder/decoder levels, and handles bilinear vs transposed-conv upsampling variants.

**`core/vit_builder.py`** (83 lines) — Builds `ArchitectureGraph` for Vision Transformer architectures. Validates `(image_height % patch_size) == 0`, constructs patch embedding, CLS token prepend, positional encoding, and N transformer encoder blocks. Handles class token and distillation token variants.

**`core/ddpm_builder.py`** (108 lines) — Builds `ArchitectureGraph` for Denoising Diffusion Probabilistic Models. Handles the U-Net-with-time-conditioning structure: time embedding injection, residual blocks with timestep conditioning, and self-attention at multiple scales.

**`core/yolo_builder.py`** (72 lines) — Builds `ArchitectureGraph` for YOLO detection architectures. Handles the backbone + neck + head structure, feature pyramid connections, and multi-scale detection head output.

**`core/blocks_resnet.py`** (51 lines) — Standard ResNet block definitions used by the ResNet builder: `BasicBlock` (two 3×3 convs with skip), `Bottleneck` (1×1 → 3×3 → 1×1 with skip). Returns `GraphNode` subgraphs.

**`core/blocks_transformer.py`** (32 lines) — Transformer block definition: Multi-Head Attention → Add & Norm → FFN → Add & Norm. Returns a `GraphNode` subgraph for embedding in the full graph.

**`core/blocks_unet.py`** (16 lines) — U-Net double-conv block: Conv 3×3 → BN → ReLU → Conv 3×3 → BN → ReLU. Minimal but used throughout the U-Net builder.

**`core/blocks_vit.py`** (26 lines) — ViT transformer block: LayerNorm → MHSA → Add → LayerNorm → MLP → Add. Includes the pre-norm variant used in standard ViT implementations.

**`core/schema_rules_*.py`** — Family-specific validation rules applied after extraction. `schema_rules_resnet.py`: stage depth limits, channel doubling pattern. `schema_rules_transformer.py`: attention head/d_model divisibility. `schema_rules_unet.py`: encoder-decoder symmetry. `schema_rules_vit.py`: patch size divisibility, CLS token requirement.

**`core/schema_refiner_*.py`** — Post-extraction schema refinement. Fills missing hyperparameters with family-specific defaults, corrects common extraction errors (wrong stride, missing bias), and normalises representations to canonical form.

**`core/generate_code_ready_schema_*.py`** — Generates implementation-ready JSON schemas from validated `ArchitectureGraph` objects. The JSON schemas drive the code mapper in `core/implementation/code_mapper.py`.

**`core/visualizer_resnet.py`** (106 lines), **`visualizer_unet.py`** (115 lines), **`visualizer_vit.py`** (66 lines) — Family-specific Graphviz DOT generation. Each applies architecture-appropriate styling: ResNet highlights residual connections in purple, U-Net colours encoder and decoder paths differently, ViT groups attention heads visually.

---

### `core/diagram/` — Diagram Files

```
core/
├── diagram_base.py    (6 lines)
├── diagram_resnet.py  (23 lines)
├── diagram_unet.py    (33 lines)
├── diagram_vit.py     (25 lines)
└── generate_diagram.py (33 lines)
```

**`core/diagram_base.py`** (6 lines) — Base Graphviz rendering configuration: default node shape, font, DPI, and colour scheme.

**`core/diagram_resnet.py`**, **`diagram_unet.py`**, **`diagram_vit.py`** — Architecture-specific diagram style overrides applied on top of the base config.

**`core/generate_diagram.py`** (33 lines) — CLI entry point for offline diagram generation. Accepts an architecture ID, fetches from database, runs the appropriate visualiser, and saves PNG/SVG output.

---

### `core/run_*.py` — Offline Execution Scripts

```
core/
├── run_codegen.py              (11 lines)
├── run_transformer_codegen.py  (12 lines)
├── run_unet_codegen.py         (21 lines)
└── run_vit_codegen.py          (15 lines)
```

**`core/run_codegen.py`** — Runs the ResNet code generation pipeline from the command line. Useful for testing code output without starting the server.

**`core/run_transformer_codegen.py`** — Same for Transformer architecture.

**`core/run_unet_codegen.py`** — Same for U-Net. Slightly longer because U-Net requires explicit encoder/decoder path specification.

**`core/run_vit_codegen.py`** — Same for Vision Transformer.

---

### `static/` — Frontend SPA

```
static/
├── index.html    (4279 lines, 264KB)
├── app.js        (1602 lines, 67KB)
├── playground.js (307 lines, 15KB)
├── design.css    (719 lines, 17KB)
└── styles.css    (1164 lines, 25KB)
```

**`static/index.html`** (4,279 lines, 264KB) — **The entire frontend.** A single-file Vanilla JavaScript SPA with hash-based routing. Contains inline `<script>` with all page rendering functions:
- `renderLibraryPage()` — architecture library grid
- `renderExplorerPage(params)` — **Phase 11A/11B/11C**: stage timeline, tensor journey, compute heatmap, Cytoscape graph
- `renderOverviewPage(params)` — architecture overview with graph and module list
- `renderModulePage(params)` — single module deep-dive with tutor
- `renderComparePage()` — architecture comparison
- `renderPlaygroundPage()` — playground with custom architecture builder
- `renderUploadPage()` — PDF upload form
- `renderGraph(containerId, graphData, nodes, mode)` — Cytoscape.js graph renderer (shared)
- `selectStage(idx)` — Explorer stage switching
- `applyHeatmap(mode)` — **Phase 11C**: FLOPs/Params/Memory node coloring
- `showNodeDetail(nodeId)` — **Phase 11C**: node detail panel
- `toggleTensorMath()` / `toggleTensorCode()` — **Phase 11B**: tensor journey toggles

No build step. No framework. Served directly by FastAPI's `StaticFiles` mount.

**`static/app.js`** (1,602 lines, 67KB) — Additional JavaScript for complex interactions that were factored out of `index.html`. Handles: Monaco Editor initialisation and configuration, Chart.js radar chart rendering, advanced comparison overlay logic, and progress dashboard chart generation.

**`static/playground.js`** (307 lines, 15KB) — Architecture Playground logic. Manages: the block palette (draggable layer types), the custom graph canvas, live TensorTracker validation on block add/remove/connect, FLOPs/params live estimation, and playground tutor context assembly.

**`static/design.css`** (719 lines, 17KB) — Primary design system. CSS custom properties for the dark theme, colour palette (CNN blue, Residual green, Transformer purple, U-Net pink), component styles (`.card`, `.badge`, `.metric-card`, `.tensor-step`, `.stage-timeline-item`), and animation keyframes (`fadeIn`, `slideIn`).

**`static/styles.css`** (1,164 lines, 25KB) — Extended styles. Cytoscape.js tooltip positioning, Monaco Editor theme overrides, responsive grid breakpoints, heatmap legend styles, node detail panel styles, tensor journey scrollbar customisation, and loading spinner variants.

---

### `migrations/` — Database Migrations

```
migrations/
├── README          (1 line)
├── env.py          (84 lines)
├── script.py.mako  (28 lines)
└── versions/
    ├── 5ca65965e66d_initial_migration.py   (80 lines)
    └── b073274a0070_add_paper_and_papermodule.py (94 lines)
```

**`migrations/env.py`** (84 lines) — Alembic environment configuration. Connects to `DATABASE_URL` from the environment, sets up the migration context for both online (live) and offline (SQL script) modes.

**`migrations/script.py.mako`** (28 lines) — Template for generating new migration files via `alembic revision`.

**`migrations/versions/5ca65965e66d_initial_migration.py`** (80 lines) — Initial database schema: creates the `papers` table with title, authors, venue, year, pdf path, extracted text, architecture category, and architecture graph JSON.

**`migrations/versions/b073274a0070_add_paper_and_papermodule.py`** (94 lines) — Adds the `paper_modules` table with per-module records: layer name, module type, `tensor_summary` (JSON), `flops_context` (JSON), `graph_nodes` (JSON), `graph_edges` (JSON), and FK to `papers`.

---

### Root-Level Test Files

```
test_agent_interfaces.py          (289 lines)
test_architecture_comparator.py   (171 lines)
test_backward_compat.py           (7 lines)
test_comparator_edge_cases.py     (136 lines)
test_comparison_explainer.py      (200 lines)
test_comprehensive_features.py    (158 lines)
test_config_extractor.py          (139 lines)
test_config_extractor_enhancements.py       (247 lines)
test_config_extractor_hardened.py           (213 lines)
test_config_extractor_hardening_enhancements.py (276 lines)
test_config_extractor_refactor.py           (403 lines)
test_config_parser.py             (189 lines)
test_config_parser_hardened.py    (226 lines)
test_enhanced_app.py              (117 lines)
test_explainers.py                (47 lines)
test_flops.py                     (13 lines)
test_params.py                    (17 lines)
test_phase11_browser.py           (312 lines)
test_phase11_simple.py            (315 lines)
test_pipeline_determinism.py      (155 lines)
test_pipeline_enhancements.py     (290 lines)
test_pipeline_rag_integration.py  (227 lines)
test_resnet_vs_vit.py             (94 lines)
test_sim.py                       (17 lines)
test_single_arch_mode.py          (72 lines)
test_transformer_builder.py       (79 lines)
test_transformer_ops.py           (49 lines)
test_transformer_tracker_upgrade.py (123 lines)
test_vit_patch_embedding.py       (151 lines)
test_visual_comparison.py         (234 lines)
test_visual_features_complete.py  (146 lines)
```

**`test_agent_interfaces.py`** (289 lines) — Verifies every agent satisfies its TypedDict contract. Tests that `ParsingAgent`, `ExplanationAgent`, and `VisualizationAgent` accept valid inputs and return correctly typed outputs.

**`test_architecture_comparator.py`** (171 lines) — Tests `architecture_comparator.py`: structural diffs, parameter deltas, FLOPs differentials, edge diff for skip connections.

**`test_backward_compat.py`** (7 lines) — Smoke test confirming that legacy imports still resolve correctly after the monorepo migration.

**`test_comparator_edge_cases.py`** (136 lines) — Edge cases for the comparator: identical architectures (zero diff), single-layer architectures, architectures with different numbers of stages.

**`test_comparison_explainer.py`** (200 lines) — Tests `comparison_explainer.py` generates coherent narratives for known architecture pairs (ResNet18 vs ResNet50, ResNet50 vs ResNet101).

**`test_comprehensive_features.py`** (158 lines) — End-to-end integration test: PDF → extraction → graph → code generation → database storage, verified for all three golden papers.

**`test_config_extractor.py`** (139 lines) — Unit tests for `config_extractor.py`: layer type detection, hyperparameter extraction, table parsing.

**`test_config_extractor_enhancements.py`** (247 lines), **`test_config_extractor_hardened.py`** (213 lines), **`test_config_extractor_hardening_enhancements.py`** (276 lines) — Progressive hardening tests: malformed input, missing sections, ambiguous notation, non-English paper text.

**`test_config_extractor_refactor.py`** (403 lines) — Comprehensive regression suite for the config extractor after each refactor. 40+ test cases covering all layer types and all architecture families.

**`test_config_parser.py`** (189 lines), **`test_config_parser_hardened.py`** (226 lines) — Tests for `config_parser.py`: ConfigDict → ArchitectureGraph conversion, skip connection inference, multi-stage assembly.

**`test_pipeline_determinism.py`** (155 lines) — **Critical test**: verifies that running the same PDF through the pipeline twice produces bit-for-bit identical graphs. Any non-determinism here breaks reproducibility guarantees.

**`test_pipeline_enhancements.py`** (290 lines), **`test_pipeline_rag_integration.py`** (227 lines) — Integration tests covering the RAG layer's integration with the extraction pipeline: knowledge graph grounding, TensorTracker validation in-pipeline, FLOPs computation at extraction time.

**`test_phase11_browser.py`** (312 lines), **`test_phase11_simple.py`** (315 lines) — Playwright-based browser automation tests for Phase 11A/11B/11C. Verify Explorer loads, timeline is interactive, Cytoscape renders, tensor shapes populate, heatmap mode switches work, node detail panels appear, and stage compute summaries are present. Tested against all 7 primary architectures.

**`test_transformer_builder.py`** (79 lines), **`test_transformer_ops.py`** (49 lines), **`test_transformer_tracker_upgrade.py`** (123 lines) — Transformer-specific tests: builder output validation, attention operation shapes, TensorTracker handling of `(B, N, D)` shapes across multi-head attention.

**`test_vit_patch_embedding.py`** (151 lines) — Vision Transformer patch embedding tests: patch size divisibility validation, CLS token prepend shape, positional encoding shape.

**`test_visual_comparison.py`** (234 lines), **`test_visual_features_complete.py`** (146 lines) — Visual rendering tests: Cytoscape element generation, comparison overlay correctness, diagram DOT output format.

**`test_resnet_vs_vit.py`** (94 lines) — Cross-family comparison test: ResNet50 vs ViT-Base, verifying the diff engine handles fundamentally different topologies.

---

### Root-Level Validation Scripts

```
validate_cross_attention_events.py   (64 lines)
validate_flops_engine.py             (88 lines)
validate_kag_explanations.py         (84 lines)
validate_kag_reasoning_engine.py     (127 lines)
validate_tensor_tracker.py           (95 lines)
validate_transformer_details.py      (102 lines)
validate_vit_extraction.py           (96 lines)
validate_vit_tensor_evolution.py     (88 lines)
verify_phase11b.py                   (87 lines)
verify_phase11c.py                   (204 lines)
verify_db_setup.py                   (66 lines)
verify_postgres_setup.py             (129 lines)
verify_quality.py                    (78 lines)
```

**`validate_flops_engine.py`** (88 lines) — Verifies FLOPs calculations against hand-computed values for Conv2d, Linear, and Self-Attention. Catches regressions in the FLOPs engine formulas.

**`validate_tensor_tracker.py`** (95 lines) — Runs TensorTracker against all 15 architectures and confirms zero validation errors. Run before any release.

**`validate_kag_explanations.py`** (84 lines) — Checks that the knowledge-augmented explanations generated by `semantic_explainer.py` contain factually correct layer descriptions (cross-references against the ontology).

**`validate_kag_reasoning_engine.py`** (127 lines) — End-to-end validation of the KAG reasoning path: query → retriever → grounding context → response quality checks.

**`validate_transformer_details.py`** (102 lines) — Validates Transformer-specific extraction: attention head counts, d_k computation, layer norm placement.

**`validate_vit_extraction.py`** (96 lines) — Validates ViT extraction: patch size, embed dim, number of transformer blocks, CLS token presence.

**`validate_vit_tensor_evolution.py`** (88 lines) — Validates that the tensor shapes through a ViT forward pass evolve correctly from patch embedding through classifier head.

**`validate_cross_attention_events.py`** (64 lines) — Validates cross-attention in encoder-decoder Transformers: shape compatibility between encoder output and decoder attention.

**`verify_phase11b.py`** (87 lines) — Playwright script that navigated to each architecture's Explorer page and verified all tensor journey shapes are populated (not `?`). Confirmed Phase 11B fix across all 7 architectures.

**`verify_phase11c.py`** (204 lines) — Playwright script verifying Phase 11C Compute Heatmap: graph renders, heatmap buttons exist, legend appears on mode switch, node detail panel opens on click, stage compute summary visible, zero console errors.

**`verify_db_setup.py`** (66 lines) — Verifies database schema matches expected structure: tables exist, columns have correct types, FKs are configured.

**`verify_postgres_setup.py`** (129 lines) — Extended verification for PostgreSQL production setup: connection pooling, table partitioning, index existence, vacuum status.

**`verify_quality.py`** (78 lines) — General quality gate: checks that all 15 architectures have non-null graphs, all modules have `tensor_summary` and `flops_context`, and all FLOPs values are positive.

---

### Root-Level Audit & Diagnostic Scripts

```
audit.py           (67 lines)
audit_db.py        (51 lines)
audit_deep.py      (187 lines)
diagnose_startup.py (197 lines)
find_syntax_error.py (197 lines)
find_syntax_error_v2.py (106 lines)
find_unclosed_backtick.py (106 lines)
check_error.py     (78 lines)
check_page.py      (95 lines)
check_page2.py     (87 lines)
check_page3.py     (87 lines)
diag_flops.py      (15 lines)
unet_edge_check.py (30 lines)
```

**`audit.py`** (67 lines) — Quick audit of all papers in the database: checks completeness of graph, module count, FLOPs availability.

**`audit_db.py`** (51 lines) — Database-level audit: row counts per table, null field detection, orphaned records.

**`audit_deep.py`** (187 lines) — Deep audit: validates each stored graph against TensorTracker, checks FLOPs consistency, reports anomalies.

**`diagnose_startup.py`** (197 lines) — Playwright-based SPA startup diagnostic. Used during Phase 11 debugging to trace the browser startup sequence and identify JavaScript errors.

**`find_syntax_error.py`** / **`find_syntax_error_v2.py`** / **`find_unclosed_backtick.py`** — JavaScript syntax analysis scripts used during Phase 11 debugging to locate the escaped-backtick bug in `static/index.html`. Found and fixed the syntax error that was causing blank page rendering.

**`check_error.py`**, **`check_page.py`**, **`check_page2.py`**, **`check_page3.py`** — Playwright-based page state inspection scripts used during Phase 11 browser debugging.

**`diag_flops.py`** (15 lines) — Minimal FLOPs diagnostic: prints FLOPs for a specific module type to verify the engine formula.

**`unet_edge_check.py`** (30 lines) — Validates that U-Net's skip connection edges are correctly wired between encoder and decoder levels.

---

### Root-Level Benchmark Scripts

```
benchmark_bert_pipeline.py  (149 lines)
benchmark_gpt_pipeline.py   (211 lines)
benchmark_vit_pipeline.py   (199 lines)
```

**`benchmark_bert_pipeline.py`** (149 lines) — Benchmarks extraction pipeline performance on BERT-style architectures: measures extraction latency, memory usage, and output quality.

**`benchmark_gpt_pipeline.py`** (211 lines) — Same for GPT-style architectures. Larger because GPT architectures have more complex decoder-only attention patterns to handle.

**`benchmark_vit_pipeline.py`** (199 lines) — Same for ViT architectures. Includes patch size sweep to benchmark performance across ViT-Small, ViT-Base, ViT-Large.

---

### Root-Level Demo & Utility Scripts

```
demo_comparator.py    (32 lines)
demo_explainer.py     (42 lines)
inject_ui.py          (208 lines)
convert_md_to_docx.py (120 lines)
fix_imports.py        (44 lines)
run_all_visual_tests.py (46 lines)
run_phase_5_5_audit.py  (141 lines)
```

**`demo_comparator.py`** (32 lines) — Quick CLI demo of `architecture_comparator.py`: compares ResNet50 vs ResNet101 and prints the diff report.

**`demo_explainer.py`** (42 lines) — Quick CLI demo of `semantic_explainer.py`: generates and prints explanations for ResNet50.

**`inject_ui.py`** (208 lines) — Utility that injects updated UI component HTML/CSS into `static/index.html` without overwriting the entire file. Used during iterative frontend development.

**`convert_md_to_docx.py`** (120 lines) — Converts Markdown documentation files to Word format using `python-docx` for external sharing.

**`fix_imports.py`** (44 lines) — One-time import path fixer used after the monorepo migration to update relative imports across `core/`.

**`run_all_visual_tests.py`** (46 lines) — Runs the complete visual test suite in sequence and aggregates results.

**`run_phase_5_5_audit.py`** (141 lines) — Phase 5.5 audit runner: verifies library page, overview page, module viewer, and tutor endpoints all respond correctly.

---

### CI/CD

```
.github/workflows/
├── ci.yml  (58 lines)
└── cd.yml  (69 lines)
```

**`.github/workflows/ci.yml`** (58 lines) — Continuous Integration. On every push and PR: installs dependencies, runs the full test suite with `pytest`, runs `validate_tensor_tracker.py` and `validate_flops_engine.py`, reports coverage.

**`.github/workflows/cd.yml`** (69 lines) — Continuous Deployment. On merge to `main`: builds the application, runs database migrations, deploys to the staging environment, and runs smoke tests against the live deployment.

---

### Documentation Files

```
AGENT_SYSTEM_DESIGN.md           (927 lines, 32KB)
AGENT_INTERFACE_REFERENCE.md     (203 lines, 4KB)
TECHNICAL_MENTOR_MASTERCLASS.md  (3691 lines, 119KB)
DELIVERABLES_INDEX.md            (301 lines, 6KB)
PHASE_11_BROWSER_AUDIT_REPORT.md (302 lines, 8KB)
PHASE_11_VERIFICATION_AUDIT.md   (304 lines, 10KB)
PHASE_3_9_B_1_COMPLETE.md        (403 lines, 12KB)
README_COMPREHENSIVE.md          (434 lines, 16KB)
PROJECT_OVERVIEW.txt             (200 lines, 12KB)
docs/
├── PAPER2CODE_ENGINEERING_HANDBOOK.md  (1712 lines, 63KB)
└── PAPER2CODE_LEARNING_CURRICULUM.md   (4786 lines, 133KB)
```

**`AGENT_SYSTEM_DESIGN.md`** (927 lines) — Full design documentation for the multi-agent system. Covers agent contracts, communication patterns, error handling, and the rationale for strict TypedDict boundaries.

**`AGENT_INTERFACE_REFERENCE.md`** (203 lines) — API reference for all agent interfaces. Documents every TypedDict field, valid values, and example inputs/outputs for each agent.

**`TECHNICAL_MENTOR_MASTERCLASS.md`** (3,691 lines, 119KB) — The most comprehensive document in the project. A full technical deep-dive covering every subsystem, design decision, and implementation detail. Written as a masterclass for engineers who want to understand the system at depth.

**`DELIVERABLES_INDEX.md`** (301 lines) — Index of all completed deliverables by phase. Tracks what was promised, what was delivered, and verification status.

**`PHASE_11_BROWSER_AUDIT_REPORT.md`** (302 lines) — Browser verification report for Phase 11A/11B. Documents the Playwright test results, selector corrections, and pass/fail status for all 7 architectures.

**`PHASE_11_VERIFICATION_AUDIT.md`** (304 lines) — Detailed audit of Phase 11A/11B/11C implementation: what was verified, how it was verified, and the final verification results.

**`docs/PAPER2CODE_ENGINEERING_HANDBOOK.md`** (1,712 lines) — Engineering handbook for contributors. Covers: system design principles, how to add a new architecture, how to extend the knowledge graph, how to add new assessment challenges, coding standards.

**`docs/PAPER2CODE_LEARNING_CURRICULUM.md`** (4,786 lines) — The learning curriculum backed by this platform. Documents all 9 concept areas, their prerequisites, recommended architecture progression, and assessment coverage.

---

## 🛠️ Tech Stack

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| Vanilla JavaScript | ES2020 | SPA routing, DOM management, all interactions |
| Cytoscape.js | 3.x | Interactive architecture graph rendering |
| Chart.js | 4.x | Radar charts, progress charts, FLOPs bar charts |
| Monaco Editor | Latest CDN | Syntax-highlighted code display in Research Engineer Mode |
| Font Awesome | 6 | Icon set throughout the UI |
| CSS Custom Properties | — | Dark theme, colour palette, component system |

The frontend is a single `static/index.html` file (4,279 lines). No build step, no Node.js, no framework.

### Backend

| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.95+ | Async REST API, static file serving, CORS |
| SQLAlchemy | 2.0+ | ORM for SQLite (dev) and PostgreSQL (prod) |
| Alembic | — | Database migrations |
| Pydantic | 2.x | Request/response validation, TypedDict contracts |

### ML & AI Infrastructure

| Technology | Purpose |
|---|---|
| Groq API | Low-latency LLM inference for AI Tutor (grounded responses only) |
| HuggingFace Transformers | Local model support, tokenisation |
| PyTorch | Architecture validation (forward pass), code generation target |
| pdfplumber | Primary PDF text extraction |
| PyMuPDF (fitz) | Fallback PDF extraction for complex layouts |

### Testing & Verification

| Technology | Purpose |
|---|---|
| pytest | Unit and integration tests |
| Playwright | Browser automation for Phase 11 verification |
| quickjs | JavaScript syntax validation without Node.js |

---

## 📊 Project Status

### ✅ Completed Phases

| Phase | Name | Key Deliverables |
|---|---|---|
| **1** | Core Parsing Engine | PDF extraction, ArchitectureGraph, ResNet parser, TensorTracker |
| **2** | Monorepo & Builders | Unified schema, all model families, RAG layer, agent system |
| **3** | Learning Platform | Semantic explainer, FastAPI backend, database, Streamlit UI |
| **4** | Paper Upload | User PDF upload, extraction pipeline, metadata storage |
| **5** | Architecture Library UI | Library page, overview page, module detail pages |
| **6** | Golden Architecture Corpus | 15 verified architectures with ground-truth graphs |
| **7** | Grounded AI Tutor | 5 tutor modes, grounding pipeline, LLM integration |
| **7.5** | Learning Analytics | Progress tracking, concept mastery metrics, dashboard |
| **8** | Interactive Assessments | 4 challenge types, deterministic backend grading |
| **9** | Adaptive Learning | Knowledge profiling, weakness detection, review plans |
| **10** | Research Engineer Mode | PyTorch code gen, pseudocode, training configs, cost cards |
| **11A** | Architecture Explorer | Stage timeline, module grouping, graph navigation, stage detail panels |
| **11B** | Tensor Journey | Per-stage shape evolution, trace fallback fix, math/code toggles |
| **11C** | Compute Heatmap | FLOPs/Params/Memory coloring, legend, node details, stage compute summary |

---

## 🗺️ Roadmap

### Completed ✅

All phases through 11C are complete and browser-verified across 7 primary architectures.

```
Phase 1     ██████████  Core Parsing Engine             COMPLETE
Phase 2     ██████████  Monorepo & Builders             COMPLETE
Phase 3     ██████████  Learning Platform               COMPLETE
Phase 4     ██████████  Paper Upload                    COMPLETE
Phase 5     ██████████  Architecture Library UI         COMPLETE
Phase 6     ██████████  Golden Corpus (15 archs)        COMPLETE
Phase 7     ██████████  Grounded AI Tutor               COMPLETE
Phase 7.5   ██████████  Learning Analytics              COMPLETE
Phase 8     ██████████  Assessments                     COMPLETE
Phase 9     ██████████  Adaptive Learning               COMPLETE
Phase 10    ██████████  Research Engineer Mode          COMPLETE
Phase 11A   ██████████  Architecture Explorer           COMPLETE
Phase 11B   ██████████  Tensor Journey                  COMPLETE
Phase 11C   ██████████  Compute Heatmap                 COMPLETE
Phase 11D   ░░░░░░░░░░  Architecture Evolution Timeline PLANNED
Phase 11E   ░░░░░░░░░░  Visual Comparison Studio        PLANNED
Phase 11F   ░░░░░░░░░░  Learning Command Center         PLANNED
Phase 12    ░░░░░░░░░░  Architecture Laboratory         FUTURE
Phase 13    ░░░░░░░░░░  High-Fidelity Paper-to-Code     FUTURE
```

### Planned Phases

| Phase | Feature | Description |
|---|---|---|
| **11D** | Architecture Evolution Timeline | Visualise how a model family evolved from paper to paper (LeNet → AlexNet → VGG → ResNet) |
| **11E** | Visual Comparison Studio | Side-by-side Explorer views with linked heatmaps, tensor journeys, and live diff overlay |
| **11F** | Learning Command Center | Unified dashboard: progress across all phases, active sessions, weak concepts, recommended next steps |
| **12** | Architecture Laboratory | Hypothesis testing, ablation study simulator, metric impact predictor, architecture mutation |
| **13** | High-Fidelity Paper-to-Code | Verified PyTorch implementations with automated correctness checks against paper benchmarks |

---

## 💻 Installation & Setup

### Prerequisites

- Python 3.8+
- pip
- SQLite (built-in, default) or PostgreSQL for production

### 1. Clone

```bash
git clone https://github.com/officialpk956-wq/paper2code.git
cd paper2code
```

### 2. Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Environment Variables

Create `.env` in the project root:

```env
# SQLite (development — no extra setup required)
DATABASE_URL=sqlite:///./paper2code.db

# Required for AI Tutor
GROQ_API_KEY=your_groq_api_key_here

# PostgreSQL (production)
# DATABASE_URL=postgresql://user:password@localhost/paper2code
```

### 5. Database Initialisation

```bash
alembic upgrade head
python golden_paper_pipeline.py   # Seed the 15 verified architectures
```

### 6. Start the Server

```bash
python backend/server.py
# App:  http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
```

The SPA is served from `static/index.html` by FastAPI. No separate frontend server needed.

### 7. Run Tests

```bash
pytest -v
python validate_tensor_tracker.py
python validate_flops_engine.py
python verify_phase11c.py   # Phase 11C browser verification (requires Playwright)
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/papers` | GET | Library listing with metadata, category, param count |
| `/api/papers/{id}` | GET | Full paper detail with architecture graph |
| `/api/papers/{id}/modules` | GET | All modules with `tensor_summary` and `flops_context` |
| `/api/papers/upload` | POST | Upload PDF → extraction pipeline → store |
| `/api/papers/{id}/tutor` | POST | Grounded tutor query (assembles context, calls LLM) |
| `/api/papers/{a}/compare/{b}` | GET | Structural and metric comparison |
| `/api/progress` | GET | User learning profile and concept mastery scores |
| `/api/progress/update` | POST | Record learning event (page visit, correct answer) |
| `/api/assessments/{id}/submit` | POST | Submit and deterministically grade a challenge |
| `/api/playground/validate` | POST | TensorTracker validation for a custom architecture |

Full interactive docs at `http://127.0.0.1:8000/docs`.

---

## 🖼️ Architecture Explorer Screenshots

### Library View

```
┌──────────────────────────────────────────────────────────────────┐
│ 📚 Paper2Code Library                            🔍 Search       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🔵 ResNet50                    🟢 DenseNet121                   │
│     Deep Residual Learning...      Densely Connected CNNs...     │
│     25.5M params · 7.3G FLOPs     8M params · 2.9G FLOPs        │
│     [Explore] [Tutor] [Code]       [Explore] [Tutor] [Code]      │
│                                                                  │
│  🟣 Transformer                 🩷 U-Net                          │
│     Attention Is All You Need      Convolutional Networks...     │
│     65M params · 11B FLOPs        31M params · 54G FLOPs         │
│     [Explore] [Tutor] [Code]       [Explore] [Tutor] [Code]      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Architecture Explorer — Stage Timeline + Compute Heatmap

```
┌──────────────────────────────────────────────────────────────────┐
│ ARCHITECTURE EXPLORER   ResNet50                                 │
│ He, Zhang, Ren, Sun                                              │
├──────────────────────────────────────────────────────────────────┤
│ Stage Progression                                                │
│ [▶ Stem] → [Stage 1] → [Stage 2] → [Classifier]                 │
├──────────────────────────────────────────────────────────────────┤
│ Stage: Stage 2               │ Architecture Graph                │
│                              │ Heatmap: [None][FLOPs][Params][Mem]│
│ ⚡ Modules: 4                │                                    │
│ ⚡ FLOPs: 1.2 GFLOPs         │  [Stem]──>[Stage1]──>[Stage2] 🔴  │
│ 📊 Parameters: 3.4 M         │             │                     │
│                              │           [Skip]                  │
│ 🔥 Stage Compute Summary     │             │                     │
│  Total FLOPs: 1.2 GFLOPs    │  [Stage3]──>[Stage4]──>[Head]     │
│  Total Params: 3.4 M         │                                    │
│  Highest: Bottleneck_3       │  ┌──────────────────────────────┐ │
│                              │  │ Layer: Bottleneck_3           │ │
│ Tensor Journey               │  │ FLOPs: 318.8 MFLOPs          │ │
│ [B,64,56,56]→Conv→[B,128,..]│  │ Params: 526K                 │ │
│                              │  │ Memory: 25.6 MB               │ │
│                              │  │ Relative Rank: Top 8%         │ │
│                              │  └──────────────────────────────┘ │
│                              │ ● Green = Low  ● Red = Very High  │
└──────────────────────────────┴───────────────────────────────────┘
```

### Comparison View

```
┌──────────────────────┬───────────────────────────────────────────┐
│ ResNet50             │ ResNet101                                 │
├──────────────────────┼───────────────────────────────────────────┤
│ Stage 1: 3× Blocks   │ Stage 1: 3× Blocks   (same)             │
│ Stage 2: 4× Blocks   │ Stage 2: 4× Blocks   (same)             │
│ Stage 3: 6× Blocks   │ Stage 3: 23× Blocks  [+17 🔴 +4.2G FLOPs]│
│ Stage 4: 3× Blocks   │ Stage 4: 3× Blocks   (same)             │
│ Params: 25.5M        │ Params: 44.5M        [+19M, +74%]        │
│ FLOPs:  7.3B         │ FLOPs:  11.5B        [+4.2B, +58%]       │
└──────────────────────┴───────────────────────────────────────────┘
💡 ResNet101 adds depth in Stage 3 only, targeting richer feature
   representations at the cost of 74% more parameters.
```

---

## 🎯 Design Principles

**1. Determinism over generation.**
Every tensor shape, FLOPs count, and parameter total is computed by a deterministic engine. The LLM never invents numerical facts.

**2. Grounding is mandatory.**
The AI Tutor cannot respond without a grounding context built from the architecture's actual data. Ungrounded responses are blocked at the API layer.

**3. Validation before output.**
TensorTracker runs before any code is generated, any module is displayed, or any assessment answer is accepted. Impossible architectures are caught early.

**4. Education over perfection.**
Generated PyTorch code is optimised for readability and understanding. Shape comments and design notes take priority over micro-benchmark performance.

**5. Client-side performance.**
Heatmap mode switches, tensor journey rendering, and graph interactions run in the browser. API calls happen once on page load; subsequent interactions are local computation.

**6. Strict agent contracts.**
Every agent communicates through TypedDict contracts. Silent failures caused by missing or mistyped fields are impossible. Any implementation can be swapped out without changing downstream code.

---

## 🤝 Contributing

Contributions are welcome. High-impact areas:

1. **New architecture support** — Add a builder in `core/` (see `vit_builder.py` as a template), schema rules, a `schema_refiner`, and a ground-truth entry in the corpus builder
2. **Assessment challenges** — Extend any of the four challenge files in `core/assessment/`
3. **Knowledge Graph expansion** — Add rules to `core/rag/knowledge_graph.py`
4. **Tutor grounding improvements** — Enhance context assembly in `core/agents/tutor_agent.py`
5. **Test coverage** — Additional integration tests in root-level `test_*.py` files

Please open an issue before submitting a large PR. The agent contract boundaries are intentional — discuss changes to `core/agents/types.py` before implementing them.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built to make deep learning architecture education rigorous, interactive, and accessible.

**[Open an Issue](https://github.com/officialpk956-wq/paper2code/issues)** · **[API Docs](http://127.0.0.1:8000/docs)** · **[Explore the Platform](http://127.0.0.1:8000)**

<br/>

*15 verified architectures · Phases 1–11C complete · Deterministic grounding throughout*

</div>
