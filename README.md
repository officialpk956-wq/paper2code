# paper2code

<div align="center">

# 🧠 paper2code

### Transform Deep Learning Research Papers into Structured Architectures, Executable Code, and Interactive Visualizations

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)]()
[![Frontend](https://img.shields.io/badge/UI-Streamlit-success.svg)]()
[![Architecture](https://img.shields.io/badge/System-Graph%20Reasoning-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

</div>

---

# 📌 Overview

**paper2code** is a research-to-implementation toolkit focused on transforming deep learning research papers into:

- structured architecture schemas
- executable PyTorch-style implementations
- semantic architecture graphs
- interactive visualizations
- explainable architectural representations

The project bridges the gap between:

> **Research Papers → Architecture Understanding → Executable Systems**

Unlike traditional “paper-to-code” generators, paper2code prioritizes:

- deterministic reasoning
- tensor correctness
- semantic understanding
- architecture explainability
- educational transparency
- graph-based validation

---

# 🚀 Core Objectives

## 🔁 Reproducibility

Research papers often contain:

- ambiguous architecture descriptions
- incomplete implementation details
- inconsistent diagrams
- undocumented tensor assumptions

paper2code converts these into:

- deterministic schemas
- validated tensor flows
- executable implementations
- structured semantic graphs

---

## 🎨 Visualization

Generate architecture diagrams featuring:

- semantic highlighting
- bottleneck visualization
- tensor-aware overlays
- graph-based rendering
- architecture comparison systems

---

## 📊 Analysis

Automatically perform:

- FLOPs estimation
- parameter counting
- tensor propagation
- structural validation
- compatibility checking
- topology analysis

---

## 🔍 Comparison

Enable side-by-side architecture comparison with:

- semantic graph comparison
- automated explanations
- bottleneck identification
- tensor compatibility analysis
- visual highlighting systems

---

# 🏗️ System Architecture

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Frontend | Streamlit |
| Backend API | FastAPI |
| Deep Learning Backend | PyTorch |
| PDF Parsing | pdfplumber, PyMuPDF |
| Visualization | Graphviz, SVG |
| Graph Engine | Custom Semantic Architecture Graph |
| Validation | Tensor Tracking + Schema Validation |
| Testing | PyTest |

---

# 🛠️ Core Technological Pillars

---

## 1️⃣ Semantic Architecture Graph

📍 `src/architecture_graph.py`

The **Semantic Architecture Graph** is the conceptual core of paper2code.

Unlike traditional computational graphs that focus purely on operations, this graph preserves:

- architectural intent
- semantic meaning
- tensor relationships
- structural hierarchy

### Semantic Role Examples

Each node is tagged with semantic roles such as:

- `token_mixer`
- `patch_embedding`
- `residual_block`
- `attention_projection`
- `encoder_stage`

This allows the system to preserve not only *how* a model works, but *why* each component exists.

### Why This Matters

This graph powers:

- code generation
- explainability
- tensor validation
- architecture comparison
- educational visualization

---

## 2️⃣ KAG-Powered Explanation System

📍 `src/rag/`

paper2code uses a **Knowledge-Augmented Generation (KAG)** approach to eliminate hallucinated explanations.

### Components

#### Knowledge Graph

A structured ontology of deep learning primitives including:

- attention
- convolutions
- normalization
- tokenization
- encoder-decoder structures

---

#### Semantic Explainer

A deterministic explanation engine that maps graph nodes to pedagogically accurate explanations.

Example:

> "This layer converts spatial image patches into token embeddings before Transformer processing."

---

#### Educational Context System

Hovering over a node in the UI provides:

- mathematically grounded explanations
- semantic descriptions
- architecture-aware context
- tensor-flow interpretation

### Goal

Transform architecture exploration into an educational reasoning experience.

---

## 3️⃣ Strict Tensor Flow Validation

📍 `src/rag/tensor_tracker.py`

To prevent hallucinated architectures that cannot execute, paper2code performs symbolic tensor propagation across the graph.

### Features

- shape propagation
- sequence-aware tensor tracking
- residual compatibility checking
- tensor dimensionality enforcement
- topology validation

### Transformer-Specific Support

Optimized specifically for:

- Vision Transformers (ViT)
- Transformer encoder-decoder systems
- token sequence architectures

### Validation Rules

The TensorTracker validates:

- MHSA head divisibility
- positional embedding alignment
- residual path connectivity
- sequence compatibility
- tensor rank consistency

This ensures architectures are structurally executable before code generation.

---

# 🧠 High-Level Pipeline

```text
Research Paper PDF
        ↓
PDF Text Extraction
        ↓
Semantic Section Splitting
        ↓
Architecture Parsing
        ↓
Semantic Graph Construction
        ↓
Schema Refinement & Validation
        ↓
Tensor Propagation
        ↓
Code Generation
        ↓
Diagram Generation
        ↓
Analysis & Explanation
        ↓
Interactive Streamlit UI
```

---

# 🔄 Detailed Pipeline Flow

## 1️⃣ Input Layer

The system accepts:

- research paper PDFs
- extracted raw text
- architecture descriptions

### Supported Architectures

| Architecture | Status |
|---|---|
| ResNet | ✅ Stable |
| U-Net | ✅ Stable |
| Vision Transformer (ViT) | ✅ Hardened |
| Transformer | ✅ Stable |

---

## 2️⃣ Extraction Layer

PDF content is extracted using:

- `pdfplumber`
- `PyMuPDF`

The extracted content is split into:

- semantic sections
- architecture blocks
- implementation-relevant segments

---

## 3️⃣ Parsing Layer

The parser converts extracted text into:

- raw model specifications
- architecture graphs
- normalized structural representations

This layer identifies:

- layers
- skip connections
- tensor transformations
- attention blocks
- encoder-decoder structures

---

## 4️⃣ Semantic Graph Construction

Architectures are internally represented as semantic graphs.

These graphs encode:

- tensor flow
- topology
- computational structure
- semantic intent
- module relationships

---

## 5️⃣ Tensor Validation

The TensorTracker validates:

- shape propagation
- transformer token dimensions
- residual compatibility
- pooling consistency
- sequence semantics

This prevents invalid architectures before executable code generation.

---

## 6️⃣ Schema Refinement

Architecture-specific rules ensure:

- dimensional consistency
- residual correctness
- attention compatibility
- topology validity
- graph consistency

---

## 7️⃣ Code Generation

The refined schema is converted into:

- executable PyTorch-style code
- deterministic architecture builders
- reusable model modules

Supported generation includes:

- ResNet builders
- U-Net builders
- Vision Transformer builders
- Transformer pipelines

---

## 8️⃣ Visualization

The visualization engine generates:

- architecture diagrams
- semantic overlays
- bottleneck highlights
- comparison graphs
- tensor-aware visualizations

---

## 9️⃣ Interactive UI

The frontend enables:

- architecture exploration
- tensor inspection
- graph visualization
- semantic explanations
- side-by-side comparison

---

# 📂 Directory Structure Highlights

```text
paper2code/
│
├── app.py                         # Main Streamlit application
├── server.py                      # FastAPI bridge for frontend/backend communication
│
├── src/
│   │
│   ├── agents/                    # Interface-driven agent architecture
│   │   ├── parsing_agent.py
│   │   ├── visualization_agent.py
│   │   ├── explanation_agent.py
│   │   └── config_parser.py
│   │
│   ├── rag/                       # Intelligence Layer
│   │   ├── retriever.py
│   │   ├── knowledge_graph.py
│   │   ├── semantic_explainer.py
│   │   └── tensor_tracker.py
│   │
│   ├── comparators/               # Architecture comparison engine
│   ├── explainers/                # Semantic explanation systems
│   ├── orchestrator/              # Pipeline orchestration logic
│   │
│   ├── architecture_graph.py      # Semantic graph engine
│   ├── codegen.py                 # Code generation engine
│   ├── model_builder.py           # Model construction
│   ├── flops_estimator.py         # FLOPs analysis
│   ├── param_counter.py           # Parameter estimation
│   └── verify_model.py            # Validation utilities
│
├── static/                        # Interactive glassmorphism frontend UI
├── templates/                     # Visualization templates
├── outputs/                       # Generated outputs
├── docs/                          # Documentation
├── tests/                         # Validation and testing suite
├── notebooks/                     # Research notebooks
├── experiments/                   # Experimental workflows
├── data/                          # Research papers
└── models/                        # Model artifacts
```

---

# 🔥 Major Achievements

## ✅ Visual Comparison Engine

Implemented a complete architecture comparison framework featuring:

- side-by-side rendering
- semantic highlighting
- synchronized graph comparison
- bottleneck detection
- ghost overlays

---

## ✅ Vision Transformer (ViT) Hardening

The project is currently finalizing a hardened Vision Transformer pipeline.

### Completed

- patch embedding extraction
- CLS token insertion logic
- MHSA topology extraction
- tensor-aware propagation
- residual validation
- executable ViT generation

### Recently Fixed

- alignment between `ArchitectureGraph`
- compatibility with `PaperToCodeGenerator`
- resolution of 500 API generation errors

### Current Focus

End-to-end validation of the explanation system to ensure:

- every ViT component is semantically mapped
- educational explanations are deterministic
- tensor semantics are preserved

---

## ✅ Tensor Tracking System

Built a symbolic tensor propagation engine capable of:

- detecting residual mismatches
- validating tensor compatibility
- enforcing sequence correctness
- preventing execution-time structural failures

---

## ✅ Multi-Agent Architecture

Implemented a modular agent architecture.

| Agent | Responsibility |
|---|---|
| ParsingAgent | Extract architecture information |
| VisualizationAgent | Generate diagrams and overlays |
| ExplanationAgent | Produce semantic explanations |

---

# 🧪 Testing & Validation

The project includes comprehensive validation suites covering:

- tensor propagation
- schema refinement
- graph consistency
- pipeline determinism
- architecture comparison
- Vision Transformer validation
- executable code correctness

## Run Tests

```bash
pytest
```

## Run Visual Validation

```bash
python run_all_visual_tests.py
```

## Run ViT Benchmark

```bash
python benchmark_vit_pipeline.py
```

---

# 🚀 Development Roadmap

---

## Phase 1 — Strong Architecture Foundations

### Goal

Build a reliable architecture extraction and validation engine focusing on:

- structured extraction
- semantic graphs
- tensor tracking
- deterministic validation

### Current Focus

- ResNet
- U-Net
- Vision Transformer

### Status

- ✅ ResNet Stable
- ✅ U-Net Stable
- ✅ Vision Transformer Hardened

---

## Phase 2 — KAG Expansion

### Goal

Introduce Knowledge-Augmented Generation (KAG) and GraphRAG reasoning.

### Planned Features

- architecture ontologies
- graph reasoning
- semantic validation
- graph-enhanced retrieval
- constraint-aware generation

---

## Phase 3 — Interactive Visualization

### Goal

Transform the platform into an architecture exploration environment.

### Planned Features

- clickable architecture graphs
- tensor tracing
- FLOP hotspot visualization
- topology exploration
- semantic graph navigation

---

## Phase 4 — Educational Sandbox

### Goal

Create an interactive architecture learning environment.

### Planned Features

- tensor mismatch debugging
- topology repair exercises
- optimization challenges
- architecture reconstruction workflows

---

# 🧠 Future Direction — KAG + GraphRAG

paper2code is evolving beyond traditional Retrieval-Augmented Generation.

The long-term vision is a hybrid reasoning engine combining:

- vector retrieval
- symbolic graph reasoning
- architecture ontologies
- tensor-aware validation

## Planned Hybrid Pipeline

```text
PDF
 ↓
Text Extraction
 ↓
Entity Linking
 ↓
Hybrid Retrieval (Vector + Graph)
 ↓
Graph-Augmented Reasoning
 ↓
Validated Schema
 ↓
Executable Code Generation
```

---

# ❌ What We Intentionally Avoid

paper2code intentionally avoids:

- opaque autonomous AI generation
- massive ML Ops infrastructure
- uncontrolled agent swarms
- "magic" architecture synthesis

The project prioritizes:

- explainability
- transparency
- architectural reasoning
- educational clarity
- deterministic validation

---

# 📌 Current Project Status

| Phase | Status |
|---|---|
| Phase 3.9.A | ✅ Complete |
| Phase 3.9.B.1 | ✅ Complete |
| Phase 3.9.B.2 | 🔄 In Progress |
| Phase 3.9.C | 📋 Planned |

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/officialpk956-wq/paper2code.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Launch Streamlit UI

```bash
streamlit run app.py
```

---

# 📚 Documentation

Important project documents:

- `AGENT_SYSTEM_DESIGN.md`
- `IMPLEMENTATION_SUMMARY_VISUAL_COMPARISON.md`
- `VALIDATION_CHECKLIST.md`
- `DELIVERABLES_INDEX.md`
- `PROJECT_OVERVIEW.txt`

---

# 🎯 Project Philosophy

paper2code is not designed to replace human understanding.

The goal is to help researchers, students, and engineers:

- understand architectures deeply
- reason about tensor flow
- validate structural correctness
- bridge theory and implementation
- explore design choices transparently

The project prioritizes deterministic reasoning over opaque automation.

---

# 📄 License

MIT License

---

# 🔗 Repository

https://github.com/officialpk956-wq/paper2code
