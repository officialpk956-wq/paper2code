# paper2code

**paper2code** is a research-oriented project that converts deep learning research paper architectures into structured, executable code and clear architecture diagrams.  
It bridges the gap between **research papers** and **practical implementation** by making model architectures explicit, reproducible, and verifiable.

---

## 🎯 Why paper2code?

Reproducing deep learning papers is often difficult because:
- Architectures are described informally in papers
- Implementation details are missing or ambiguous
- Diagrams are incomplete or inconsistent

paper2code solves this by converting paper-level descriptions into:
- **Structured schemas** — machine-readable architecture definitions
- **Executable code** — ready-to-use model representations
- **Clear diagrams** — visual architecture representations
- **Semantic graphs** — queryable architecture knowledge

---

## ✨ Key Features

### Core Capabilities
- 📄 **Architecture extraction** from research papers
- 🏗️ **Modular support** for multiple model families:
  - ResNet (CNN-based classification)
  - U-Net (Encoder-decoder segmentation)
  - Vision Transformer (ViT) (Transformer-based vision)
  - Transformer (Encoder–Decoder) (Sequence-to-sequence)
  
### Advanced Features
- 🎨 **Automatic diagram generation** with semantic highlighting
- 📊 **Code-ready schema generation** for implementation
- 📈 **Parameter counting** and FLOPs estimation
- ✅ **Model verification** utilities
- 🔍 **Architecture comparison** between models
- 💬 **Natural language explanations** of architectural differences
- 🤖 **Agent system** for parsing, visualization, and explanation (Phase 3.9.B.1)

### User Interface
- 🖥️ **Interactive Streamlit UI** for architecture exploration
- 🔗 **Side-by-side comparison** with visual highlighting
- 🎯 **Bottleneck identification** and visual emphasis
- 📝 **Human-readable explanations** of architectural choices

---

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

### Run Tests
```bash
# All agent interface tests
python test_agent_interfaces.py

# All visual comparison tests
python run_all_visual_tests.py
```

---

## 📁 Project Structure

```
paper2code/
├── app.py                      # Main Streamlit UI with visual comparison
├── main.py                     # Legacy entry point
├── src/
│   ├── agents/                 # Agent system interfaces (Phase 3.9.B.1)
│   │   ├── __init__.py
│   │   ├── types.py            # 30 TypedDict definitions
│   │   ├── parsing_agent.py    # ParsingAgent interface
│   │   ├── visualization_agent.py  # VisualizationAgent interface
│   │   └── explanation_agent.py    # ExplanationAgent interface
│   ├── comparators/            # Architecture comparison engine
│   ├── blocks_*.py             # Model building blocks (ResNet, U-Net, ViT, Transformer)
│   ├── diagram_*.py            # Diagram generation and visualization
│   ├── schema_*.py             # Schema definitions
│   ├── schema_refiner_*.py     # Architecture-specific refinement rules
│   ├── model_builder.py        # Model construction utilities
│   ├── param_counter.py        # Parameter enumeration
│   ├── flops_estimator.py      # FLOPs calculation
│   ├── verify_model.py         # Model validation
│   └── utils.py                # Utility functions
├── tests/                      # Test suite
├── templates/                  # Diagram templates
├── outputs/                    # Generated outputs (schemas, diagrams, code)
├── docs/                       # Documentation
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🔬 Architecture Support Matrix

| Model | Type | Status | Features |
|-------|------|--------|----------|
| **ResNet** | CNN | ✅ Stable | Building blocks, diagram generation, comparison |
| **U-Net** | Encoder-Decoder | ✅ Stable | Semantic segmentation, encoder/decoder layers |
| **Vision Transformer (ViT)** | Transformer | ✅ Stable | Patch embedding, attention blocks |
| **Transformer** | Seq2Seq | ✅ Stable | Multi-head attention, encoder-decoder |

---

## 📊 Generated Outputs

The project generates several output types:

### Schemas
- **code_ready/**: Implementation-ready schemas
- **modelspecs/**: Full model specifications
- **sections/**: Modular architecture sections

### Diagrams
- **diagrams/**: Visual architecture representations
- **SVG format**: Web-compatible vector graphics

### Code
- **generated_scripts/**: Executable model code

---

## 🤖 Agent System (Phase 3.9.B.1)

The project includes a clean **three-agent architecture** for future implementation:

### ParsingAgent
Responsible for:
- Extracting architecture information from papers
- Parsing schemas into structured representations
- Validating architecture specifications

### VisualizationAgent
Responsible for:
- Rendering architecture diagrams
- Generating visual cues and highlights
- Creating semantic graph representations

### ExplanationAgent
Responsible for:
- Generating natural language descriptions
- Explaining architectural choices
- Comparing architectures and highlighting differences

**Status**: Interface design complete (Phase 3.9.B.1). Ready for implementation in Phase 3.9.B.2.

See [AGENT_SYSTEM_DESIGN.md](docs/AGENT_SYSTEM_DESIGN.md) for complete architecture details.

---

## 🎨 Visual Comparison Features

### Side-by-Side Comparison
- Interactive architecture visualization
- Synchronized graph rendering
- Detailed parameter comparisons

### Visual Highlighting System
1. **Bottleneck Badges** — Identifies layers with highest impact
2. **Compute Highlighting** — Emphasizes parameter-heavy layers
3. **Scaling Highlighting** — Shows adaptive layers
4. **Spatial Highlighting** — Marks large spatial dimensions
5. **Ghost Overlay** — Dims non-matching layers

### Interactive Legend
- Explains each highlighting type
- Toggleable layer visibility
- Comparison context display

---

## 🧪 Testing

Comprehensive test suite with 100% pass rate:

```bash
# Test agent interfaces
python test_agent_interfaces.py

# Test visual comparison features
python test_visual_comparison.py

# Test architecture comparators
python test_architecture_comparator.py

# Run all visual tests
python run_all_visual_tests.py
```

**Coverage**: 
- Agent interface compliance (8 tests)
- Visual comparison features (10+ test suites)
- Edge cases and backward compatibility
- Windows-compatible test harness

---

## 📚 Documentation

Key documentation files:
- **AGENT_SYSTEM_DESIGN.md** — Complete agent architecture
- **PHASE_3_9_B_1_COMPLETE.md** — Phase completion report
- **IMPLEMENTATION_SUMMARY_VISUAL_COMPARISON.md** — Visual features guide
- **VALIDATION_CHECKLIST.md** — QA verification checklist
- **GITHUB_PUSH_SUMMARY.md** — Latest push summary

---

## 💾 About Data and Generated Files

This repository intentionally does **not** include:
- Datasets (`data/`)
- Trained models (`models/`)
- Generated outputs (`outputs/`) — *except templates and reference schemas*
- Large experiment artifacts

This keeps the repository:
- ✅ Lightweight and focused on code
- ✅ Free from copyrighted materials
- ✅ Reproducible through provided scripts

**All diagrams, schemas, and models** can be regenerated locally using the provided scripts.

---

## 🔗 Project Timeline

| Phase | Status | Details |
|-------|--------|---------|
| **3.9.A** | ✅ Complete | Visual comparison UI with highlighting and bottleneck identification |
| **3.9.B.1** | ✅ Complete | Agent interface design (ParsingAgent, VisualizationAgent, ExplanationAgent) |
| **3.9.B.2** | 🔜 Next | Concrete agent implementations |
| **3.9.C** | 📋 Planned | RAG integration and batch processing |

---

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Last Updated**: February 4, 2026  
**Status**: ✅ Production Ready (Phase 3.9.B.1 Complete)  
**Repository**: https://github.com/officialpk956-wq/paper2code
