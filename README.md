# 📄 paper2code: Research-to-Implementation Intelligence

> Transform Deep Learning Research Papers into Structured Architectures, Executable Code, and Interactive Visualizations.

**paper2code** is a state-of-the-art research-to-implementation toolkit designed to bridge the "reproducibility gap" in deep learning. It transforms informal architecture descriptions from research papers into structured, verifiable, and pedagogically rich code artifacts.

---

## 🌟 What We Have Done (Achievements & Current State)

We have successfully built a robust, multi-architecture paper-to-code pipeline. The project has moved beyond the prototype phase into a fully maintained feature set. Key accomplishments include:

- **Multi-Architecture Support:** Built complete parsing, refinement, and code generation pipelines for ResNet, U-Net, Vision Transformer (ViT), and standard Transformers.
- **Universal Architecture Graph:** Developed a semantic `ArchitectureGraph` representation that serves as the single source of truth for all models.
- **Deterministic KAG Explanation System:** Implemented a Knowledge-Augmented Generation (KAG) system using a hardcoded Deep Learning Ontology to provide hallucination-free educational context and semantic explanations.
- **Mathematical Tensor Tracking:** Integrated a symbolic forward-pass engine (`TensorTracker`) that mathematically validates tensor shapes `(B, N, D)` and detects topological mismatches prior to code generation.
- **Interactive Glassmorphism UI:** Deployed a premium Streamlit-based dashboard (`app.py`) featuring real-time graph exploration, bottleneck highlighting, ghost overlays, and side-by-side model comparisons.
- **Agent System Foundation:** Designed and completed the interface layer for Phase 3.9.B.1 agents (Parsing, Visualization, and Explanation agents).
- **Extensive Validation:** Created a comprehensive suite of unit tests, visual validation scripts, and benchmark tests ensuring robust pipeline behavior.

---

## 📂 Project Directory Structure & File Manifest

Here is a detailed breakdown of **every file** and directory in the project, explaining their precise role and responsibilities.

### 📁 Root Directory
**Documentation & Metadata**
- `AGENT_INTERFACE_REFERENCE.md`: Detailed API reference and documentation for the Agent Interface layer.
- `AGENT_SYSTEM_DESIGN.md`: High-level design document detailing the multi-agent orchestration architecture.
- `DELIVERABLES_INDEX.md`: An index tracking project deliverables across different phases.
- `GITHUB_PUSH_SUMMARY.md`: Summary logs of GitHub commits and pushes.
- `IMPLEMENTATION_SUMMARY_VISUAL_COMPARISON.md`: Summary of the UI visual comparison features and enhancements.
- `PHASE_3_9_B_1_COMPLETE.md` & `PHASE_3_9_B_1_SUMMARY.md`: Documentation detailing the completion and summary of Phase 3.9.B.1.
- `PROJECT_OVERVIEW.txt`: A high-level, human-readable overview of the project's goals, pipeline, and state.
- `README.md`: The master documentation file you are currently reading.
- `README_PHASE_3_9_B_1.md`: Phase-specific documentation for the recent agent architecture work.
- `VALIDATION_CHECKLIST.md`: A checklist tracking testing and validation progress.

**Core Application & Entry Points**
- `app.py`: The main Streamlit application providing the interactive Glassmorphism UI.
- `server.py`: FastAPI backend server handling API requests.
- `main.py`: The primary entry point for extracting text from PDF papers using `pdfplumber`/`PyMuPDF`.

**Demonstration & Benchmarking**
- `demo_comparator.py`: Demonstration script showcasing side-by-side architecture comparisons.
- `demo_explainer.py`: Demonstration script showcasing the architecture explanation engine.
- `benchmark_bert_pipeline.py`, `benchmark_gpt_pipeline.py`, `benchmark_vit_pipeline.py`: Scripts to benchmark the extraction and processing speeds/accuracy for specific model families.

**Validation & Testing (Root Level)**
- `test_*.py`: A large collection of pytest files targeting specific components (e.g., `test_agent_interfaces.py`, `test_architecture_comparator.py`, `test_config_extractor.py`, `test_resnet_vs_vit.py`, etc.). They ensure correctness of extraction, parsing, explanation, and backward compatibility.
- `validate_*.py`: Scripts designed for deeper validation of specific sub-systems (e.g., `validate_tensor_tracker.py`, `validate_vit_extraction.py`, `validate_flops_engine.py`).
- `run_all_visual_tests.py`: A utility to run all visual regression tests automatically.

**Configuration & Dependencies**
- `requirements.txt` & `requirements.in`: Lists the Python package dependencies needed to run the project.
- `.env`: Environment variables configuration.
- `.gitignore` & `.gitattributes`: Git version control configuration files.
- `PHASE_3_9_B_1_CERTIFICATE.py`: Script serving as a certificate of completion for Phase 3.9.B.1.

### 📁 `src/` (Core Source Code)
The engine room of `paper2code`.
- `architecture_extractor.py`: Converts raw paper text sections into structured model specifications.
- `architecture_graph.py`: Defines the `GraphNode`, `GraphEdge`, and `ArchitectureGraph` core data structures.
- `codegen.py`: Converts the final architecture graph into executable PyTorch `nn.Module` code.
- `diagram_base.py`, `diagram_resnet.py`, `diagram_unet.py`, `diagram_vit.py`: Logic for rendering static Graphviz-style diagrams for different architectures.
- `generate_code_ready_schema*.py`: Scripts to generate implementation-ready JSON schemas from the refined extractions.
- `generate_diagram.py`: Orchestrates the production of visual diagram artifacts.
- `llm_client.py`: Client wrapper for connecting to Large Language Models used in extraction.
- `metrics_estimator.py` & `flops_estimator.py` & `param_counter.py`: Modules for estimating FLOPs, parameters, and computational complexity of nodes.
- `model_builder.py`, `transformer_builder.py`, `unet_builder.py`, `vit_builder.py`, `ddpm_builder.py`, `yolo_builder.py`: PyTorch-style model builders that translate schemas into actual neural network code.
- `normalizer.py`: Standardizes extracted structures across varying formats.
- `paper_to_code_generator.py`: The high-level orchestrator connecting PDF extraction to code generation.
- `radar_chart.py`: Generates visual performance and complexity trade-off charts.
- `run_*_codegen.py`: Family-specific entry points for running the code generation pipeline.
- `schema.py` & `schemas_base.py`: Foundations and shared templates for the data models.
- `schema_refiner*.py` & `schema_rules*.py`: Architecture-specific rules that normalize, clean, and validate raw extractions into stable schemas.
- `section_splitter.py`: Breaks extracted raw paper text into logical sections (Methodology, Experiments, etc.).
- `utils.py`: Shared helper utilities and functions.
- `verify_model.py`: Performs analytical validation on the generated model structures.
- `visualizer_resnet.py`, `visualizer_unet.py`, `visualizer_vit.py`: Converts family-specific architectural structures into Streamlit graph views.
- `blocks_resnet.py`, `blocks_transformer.py`, `blocks_unet.py`, `blocks_vit.py`: Defines reusable architectural components (residual blocks, attention mechanisms).

### 📁 `src/rag/` (Retrieval-Augmented Generation & Intelligence)
Handles the "reasoning" and deterministic knowledge logic.
- `knowledge_graph.py`: Contains the deep learning ontology mapping architecture types to semantic roles (e.g., `mhsa` -> `token_mixer`).
- `semantic_explainer.py`: Generates hallucination-free educational text explaining *why* layers exist.
- `tensor_tracker.py`: Performs symbolic math to validate input/output shape alignments across blocks.
- `config_extractor.py`: Extracts architectural hyperparameters (e.g., patch size, embedding dim) from raw text.
- `retriever.py`: Retrieves context for specific terms to ground the parsing process.
- `diff_engine.py`: Engine to compute semantic differences between architectures.
- `flops_engine.py`: Advanced engine for FLOP calculation context.
- `symbolic_parser.py`: Parses symbolic tensor shapes and operations.
- `normalizer.py`: Normalizes RAG inputs and context.
- `section_splitter.py`: specialized section splitting for RAG context.

### 📁 `src/agents/` (Multi-Agent System)
Orchestrates autonomous agents for specialized tasks.
- `parsing_agent.py` & `parsing_agent_impl.py`: Agents responsible for converting text to the `ArchitectureGraph`.
- `visualization_agent.py` & `visualization_agent_impl.py`: Agents managing styling, rendering, colors, and hover-cards of the graph.
- `explanation_agent.py` & `explanation_agent_impl.py`: Agents generating human-readable context summaries.
- `config_parser.py`: Agent logic for parsing complex configurations.
- `types.py`: Defines shared typed contracts and ABCs for the agents.

### 📁 `src/comparators/` & `src/explainers/`
- `src/comparators/architecture_comparator.py`: Deterministic comparison logic to find differences between two architectures.
- `src/comparators/comparison_explainer.py`: Explains the differences found by the comparator in natural language.
- `src/explainers/graph_explainer.py`: Utilities for explaining graph structures visually and textually.

### 📁 `src/orchestrator/`
- `pipeline.py`: Defines the master orchestration pipeline that links extraction, agents, and RAG.

### 📁 `outputs/`
Stores all generated artifacts produced by the pipeline.
- `texts/`: Raw text extracted from PDFs.
- `sections/`: Section-organized text data.
- `modelspecs/`: Raw extracted architecture specifications.
- `code_ready/` & `code_ready_unet/`: Refined, implementation-ready schemas.
- `diagrams/`: Generated architecture visual diagrams (`.png`).
- `generated_scripts/`: Python artifacts generated from schemas.

### 📁 Other Supporting Directories
- `data/pdfs/`: Storage for source research papers (e.g., Attention Is All You Need, ResNet, U-Net).
- `docs/`: Additional documentation and image assets.
- `experiments/` & `notebooks/`: Playgrounds for Jupyter notebooks and exploratory data analysis.
- `models/`: Storage for saved PyTorch models or embeddings.
- `scripts/`: Assorted bash/python scripts for maintenance.
- `static/` & `templates/`: HTML, CSS, and JS assets for web-based frontends outside of Streamlit.
- `tests/`: Cache and output directories for Pytest runs.
- `paper2code/`: A sub-package containing data handling (`data.py`), model definition wrappers (`models.py`), training loops (`train.py`), and local utilities (`utils.py`).

---

## 🗺️ What We Have Planned & Project Direction

While the foundation is solid, the project is actively moving towards deeper automation, improved accuracy, and broader model support. Our future direction is guided by the following roadmap:

### **1. Concrete Agent Implementations**
We have defined the interfaces and basic implementations for the multi-agent system (Phase 3.9.B.1). The next major goal is to fully empower these agents to operate autonomously, handling complex edge cases and negotiating API schema mismatches without human intervention.

### **2. Multi-Modal Architecture Extraction**
Currently, we rely heavily on text extraction. The future direction involves parsing **architecture diagrams directly from images** in the papers, combining OCR with visual layout understanding to cross-verify the text-based extraction.

### **3. Specialized "Architecture-LLM" Fine-Tuning**
We plan to fine-tune an open-source LLM specifically on deep learning architecture descriptions, ontology mapping, and tensor shape math to significantly boost extraction precision and reduce dependency on proprietary APIs.

### **4. Expanded Architecture Support (State Space Models & LLMs)**
Expanding our ontology and builder modules to natively support modern architectures like **Llama, Mistral**, and State Space Models (**Mamba**). This involves upgrading the `TensorTracker` to handle KV-caching semantics and recurrent unrolling.

### **5. Automatic Fix-Agent (Self-Healing Pipelines)**
Developing an agent that not only detects topological or mathematical impossibilities (via the `TensorTracker`) but also suggests code-level or schema-level corrections automatically, creating a self-healing pipeline.

---

## 🛠️ Usage & Data Flow

### The Data Pipeline
1. **Input:** PDF or paper text (`main.py` -> `data/pdfs/`)
2. **Extraction:** Split into logical sections (`src/section_splitter.py`)
3. **Parsing:** Semantic extraction into raw schemas (`src/architecture_extractor.py`)
4. **Refinement:** Normalization via rules (`src/schema_refiner.py`)
5. **Code-Ready:** Conversion to implementation JSON (`src/generate_code_ready_schema.py`)
6. **Graphing & Validation:** Building `ArchitectureGraph` and checking tensors (`src/rag/tensor_tracker.py`)
7. **Presentation:** Viewing in Streamlit (`app.py`)

### Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```
3. **Extract a Paper to Code:**
   ```bash
   python main.py
   python src/paper_to_code_generator.py
   ```

---
*Maintained by officialpk956-wq | paper2code is an ongoing journey to make deep learning research accessible, reproducible, and verifiable.*
