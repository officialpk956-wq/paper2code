# Paper2Code Masterclass: From Beginner to System Architect

A complete learning curriculum for rebuilding the Paper2Code project from scratch.

## PART 1: PROJECT DECONSTRUCTION

### What Problem Does This Project Solve?

Paper2Code bridges the gap between academic machine learning papers and executable, validated code implementations. The problem it addresses:

- **Reproducibility Crisis**: ML papers publish algorithms without verified implementations
- **Implementation Complexity**: Building production-grade models from papers requires extensive engineering
- **Version Control**: No standard way to track which paper version corresponds to which code version
- **Architecture Transparency**: "Black box" implementations lack explainability

Paper2Code creates a system where:
1. You parse an ML paper into a computational graph
2. The system generates executable module code
3. Every tensor transformation is tracked and validated
4. Comparisons between architectures are deterministic and reproducible
5. Frontend visualizations show the complete computational flow

**Reference**: core/orchestrator/pipeline.py shows the end-to-end orchestration of this process.

### Why Does It Exist?

The project was created to address real pain points in ML research and engineering:

- Researchers wanted a tool to validate if their paper implementations matched the original designs
- Engineers needed deterministic, auditable ML pipelines for production systems
- Data scientists wanted to understand architectural differences between models (ResNet vs ViT)
- Teams needed reproducible comparisons for decision-making

### Who Would Use It?

1. **ML Researchers**: Validate implementations against paper specifications
2. **Data Engineers**: Build reproducible ML pipelines
3. **ML Ops Teams**: Monitor tensor flows and FLOP calculations in production
4. **Students**: Learn how papers translate to code
5. **Teams Comparing Architectures**: Deterministic architectural analysis and comparison

### What Makes It Unique?

Unlike generic neural network frameworks (TensorFlow, PyTorch), Paper2Code:

1. **Traces Papers→Code**: Directly extracts computational graphs from PDF papers
2. **Deterministic Parsing**: RAG pipeline ensures consistent parsing results (see core/rag/ directory)
3. **Tensor Validation**: Real-time tensor shape tracking prevents silent errors (core/rag/tensor_tracker.py)
4. **Knowledge Graphs**: Builds semantic graphs of architectural relationships (core/rag/knowledge_graph.py)
5. **End-to-End Comparison**: Compares architectures deterministically across full pipelines
6. **Explainable Reasoning**: Generates human-readable explanations for architectural decisions

**Reference**: The deterministic RAG pipeline in core/rag/config_extractor.py uses BM25 retrieval + LLM verification loops to ensure reproducible parsing.

### How Did It Evolve?

Based on the codebase analysis:

**Phase 1 (Foundation)**: Core infrastructure
- Built core/orchestrator/pipeline.py as central orchestrator
- Implemented core/agents/parsing_agent_impl.py for paper parsing
- Created core/rag/tensor_tracker.py for validation

**Phase 2 (Determinism)**: Reproducibility layer
- Developed deterministic config extraction (core/rag/config_extractor.py)
- Implemented caching and layer capping mechanisms
- Created comparison engine with deterministic outputs

**Phase 3 (Frontend & API)**: User-facing interfaces
- Built FastAPI backend (ackend/server.py) with 8 REST endpoints
- Implemented static HTML/JS frontend (static/index.html, static/app.js)
- Connected to SQLAlchemy ORM (ackend/models.py)

**Phase 4 (Testing & Ops)**: Production readiness
- Created 20+ comprehensive test suites (see 	ests/ directory)
- Implemented CI/CD pipelines (.github/workflows/)
- Added Alembic migrations for database versioning (migrations/env.py)

### What Is The Final Vision?

The complete vision for Paper2Code:

1. **Universal Paper Parser**: Parse any ML architecture paper into executable code
2. **Deterministic ML Pipelines**: Every execution produces identical results with identical inputs
3. **Complete Architectural Analysis**: Compare any two architectures with full transparency
4. **Production-Ready Framework**: Use Paper2Code as foundation for production ML systems
5. **Knowledge Base**: Build ontologies of ML architectures and their relationships
6. **Explainable AI**: Generate human-readable reasoning for architectural choices

The system should enable: **"Take any paper, get production-ready code, with full reproducibility and explainability."**

---


## PART 2: BUILD FROM SCRATCH ROADMAP

If you were to rebuild Paper2Code from scratch starting with mkdir paper2code, here's the exact roadmap:

### Phase 1: Project Foundation & Environment (Week 1)

**Knowledge Required:**
- Python 3.9+
- Virtual environments (venv)
- Package management (pip)
- Git & version control
- Project structure conventions

**Concepts to Learn:**
- Python project scaffolding
- Dependency management
- Git workflows

**Files to Create:**
`
mkdir paper2code
cd paper2code
git init
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Create initial structure
mkdir core backend frontend tests docs scripts data
touch requirements.txt README.md .gitignore
touch main.py app.py
`

**Why These Files:**
- equirements.txt: Declares all Python dependencies (see actual: 30+ packages in ML stack)
- main.py: Entry point for CLI operations
- pp.py: Entry point for backend server
- Directory structure separates concerns: core logic, backend API, frontend UI, testing

**Actual Implementation Reference**: See equirements.txt in repository root for complete dependency list.

---

### Phase 2: Core Architecture & Orchestration (Week 2-3)

**Knowledge Required:**
- Object-oriented programming
- Design patterns (Strategy, Observer, Repository)
- Data structures (Graphs, DAGs)
- Python decorators and metaclasses
- Type hints and dataclasses

**Concepts to Learn:**
- Pipeline orchestration patterns
- Dependency injection
- Data flow pipelines
- State management

**Files to Create:**
`
core/
├── orchestrator/
│   ├── __init__.py
│   └── pipeline.py          # Central orchestration logic
├── agents/
│   ├── __init__.py
│   ├── parsing_agent.py     # Agent interface
│   └── parsing_agent_impl.py # Concrete parsing implementation
└── models/
    ├── __init__.py
    └── representations.py    # Data structures for pipeline
`

**Key Implementation Details:**

core/orchestrator/pipeline.py - The heart of the system:
`python
class Pipeline:
    def __init__(self):
        self.parsing_agent = ParsingAgent()
        self.tensor_tracker = TensorTracker()
        self.knowledge_graph = KnowledgeGraph()
    
    def execute(self, paper_config):
        # 1. Parse paper into config
        config = self.parsing_agent.parse(paper_config)
        # 2. Track tensors through architecture
        tensor_flow = self.tensor_tracker.track(config)
        # 3. Build knowledge graph
        kg = self.knowledge_graph.build(tensor_flow)
        # 4. Return complete pipeline state
        return PipelineOutput(config, tensor_flow, kg)
`

**Why This Structure:**
- Separation of concerns: parsing, validation, graph building are independent
- Extensibility: Easy to swap agents or add new validators
- Testability: Each component can be tested in isolation

**Actual Implementation Reference**: core/orchestrator/pipeline.py (130+ lines) orchestrates the complete data flow.

---

### Phase 3: Paper Parsing & Configuration Extraction (Week 4-5)

**Knowledge Required:**
- Regular expressions
- PDF parsing / text extraction
- JSON schema validation
- Language models and prompting
- Retrieval-Augmented Generation (RAG)

**Concepts to Learn:**
- BM25 full-text search
- LLM verification loops
- Configuration parsing from unstructured text
- Deterministic output caching

**Files to Create:**
`
core/rag/
├── __init__.py
├── config_extractor.py      # Extracts configs from paper text
├── tensor_tracker.py        # Validates tensor shapes/FLOPs
├── knowledge_graph.py       # Builds semantic graphs
├── flops_engine.py          # Calculates FLOPs
└── prompt_templates.py      # LLM prompts for extraction
`

**Key Implementation Details:**

core/agents/config_parser.py - Parameter assignment by type:
`python
def assign_parameters(layer_config):
    for param, value in layer_config.items():
        if is_numeric(value):
            return int(value)
        elif is_choice(value):
            return resolve_from_choices(value)
        elif is_conditional(value):
            return resolve_conditionally(value)
`

**Why This Is Critical:**
- Papers describe layers in natural language ("3x3 convolution with ReLU")
- System must deterministically convert to code ("Conv2d(3, 3, activation='relu')")
- RAG pipeline ensures reproducibility (same input → same output every time)

**Actual Implementation Reference**: 
- core/rag/config_extractor.py (200+ lines) - deterministic extraction with BM25 + LLM loop
- core/agents/config_parser.py - type-based parameter resolution

---

### Phase 4: Tensor Tracking & Validation (Week 6)

**Knowledge Required:**
- Tensor operations and shapes
- Broadcasting rules
- FLOP calculations
- Error handling and validation
- Logging and monitoring

**Concepts to Learn:**
- Shape inference through layers
- FLOP counting algorithms
- Validation frameworks
- Real-time tensor monitoring

**Files to Create:**
`
core/rag/tensor_tracker.py:
- Track shape at every layer
- Validate shapes don't conflict
- Calculate FLOPs for each operation
- Raise TensorMismatchError on inconsistency

core/rag/flops_engine.py:
- FLOP calculation for each operation type
- Batched FLOP counting
- Memory usage estimation
`

**Key Implementation Details:**

Tensor validation (from core/rag/tensor_tracker.py):
`python
class TensorTracker:
    def track_layer(self, layer_type, input_shape, layer_params):
        # Calculate expected output shape based on layer type
        output_shape = self.infer_output_shape(layer_type, input_shape, layer_params)
        
        # Validate shape is valid
        if not self.is_valid_shape(output_shape):
            raise TensorMismatchError(f"Invalid shape: {output_shape}")
        
        # Log FLOPs
        flops = self.flops_engine.calculate(layer_type, input_shape, layer_params)
        self.log_event(f"layer={layer_type}, flops={flops}")
        
        return output_shape
`

**Why This Is Essential:**
- Silent tensor shape mismatches cause hard-to-debug errors
- Early validation catches bugs at parse time, not runtime
- FLOP tracking helps understand model computational cost

**Actual Implementation Reference**: core/rag/tensor_tracker.py (150+ lines) tracks every tensor transformation.

---

### Phase 5: Knowledge Graph Construction (Week 7)

**Knowledge Required:**
- Graph data structures (nodes, edges)
- DAG algorithms (topological sort, dependency analysis)
- Semantic representations
- Graph visualization

**Concepts to Learn:**
- Computational graphs
- Dependency resolution
- Graph traversal algorithms
- DAG properties (cycles, paths)

**Files to Create:**
`
core/rag/knowledge_graph.py:
- Node types: Layer, Parameter, Tensor, Operation
- Edge types: input_to, output_from, depends_on
- Methods: build(), query(), validate_acyclic(), find_paths()
`

**Key Implementation Details:**

Knowledge graph structure:
`python
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}  # id -> Node
        self.edges = []  # List of (source_id, dest_id, edge_type)
    
    def add_layer(self, layer_id, layer_type, params):
        node = LayerNode(layer_id, layer_type, params)
        self.nodes[layer_id] = node
    
    def add_dependency(self, source_layer_id, dest_layer_id):
        self.edges.append((source_layer_id, dest_layer_id, 'input_to'))
    
    def validate_acyclic(self):
        # Ensure it's a DAG (no cycles)
        if self.has_cycle():
            raise ValueError("Graph contains cycle - not a valid DAG")
    
    def topological_sort(self):
        # Returns execution order of layers
        return self._toposort(self.nodes.keys())
`

**Why This Matters:**
- KGs enable semantic understanding of architectures
- Detect impossible designs (cycles in computation)
- Enable comparison between architectures
- Support reasoning about data flow

**Actual Implementation Reference**: core/rag/knowledge_graph.py - builds semantic graph of all layers and their relationships.

---

### Phase 6: Backend API & Database Layer (Week 8-9)

**Knowledge Required:**
- REST API design (HTTP, methods, status codes)
- FastAPI framework
- SQLAlchemy ORM
- Database schema design
- Alembic migrations

**Concepts to Learn:**
- Request/response serialization
- Dependency injection in FastAPI
- ORM entity mapping
- Database versioning
- ACID properties

**Files to Create:**
`
backend/
├── __init__.py
├── server.py          # FastAPI routes (8 endpoints)
├── models.py          # SQLAlchemy ORM models
├── database.py        # Session management
└── schemas.py         # Request/response models

migrations/
├── env.py
├── script.py.mako
└── versions/
    └── 001_initial_schema.py
`

**FastAPI Endpoints** (from ackend/server.py):

1. POST /api/parse - Parse paper and return config
2. POST /api/generate - Generate module code
3. POST /api/compare - Compare two architectures
4. POST /api/analyze - Deep analysis of architecture
5. GET /api/papers - List all papers
6. GET /api/modules - List modules for paper
7. GET /api/comparisons - List comparisons
8. POST /api/explain - Get explanation of architecture

**Database Schema** (from ackend/models.py):

`python
class Paper(Base):
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    parsed_config = Column(JSON)

class Module(Base):
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('paper.id'))
    name = Column(String)
    code = Column(Text)
    tensor_flow = Column(JSON)

class Comparison(Base):
    id = Column(Integer, primary_key=True)
    architecture1_id = Column(Integer, ForeignKey('module.id'))
    architecture2_id = Column(Integer, ForeignKey('module.id'))
    comparison_result = Column(JSON)
`

**Why This Architecture:**
- FastAPI: Modern, fast, type-safe framework
- SQLAlchemy: Pythonic ORM, works with any SQL database
- Alembic: Track schema changes over time
- Separation: API layer independent of business logic

**Actual Implementation Reference**: 
- ackend/server.py (200+ lines) - all API endpoints
- ackend/models.py - ORM entity definitions
- ackend/database.py - session management

---

### Phase 7: Frontend & User Interaction (Week 10)

**Knowledge Required:**
- HTML/CSS/JavaScript basics
- REST API consumption
- Frontend state management
- Visualization libraries
- Responsive design

**Concepts to Learn:**
- Client-side frameworks (Next.js, React)
- Form handling and validation
- API integration patterns
- Chart/graph visualization

**Files to Create:**
`
frontend/              # Next.js stubs
static/
├── index.html        # Main HTML page
├── app.js            # Frontend logic
├── style.css         # Styling
└── viz.js            # Visualization library
`

**Frontend Capabilities:**
- Upload paper → triggers /api/parse
- View parsed architecture as interactive graph
- Compare two architectures side-by-side
- Export results as JSON/PDF

**Why This Structure:**
- Static HTML/JS is lightweight and deployable
- Next.js stubs available for future enhancement
- API-first design allows multiple frontends

**Actual Implementation Reference**: 
- static/index.html - main interface
- static/app.js - API integration and state management

---

### Phase 8: Testing & Quality Assurance (Week 11)

**Knowledge Required:**
- Unit testing (pytest)
- Integration testing
- Test fixtures and mocking
- Coverage analysis
- Determinism testing

**Concepts to Learn:**
- TDD methodology
- Mock objects and patches
- Parametrized tests
- CI/CD integration

**Files to Create:**
`
tests/
├── conftest.py                      # Shared fixtures
├── test_pipeline_determinism.py     # Ensure reproducibility
├── test_config_parser.py            # Parser tests
├── test_tensor_tracker.py           # Validation tests
├── test_knowledge_graph.py          # Graph tests
├── test_api_endpoints.py            # API tests
├── test_comparator.py               # Comparison tests
└── ...20+ more test files
`

**Example Test Pattern** (Determinism):

`python
def test_pipeline_determinism():
    paper_config = load_fixture('resnet_config.json')
    
    # Run pipeline twice with identical inputs
    result1 = pipeline.execute(paper_config)
    result2 = pipeline.execute(paper_config)
    
    # Results must be identical (deterministic)
    assert result1.tensor_flow == result2.tensor_flow
    assert result1.knowledge_graph == result2.knowledge_graph
`

**Why This Matters:**
- 20+ test files ensure reliability
- Determinism tests guarantee reproducible outputs
- Coverage analysis prevents untested code paths
- Regression tests catch breaking changes

**Actual Implementation Reference**: 
- 	ests/ directory contains 20+ comprehensive test files
- Determinism is a core requirement (see 	est_pipeline_determinism.py)

---

### Phase 9: CI/CD & Deployment (Week 12)

**Knowledge Required:**
- GitHub Actions workflows
- Linting and code quality (black, flake8)
- Automated testing in CI
- Deployment strategies
- Environment management

**Concepts to Learn:**
- YAML workflow syntax
- Matrix testing (multiple Python versions)
- Artifact caching
- Deploy triggers

**Files to Create:**
`
.github/workflows/
├── ci.yml           # Run tests on every push
└── cd.yml           # Deploy on tag

scripts/
├── lint.sh          # Run linters
├── test.sh          # Run tests
└── deploy.sh        # Deployment script
`

**CI Workflow** (from .github/workflows/ci.yml):

`yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: black --check .
      - run: flake8 .
      - run: pytest --cov=core --cov=backend tests/
`

**Why This Matters:**
- Automated testing prevents regressions
- Linting enforces code quality standards
- CI/CD enables rapid, safe iterations
- Deployment automation reduces human error

**Actual Implementation Reference**: .github/workflows/ contains production CI/CD workflows.

---

### Summary: Build Phases in Sequence

| Phase | Duration | Focus | Key Output |
|-------|----------|-------|-----------|
| 1 | Week 1 | Foundation | Project structure, venv, git |
| 2 | Week 2-3 | Architecture | Pipeline orchestrator, core models |
| 3 | Week 4-5 | Parsing | RAG pipeline, config extraction |
| 4 | Week 6 | Validation | Tensor tracking, FLOP calculation |
| 5 | Week 7 | Semantics | Knowledge graphs, DAGs |
| 6 | Week 8-9 | Backend | FastAPI, SQLAlchemy, migrations |
| 7 | Week 10 | Frontend | HTML/JS UI, API integration |
| 8 | Week 11 | Testing | 20+ test suites, determinism validation |
| 9 | Week 12 | Operations | CI/CD, linting, automated deployment |

**Total timeline: 12 weeks for a complete, production-ready system.**

---


## PART 3: KNOWLEDGE PREREQUISITES

This section covers every major concept used in Paper2Code. Understanding these is prerequisite knowledge for rebuilding the system.

### 3.1 Python Programming Fundamentals

**What Is It?**
Python is a high-level, dynamically-typed programming language designed for readability and rapid development.

**Why It Exists:**
- Easy to learn and use
- Extensive ecosystem for ML (NumPy, TensorFlow, PyTorch, pandas)
- Excellent for prototyping and research
- First-class support for scientific computing

**How Paper2Code Uses It:**
- **Core business logic**: All parsing, tracking, and orchestration in Python
- **Backend**: FastAPI server written entirely in Python
- **Testing**: Comprehensive test suites using pytest
- **Data processing**: pandas and NumPy for numerical operations

**Reference Files:**
- Entry points: main.py, pp.py
- Core: core/orchestrator/pipeline.py, core/agents/parsing_agent_impl.py
- Backend: ackend/server.py, ackend/models.py

**Python Concepts Used:**
- Classes and OOP
- Decorators (used extensively in FastAPI)
- Type hints (throughout codebase)
- Async/await (FastAPI endpoints)
- Context managers (database sessions)
- List comprehensions and generators
- Dataclasses (used in models)

---

### 3.2 Object-Oriented Programming (OOP)

**What Is It?**
OOP is a programming paradigm based on objects (data + behavior) and classes (blueprints for objects).

**Why It Exists:**
- Models real-world concepts naturally
- Enables code reuse through inheritance
- Provides encapsulation (data hiding)
- Supports polymorphism (method overriding)

**How Paper2Code Uses It:**
- **Agents**: ParsingAgent base class with ParsingAgentImpl concrete implementation
- **Components**: Pipeline, TensorTracker, KnowledgeGraph, ConfigExtractor as independent objects
- **Models**: Paper, Module, Comparison as SQLAlchemy entity classes
- **Design patterns**: Repository pattern, Strategy pattern, Observer pattern

**Key OOP Patterns in Paper2Code:**

1. **Strategy Pattern** - Interchangeable algorithms:
   `python
   class ParsingAgent:
       def parse(self, config): pass
   
   class ParsingAgentImpl(ParsingAgent):
       def parse(self, config):
           # Actual implementation
           return parsed_config
   `

2. **Repository Pattern** - Abstract data access:
   `python
   class PaperRepository:
       def get(self, id): pass
       def save(self, paper): pass
   `

3. **Dependency Injection** - Pass dependencies to constructors:
   `python
   class Pipeline:
       def __init__(self, parsing_agent, tensor_tracker, knowledge_graph):
           self.parsing_agent = parsing_agent
           self.tensor_tracker = tensor_tracker
           self.knowledge_graph = knowledge_graph
   `

**Reference Files:**
- core/agents/parsing_agent.py - agent interface
- core/agents/parsing_agent_impl.py - concrete implementation
- ackend/models.py - SQLAlchemy ORM entities
- core/orchestrator/pipeline.py - dependency injection

---

### 3.3 Dataclasses

**What Is It?**
Dataclasses are a Python feature for creating classes that primarily store data with minimal code.

**Why It Exists:**
- Reduces boilerplate code (no need to write __init__, __repr__, __eq__)
- Generates helpful methods automatically
- Type-safe representation of data
- Cleaner, more readable code

**How Paper2Code Uses It:**
- **Model definitions**: Data structures for pipeline state, tensor information, graph nodes
- **Configuration**: Represents parsed paper configurations
- **Type safety**: Ensures type hints are enforced

**Example Dataclass Usage:**

`python
from dataclasses import dataclass

@dataclass
class TensorInfo:
    name: str
    shape: List[int]
    dtype: str
    flops: int

@dataclass
class LayerConfig:
    layer_type: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: Dict[str, Any]
`

**Reference Files:**
- core/models/representations.py - likely uses dataclasses for core data structures
- ackend/models.py - SQLAlchemy models (similar pattern)

---

### 3.4 JSON & Serialization

**What Is It?**
JSON (JavaScript Object Notation) is a text format for structuring and serializing data.

**Why It Exists:**
- Language-independent data interchange format
- Human-readable
- Supports nested structures (objects, arrays)
- Standard for REST APIs and configuration files

**How Paper2Code Uses It:**
- **Configuration storage**: Parsed paper configs stored as JSON in database
- **API responses**: FastAPI automatically serializes responses to JSON
- **Knowledge graphs**: Serialized as JSON for transmission to frontend
- **Tensor flows**: Represented as JSON for persistence and visualization

**JSON Structure in Paper2Code:**

`json
{
  "paper_id": "resnet_v1",
  "architecture": {
    "layers": [
      {
        "id": "conv1",
        "type": "Conv2d",
        "input_shape": [1, 224, 224, 3],
        "output_shape": [1, 112, 112, 64],
        "parameters": {
          "kernel_size": 7,
          "stride": 2,
          "padding": 3
        },
        "flops": 11689472000
      }
    ],
    "connections": [
      {"from": "conv1", "to": "batch_norm1"},
      {"from": "batch_norm1", "to": "relu1"}
    ]
  }
}
`

**Reference Files:**
- ackend/models.py - parsed_config JSON field in Paper model
- ackend/server.py - JSON responses from API endpoints

---

### 3.5 Graph Data Structures & DAGs

**What Is It?**
Graphs are data structures consisting of nodes (vertices) and edges (connections). DAGs (Directed Acyclic Graphs) are graphs with directed edges and no cycles.

**Why They Exist:**
- Model relationships and dependencies naturally
- Enable efficient algorithms (BFS, DFS, topological sort)
- Detect cycles and impossible configurations
- Represent computational flows

**How Paper2Code Uses Them:**

1. **Knowledge Graphs** - Represent architecture semantics:
   - Nodes: Layers, parameters, tensors
   - Edges: input_to, output_from, depends_on
   - Used for: Architecture comparison, analysis, reasoning

2. **Computational DAGs** - Represent execution order:
   - Nodes: Operations (Conv, ReLU, etc.)
   - Edges: Data flow between operations
   - Used for: Tensor tracking, FLOP calculation

**DAG Properties Verified:**
`python
def validate_dag(graph):
    # No cycles (DAG property)
    if has_cycle(graph):
        raise ValueError("Graph contains cycle")
    
    # Topological ordering possible
    execution_order = topological_sort(graph)
    return execution_order
`

**Reference Files:**
- core/rag/knowledge_graph.py - builds and manages KGs
- Implicit in core/rag/tensor_tracker.py - tracks flow through DAG

---

### 3.6 Neural Network Architectures

**What Is It?**
Neural networks are computational models inspired by biological neurons. They consist of layers (linear transformations + activations) arranged in sequences.

**Why They Exist:**
- Universal function approximators
- Proven effective for classification, regression, generation
- Foundation for deep learning
- Enable learning from data

**How Paper2Code Uses Them:**
- **Parsing papers**: Extracts layer specifications from papers
- **Comparisons**: Compares architectural differences
- **Validation**: Ensures tensor shapes are compatible
- **Analysis**: Calculates FLOPs, parameter counts

**Common Architectures in Paper2Code:**

1. **ResNet (Residual Networks)**
   - Key innovation: Skip connections
   - Why: Enables deeper networks without vanishing gradients
   - Used in: Default example for tracing

2. **ViT (Vision Transformers)**
   - Key innovation: Patch-based attention
   - Why: Superior to CNNs on large datasets
   - Used in: Architecture comparison examples

3. **Other Architectures**
   - Referenced: Transformers, CNNs, RNNs, Autoencoders

**Reference Files:**
- Test files: 	est_resnet_vs_vit.py, 	est_vit_patch_embedding.py
- Parsing: Extracts these architecture specifications from papers

---

### 3.7 Convolutional Neural Networks (CNNs)

**What Is It?**
CNNs are neural networks using convolutional layers for spatial pattern recognition.

**Key Components:**
- **Convolution**: Sliding window operation extracting features
- **Pooling**: Reducing spatial dimensions (max/average)
- **Activation**: Non-linearity (ReLU, sigmoid)

**Why They Exist:**
- Efficiently capture spatial hierarchies
- Parameter sharing reduces memory
- Translation equivariance
- State-of-the-art on image tasks (before Transformers)

**How Paper2Code Uses CNNs:**
- **Parsing**: Extracts Conv2d, Conv1d specifications
- **Comparison**: Contrasts CNN vs Transformer approaches
- **Analysis**: Validates convolution operations

**CNN Tensor Operations:**

`python
# Conv2d: (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
input_shape = [32, 224, 224, 3]  # batch=32, H=224, W=224, C=3
kernel_size = 3
stride = 1
padding = 1
out_channels = 64

output_height = (224 - 3 + 2*1) // 1 + 1 = 224
output_width = (224 - 3 + 2*1) // 1 + 1 = 224
output_shape = [32, 224, 224, 64]
`

**Reference Files:**
- 	est_comparator.py - compares CNN vs other architectures
- Implicit in tensor tracker validation

---

### 3.8 Transformers & Attention Mechanisms

**What Is It?**
Transformers use self-attention to process sequences in parallel, enabling:
- Parallel processing (vs RNN sequentiality)
- Long-range dependencies without recurrence
- State-of-the-art results across domains

**Key Components:**
- **Self-Attention**: Compute attention between all positions
- **Multi-Head Attention**: Multiple attention subspaces
- **Feed-Forward**: Dense layers between attention layers
- **Positional Encoding**: Add position information (Transformers lack recurrence)

**Why They Exist:**
- Overcome RNN limitations (slow, hard to parallelize)
- Enable parallelization of sequence processing
- Capture long-range dependencies efficiently
- Foundation for large language models (GPT, BERT)

**How Paper2Code Uses Them:**
- **ViT Parsing**: Extracts patch embedding and attention specifications
- **Comparison**: ViT vs ResNet is primary comparison example
- **Analysis**: Validates attention head configurations

**Transformer Tensor Operations:**

`python
# Multi-head attention:
batch_size = 32
seq_length = 196  # For ViT with 14x14 patches
hidden_dim = 768
num_heads = 12
head_dim = hidden_dim // num_heads = 64

# Input: (batch, seq_len, hidden_dim)
Q = (32, 196, 768)  # Query projection
K = (32, 196, 768)  # Key projection
V = (32, 196, 768)  # Value projection

# Attention: (seq_len, seq_len)
attention_scores = Q @ K^T / sqrt(head_dim)  # (196, 196)
attention_output = softmax(attention_scores) @ V  # (32, 196, 768)
`

**Reference Files:**
- 	est_vit_patch_embedding.py - ViT-specific tests
- 	est_transformer_builder.py - Transformer construction
- core/rag/config_extractor.py - parsing Transformer specs

---

### 3.9 Tensor Shapes & Broadcasting

**What Is It?**
Tensors are multi-dimensional arrays. Shape defines dimensions. Broadcasting allows operations on different-shaped tensors.

**Why It Matters:**
- Silent shape mismatches cause hard-to-debug errors
- Broadcasting can mask errors
- Understanding shapes is prerequisite for deep learning

**Tensor Rank:**
- Rank 0: Scalar (5)
- Rank 1: Vector [1, 2, 3] - shape (3,)
- Rank 2: Matrix [[1,2],[3,4]] - shape (2, 2)
- Rank 3: Tensor [[[1]]] - shape (1, 1, 1)
- Rank 4: Batch of images - shape (batch, height, width, channels) = (32, 224, 224, 3)

**Broadcasting Rules:**
`python
# Shapes are compatible if dimensions match or one is 1
(32, 1, 224, 3) broadcasts with (1, 224, 224, 1) -> (32, 224, 224, 3)

# Error: incompatible shapes
(32, 224, 224) cannot broadcast with (32, 224, 3)  # dimension mismatch
`

**How Paper2Code Uses Shape Validation:**

`python
# From tensor_tracker.py
def track_conv2d(input_shape, kernel_size, stride, padding):
    batch, height, width, channels = input_shape
    out_height = (height - kernel_size + 2*padding) // stride + 1
    out_width = (width - kernel_size + 2*padding) // stride + 1
    output_shape = [batch, out_height, out_width, out_channels]
    
    # Validate output shape is sensible
    if any(dim <= 0 for dim in output_shape):
        raise TensorMismatchError(f"Invalid output shape: {output_shape}")
    
    return output_shape
`

**Reference Files:**
- core/rag/tensor_tracker.py - shape inference and validation
- All test files validate expected tensor shapes

---

### 3.10 FLOPs (Floating-Point Operations)

**What Is It?**
FLOPs measure computational complexity - the number of floating-point operations.

**Why It Matters:**
- Predicts runtime and energy consumption
- Compares model efficiency
- Hardware-independent measure of computation

**FLOP Calculations by Layer:**

`python
# Convolution:
# FLOPs = 2 × kernel_height × kernel_width × in_channels × out_height × out_width × out_channels
conv_params = {"kernel_size": 3, "in_channels": 3, "out_channels": 64}
input_shape = [1, 224, 224, 3]
out_h, out_w = 112, 112
flops = 2 * 3 * 3 * 3 * 112 * 112 * 64 = 11,689,472,000

# Linear/Dense:
# FLOPs = 2 × input_features × output_features × batch_size
flops = 2 * 1000 * 10 * 32 = 640,000

# Attention:
# FLOPs ≈ 4 × seq_length^2 × hidden_dim + 4 × seq_length × hidden_dim^2
flops = 4 * 196^2 * 768 + 4 * 196 * 768^2 = (large number)
`

**How Paper2Code Uses FLOPs:**
- **Calculation**: core/rag/flops_engine.py computes FLOPs per layer
- **Tracking**: Logged via core/rag/tensor_tracker.py
- **Comparison**: Compared between architectures (ResNet vs ViT FLOPs)
- **Analysis**: Helps understand computational cost differences

**Reference Files:**
- core/rag/flops_engine.py - FLOP calculation engine
- core/rag/tensor_tracker.py - logs FLOPs during tracking

---

### 3.11 FastAPI & REST APIs

**What Is It?**
FastAPI is a modern Python web framework for building REST APIs with automatic documentation and validation.

**Why It Exists:**
- Type safety with Python type hints
- Automatic request/response validation
- Auto-generated OpenAPI/Swagger documentation
- Async support for high concurrency
- Fast performance

**How Paper2Code Uses It:**

API Structure (ackend/server.py):
`python
from fastapi import FastAPI
app = FastAPI()

# Endpoint 1: Parse paper
@app.post("/api/parse")
async def parse_paper(paper_config: PaperConfigSchema):
    result = pipeline.execute(paper_config)
    return result

# Endpoint 2: Compare architectures
@app.post("/api/compare")
async def compare_architectures(arch1_id: int, arch2_id: int):
    comparison = comparator.compare(arch1_id, arch2_id)
    return comparison
`

**Key Features Used:**
- Type hints for request validation
- Async/await for concurrency
- Dependency injection (FastAPI's Depends())
- JSON response serialization
- 8 main endpoints covering all operations

**REST Principles Applied:**
- GET for retrieving data
- POST for creating/processing
- JSON request/response bodies
- HTTP status codes (200 OK, 400 Bad Request, 500 Internal Error)

**Reference Files:**
- ackend/server.py - all API implementations
- ackend/schemas.py - request/response models

---

### 3.12 SQL & SQLAlchemy ORM

**What Is It?**
SQLAlchemy is a Python ORM (Object-Relational Mapping) that maps Python classes to database tables.

**Why It Exists:**
- Abstraction over SQL databases
- Type-safe data access
- Automatic query generation
- Migration support via Alembic
- Declarative syntax

**How Paper2Code Uses It:**

**Database Schema** (ackend/models.py):
`python
from sqlalchemy import Column, Integer, String, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    parsed_config = Column(JSON)

class Module(Base):
    __tablename__ = 'modules'
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'))
    name = Column(String)
    code = Column(Text)
    tensor_flow = Column(JSON)
`

**Session Management** (ackend/database.py):
`python
from sqlalchemy.orm import sessionmaker

SessionLocal = sessionmaker(bind=engine)

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Used in FastAPI:
@app.get("/api/papers")
def list_papers(session = Depends(get_session)):
    return session.query(Paper).all()
`

**Reference Files:**
- ackend/models.py - ORM entity definitions
- ackend/database.py - session management
- ackend/server.py - usage in endpoints

---

### 3.13 Alembic Migrations

**What Is It?**
Alembic is a version control system for database schemas, allowing tracking of schema changes over time.

**Why It Exists:**
- Track schema evolution
- Rollback/forward schema changes
- Collaborate on database changes
- Reproducible deployments

**How Paper2Code Uses It:**

Migration Structure (migrations/env.py, migrations/versions/):
`python
# Auto-generate migration when model changes
alembic revision --autogenerate -m "Add comparison table"

# Migration file: alembic/versions/001_add_comparison_table.py
def upgrade():
    op.create_table(
        'comparisons',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('architecture1_id', sa.Integer()),
        sa.Column('architecture2_id', sa.Integer()),
        sa.ForeignKeyConstraint(['architecture1_id'], ['modules.id']),
        sa.ForeignKeyConstraint(['architecture2_id'], ['modules.id']),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('comparisons')
`

**Workflow:**
`ash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
`

**Reference Files:**
- migrations/env.py - Alembic configuration
- migrations/versions/ - individual migration files

---

### 3.14 Repository Pattern

**What Is It?**
Repository pattern abstracts data access, providing a collection-like interface.

**Why It Exists:**
- Decouple business logic from data access
- Testability (mock repositories)
- Reusability across services
- Single source of truth for queries

**How Paper2Code Could Use It:**

`python
class PaperRepository:
    def __init__(self, session):
        self.session = session
    
    def get(self, paper_id):
        return self.session.query(Paper).filter_by(id=paper_id).first()
    
    def save(self, paper):
        self.session.add(paper)
        self.session.commit()
    
    def get_all(self):
        return self.session.query(Paper).all()

# Usage in business logic:
repo = PaperRepository(session)
paper = repo.get(1)
paper.parsed_config = new_config
repo.save(paper)
`

**Reference Files:**
- Implicit pattern in ackend/database.py and usage in ackend/server.py

---

### 3.15 Service Layer Pattern

**What Is It?**
Service layer pattern encapsulates business logic, separating it from API and data layers.

**Why It Exists:**
- Reusable business logic
- Testable in isolation
- Clear responsibilities
- Easy to mock in tests

**How Paper2Code Uses It:**

`python
class ParsingService:
    def __init__(self, parsing_agent, tensor_tracker):
        self.parsing_agent = parsing_agent
        self.tensor_tracker = tensor_tracker
    
    def parse_paper(self, paper_content):
        # Business logic
        config = self.parsing_agent.parse(paper_content)
        tensor_flow = self.tensor_tracker.track(config)
        return {"config": config, "tensor_flow": tensor_flow}

# FastAPI endpoint uses service:
@app.post("/api/parse")
def parse(paper: PaperSchema, service: ParsingService = Depends()):
    return service.parse_paper(paper.content)
`

**Reference Files:**
- core/orchestrator/pipeline.py - implements service logic
- ackend/server.py - FastAPI endpoints use pipeline (service)

---

### 3.16 Dependency Injection

**What Is It?**
Dependency Injection (DI) passes dependencies to objects instead of having them create their own.

**Why It Exists:**
- Loose coupling between components
- Testability (inject mocks)
- Flexibility (swap implementations)
- Explicit dependencies

**How Paper2Code Uses It:**

`python
# Constructor injection
class Pipeline:
    def __init__(self, parsing_agent, tensor_tracker, knowledge_graph):
        self.parsing_agent = parsing_agent
        self.tensor_tracker = tensor_tracker
        self.knowledge_graph = knowledge_graph

# FastAPI dependency injection
from fastapi import Depends

def get_pipeline() -> Pipeline:
    # Build and return pipeline
    parsing_agent = ParsingAgentImpl()
    tensor_tracker = TensorTracker()
    knowledge_graph = KnowledgeGraph()
    return Pipeline(parsing_agent, tensor_tracker, knowledge_graph)

@app.post("/api/parse")
def parse(config: ConfigSchema, pipeline: Pipeline = Depends(get_pipeline)):
    return pipeline.execute(config)
`

**Benefits:**
- Testing: Inject mock agents
- Flexibility: Swap implementations without changing code
- Clear dependencies visible in constructor

**Reference Files:**
- core/orchestrator/pipeline.py - uses DI in __init__
- ackend/server.py - FastAPI Depends() for DI

---

### 3.17 Testing with Pytest

**What Is It?**
Pytest is a Python testing framework enabling simple, powerful testing.

**Why It Exists:**
- Simple assertion syntax
- Auto-discovery of test functions
- Fixtures for setup/teardown
- Parametrized testing
- Excellent error messages

**How Paper2Code Uses It:**

**Basic Test** (from existing test files):
`python
import pytest
from core.orchestrator.pipeline import Pipeline

def test_pipeline_executes():
    pipeline = Pipeline()
    result = pipeline.execute(sample_config)
    assert result is not None
    assert result.tensor_flow is not None
`

**Determinism Test**:
`python
def test_pipeline_determinism():
    pipeline = Pipeline()
    
    # Run twice with identical inputs
    result1 = pipeline.execute(config)
    result2 = pipeline.execute(config)
    
    # Assert identical outputs (deterministic)
    assert result1.tensor_flow == result2.tensor_flow
`

**Parametrized Test**:
`python
@pytest.mark.parametrize("layer_type,input_shape,expected_out", [
    ("Conv2d", [1, 224, 224, 3], [1, 112, 112, 64]),
    ("MaxPool", [1, 112, 112, 64], [1, 56, 56, 64]),
])
def test_layer_shapes(layer_type, input_shape, expected_out):
    tracker = TensorTracker()
    output = tracker.track_layer(layer_type, input_shape, {})
    assert output == expected_out
`

**Fixtures**:
`python
@pytest.fixture
def pipeline():
    """Fixture providing a pipeline instance"""
    return Pipeline()

def test_with_fixture(pipeline):
    result = pipeline.execute(config)
    assert result is not None
`

**Reference Files:**
- 	ests/ - 20+ test files using these patterns
- conftest.py - shared fixtures
- All test files follow pytest conventions

---

### 3.18 CI/CD with GitHub Actions

**What Is It?**
CI/CD automates testing, linting, and deployment on every code change.

**Why It Exists:**
- Catch bugs early
- Enforce code quality
- Automate deployment
- Prevent human error
- Enable rapid iteration

**How Paper2Code Uses It:**

**CI Workflow** (.github/workflows/ci.yml):
`yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install black flake8 pytest pytest-cov
      
      - name: Lint with black
        run: black --check .
      
      - name: Lint with flake8
        run: flake8 .
      
      - name: Run tests
        run: pytest --cov=core --cov=backend tests/
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
`

**Key CI Jobs:**
1. Checkout code
2. Set up Python environment
3. Install dependencies
4. Run linters (black, flake8)
5. Run tests with coverage
6. Upload coverage reports

**CD Workflow** (.github/workflows/cd.yml):
- Triggered on version tags (v1.0.0)
- Builds and pushes Docker images
- Deploys to production

**Reference Files:**
- .github/workflows/ci.yml - CI configuration
- .github/workflows/cd.yml - CD configuration

---

### 3.19 Monorepos

**What Is It?**
Monorepo stores multiple projects in a single repository.

**Why It Matters:**
- Easier coordination across services
- Shared code and utilities
- Single CI/CD pipeline
- Atomic commits across services

**How Paper2Code Structures It:**

`
paper2code/                    # Root
├── core/                      # Core business logic
│   ├── orchestrator/         # Orchestration layer
│   ├── agents/               # Parsing agents
│   ├── rag/                  # RAG pipeline
│   └── models/               # Data models
├── backend/                  # FastAPI backend
│   ├── server.py
│   ├── models.py
│   ├── database.py
│   └── schemas.py
├── frontend/                 # Next.js (stubs)
├── static/                   # Static HTML/JS UI
├── tests/                    # Test suite
├── migrations/               # Alembic migrations
├── scripts/                  # Utility scripts
├── docs/                     # Documentation
└── requirements.txt          # All dependencies
`

**Monorepo Benefits in Paper2Code:**
- Core logic used by both CLI (main.py) and API (pp.py)
- Shared tests across components
- Single requirements.txt for all dependencies
- Atomic commits (parse + test + deploy together)

**Reference Files:**
- Directory structure as shown above
- equirements.txt - centralized dependencies

---

### Summary: Knowledge Prerequisites Table

| Concept | Relevance | Used For | Key Files |
|---------|-----------|----------|-----------|
| Python OOP | Critical | All business logic | core/, backend/ |
| Dataclasses | High | Data structures | backend/models.py |
| JSON | High | Config/API storage | backend/ |
| Graphs/DAGs | Critical | Knowledge graphs | core/rag/knowledge_graph.py |
| Neural Networks | Critical | Architecture parsing | tests/test_resnet_vs_vit.py |
| CNNs | High | ResNet parsing | test files |
| Transformers | High | ViT parsing | test_vit_patch_embedding.py |
| Tensor Shapes | Critical | Validation | core/rag/tensor_tracker.py |
| FLOPs | High | Computation tracking | core/rag/flops_engine.py |
| FastAPI | Critical | API layer | backend/server.py |
| SQLAlchemy | High | Data persistence | backend/models.py |
| Alembic | Medium | Schema versioning | migrations/ |
| Pytest | High | Test framework | tests/ |
| CI/CD | Medium | Deployment automation | .github/workflows/ |

**Study Order (Recommended):**
1. Python & OOP
2. Data structures (JSON, dataclasses, graphs)
3. Neural networks (CNNs, Transformers)
4. Backend (FastAPI, SQLAlchemy, Alembic)
5. Testing (pytest)
6. Operations (CI/CD)

---


## PART 4: CODEBASE WALKTHROUGH

Walk through every major folder. For each folder: purpose, responsibilities, dependencies, important files, and interactions.

---

## 4.1 core/ - Business Logic Core

**Purpose:**
Core contains all business logic: parsing, RAG pipeline, tensor tracking, knowledge graphs. This is the intellectual property of Paper2Code.

**Responsibilities:**
- Parse papers into configurations
- Extract tensor shapes and flow
- Build semantic knowledge graphs
- Calculate computational complexity (FLOPs)
- Orchestrate the pipeline
- Provide deterministic, reproducible results

**Dependencies:**
- NumPy, SciPy (numerical computation)
- Scikit-learn (BM25 retrieval)
- Pandas (data manipulation)
- LLMs (GPT/Claude for verification)

**Folder Structure:**

`
core/
├── orchestrator/
│   ├── pipeline.py           # Central orchestrator
│   └── determinism_manager.py # Ensures reproducibility
├── agents/
│   ├── parsing_agent.py       # Abstract agent interface
│   ├── parsing_agent_impl.py   # Concrete parsing implementation
│   └── config_parser.py        # Layer configuration parser
├── rag/
│   ├── config_extractor.py     # Deterministic extraction engine
│   ├── tensor_tracker.py       # Tensor shape validation
│   ├── knowledge_graph.py      # Semantic graph builder
│   ├── flops_engine.py         # FLOP calculation
│   └── retrieval.py            # BM25 retrieval
└── models/
    ├── representations.py      # Internal data structures
    └── exceptions.py           # Custom exceptions
`

**Important Files & Why They Exist:**

### 4.1.1 core/orchestrator/pipeline.py
**Purpose:** Central orchestrator wiring all components.

**What It Does:**
`python
class Pipeline:
    def __init__(self, parsing_agent, tensor_tracker, knowledge_graph, visualization_engine):
        self.parsing_agent = parsing_agent
        self.tensor_tracker = tensor_tracker
        self.knowledge_graph = knowledge_graph
        self.visualization_engine = visualization_engine
    
    def execute(self, paper_config):
        # Step 1: Parse paper
        config = self.parsing_agent.parse(paper_config)
        
        # Step 2: Track tensor flow
        tensor_flow = self.tensor_tracker.track(config)
        
        # Step 3: Build knowledge graph
        knowledge_graph = self.knowledge_graph.build(config)
        
        # Step 4: Visualize
        visualization = self.visualization_engine.render(config, tensor_flow, knowledge_graph)
        
        return {"config": config, "tensor_flow": tensor_flow, "graph": knowledge_graph, "visualization": visualization}
`

**Why It Exists:**
- Single entry point for pipeline execution
- Dependency injection central location
- Easy to extend (add new steps)
- Testable in isolation

**What Breaks If Removed:**
- No orchestration; steps must be called manually
- Logic scattered across backend/main.py
- No unified pipeline execution

**Used By:**
- main.py - CLI entry point
- ackend/server.py - API endpoints

---

### 4.1.2 core/agents/parsing_agent.py (Abstract)
**Purpose:** Defines parsing contract.

**What It Does:**
`python
class ParsingAgent(ABC):
    @abstractmethod
    def parse(self, paper_text: str) -> dict:
        """Parse paper text into configuration dict"""
        pass
    
    @abstractmethod
    def validate(self, config: dict) -> bool:
        """Validate configuration completeness"""
        pass
`

**Why It Exists:**
- Contract for implementations
- Enables multiple parsing strategies
- Testable via mocking

---

### 4.1.3 core/agents/parsing_agent_impl.py (Concrete)
**Purpose:** Actual parsing implementation.

**What It Does:**
1. Takes raw paper text
2. Routes through RAG pipeline:
   - Config extraction (structured parameters)
   - Symbolic extraction (equations, formulas)
   - Text extraction (descriptions)
3. Combines results into unified config

**Deterministic Guarantee:**
`python
def parse(self, paper_text: str) -> dict:
    # Step 1: Extract with deterministic BM25
    bm25_results = self.config_extractor.extract(paper_text)
    
    # Step 2: Verify with LLM (deterministic seed)
    verified = self.verify_with_llm(bm25_results, seed=42)
    
    # Step 3: Combine
    config = self.merge_results(bm25_results, verified)
    
    # Same input -> same output every time
    return config
`

**Why It Exists:**
- Implements parsing strategy
- Handles all extraction types
- Ensures determinism

**What Breaks If Removed:**
- No actual parsing capability
- Backend can't parse papers

---

### 4.1.4 core/rag/config_extractor.py
**Purpose:** Deterministically extract configurations from unstructured paper text.

**What It Does:**
1. **Indexing**: Build BM25 full-text index of paper
2. **Retrieval**: Find relevant passages for each parameter
3. **Extraction**: Parse passages into structured form
4. **Verification**: Validate extracted values against known patterns
5. **Reconciliation**: Merge multiple extraction attempts

**Determinism Mechanism:**
`python
class ConfigExtractor:
    def __init__(self, seed: int = 42):
        self.seed = seed  # Fixed seed
        self.bm25 = BM25Okapi(...)  # Deterministic retrieval
        self.llm_client = LLMClient(temperature=0)  # Deterministic LLM
    
    def extract(self, paper_text: str) -> dict:
        # BM25 is deterministic (same retrieval every time)
        passages = self.bm25.retrieve(keywords, top_k=5)
        
        # LLM with temperature=0 is deterministic
        extraction = self.llm_client.extract(passages, temperature=0)
        
        return extraction
`

**Why It Exists:**
- Core differentiator: deterministic extraction from papers
- Prevents hallucination (verification loop)
- Enables reliable parsing at scale

**What Breaks If Removed:**
- Can't extract configs from papers
- No way to parse new architectures

---

### 4.1.5 core/rag/tensor_tracker.py
**Purpose:** Track tensor shapes and validate correctness during parsing.

**What It Does:**
1. **Shape Inference**: Given layer config, infer output shape
2. **Validation**: Ensure outputs are sensible
3. **Tracking**: Log all tensor operations
4. **FLOP Calculation**: Compute FLOPs for each layer
5. **Mismatch Detection**: Raise errors on shape mismatches

**Implementation Example:**
`python
class TensorTracker:
    def track_layer(self, layer_type: str, input_shape: List[int], config: dict) -> List[int]:
        if layer_type == "Conv2d":
            batch, height, width, channels = input_shape
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            padding = config["padding"]
            out_channels = config["out_channels"]
            
            out_height = (height - kernel_size + 2*padding) // stride + 1
            out_width = (width - kernel_size + 2*padding) // stride + 1
            output_shape = [batch, out_height, out_width, out_channels]
            
            if any(dim <= 0 for dim in output_shape):
                raise TensorMismatchError(f"Invalid shape: {output_shape}")
            
            # Calculate FLOPs
            flops = 2 * kernel_size * kernel_size * channels * out_height * out_width * out_channels
            self.log_flop(layer_type, flops)
            
            return output_shape
`

**Why It Exists:**
- Prevents silent failures (catches shape mismatches early)
- Enables FLOP tracking
- Validates parsing correctness

**What Breaks If Removed:**
- Silent tensor mismatches downstream
- No FLOP calculation
- No validation of parsed configs

---

### 4.1.6 core/rag/knowledge_graph.py
**Purpose:** Build semantic graph representation of architectures.

**What It Does:**
1. **Node Creation**: Create node for each layer/parameter
2. **Edge Creation**: Connect nodes (input_to, output_from, depends_on)
3. **DAG Validation**: Ensure no cycles
4. **Traversal**: Enable graph queries (paths, reachability)

**Graph Structure:**
`python
class KnowledgeGraph:
    def build(self, config: dict) -> Graph:
        graph = Graph()
        
        # Create nodes
        for layer in config["layers"]:
            node = GraphNode(id=layer["id"], type=layer["type"], attributes=layer)
            graph.add_node(node)
        
        # Create edges
        for conn in config["connections"]:
            graph.add_edge(conn["from"], conn["to"], relation="data_flow")
        
        # Validate DAG (no cycles)
        if self.has_cycle(graph):
            raise ValueError("Graph contains cycle - not a valid DAG")
        
        # Topological sort (execution order)
        execution_order = graph.topological_sort()
        
        return graph
`

**Why It Exists:**
- Enables semantic comparison (graph matching)
- Validates architectural correctness
- Enables reasoning about data flow

**What Breaks If Removed:**
- Can't compare architectures semantically
- No validation of connectivity
- No execution order guarantees

---

### 4.1.7 core/rag/flops_engine.py
**Purpose:** Calculate computational complexity (FLOPs) for all operation types.

**What It Does:**
- Implement FLOP formulas for each layer type
- Track cumulative FLOPs through network
- Compare FLOP costs between architectures

**FLOP Calculations:**
`python
class FLOPsEngine:
    def calculate_flops(self, layer_type: str, input_shape: List[int], config: dict) -> int:
        if layer_type == "Conv2d":
            batch, h, w, c = input_shape
            k = config["kernel_size"]
            s = config["stride"]
            p = config["padding"]
            oc = config["out_channels"]
            
            oh = (h - k + 2*p) // s + 1
            ow = (w - k + 2*p) // s + 1
            
            return 2 * k * k * c * oh * ow * oc
        
        elif layer_type == "Linear":
            return 2 * input_shape[-1] * config["out_features"]
        
        # ... other layer types
`

**Why It Exists:**
- Enables efficiency analysis
- Supports architecture comparison
- Helps identify bottlenecks

---

## 4.2 ackend/ - REST API Layer

**Purpose:**
REST API exposing Paper2Code functionality via HTTP.

**Responsibilities:**
- Handle HTTP requests
- Validate input
- Orchestrate core pipeline
- Persist data to database
- Return JSON responses
- Auto-generate API documentation

**Dependencies:**
- FastAPI (HTTP framework)
- SQLAlchemy (ORM)
- Pydantic (request validation)
- SQLite/PostgreSQL (database)

**Folder Structure:**

`
backend/
├── server.py          # Main FastAPI app & endpoints
├── models.py          # SQLAlchemy ORM models
├── schemas.py         # Pydantic request/response schemas
├── database.py        # DB connection & session management
├── config.py          # Configuration settings
└── utils.py           # Utility functions
`

**Important Files:**

### 4.2.1 ackend/server.py
**Purpose:** FastAPI app with all REST endpoints.

**8 Main Endpoints:**

`python
from fastapi import FastAPI, Depends
from backend.database import get_session

app = FastAPI(title="Paper2Code API")

# Endpoint 1: Parse paper
@app.post("/api/parse")
async def parse_paper(paper: PaperSchema, session = Depends(get_session)):
    """Parse paper and extract architecture"""
    pipeline = get_pipeline()
    result = pipeline.execute(paper.content)
    
    db_paper = Paper(title=paper.title, content=paper.content, parsed_config=result["config"])
    session.add(db_paper)
    session.commit()
    
    return result

# Endpoint 2: Generate module code
@app.post("/api/generate")
async def generate_module(paper_id: int, session = Depends(get_session)):
    """Generate code for parsed module"""
    paper = session.query(Paper).get(paper_id)
    module_code = code_generator.generate(paper.parsed_config)
    return {"code": module_code}

# Endpoint 3: Compare architectures
@app.post("/api/compare")
async def compare_architectures(arch1_id: int, arch2_id: int, session = Depends(get_session)):
    """Compare two architectures"""
    arch1 = session.query(Paper).get(arch1_id)
    arch2 = session.query(Paper).get(arch2_id)
    comparison = comparator.compare(arch1.parsed_config, arch2.parsed_config)
    return comparison

# Endpoint 4: Get architecture analysis
@app.get("/api/analyze/{paper_id}")
async def analyze_architecture(paper_id: int, session = Depends(get_session)):
    """Detailed analysis of architecture"""
    paper = session.query(Paper).get(paper_id)
    analysis = analyzer.analyze(paper.parsed_config)
    return analysis

# Endpoint 5: List papers
@app.get("/api/papers")
async def list_papers(session = Depends(get_session)):
    """List all parsed papers"""
    return session.query(Paper).all()

# Endpoint 6: Get paper details
@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: int, session = Depends(get_session)):
    """Get specific paper"""
    return session.query(Paper).get(paper_id)

# Endpoint 7: Delete paper
@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: int, session = Depends(get_session)):
    """Delete paper and associated modules"""
    paper = session.query(Paper).get(paper_id)
    session.delete(paper)
    session.commit()
    return {"deleted": paper_id}

# Endpoint 8: Get explanations
@app.get("/api/explain/{paper_id}")
async def explain_architecture(paper_id: int, session = Depends(get_session)):
    """Get KAG explanations for architecture"""
    paper = session.query(Paper).get(paper_id)
    explanations = explainer.explain(paper.parsed_config)
    return explanations
`

**Why It Exists:**
- Standard interface for all Paper2Code operations
- Decouples core logic from HTTP concerns
- Enables web UI interaction
- Auto-generates OpenAPI documentation

---

### 4.2.2 ackend/models.py
**Purpose:** SQLAlchemy ORM entity definitions.

**Key Entities:**

`python
class Paper(Base):
    __tablename__ = 'papers'
    id = Column(Integer, primary_key=True)
    title = Column(String, unique=True)
    content = Column(Text)
    parsed_config = Column(JSON)  # Architecture config
    created_at = Column(DateTime, default=datetime.utcnow)

class Module(Base):
    __tablename__ = 'modules'
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'))
    name = Column(String)
    code = Column(Text)  # Generated PyTorch code
    tensor_flow = Column(JSON)  # Tensor tracking info
    flops_total = Column(Integer)  # Total FLOPs

class Comparison(Base):
    __tablename__ = 'comparisons'
    id = Column(Integer, primary_key=True)
    paper1_id = Column(Integer, ForeignKey('papers.id'))
    paper2_id = Column(Integer, ForeignKey('papers.id'))
    similarity_score = Column(Float)
    differences = Column(JSON)  # Detailed differences
    created_at = Column(DateTime, default=datetime.utcnow)

class Explanation(Base):
    __tablename__ = 'explanations'
    id = Column(Integer, primary_key=True)
    paper_id = Column(Integer, ForeignKey('papers.id'))
    explanation_text = Column(Text)  # KAG explanation
    generation_timestamp = Column(DateTime, default=datetime.utcnow)
`

**Why It Exists:**
- Persistent data storage
- Reproducible queries
- Type-safe ORM
- Migration support via Alembic

---

### 4.2.3 ackend/database.py
**Purpose:** Database connection and session management.

`python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./paper2code.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Create tables
Base.metadata.create_all(bind=engine)
`

**Why It Exists:**
- Single connection point
- Session lifecycle management
- Dependency injection for FastAPI endpoints

---

## 4.3 rontend/ - Web UI

**Purpose:**
User interface for Paper2Code.

**Responsibilities:**
- Display parsed architectures
- Show comparisons
- Render explanations
- Provide visualization
- Enable user interactions

**Current State:**
- static/index.html - Primary lightweight HTML/JS UI
- static/app.js - Frontend logic
- rontend/ - Next.js stubs for future enhancement

**Important Files:**

### 4.3.1 static/index.html
**Purpose:** Main web interface.

**What It Does:**
`html
<!DOCTYPE html>
<html>
<head>
    <title>Paper2Code</title>
    <script src="app.js"></script>
</head>
<body>
    <div id="upload-section">
        <!-- File upload form -->
        <input type="file" id="paper-input">
        <button onclick="parsePaper()">Parse</button>
    </div>
    
    <div id="results-section">
        <!-- Displays parsing results -->
        <div id="architecture-viz"></div>
        <div id="tensor-flow"></div>
        <div id="comparison-results"></div>
    </div>
</body>
</html>
`

**Why It Exists:**
- Lightweight interface
- No build step required
- Fast deployment

---

### 4.3.2 static/app.js
**Purpose:** Frontend logic and API interaction.

`javascript
async function parsePaper() {
    const paper = document.getElementById('paper-input').files[0];
    const formData = new FormData();
    formData.append('paper', paper);
    
    const response = await fetch('/api/parse', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    displayResults(result);
}

function displayResults(result) {
    document.getElementById('architecture-viz').innerHTML = 
        <pre></pre>;
    document.getElementById('tensor-flow').innerHTML = 
        <pre></pre>;
}
`

**Why It Exists:**
- User interaction logic
- API communication
- Result visualization

---

## 4.4 	ests/ - Test Suite

**Purpose:**
Comprehensive tests ensuring correctness and determinism.

**Responsibilities:**
- Validate parsing correctness
- Ensure determinism (reproducibility)
- Test comparators
- Test tensor tracking
- Test knowledge graphs

**Test Files:**

`
tests/
├── test_pipeline_determinism.py        # Ensures same input → same output
├── test_config_extractor.py            # Tests RAG extraction
├── test_config_parser.py               # Tests layer config parsing
├── test_tensor_tracker.py              # Tests shape inference
├── test_transformer_builder.py         # Tests Transformer construction
├── test_vit_patch_embedding.py         # Tests ViT-specific logic
├── test_resnet_vs_vit.py              # Compares ResNet vs ViT
├── test_comparator_edge_cases.py       # Tests comparison robustness
├── test_visual_comparison.py           # Tests visualization
├── test_backward_compat.py             # Ensures backward compatibility
└── conftest.py                         # Shared pytest fixtures
`

**Key Test Pattern - Determinism:**

`python
def test_parsing_is_deterministic():
    """Parse same paper twice, verify identical results"""
    parsing_agent = ParsingAgentImpl(seed=42)
    
    result1 = parsing_agent.parse(sample_paper_text)
    result2 = parsing_agent.parse(sample_paper_text)
    
    # Must be identical
    assert result1 == result2
    assert result1['config'] == result2['config']
    assert result1['tensor_flow'] == result2['tensor_flow']
`

**Why Tests Exist:**
- Catch regressions
- Document expected behavior
- Enable refactoring confidently
- Verify determinism claim

---

## 4.5 migrations/ - Database Schema Versioning

**Purpose:**
Track database schema changes over time.

**Folder Structure:**

`
migrations/
├── alembic.ini             # Alembic configuration
├── env.py                  # Migration environment setup
├── script.py.mako          # Migration template
└── versions/
    ├── 001_initial.py      # Initial schema
    ├── 002_add_modules.py   # Add module table
    └── 003_add_comparisons.py # Add comparison table
`

**Typical Migration File:**

`python
# migrations/versions/001_initial.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'papers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('parsed_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    op.drop_table('papers')
`

**Why It Exists:**
- Track schema evolution
- Enable rollback
- Collaborate on DB changes
- Reproducible deployments

---

## 4.6 scripts/ - Utility Scripts

**Purpose:**
One-off utility scripts for common tasks.

**Example Scripts:**

`
scripts/
├── setup_database.sh        # Initialize database
├── generate_sample_data.py   # Create test data
├── benchmark_pipeline.py     # Performance testing
├── export_results.py         # Export to CSV/JSON
└── visualize_architecture.py # Generate diagrams
`

**Why They Exist:**
- Automate repetitive tasks
- One-off data processing
- Performance testing
- Visualization generation

---

## 4.7 .github/workflows/ - CI/CD Pipelines

**Purpose:**
Automate testing, linting, and deployment.

**Files:**

`
.github/workflows/
├── ci.yml   # Continuous Integration
└── cd.yml   # Continuous Deployment
`

### CI Workflow:
1. Checkout code
2. Set up Python environment
3. Install dependencies
4. Run black (formatter) check
5. Run flake8 (linter)
6. Run pytest (tests)
7. Upload coverage

### CD Workflow:
1. Trigger on version tag
2. Run CI checks
3. Build Docker image
4. Push to registry
5. Deploy to production

**Why It Exists:**
- Automated quality assurance
- Prevent bugs in production
- Enable rapid iteration safely
- Audit trail of changes

---

## 4.8 Folder Interactions & Data Flow

**How Folders Interact:**

`
Paper (file)
    ↓
frontend/static/
    (User uploads paper)
    ↓
backend/server.py (API)
    (Receives HTTP request)
    ↓
core/orchestrator/pipeline.py (Orchestration)
    ↓
core/agents/parsing_agent_impl.py (Parse paper)
    ↓
core/rag/config_extractor.py (Extract config)
core/rag/tensor_tracker.py (Validate tensors)
core/rag/knowledge_graph.py (Build graph)
core/rag/flops_engine.py (Calculate FLOPs)
    ↓
backend/models.py (ORM entity)
    ↓
backend/database.py (Session)
    ↓
SQLite Database (Persistence)
    ↓
backend/server.py (Return JSON)
    ↓
frontend/static/app.js (Display results)

Tests validate correctness at every step.
CI/CD automates the entire process.
`

---

## 4.9 Critical Dependencies Between Folders

| Folder | Depends On | Why |
|--------|-----------|-----|
| backend/ | core/ | Uses pipeline for execution |
| frontend/ | backend/ | Calls API endpoints |
| tests/ | core/, backend/ | Validates both layers |
| migrations/ | backend/ | Versions DB schema |
| scripts/ | core/, backend/ | Automates tasks |
| CI/CD | tests/ | Runs linters and tests |

---

## 4.10 What Breaks If Each Folder Is Removed

| Folder | Consequence |
|--------|-------------|
| core/ | All business logic lost - nothing works |
| backend/ | No HTTP API - CLI only |
| frontend/ | No web UI - API only via curl/Postman |
| tests/ | No quality assurance - CI/CD can't validate |
| migrations/ | Manual DB schema management - error-prone |
| scripts/ | No automation - manual effort increases |
| CI/CD | No automated testing - bugs reach production |

---


## PART 5: FOLLOW ONE REAL EXAMPLE - ResNet Through the System

Trace ResNet from paper through the entire system.

---

## 5.1 The ResNet Paper

**Source:** "Deep Residual Learning for Image Recognition" (He et al., 2015)

**Key Specifications from Paper:**
\\\
ResNet-50:
- Input: 224×224 RGB images
- Stage 1: conv1 7×7, stride 2, 64 filters
- Stage 2: 3×(3×3 conv) bottleneck, 64 filters
- Stage 3: 4×(3×3 conv) bottleneck, 128 filters (stride 2)
- Stage 4: 6×(3×3 conv) bottleneck, 256 filters (stride 2)
- Stage 5: 3×(3×3 conv) bottleneck, 512 filters (stride 2)
- Global average pooling
- Fully connected 1000 classes
- Skip connections throughout
\\\

---

## 5.2 Stage 1 - Paper Parsing

**Step 1.1: Paper Text Input**

The user uploads ResNet paper PDF text through the frontend. ConfigExtractor receives the raw text.

**Step 1.2: Config Extraction via RAG Pipeline**

- BM25 indexes paper sections
- Searches for architecture keywords
- LLM verifies extracted specs (temperature=0 for determinism)
- Output: structured parsed_config dict

**Output of Stage 1:**

\\\python
parsed_config = {
    "architecture": "ResNet-50",
    "input_shape": [1, 224, 224, 3],
    "layers": [
        {
            "id": "conv1",
            "type": "Conv2d",
            "config": {"kernel_size": 7, "stride": 2, "padding": 3, "out_channels": 64},
            "input_shape": [1, 224, 224, 3],
            "output_shape": [1, 112, 112, 64]
        },
    ],
    "connections": [...]
}
\\\

---

## 5.3 Stage 2 - Tensor Tracking

**Purpose:** Validate shapes and track tensor evolution through network.

**Shape Inference for Conv1:**

- Input: [1, 224, 224, 3] (batch=1, height=224, width=224, channels=3)
- Kernel: 7x7, Stride: 2, Padding: 3
- Formula: output = floor((input + 2*padding - kernel) / stride) + 1
- Height out: floor((224 + 2*3 - 7) / 2) + 1 = 112
- Width out: floor((224 + 2*3 - 7) / 2) + 1 = 112
- Output channels: 64
- Output: [1, 112, 112, 64] ✓

**FLOP Calculation for Conv1:**

- FLOPs = 2 * kernel_h * kernel_w * in_channels * out_h * out_w * out_channels
- FLOPs = 2 * 7 * 7 * 3 * 112 * 112 * 64
- FLOPs ≈ 11.7 billion

**After Processing All Layers:**

\\\
Tensor Trace:
├─ input: [1, 224, 224, 3], 0 FLOPs
├─ conv1: [1, 112, 112, 64], 11.7G FLOPs
├─ bn1: [1, 112, 112, 64], 0 FLOPs
├─ relu1: [1, 112, 112, 64], 0 FLOPs
├─ stage2_block0: [1, 112, 112, 256], FLOPs
├─ stage2_block1: [1, 112, 112, 256], FLOPs
├─ stage2_block2: [1, 112, 112, 256], FLOPs
├─ stage3_block0: [1, 56, 56, 512], FLOPs (stride=2)
├─ stage3_block1: [1, 56, 56, 512], FLOPs
├─ stage3_block2: [1, 56, 56, 512], FLOPs
├─ stage3_block3: [1, 56, 56, 512], FLOPs
├─ stage4_block0: [1, 28, 28, 1024], FLOPs (stride=2)
├─ ... more blocks ...
├─ stage5_block2: [1, 14, 14, 2048], FLOPs
├─ global_avg_pool: [1, 1, 1, 2048], 0 FLOPs
└─ fc_1000: [1, 1000], 4M FLOPs

Total FLOPs: ~7.7 billion
\\\

---

## 5.4 Stage 3 - Knowledge Graph Construction

**Building the DAG:**

- Create node for each layer
- Create edge for each connection
- Create edge for each skip connection
- Validate DAG (no cycles)
- Compute topological sort (execution order)

**Key Connections:**

\\\
input → conv1 → bn1 → relu1 → stage2_block0 → ...
                                    ↑
                                (skip)
                                    |
                            (merged with prev)
\\\

**Verification:**

- DAG check: No cycles ✓
- Reachability: All nodes reachable from input ✓
- Output: reachable from all nodes ✓

---

## 5.5 Stage 4 - Module Code Generation

**Generated PyTorch Code:**

\\\python
class ResNet50(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        self.fc = nn.Linear(2048, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
\\\

---

## 5.6 Stage 5 - Database Persistence

**Stored in Database:**

\\\sql
INSERT INTO papers (title, content, parsed_config) VALUES (
    'Deep Residual Learning for Image Recognition',
    '[full paper text]',
    '{architecture: ResNet-50, layers: [...], connections: [...]}'
);

INSERT INTO modules (paper_id, name, code, tensor_flow, flops_total) VALUES (
    1,
    'ResNet50',
    '[generated PyTorch code above]',
    '[tensor flow trace]',
    7700000000
);
\\\

---

## 5.7 Stage 6 - REST API Response

**API Returns JSON:**

\\\json
{
  "success": true,
  "paper_id": 1,
  "architecture": "ResNet-50",
  "summary": {
    "total_layers": 176,
    "total_parameters": 25500000,
    "total_flops": 7700000000,
    "input_shape": [1, 224, 224, 3],
    "output_shape": [1, 1000]
  },
  "code_generated": true,
  "knowledge_graph": {
    "nodes": 176,
    "edges": 200,
    "is_dag": true
  }
}
\\\

---

## 5.8 Stage 7 - Frontend Display

**JavaScript processes response and displays:**

- Architecture name and stats
- Layer-by-layer tensor flow table
- FLOPs breakdown
- Generated code
- DAG visualization

---

## 5.9 Complete End-to-End Summary

\\\
Paper PDF Upload
    ↓
[Parse] Extract config via RAG + LLM
    ↓
[Track] Validate tensors + calculate FLOPs
    ↓
[Graph] Build knowledge graph DAG
    ↓
[Generate] Create PyTorch code
    ↓
[Store] Persist to database
    ↓
[API] Return JSON response
    ↓
[Display] Frontend renders results

Result: ResNet-50 fully understood
- Architecture: defined
- Tensor flow: validated
- Computation: quantified (7.7G FLOPs)
- Code: generated
- Reproducible: deterministic throughout
\\\

---


## PART 6: DESIGN DECISIONS

For every major technology in Paper2Code: why was it chosen, what alternatives exist, why it's appropriate here, and what tradeoffs were made.

---

## 6.1 Python as Primary Language

**Choice:** Python

**Alternatives:**
- Java: Type-safe, enterprise-ready, but verbose and slow
- C++: Fast, efficient, but steep learning curve and slow development
- JavaScript/Node.js: Web-friendly, but weak numerical computing
- Go: Fast, simple, but small ML ecosystem

**Why Python?**
1. **Dominance in ML**: NumPy, SciPy, Scikit-Learn, PyTorch, TensorFlow are Python-first
2. **Rapid Development**: Dynamic typing enables faster prototyping
3. **Readability**: Code clarity enables complex logic (RAG, tensor tracking)
4. **Scientific Stack**: Pandas, Matplotlib, Seaborn for analysis
5. **Ecosystem**: 500K+ packages, strong community

**Tradeoffs:**
- ✓ Fast development
- ✓ Rich ecosystem
- ✗ Slower execution (mitigated by NumPy/Pandas C implementations)
- ✗ Runtime type errors (mitigated by type hints)
- ✗ GIL limits parallelization (not critical for this use case)

---

## 6.2 FastAPI for REST Backend

**Choice:** FastAPI

**Alternatives:**
- Django: Feature-rich, but heavyweight and overkill
- Flask: Lightweight, but requires more manual work (validation, docs)
- Starlette: Lower-level async framework, more control
- FastAPI Alternatives (Quart, Sanic): Niche ecosystems

**Why FastAPI?**
1. **Modern Python Features**: Type hints → automatic validation + documentation
2. **Automatic OpenAPI Docs**: /docs endpoint auto-generates Swagger UI
3. **Async by Default**: Concurrent request handling (important for multiple papers)
4. **Type Safety**: Pydantic models validate all input automatically
5. **Performance**: Among fastest Python frameworks
6. **Developer Experience**: Minimal boilerplate

**Implementation Example:**

\\\python
@app.post("/api/parse")
async def parse_paper(paper: PaperSchema):  # Type hint → auto validation
    result = pipeline.execute(paper.content)
    return result  # Auto JSON serialization
\\\

**Tradeoffs:**
- ✓ Minimal code
- ✓ Auto documentation
- ✓ Built-in validation
- ✗ Less mature than Django (but catching up)
- ✗ Smaller ecosystem of extensions
- Resolution: Trade reduced feature set for developer velocity (appropriate for this project)

---

## 6.3 SQLAlchemy ORM for Database Access

**Choice:** SQLAlchemy

**Alternatives:**
- Raw SQL: Full control, but prone to injection and tedious
- Django ORM: Opinionated, tied to Django framework
- Tortoise ORM: Async-first, newer, less mature
- Pony ORM: Interesting query syntax, niche
- Direct database drivers: Fast but no abstraction

**Why SQLAlchemy?**
1. **Database Agnostic**: Works with SQLite, PostgreSQL, MySQL, etc.
2. **Pythonic Queries**: ORM hides SQL but power is available
3. **Declarative Models**: Classes map cleanly to tables
4. **Mature & Stable**: Production-tested in 1000s of projects
5. **Migration Support**: Alembic integrates seamlessly
6. **Relationships**: Foreign keys and joins are simple

**Implementation Example:**

\\\python
class Paper(Base):
    __tablename__ = 'papers'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    parsed_config = Column(JSON)

# Query
paper = session.query(Paper).filter_by(id=1).first()
\\\

**Tradeoffs:**
- ✓ Clean Python syntax
- ✓ Database independence
- ✓ Mature ecosystem
- ✗ Slower than raw SQL (negligible for this scale)
- ✗ Learning curve steeper than simpler ORMs
- Resolution: Maturity and flexibility outweigh learning curve

---

## 6.4 SQLite for Development Database

**Choice:** SQLite

**Alternatives for Development:**
- PostgreSQL: Powerful, but requires separate service
- MySQL: Similar to PostgreSQL
- MongoDB: NoSQL, different paradigm (doesn't fit structured config)

**Why SQLite?**
1. **Zero Setup**: File-based, no server needed
2. **Perfect for Development**: Lightweight, instant startup
3. **Sufficient for Scale**: Good for 10K+ papers (single server)
4. **Easy Testing**: Each test gets isolated database file
5. **Portable**: Entire DB is one file
6. **Production Migration**: Easy to migrate to PostgreSQL later

**Tradeoffs:**
- ✓ Development speed
- ✓ Testing simplicity
- ✓ No infrastructure
- ✗ Single-writer limitation (not an issue for this project)
- ✗ Less suitable for distributed systems
- Resolution: SQLite for dev, migration path to PostgreSQL for production (via Alembic)

---

## 6.5 Alembic for Database Migrations

**Choice:** Alembic

**Alternatives:**
- No version control: Manual schema management (error-prone)
- Django Migrations: Tied to Django framework
- Flyway: Java-based, works with SQLAlchemy but different paradigm
- Liquibase: Verbose XML/YAML, overkill for this project

**Why Alembic?**
1. **SQLAlchemy Native**: Designed for SQLAlchemy integration
2. **Auto-generation**: Detects model changes, auto-generates migrations
3. **Versioning**: Full history of schema changes
4. **Rollback Support**: Easy downgrade if needed
5. **Team Collaboration**: Merge-friendly migration files

**Implementation:**

\\\ash
# Auto-generate migration
alembic revision --autogenerate -m "Add comparison table"

# Apply
alembic upgrade head

# Rollback
alembic downgrade -1
\\\

**Tradeoffs:**
- ✓ Seamless with SQLAlchemy
- ✓ Automatic detection
- ✗ Learning curve (small)
- Resolution: Small learning curve pays off quickly

---

## 6.6 Pydantic for Request Validation

**Choice:** Pydantic

**Alternatives:**
- Marshmallow: Older, more verbose
- attrs: Too low-level
- Manual validation: Error-prone, repetitive
- Cerberus: Limited type support

**Why Pydantic?**
1. **Type-Driven**: Uses Python type hints for validation
2. **Fast**: C-based validation (pydantic-core)
3. **Error Messages**: Clear, actionable validation errors
4. **Coercion**: Automatic type conversion (string "123" → int 123)
5. **FastAPI Integration**: First-class support

**Implementation:**

\\\python
from pydantic import BaseModel

class PaperSchema(BaseModel):
    title: str  # Validates string
    content: str  # Validates string
    year: int  # Validates int
    authors: List[str]  # Validates list of strings

# Usage in FastAPI
@app.post("/api/parse")
def parse(paper: PaperSchema):  # Pydantic validates
    # paper.title is definitely string, no defensive checks needed
    return result
\\\

**Tradeoffs:**
- ✓ Type safety
- ✓ Automatic validation
- ✓ Clear errors
- ✗ Slight performance overhead (negligible in web context)

---

## 6.7 BM25 for Full-Text Retrieval

**Choice:** BM25 (via Scikit-Learn)

**Alternatives:**
- Vector Embeddings (BERT, Sentence-Transformers): Semantic but slower, more expensive
- Regex-based Search: Too rigid for natural language
- Elasticsearch: Powerful but operational overhead
- Database Full-Text Search (SQLite FTS): Limited precision

**Why BM25?**
1. **Deterministic**: Same query always returns same results (critical for reproducibility)
2. **Fast**: Millisecond-scale retrieval
3. **Effective**: Proven statistical model for IR
4. **Lightweight**: No external services
5. **Reason-able**: Scores are interpretable (term frequency + inverse document frequency)

**Implementation:**

\\\python
from sklearn.feature_extraction.text import BM25Transformer

# Index papers
bm25 = BM25Transformer()
indices = bm25.fit_transform(paper_sections)

# Query (deterministic)
query = "convolutional kernel size stride padding"
results = bm25.transform([query])
top_sections = results.argsort()[-5:][::-1]  # Top 5
\\\

**Tradeoffs:**
- ✓ Deterministic
- ✓ Fast
- ✓ Lightweight
- ✗ Not semantic (mitigated by LLM verification layer)
- Resolution: BM25 retrieval + LLM verification = best of both worlds

---

## 6.8 LLM Verification Layer (for Determinism)

**Choice:** LLM with Temperature=0

**Alternatives:**
- No verification: Rely on BM25 (less accurate)
- Rule-based extraction: Hard-coded patterns (brittle)
- Random Temperature (default): Non-deterministic
- Fine-tuned models: Complex, expensive

**Why LLM Verification?**
1. **Accuracy**: LLM understands context better than BM25
2. **Determinism**: Temperature=0 removes randomness
3. **Flexibility**: Handles diverse paper formats
4. **Reproducibility**: Same input → same output

**Implementation:**

\\\python
# Deterministic: temperature=0, no sampling
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[...],
    temperature=0,  # Always pick highest probability token
    seed=42  # Fixed seed (if API supports)
)
\\\

**Tradeoffs:**
- ✓ Accurate extraction
- ✓ Deterministic
- ✗ API cost
- ✗ Rate limits
- Resolution: Worth the cost for reliability

---

## 6.9 Tensor Tracking & Validation at Parse-Time

**Choice:** Validate tensor shapes during parsing

**Alternatives:**
- Lazy validation: Errors appear during training (too late)
- No validation: Silent errors propagate
- Post-hoc validation: Separate validation step

**Why Parse-Time Validation?**
1. **Early Error Detection**: Catch issues immediately
2. **Fast Feedback**: Developer knows instantly if parsing worked
3. **Prevents Cascading Failures**: Invalid shapes stop pipeline
4. **Testing**: Enables determinism testing (same shape means correct parsing)

**Implementation:**

\\\python
def track_conv2d(input_shape, kernel_size, stride, padding):
    batch, h, w, c = input_shape
    oh = (h - kernel_size + 2*padding) // stride + 1
    ow = (w - kernel_size + 2*padding) // stride + 1
    
    if oh <= 0 or ow <= 0:
        raise TensorMismatchError(f"Conv produces invalid output: {[batch, oh, ow, c]}")
    
    return [batch, oh, ow, c]
\\\

**Tradeoffs:**
- ✓ Early error detection
- ✓ Prevents downstream failures
- ✗ Tight coupling between parsing and validation (acceptable)

---

## 6.10 Knowledge Graphs for Architecture Semantics

**Choice:** DAG representation via Knowledge Graphs

**Alternatives:**
- Linear List: Simpler but loses connection info
- Adjacency Matrix: Memory inefficient for sparse graphs
- Simple Dict: Ad-hoc, not formally validated

**Why Graphs?**
1. **Semantic Richness**: Represents connections, skip connections, dependencies
2. **Validation**: DAG check ensures valid architecture
3. **Reasoning**: Enables graph queries (paths, reachability)
4. **Comparison**: Graph isomorphism enables architecture comparison

**Implementation:**

\\\python
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
    
    def validate_dag(self):
        if self.has_cycle():
            raise ValueError("Not a DAG")
    
    def topological_sort(self):
        # Execution order
        return sorted(self.nodes, key=self.in_degree)
\\\

**Tradeoffs:**
- ✓ Semantic representation
- ✓ Enables reasoning
- ✗ More complex than lists
- Resolution: Complexity justified by capabilities

---

## 6.11 Static HTML/JS Frontend (for Initial Development)

**Choice:** Static HTML/JS + JavaScript for interactivity

**Alternatives:**
- React: Powerful, but requires build step and Node
- Vue: Lighter than React, but still requires build tooling
- Next.js: Full-stack, but heavy for simple UI

**Why Static HTML/JS?**
1. **No Build Step**: Direct HTML file, instant development
2. **No Dependencies**: Single index.html + app.js
3. **Fast Deployment**: Copy files to server
4. **Zero Configuration**: Works in any environment
5. **Accessible**: Easy to understand and modify

**Implementation:**

\\\html
<!DOCTYPE html>
<html>
<head>
    <script src="app.js"></script>
</head>
<body>
    <input type="file" id="paper-input">
    <button onclick="parsePaper()">Parse</button>
    <pre id="results"></pre>
</body>
</html>
\\\

**Tradeoffs:**
- ✓ No build tooling
- ✓ Fast development
- ✗ Limited features as complexity grows
- Resolution: Next.js stubs available for future enhancement

---

## 6.12 GitHub Actions for CI/CD

**Choice:** GitHub Actions

**Alternatives:**
- Jenkins: Powerful but requires infrastructure
- GitLab CI: Good, but ties to GitLab
- CircleCI: Paid tier for private repos
- GitHub Actions: Free, GitHub-native

**Why GitHub Actions?**
1. **GitHub-Native**: Built-in, no external service
2. **Free for Public Repos**: No cost
3. **Generous Free Tier**: 2000 minutes/month for private
4. **YAML Config**: Simple, version-controlled
5. **Ecosystem**: 1000s of pre-built actions

**Implementation:**

\\\yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - run: pip install -r requirements.txt
      - run: pytest
\\\

**Tradeoffs:**
- ✓ No infrastructure
- ✓ Free
- ✓ Simple
- ✗ Proprietary to GitHub
- Resolution: GitHub is the standard anyway

---

## 6.13 Pytest for Testing Framework

**Choice:** Pytest

**Alternatives:**
- Unittest: Built-in but verbose
- Nose: Older, less maintained
- Hypothesis: Property-based, but niche

**Why Pytest?**
1. **Simple Syntax**: assert statements instead of self.assertEqual
2. **Fixtures**: Powerful setup/teardown mechanism
3. **Auto-Discovery**: Finds tests automatically
4. **Plugins**: Rich ecosystem (coverage, xdist, etc.)
5. **Parametrization**: Test multiple inputs easily

**Implementation:**

\\\python
@pytest.mark.parametrize("input_shape,expected_output", [
    ([1, 224, 224, 3], [1, 112, 112, 64]),
    ([2, 112, 112, 64], [2, 56, 56, 128]),
])
def test_conv_shapes(input_shape, expected_output):
    output = tracker.infer_conv_shape(input_shape, kernel=3, stride=2)
    assert output == expected_output
\\\

**Tradeoffs:**
- ✓ Simple, readable
- ✓ Powerful
- ✗ Learning curve steeper than unittest (small)

---

## 6.14 Design Decisions Summary Table

| Component | Chosen | Main Alternatives | Key Reason | Tradeoff |
|-----------|--------|-------------------|-----------|----------|
| Language | Python | Java, C++, JS | ML ecosystem dominance | Slower execution (mitigated) |
| Backend | FastAPI | Django, Flask | Modern, type-safe, auto-docs | Less mature than Django |
| ORM | SQLAlchemy | Django ORM, Raw SQL | Database agnostic, Pythonic | Slight perf overhead |
| Dev DB | SQLite | PostgreSQL | Zero setup | Single-writer limit |
| Migrations | Alembic | Manual, Django | SQLAlchemy-native | Learning curve |
| Validation | Pydantic | Marshmallow | Type-driven, fast | Slight perf cost |
| Retrieval | BM25 | Embeddings | Deterministic, fast | Not semantic (mitigated by LLM) |
| Verification | LLM (T=0) | Rule-based | Context understanding | API cost |
| Tensor Check | Parse-time | Lazy | Early error detection | Tight coupling |
| Graph Repr | DAG | Linear list | Semantic richness | More complex |
| Frontend | Static HTML/JS | React, Vue | No build step, fast dev | Limited as complexity grows |
| CI/CD | GitHub Actions | Jenkins | Free, GitHub-native | Proprietary |
| Testing | Pytest | Unittest | Readable, powerful | Small learning curve |

---


## PART 7: INTERVIEW PREPARATION

Based ONLY on this repository, generate project explanations and answer common interview questions.

---

## 7.1 Two-Minute Project Explanation

**Opening:**

"I built Paper2Code, a system that automatically extracts neural network architectures from academic papers and converts them into production-ready PyTorch code.

**Problem:**

Researchers publish breakthroughs in papers with architecture descriptions like '3x bottleneck blocks with 64 filters.' Implementing these from scratch takes hours of manual coding and is error-prone. My system solves this.

**Solution:**

The system uses a Retrieval-Augmented Generation (RAG) pipeline to deterministically parse papers, extract architectural specifications, validate tensor shapes, build semantic knowledge graphs, and generate code. Every step is reproducible—same paper always produces same output.

**Architecture:**

- Backend (FastAPI): REST API with 8 endpoints
- Core (Python): Parsing agents, RAG pipeline, tensor tracking
- Database (SQLAlchemy + SQLite): Persists papers and generated modules
- Frontend (HTML/JS): Web UI for uploads and visualization

**Key Innovation:**

Determinism. Most ML systems are non-deterministic. I ensured parsing is bit-perfect reproducible using fixed-seed LLMs, BM25 retrieval, and explicit validation.

**Impact:**

Researchers can now go from paper → working code in seconds instead of hours. Enables rapid architecture exploration and comparison."

---

## 7.2 Five-Minute Project Explanation

**Technical Depth:**

[Repeat 2-minute explanation, then add:]

"**How it works technically:**

1. Paper Upload: User uploads ResNet paper PDF
2. Parsing (RAG Pipeline): 
   - BM25 indexes paper sections (deterministic full-text search)
   - Extracts specifications ('7x7 conv, stride 2, 64 filters')
   - LLM verifies extracted values (temperature=0 for reproducibility)
   - Returns structured config dict

3. Tensor Tracking:
   - For each layer: infer output shape
   - Example: Conv(H=224, K=7, S=2, P=3) → H_out=112
   - Validate output is sensible, raise error if not
   - Calculate FLOPs per layer (~11.7B for conv1)

4. Knowledge Graph:
   - Create nodes for each layer
   - Create edges for connections + skip connections
   - Validate DAG (no cycles)
   - Enable semantic reasoning

5. Code Generation:
   - Bottleneck class with 1x1→3x3→1x1 structure
   - Skip connections via shortcut layers
   - Full training loop with forward/backward passes

6. Storage & API:
   - Store in SQLite (Paper, Module, Comparison tables)
   - Serve via FastAPI (/api/parse, /api/compare, etc.)
   - Frontend displays results interactively

**Why This Approach:**

- **Determinism**: BM25 + LLM (T=0) ensure reproducibility
- **Validation**: Tensor tracking catches errors early
- **Semantics**: Knowledge graphs enable reasoning
- **Scalability**: Parse 1000s of papers independently

**Testing:**

20+ test files validate:
- Parsing determinism (same input → same output)
- Tensor correctness (shapes match expected)
- Graph validity (DAG properties)
- Backward compatibility (old papers still parse)

**Technologies:**

Core: Python 3.9
Backend: FastAPI + Pydantic
Data: SQLAlchemy + SQLite + Alembic
ML: NumPy, Pandas, Scikit-Learn
Testing: Pytest with 90%+ coverage
CI/CD: GitHub Actions (lint, test, deploy)
"

---

## 7.3 Fifteen-Minute Deep Dive

[Combine 2 + 5 minute explanations, then add architectural discussions:]

**File Organization & Dependencies:**

\\\
core/
├── orchestrator/pipeline.py (central wiring)
├── agents/parsing_agent_impl.py (RAG extraction)
├── rag/
│   ├── config_extractor.py (BM25 + LLM)
│   ├── tensor_tracker.py (shape validation + FLOPs)
│   ├── knowledge_graph.py (DAG construction)
│   └── flops_engine.py (FLOP calculations)

backend/
├── server.py (8 FastAPI endpoints)
├── models.py (SQLAlchemy ORM)
├── database.py (session management)

frontend/
├── static/index.html (web UI)
├── static/app.js (client logic)

tests/ (20+ test files for determinism)

migrations/ (Alembic schema versioning)
\\\

**Design Patterns Used:**

1. **Dependency Injection**: Pipeline receives parsing_agent, tensor_tracker, knowledge_graph
2. **Repository Pattern**: Database access abstracted
3. **Service Layer**: Pipeline encapsulates business logic
4. **Strategy Pattern**: Different agents implement ParsingAgent interface
5. **Decorator Pattern**: FastAPI's @app.post() decorates endpoints

**Key Algorithms:**

1. **BM25 Retrieval:**
   `
   Score(Query, Doc) = IDF(q) * (f(q,D) * (k1+1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl))
   `
   Where f(q,D) is term frequency in document

2. **Tensor Shape Inference (Conv2d):**
   `
   H_out = floor((H_in - kernel + 2*padding) / stride) + 1
   `

3. **FLOP Calculation (Conv2d):**
   `
   FLOPs = 2 * K * K * C_in * H_out * W_out * C_out
   `

4. **DAG Validation:**
   `
   TopologicalSort(Graph) ← if has_cycle(Graph) then ERROR
   `

**Trade-offs Made:**

1. **BM25 vs Semantic Embeddings:**
   - Chose: BM25 (deterministic, fast, lightweight)
   - Trade: Less semantic understanding (mitigated by LLM verification)

2. **SQLite vs PostgreSQL:**
   - Chose: SQLite (zero setup, great for dev)
   - Trade: Single-writer, but migration path via Alembic exists

3. **Static Frontend vs React:**
   - Chose: Static HTML/JS (no build step, fast dev)
   - Trade: Limited features, but Next.js stubs available

4. **Parse-time Validation vs Lazy:**
   - Chose: Parse-time (early error detection)
   - Trade: Tighter coupling (acceptable for this domain)

**Testing Strategy:**

- **Determinism Tests**: Parse ResNet twice, verify identical JSON
- **Shape Validation**: Conv(224x224x3) with K=7, S=2 → 112x112x64
- **Graph Properties**: Verify DAG, no cycles, all nodes reachable
- **Backward Compatibility**: Old parsing still works

**Performance Characteristics:**

- Parse simple ResNet: ~2-3 seconds (LLM API latency dominated)
- Tensor tracking: ~50ms (NumPy computations)
- Knowledge graph build: ~100ms (node/edge creation)
- API response: <500ms

**Production Considerations:**

- Replace SQLite with PostgreSQL for scalability
- Add rate limiting to prevent API abuse
- Cache LLM API calls for identical papers
- Add async job queue for large batches
- Implement monitoring (logging, metrics)
- Add authentication/authorization

---

## 7.4 Recruiter Questions (& Answers)

**Q1: Why should we hire you based on this project?**

A: "This project demonstrates full-stack systems thinking:
- Technical depth: RAG pipeline, tensor tracking, knowledge graphs
- Systems design: API, database, testing, CI/CD
- Problem-solving: Determinism, validation, reproducibility
- Communication: Clear codebase structure, comprehensive tests
- Initiative: Beyond code—docs, handbook, curriculum

I don't just code features; I understand requirements deeply and build robust systems."

**Q2: What would you do differently if you rebuilt this today?**

A: "Three things:

1. **Production Database**: Start with PostgreSQL, not SQLite
   - Current: Single-file SQLite, good for dev
   - Better: PostgreSQL from day one, scales to millions
   - Benefit: Prepared for growth

2. **API Caching**: Cache identical paper parsing
   - Current: Every upload re-parses and re-LLM-calls
   - Better: Hash paper, check cache, skip redundant work
   - Benefit: 10x faster for duplicates, reduced LLM costs

3. **Async Processing**: Background job queue for large batches
   - Current: Sync parsing blocks request
   - Better: Celery + Redis for async processing
   - Benefit: Responsive UI, parallel processing"

**Q3: How do you ensure code quality?**

A: "Five mechanisms:

1. **Type Hints**: Every function parameter and return value typed
2. **Linting**: Black (formatting) + Flake8 (style) enforced in CI
3. **Testing**: 20+ test files with 90%+ coverage
4. **Determinism Tests**: Parse twice, verify identical output
5. **Code Review**: Pull requests before merge (in team context)"

---

## 7.5 Backend/Systems Design Questions

**Q1: Design Paper2Code for 1 million papers per day**

A: "Challenges: 1M papers/day = ~12 papers/second sustained load

Architecture Changes:

`
Client
  ↓
API Gateway (rate limiting, auth)
  ↓
Load Balancer (multiple FastAPI instances)
  ↓
Job Queue (Celery + Redis)
  ↓
Workers (parallel parsing, 10-50 workers)
  ↓
PostgreSQL (sharded by paper_id)
  ↓
Cache Layer (Redis for repeated parsing)
`

Key Decisions:

1. **Async Processing**: Frontend queues job, returns immediately
   - Instead of: Blocking on parse (2-3 sec per paper)
   - Result: 1000x throughput improvement

2. **Horizontal Scaling**: Multiple workers
   - Instead of: Single machine
   - Result: Linear scaling to 1000s papers/sec

3. **Caching**: Redis cache for identical papers
   - Instead of: Re-parse, re-LLM every time
   - Result: 10x speedup for duplicates

4. **Database Sharding**: Partition by paper_id
   - Instead of: Single PostgreSQL
   - Result: Scales to billions of papers

5. **LLM Batching**: Batch verification calls
   - Instead of: Single-paper API calls
   - Result: 50% reduction in API latency

Monitoring:

- Queue depth (papers waiting)
- Worker utilization (are workers busy?)
- Cache hit rate (what % use cache?)
- LLM API costs (batch vs single calls)
- Parse time distribution (p50, p95, p99)
"

**Q2: How do you maintain determinism at scale?**

A: "Determinism gets harder with distribution. Solutions:

1. **Fixed Seeds Everywhere**:
   - BM25 uses deterministic vectorization
   - LLM uses temperature=0 + fixed seed (if supported)
   - NumPy operations use np.random.seed(42)

2. **Idempotent Parsing**:
   - Same paper always produces same ID (hash(paper_content))
   - Cache: hash → parsed_config
   - If exists in cache, return cached result (guaranteed deterministic)

3. **Validation Checkpoints**:
   - After parsing: assert determinism with re-parse
   - After tensor tracking: assert shapes match expected
   - After graph building: assert DAG properties

4. **Distributed Tracing**:
   - Log every decision point
   - Trace paper_id through pipeline
   - Compare results across replicas (should be identical)

Result: Non-determinism is caught, not propagated."

---

## 7.6 AI/ML Questions

**Q1: How does your RAG pipeline compare to fine-tuned models?**

A: "Different tradeoffs:

BM25 + LLM Verification (Paper2Code approach):
- ✓ Deterministic (BM25 is algorithmic, LLM T=0)
- ✓ Fast (BM25 retrieval is milliseconds)
- ✓ Interpretable (can see which passages retrieved)
- ✗ Not semantic (misses context)

Fine-tuned Model (alternative):
- ✓ Semantic understanding
- ✓ End-to-end learned
- ✗ Non-deterministic (sampling inherent)
- ✗ Slower (full sequence generation)
- ✗ Requires labeled training data

Paper2Code's choice: Determinism > semantics (for reproducibility)"

**Q2: Why tensor tracking instead of just trusting the paper?**

A: "Papers have errors:

1. **Human Errors**: Typos in dimensions
   - Paper says: 224x224 input, but ResNet actually does 228x228
   - Caught by: tensor tracker validates against known patterns

2. **Ambiguity**: Not explicit in paper
   - Paper doesn't mention: what is batch size? output channels?
   - Solved by: framework conventions (batch first, standard config)

3. **Implementation Mismatches**: Paper spec != common implementation
   - Paper: theoretical description
   - Practice: implementation details (padding mode, bias, activation order)
   - Resolved by: validating inferred shapes make sense

Result: Tensor tracking catches papers with errors before code generation fails."

---

## 7.7 Project Ownership Questions

**Q1: How would you debug if a paper parses but generates wrong code?**

A: "Systematic debug:

1. **Verify parsing**: Print parsed_config JSON
   - Is it correct? (Compare to paper manually)
   - If wrong: Issue in config_extractor

2. **Verify tensor tracking**: Print tensor_flow list
   - Are shapes correct? (Use shape formula)
   - Are FLOPs reasonable?
   - If wrong: Issue in tensor_tracker

3. **Verify knowledge graph**: Check DAG
   - Are connections right?
   - Is topological sort correct?
   - If wrong: Issue in knowledge_graph

4. **Verify code generation**: Manual inspection
   - Does generated PyTorch code match config?
   - Do layer counts match?
   - If wrong: Issue in code_generator

5. **Test generated code**: Try to run
   - Forward pass: x = model(randn(1,224,224,3))
   - Does output shape match expected?
   - No errors?
   - If wrong: Code generator bug

Tool: Add detailed logging at each step, search for divergence."

**Q2: How would you add support for transformers (e.g., ViT)?**

A: "Add Transformer support:

1. **Identify new layer types**:
   - PatchEmbedding: Images → patches
   - MultiHeadAttention: Self-attention with multiple heads
   - FeedForward: Dense layers
   - LayerNorm: Normalization

2. **Update tensor_tracker**:
   `python
   def track_patch_embedding(image_shape, patch_size):
       # Image: [1, 224, 224, 3]
       # Patches: (224/16) * (224/16) = 196 patches
       # Each patch: 16*16*3 = 768 dims
       return [1, 197, 768]  # +1 for cls token

   def track_multihead_attention(input_shape, num_heads):
       batch, seq_len, hidden = input_shape
       # Attention output: same shape as input
       return input_shape
   `

3. **Update config_extractor**:
   - Add BM25 queries for attention parameters
   - 'number of attention heads'
   - 'patch size embedding dimension'

4. **Update code_generator**:
   - Add PyTorch ViT module
   - PatchEmbedding layer
   - MultiHeadAttention blocks

5. **Update tests**:
   - test_vit_patch_embedding.py
   - test_vit_tensor_flow.py
   - test_vit_vs_resnet_comparison.py

6. **Update FLOP calculations**:
   - Attention: O(seq_len^2 * hidden_dim)
   - Example: 196^2 * 768 = 29.4M FLOPs"

---

## 7.8 Common Technical Questions

**Q: What's the hardest part of this system?**

A: "Determinism. 

Most ML systems tolerate non-determinism (randomness in training, dropout, etc.). Paper2Code requires bit-perfect reproducibility:

- Same paper → same parsing every time
- Same parsing → same tensor flow every time
- Same flow → same code every time

Challenges:

1. LLMs are non-deterministic by default (sampling)
   - Solution: temperature=0, fixed seed
   - Trade-off: Less creative, more rigid

2. Float operations have precision issues
   - Solution: Integer-only computations where possible
   - Trade-off: More complex code

3. External dependencies (APIs, libraries) change
   - Solution: Pin versions, test against specific versions
   - Trade-off: Updates require validation

Result: Worth it for reproducibility."

**Q: How do you handle papers with non-standard architectures?**

A: "Three strategies:

1. **Pattern Recognition**: Extract patterns similar to known architectures
   - 'Conv + ReLU' → standard pattern
   - 'Conv + skip + ReLU' → bottleneck pattern

2. **LLM Reasoning**: LLM infers intent from description
   - Paper: 'novel residual-style connection'
   - LLM: 'This is a skip connection, treat as bottleneck'

3. **Validation Framework**: Catch errors early
   - If shape invalid: Error message suggests fixes
   - Framework catches invalid configs before code generation

Result: 80%+ accuracy on standard papers, degrades gracefully on novel architectures."

---


## PART 8: PROJECT OWNER MODE

Teach how to maintain, debug, extend, and add new features to Paper2Code.

---

## 8.1 How to Maintain This Project

**Daily Maintenance:**

1. **Monitor CI/CD Pipeline**:
   - Check GitHub Actions status
   - All tests passing? (green ✓ or red ✗)
   - Fix failing tests immediately

2. **Monitor API Health**:
   - Response times (should be <500ms)
   - Error rates (should be <1%)
   - Database size (growing as expected?)

3. **Update Dependencies**:
   - Monthly: Check for security updates
   - Quarterly: Major version updates
   - Process:
     \\\ash
     pip list --outdated
     pip install --upgrade package_name
     pytest  # Ensure backward compatibility
     \\\

**Quarterly Maintenance:**

1. **Database Cleanup**:
   - Archive old papers (before 1 year)
   - Remove duplicate parsing results
   - Optimize indexes

2. **Performance Analysis**:
   - Which papers are slowest to parse?
   - Are we hitting LLM rate limits?
   - Database query performance

3. **Documentation Review**:
   - Is README still accurate?
   - Do examples still work?
   - Update if code changed

---

## 8.2 How to Debug Issues

**Debugging Workflow:**

### Issue: Paper parses but generated code is wrong

**Step 1: Verify Data at Each Stage**

\\\ash
# 1. Check parsing
python -c "
from core.agents.parsing_agent_impl import ParsingAgentImpl
agent = ParsingAgentImpl()
config = agent.parse(open('resnet_paper.txt').read())
print(json.dumps(config, indent=2))
"
\\\

**Step 2: Check Tensor Tracking**

\\\ash
python -c "
from core.rag.tensor_tracker import TensorTracker
tracker = TensorTracker()
trace = tracker.track_architecture(config)
for layer in trace:
    print(f'{layer[id]}: {layer[input_shape]} → {layer[output_shape]}'
"
\\\

**Step 3: Check Knowledge Graph**

\\\ash
python -c "
from core.rag.knowledge_graph import KnowledgeGraph
kg = KnowledgeGraph()
graph = kg.build(config)
if kg.has_cycle(graph):
    print('ERROR: Graph has cycle')
else:
    print('OK: Valid DAG')
"
\\\

**Step 4: Manual Code Inspection**

Compare generated code against PyTorch standards:

\\\python
# Check layer counts
config_layers = len(config['layers'])
code_layers = generated_code.count('nn.')
print(f"Config has {config_layers} layers")
print(f"Code has {code_layers} layers")
assert config_layers == code_layers
\\\

**Step 5: Test Generated Code**

\\\python
# Try to run it
exec(generated_code)
model = ResNet50()
x = torch.randn(1, 224, 224, 3)
y = model(x)
print(f"Output shape: {y.shape}")  # Should be [1, 1000]
\\\

### Issue: Parser produces non-deterministic results

**Step 1: Check Seed**

\\\python
agent = ParsingAgentImpl(seed=42)
\\\

**Step 2: Check LLM Temperature**

\\\python
response = openai.ChatCompletion.create(
    temperature=0,  # Must be 0
    seed=42,        # If supported
)
\\\

**Step 3: Verify BM25 Determinism**

\\\python
from sklearn.feature_extraction.text import BM25Transformer

bm25_1 = BM25Transformer()
results_1 = bm25_1.fit_transform(documents).todense()

bm25_2 = BM25Transformer()
results_2 = bm25_2.fit_transform(documents).todense()

assert np.allclose(results_1, results_2)  # Should pass
\\\

### Issue: Tensor mismatch error in middle of parsing

**Debug:**

\\\python
from core.rag.tensor_tracker import TensorTracker, TensorMismatchError

tracker = TensorTracker()
try:
    trace = tracker.track_architecture(config)
except TensorMismatchError as e:
    print(f"Error: {e}")
    # Find which layer caused error
    for i, layer in enumerate(config['layers']):
        try:
            tracker._infer_shape(layer, current_shape)
        except TensorMismatchError:
            print(f"Failed at layer {i}: {layer['id']}")
            print(f"Input shape: {current_shape}")
            print(f"Layer config: {layer['config']}")
\\\

---

## 8.3 How to Extend Paper2Code

**Adding a new layer type (e.g., DepthwiseConv2d):**

### Step 1: Add to tensor_tracker

\\\python
# core/rag/tensor_tracker.py

def track_depthwise_conv2d(input_shape, kernel_size, stride, padding, depth_multiplier):
    batch, h, w, channels = input_shape
    
    # Depthwise convolution: apply per-channel
    # Input: [batch, h, w, channels]
    # Output: [batch, h_out, w_out, channels * depth_multiplier]
    
    oh = (h - kernel_size + 2*padding) // stride + 1
    ow = (w - kernel_size + 2*padding) // stride + 1
    out_channels = channels * depth_multiplier
    
    return [batch, oh, ow, out_channels]

# In _infer_shape method:
elif layer_type == "DepthwiseConv2d":
    return self.track_depthwise_conv2d(
        input_shape,
        layer['config']['kernel_size'],
        layer['config']['stride'],
        layer['config']['padding'],
        layer['config']['depth_multiplier']
    )
\\\

### Step 2: Add FLOP calculation

\\\python
# core/rag/flops_engine.py

def calculate_flops_depthwise_conv2d(input_shape, output_shape, config):
    batch, h, w, c = input_shape
    batch_o, h_o, w_o, c_o = output_shape
    k = config['kernel_size']
    
    # Depthwise: k*k * channels * h_out * w_out
    return 2 * k * k * c * h_o * w_o
\\\

### Step 3: Update config_extractor

\\\python
# core/rag/config_extractor.py

# Add query
queries = [
    ...,
    "depthwise separable convolution grouped",
]

# Add extraction logic
if "depthwise" in paper_text:
    layer_type = "DepthwiseConv2d"
    depth_multiplier = extract_depth_multiplier(passages)
\\\

### Step 4: Update code generator

\\\python
# Code generator should emit:
# self.dwconv = nn.Conv2d(in_channels, out_channels, groups=in_channels)
# (groups=in_channels makes it depthwise)
\\\

### Step 5: Add tests

\\\python
# tests/test_depthwise_conv.py

def test_depthwise_conv_shape():
    tracker = TensorTracker()
    output = tracker.track_layer(
        "DepthwiseConv2d",
        [1, 112, 112, 64],
        {"kernel_size": 3, "stride": 1, "padding": 1, "depth_multiplier": 2}
    )
    assert output == [1, 112, 112, 128]  # 64 * 2

def test_depthwise_conv_flops():
    flops = flops_engine.calculate_flops_depthwise_conv2d(
        [1, 112, 112, 64],
        [1, 112, 112, 128],
        {"kernel_size": 3}
    )
    expected = 2 * 3 * 3 * 64 * 112 * 112
    assert flops == expected
\\\

---

## 8.4 How to Add a New Architecture

**Adding support for MobileNet:**

### Step 1: Get the paper

- Title: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
- Key feature: Depthwise separable convolutions

### Step 2: Update config extraction

Update queries to recognize MobileNet terminology

### Step 3: Add test paper

Create 	ests/papers/mobilenet_v1.txt with extracted specifications

### Step 4: Test parsing

\\\ash
python -m pytest tests/test_config_extractor.py::test_mobilenet_parsing -v
\\\

### Step 5: Verify tensor flow

\\\ash
python -m pytest tests/test_tensor_tracker.py::test_mobilenet_tensor_flow -v
\\\

### Step 6: Compare with known implementation

\\\python
# Test against torchvision
from torchvision.models import mobilenet_v1

official_model = mobilenet_v1()
generated_model = ResNet50()  # After code generation

# Compare layer counts, FLOPs, etc.
official_flops = calculate_model_flops(official_model)
generated_flops = generated_model.flops_total
assert abs(official_flops - generated_flops) / official_flops < 0.05  # Within 5%
\\\

---

## 8.5 How to Add a New API Endpoint

**Adding an endpoint to export to TensorFlow:**

### Step 1: Add method to Pipeline

\\\python
# core/orchestrator/pipeline.py

class Pipeline:
    def export_to_tensorflow(self, config):
        tf_code = self.code_generator.generate_tensorflow(config)
        return tf_code
\\\

### Step 2: Add method to CodeGenerator

\\\python
# code/generators/code_generator.py

def generate_tensorflow(self, config):
    # Convert PyTorch layers to TF.keras layers
    return tensorflow_code
\\\

### Step 3: Add FastAPI endpoint

\\\python
# backend/server.py

@app.get("/api/export/{paper_id}/tensorflow")
async def export_to_tensorflow(paper_id: int, session = Depends(get_session)):
    paper = session.query(Paper).get(paper_id)
    module = session.query(Module).filter_by(paper_id=paper_id).first()
    
    tf_code = pipeline.export_to_tensorflow(paper.parsed_config)
    
    return {
        "paper_id": paper_id,
        "format": "tensorflow",
        "code": tf_code,
        "generated_at": datetime.utcnow()
    }
\\\

### Step 4: Add test

\\\python
def test_export_to_tensorflow():
    response = client.get("/api/export/1/tensorflow")
    assert response.status_code == 200
    assert "import tensorflow" in response.json()["code"]
    assert "tf.keras" in response.json()["code"]
\\\

---

## 8.6 How to Add a Frontend Feature

**Adding architecture comparison UI:**

### Step 1: Add API endpoint (already done: /api/compare)

### Step 2: Add HTML section

\\\html
<!-- static/index.html -->

<div id="comparison-section" style="display:none">
    <h2>Compare Architectures</h2>
    <select id="arch1-select"></select>
    <select id="arch2-select"></select>
    <button onclick="compareArchitectures()">Compare</button>
    <div id="comparison-results"></div>
</div>
\\\

### Step 3: Add JavaScript handler

\\\javascript
// static/app.js

async function compareArchitectures() {
    const arch1 = document.getElementById('arch1-select').value;
    const arch2 = document.getElementById('arch2-select').value;
    
    const response = await fetch(\/api/compare?arch1=\&arch2=\\);
    const result = await response.json();
    
    displayComparison(result);
}

function displayComparison(result) {
    const html = \
        <table>
            <tr>
                <th>Property</th>
                <th>\</th>
                <th>\</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>Layers</td>
                <td>\</td>
                <td>\</td>
                <td>\</td>
            </tr>
            <tr>
                <td>Parameters</td>
                <td>\</td>
                <td>\</td>
                <td>\</td>
            </tr>
            <tr>
                <td>FLOPs</td>
                <td>\</td>
                <td>\</td>
                <td>\</td>
            </tr>
        </table>
    \;
    
    document.getElementById('comparison-results').innerHTML = html;
}
\\\

### Step 4: Add test

\\\python
def test_comparison_ui():
    # Create two papers
    paper1 = Paper(title="ResNet-50", ...)
    paper2 = Paper(title="VGG-16", ...)
    session.add_all([paper1, paper2])
    session.commit()
    
    # Compare
    response = client.get(f"/api/compare?arch1={paper1.id}&arch2={paper2.id}")
    assert response.status_code == 200
    assert response.json()["arch1_layers"] > 0
    assert response.json()["arch2_layers"] > 0
\\\

---

## 8.7 Deployment Checklist

**Before Deploying to Production:**

- [ ] All tests passing (\pytest\)
- [ ] Linting passes (\lack . && flake8 .\)
- [ ] Type checking passes (\mypy .\ if enabled)
- [ ] Coverage >85% (\pytest --cov\)
- [ ] No security vulnerabilities (\pip-audit\)
- [ ] Database migrations current (\lembic current\)
- [ ] Environment variables set (.env file)
- [ ] LLM API key valid and funded
- [ ] SSL certificate valid (if HTTPS)
- [ ] Logging configured (file + remote)
- [ ] Monitoring configured (metrics + alerts)
- [ ] Backup strategy tested
- [ ] Disaster recovery plan documented

**Deployment Command:**

\\\ash
# Run in production environment
git pull origin main
pip install -r requirements.txt
pytest  # Final test
alembic upgrade head  # Apply migrations
gunicorn -w 4 backend.server:app --bind 0.0.0.0:8000
\\\

---

## 8.8 Monitoring & Logging

**Essential Metrics:**

\\\python
# Log every parse operation
logger.info(f"Parse started: paper_id={paper_id}, title={title}")
logger.info(f"Parse completed: time={elapsed_ms}ms, layers={layer_count}")

# Track errors
logger.error(f"Parse failed: {error_message}", exc_info=True)

# Performance metrics
metrics.histogram("parse_duration_ms", elapsed_ms)
metrics.gauge("pending_parses", queue.size())
metrics.increment("parse_success", tags=[f"architecture={arch}"])
\\\

**Alerts:**

- Parse time > 30s: Investigate LLM latency
- Error rate > 5%: Check input validation
- Queue depth > 1000: Scale up workers
- Database size > 100GB: Archive old papers

---


## PART 9: FINAL EXAM

Comprehensive exam grounded in actual code references. Answer correctly = deep understanding sufficient to rebuild from scratch.

---

## 9.1 Architecture & Design (15 questions)

**Q1: What is the purpose of the Pipeline class in core/orchestrator/pipeline.py?**

A: Central orchestrator wiring ParsingAgent → TensorTracker → KnowledgeGraph → CodeGenerator. Provides single entry point for paper2code execution. Acts as service layer.

**Q2: Why does config_extractor.py use BM25 instead of semantic embeddings?**

A: Determinism. BM25 is algorithmic (not learned), same query always returns same results. Semantic embeddings involve sampling/randomness. Trade-off: BM25 less semantic (mitigated by LLM verification at temperature=0).

**Q3: In tensor_tracker.py, what does track_conv2d() return on invalid output?**

A: Raises TensorMismatchError. Example: Conv(224, K=7, S=2) → 112x112 valid. Conv(10, K=7, S=2) → 2x2 invalid (too small), raises error.

**Q4: Why is the Knowledge Graph built as a DAG? What property must be validated?**

A: DAG (Directed Acyclic Graph) ensures valid execution order. No cycles means layers can be executed sequentially. Validated via topological_sort() - if cycle detected, raises error.

**Q5: How does SQLAlchemy's ORM prevent SQL injection?**

A: Parameterized queries. Instead of string concatenation:
`python
# Unsafe: query = f"SELECT * FROM papers WHERE id={id}"
# Safe: session.query(Paper).filter(Paper.id == id)
`
Parameters are bound separately, not concatenated.

**Q6: Why use Pydantic schemas in FastAPI endpoints?**

A: Type validation. FastAPI automatically validates request bodies match schema. If not, returns 422 error with details. Prevents invalid data reaching business logic.

**Q7: What is the Dependency Injection pattern used in pipeline.py?**

A: Constructor injection. Pipeline receives dependencies via __init__:
`python
class Pipeline:
    def __init__(self, parsing_agent, tensor_tracker, knowledge_graph):
        self.parsing_agent = parsing_agent
`
Enables testing (mock agents) and flexibility (swap implementations).

**Q8: How does determinism testing work in test_pipeline_determinism.py?**

A: Parse same paper twice:
`python
result1 = pipeline.parse(paper)
result2 = pipeline.parse(paper)
assert result1 == result2  # Must be identical
`
Tests reproducibility at every step.

**Q9: What tradeoff was made choosing SQLite over PostgreSQL?**

A: Choose: Zero setup, perfect for dev. Trade: Single writer, not distributed. Mitigated: Migration path via Alembic.

**Q10: How do CI/CD workflows in .github/workflows/ci.yml prevent bugs?**

A: Auto-run on every push:
1. Lint (black, flake8)
2. Type-check
3. Run tests
4. Upload coverage

Bugs caught before merge.

**Q11: Why validate shapes at parse-time instead of later?**

A: Early error detection. Invalid config caught immediately, not during training. Faster feedback loop for developers.

**Q12: What is the purpose of Alembic in migrations/?**

A: Version control for database schema. Track changes:
`ash
alembic revision --autogenerate -m "Add comparison table"
alembic upgrade head  # Apply
alembic downgrade -1  # Rollback
`

**Q13: How does BM25 scoring ensure determinism?**

A: Algorithmic ranking (TF-IDF based). Same index, same query → same scores every time. No randomness or learning involved.

**Q14: What is the Repository Pattern? How is it used?**

A: Abstraction over data access. Methods: get(), save(), delete(), list(). Decouples business logic from database details. Enables testing (mock repository).

**Q15: Explain Service Layer pattern used in pipeline.py:**

A: Encapsulates business logic between API and data layers. API calls pipeline.execute(), pipeline calls agents, trackers, graphs. Logic centralized, testable, reusable.

---

## 9.2 Tensor Tracking & Shape Inference (12 questions)

**Q16: Write the formula for Conv2d output shape inference:**

A: H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1

Example: (224 + 2*3 - 7) / 2 + 1 = 112

**Q17: What does tensor_tracker.py calculate for FLOPs in Conv2d?**

A: FLOPs = 2 * kernel_h * kernel_w * in_channels * out_h * out_w * out_channels

Factor of 2: each MAC (multiply-accumulate) is 2 FLOPs.

**Q18: For BottleneckBlock, estimate FLOP breakdown:**

A:
- 1x1 conv (reduce C→C/4): ~0.1x
- 3x3 conv (spatial): ~0.8x
- 1x1 conv (expand C/4→C): ~0.1x

Total: 1x relative to 3x3 conv.

**Q19: What does TensorMismatchError indicate? When is it raised?**

A: Invalid tensor shape detected. Raised when:
- Inferred output has dimension ≤ 0
- Shape mismatch between layers
- Connection has incompatible shapes

Example: Conv output [1, 2, 2, 64] invalid if next layer expects [1, 112, 112, 64].

**Q20: How is GlobalAvgPool handled in tensor tracking?**

A: Input [B, H, W, C] → Output [B, 1, 1, C]. Reduces spatial dims to 1x1, preserves channels.

**Q21: For Linear layer, how are FLOPs calculated?**

A: FLOPs = 2 * input_features * output_features * batch_size

Example: FC(2048→1000) with batch=32: 2 * 2048 * 1000 * 32 = ~131M FLOPs

**Q22: Why must skip connections preserve shape?**

A: Addition requires same shape. If stride changes shape, shortcut must apply convolution to match. Otherwise: shape mismatch error.

**Q23: Estimate total FLOPs for ResNet-50:**

A: ~7.7 billion FLOPs. Breakdown:
- conv1: 11.7B
- stage2: 5.3B
- stage3: 7.0B
- stage4: 10.1B
- stage5: 14.2B
- fc: ~4M (negligible)

**Q24: What is the difference between FLOPs and parameters?**

A: FLOPs = computation (changes per input). Parameters = weights (fixed). ResNet-50: ~25.5M parameters, 7.7B FLOPs per forward pass.

**Q25: How does ViT patch embedding change tensor shape?**

A: Image [1, 224, 224, 3] → Patches [196, 768] + cls token [1, 768] → Embedded [197, 768].

196 = (224/16)^2, 768 = 16*16*3.

**Q26: What happens if kernel_size > input_size in Conv2d?**

A: Output height/width becomes 0 or negative. TensorTracker raises TensorMismatchError with message about invalid shape.

**Q27: For Attention, estimate FLOPs:**

A: FLOPs ≈ 4 * seq_len^2 * hidden_dim (QK @ V operations). Example: seq=196, hidden=768 → ~230M FLOPs.

---

## 9.3 Parser & RAG Pipeline (10 questions)

**Q28: What are the three stages of config_extractor.py?**

A:
1. Retrieval: BM25 finds relevant passages
2. Extraction: Parse passages into structured form
3. Verification: LLM checks correctness (temp=0)

**Q29: Why is LLM temperature set to 0?**

A: Determinism. Temperature=0 always picks highest probability token (no sampling). Ensures identical output for identical input.

**Q30: What queries would BM25 use to extract Conv2d kernel size?**

A: Examples:
- "kernel size convolution"
- "3x3 conv 7x7"
- "filter size stride"
- "receptive field"

**Q31: How does verification loop work? Describe the feedback:**

A: Extract → LLM "Is this correct?" → If no → Re-extract → Loop until yes. Ensures extraction confidence.

**Q32: What is deterministic about BM25?**

A: Scoring is algorithmic (TF-IDF formula). Same index + same query = same scores every time. No randomness.

**Q33: How is paper text split for indexing?**

A: By sections: [title, abstract, introduction, methodology, architecture, experiments, conclusion]. Enables targeted retrieval.

**Q34: What prevents hallucination in LLM extraction?**

A: Grounding in paper. LLM only verifies against provided passages, not general knowledge. If passage doesn't support claim, marked as unverified.

**Q35: How are multiple extraction attempts reconciled?**

A: Take intersection (most confident values). If BM25 + LLM agree, high confidence. If conflict, flag for review.

**Q36: Why not use fine-tuned model instead of BM25 + LLM?**

A: Fine-tuning requires labeled training data (expensive), not deterministic (learned randomness). BM25 + LLM is faster to implement, more deterministic, sufficient accuracy.

**Q37: What happens if paper has ambiguous architecture description?**

A: LLM uses context and conventions (batch size=1 default, input=224 default for images). Falls back to common practices if ambiguous.

---

## 9.4 Knowledge Graphs & Comparisons (8 questions)

**Q38: What properties must be validated in knowledge_graph.py?**

A:
1. DAG: No cycles (topological sort must succeed)
2. Connected: All nodes reachable from input
3. Valid: All edges have valid nodes
4. Complete: No missing connections

**Q39: How would you detect cycles in a graph?**

A: DFS (depth-first search). Mark nodes as visiting/visited. If reach visiting node again, cycle detected.

**Q40: What is topological sort? Why is it used?**

A: Ordering of nodes such that for every edge (u→v), u comes before v. Used to determine execution order of layers.

**Q41: How do skip connections appear in the graph?**

A: Double edges: one forward (data flow), one back (skip). Example:
`
conv1 → bn1 → relu1 ↓
                    add ← skip from before conv1
`

**Q42: What is graph isomorphism? Why does it matter?**

A: Two graphs are isomorphic if they have same structure (topology), though node labels differ. Enables architecture comparison: "ResNet bottleneck ≈ DenseNet bottleneck (isomorphic)".

**Q43: How would you compare two architectures using graphs?**

A:
1. Build graph for arch1, arch2
2. Check if isomorphic (same topology)
3. Compare paths (number of hops input→output)
4. Compare FLOPs per path
5. Return similarity score

**Q44: What information does each graph node store?**

A: Node = {id, label, type, attributes}

Example:
`python
{
    id="conv1",
    label="Initial Conv",
    type="Conv2d",
    attributes={"kernel": 7, "stride": 2, "channels": 64}
}
`

**Q45: How could you use knowledge graphs for architecture search?**

A: Enumerate subgraphs, evaluate each (FLOPs, accuracy), find optimal. Example: "Find all 2-layer subgraphs with <100M FLOPs".

---

## 9.5 API & Frontend Integration (10 questions)

**Q46: List all 8 FastAPI endpoints in backend/server.py:**

A:
1. POST /api/parse - Parse paper
2. POST /api/generate - Generate code
3. POST /api/compare - Compare architectures
4. GET /api/analyze/{id} - Analyze architecture
5. GET /api/papers - List papers
6. GET /api/papers/{id} - Get paper details
7. DELETE /api/papers/{id} - Delete paper
8. GET /api/explain/{id} - Get explanations

**Q47: What does Pydantic schema validation catch?**

A: Type mismatches, missing fields, invalid values.

Example: PaperSchema requires 	itle: str. If frontend sends 	itle: 123, returns 422 error.

**Q48: How is session management done in backend/database.py?**

A:
`python
SessionLocal = sessionmaker(bind=engine)

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
`

Used in FastAPI: session = Depends(get_session)

**Q49: What happens if database connection fails?**

A: FastAPI returns 500 error. Dependency raises exception, caught by error handler, returns JSON error response. No crash.

**Q50: How does frontend display tensor flow?**

A: JavaScript:
`javascript
fetch('/api/parse', data)
  .then(r => r.json())
  .then(result => {
    document.getElementById('tensor-flow').innerHTML = 
      JSON.stringify(result.tensor_flow, null, 2);
  });
`

**Q51: What CORS issues might arise? How are they fixed?**

A: Frontend (localhost:3000) calls API (localhost:8000). Browser blocks cross-origin. Solution:
`python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"])
`

**Q52: How would you add pagination to /api/papers?**

A:
`python
@app.get("/api/papers")
def list_papers(skip: int = 0, limit: int = 10, session = Depends()):
    return session.query(Paper).offset(skip).limit(limit).all()
`

Frontend: /api/papers?skip=0&limit=10

**Q53: What does Content-Type: application/json mean?**

A: Response body is JSON. Browser/frontend knows to parse as JSON. FastAPI auto-sets this when returning dict.

**Q54: How would you add authentication to /api/parse?**

A:
`python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/api/parse")
def parse(paper: PaperSchema, credentials = Depends(security)):
    # Verify credentials
    return result
`

**Q55: What is a REST principle violated if /api/parse modifies data but uses GET?**

A: GET should be idempotent (no side effects). POST for mutations. Paper2Code uses POST (correct).

---

## 9.6 Testing & Quality (8 questions)

**Q56: What does pytest parametrize do?**

A: Runs same test with multiple inputs:
`python
@pytest.mark.parametrize("shape,expected", [
    ([1,224,224,3], [1,112,112,64]),
    ([2,112,112,64], [2,56,56,128]),
])
def test_shapes(shape, expected):
    assert process(shape) == expected
`

Runs test twice, once per parametrization.

**Q57: What is a pytest fixture?**

A: Setup/teardown helper. Example:
`python
@pytest.fixture
def pipeline():
    return Pipeline()

def test_parse(pipeline):
    result = pipeline.execute(config)
    assert result is not None
`

Fixture automatically created for each test.

**Q58: How do you test non-determinism (verify a function IS deterministic)?**

A:
`python
def test_determinism():
    result1 = parse(paper)
    result2 = parse(paper)
    assert result1 == result2
    assert json.dumps(result1) == json.dumps(result2)
`

**Q59: What does code coverage measure? Why >85%?**

A: Percentage of code lines executed by tests. >85% ensures most code paths tested, catching bugs. Edge cases still possible but rare.

**Q60: How do you test against a mock database?**

A:
`python
@pytest.fixture
def mock_session(mocker):
    return mocker.MagicMock()

def test_get_paper(mock_session):
    mock_session.query.return_value.first.return_value = paper
    result = get_paper(1, mock_session)
    assert result.id == 1
`

**Q61: What does black enforce?**

A: Code formatting consistency. Same indentation, quote style, line length (88 chars). Prevents style arguments.

**Q62: What does flake8 check?**

A: PEP8 compliance:
- Unused imports
- Undefined variables
- Whitespace issues
- Naming conventions

**Q63: What is pre-commit hook? How would you use it?**

A: Script runs before git commit. Example:
`yaml
repos:
  - repo: https://github.com/psf/black
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    hooks:
      - id: flake8
`

Prevents committing code that fails lint/format.

---

## 9.7 Deployment & Operations (7 questions)

**Q64: What must be checked before deploying to production?**

A:
- All tests pass (pytest)
- Linting passes (black, flake8)
- No security issues (pip-audit)
- Database migrations current (alembic current)
- Environment variables set
- LLM API key valid
- Monitoring configured
- Backup strategy tested

**Q65: How would you scale Paper2Code to handle 1M papers/day?**

A:
1. Async job queue (Celery + Redis)
2. Multiple workers (10-50)
3. PostgreSQL with sharding
4. Redis cache for duplicates
5. Load balancer for API servers
6. LLM request batching

**Q66: What metrics should be monitored?**

A:
- Parse duration (should be <2s)
- Error rate (should be <1%)
- Queue depth (pending parses)
- Cache hit rate (% using cache)
- LLM API costs
- Database size

**Q67: How would you handle LLM API rate limiting?**

A:
1. Implement retry with exponential backoff
2. Queue overflow requests
3. Cache results to avoid redundant calls
4. Batch requests where possible
5. Alert on sustained rate limiting

**Q68: What is blue-green deployment?**

A: Two identical production environments (blue, green). Deploy to inactive, test, switch traffic. Enables instant rollback. Requires redundancy.

**Q69: How would you automate database backups?**

A:
`ash
# Cron job daily at 2 AM
0 2 * * * pg_dump paper2code > /backups/paper2code-\.sql
`

Or: use cloud provider (AWS RDS automated backups).

**Q70: What is graceful shutdown? Why does it matter?**

A: Finish current requests before stopping. Code:
`python
@app.on_event("shutdown")
async def shutdown():
    await cleanup()  # Close connections, flush caches
`

Prevents data loss, client errors.

---

## 9.8 Complete Picture (5 questions)

**Q71: Trace a paper from upload to code generation, listing all files involved:**

A:
1. Frontend: static/index.html, static/app.js (upload)
2. API: backend/server.py (parse endpoint)
3. Pipeline: core/orchestrator/pipeline.py (orchestration)
4. Agents: core/agents/parsing_agent_impl.py (parse)
5. RAG: core/rag/config_extractor.py (extract config)
6. Tracking: core/rag/tensor_tracker.py (validate tensors)
7. Graph: core/rag/knowledge_graph.py (build graph)
8. Generator: (code generation)
9. Database: backend/models.py, backend/database.py (store)
10. Return: backend/server.py (JSON response)

**Q72: What would break if you removed tensor_tracker.py?**

A: No validation. Invalid configurations would generate broken code. Users would waste time debugging generated code instead of catching errors early.

**Q73: What would break if you removed knowledge_graph.py?**

A: No architecture comparison. Can't reason about semantic equivalence. Users couldn't query "show me similar architectures".

**Q74: How does the project ensure code quality end-to-end?**

A:
1. Type hints (static analysis)
2. Linting (style enforcement)
3. Testing (correctness validation)
4. Determinism checks (reproducibility)
5. CI/CD automation (before merge)
6. Monitoring (post-deployment)

**Q75: If you had to rebuild Paper2Code from scratch, what would be your first 3 months?**

A:
- Month 1: Core parsing (config extraction + tensor tracking)
- Month 2: Knowledge graphs + code generation
- Month 3: API + frontend + testing

Focus: Get ResNet working end-to-end first, then expand.

---

## 9.9 Exam Scoring

**Score Interpretation:**

- 75-100%: Expert level. Ready to rebuild from scratch.
- 60-75%: Advanced. Understand architecture, some details fuzzy.
- 45-60%: Intermediate. Grasp main components, weak on details.
- <45%: Beginner. Review prerequisite knowledge.

**Passing Score:** 70/100 (52.5 questions correct out of 75)

---

