# Phase 3.9.B.1 — Agent Interfaces
## Clean Separation of Concerns for paper2code

**Status:** ✅ COMPLETE  
**Date:** February 4, 2026  
**Scope:** Interface design only (no implementation)

---

## Overview

Phase 3.9.B.1 introduces three thin, deterministic agents that cleanly separate parsing, visualization, and explanation concerns. These agents will serve as the foundation for a composable, extensible architecture.

**Key Principle:** Interfaces define contracts, not suggestions. Incorrect usage should be impossible at the type level.

---

## 📁 Deliverables

### Directory Structure
```
src/agents/
├── __init__.py              # Exports all agents and types
├── types.py                 # TypedDict definitions (no logic)
├── parsing_agent.py         # ParsingAgent interface
├── visualization_agent.py   # VisualizationAgent interface
└── explanation_agent.py     # ExplanationAgent interface
```

### Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `types.py` | 180 | All TypedDict definitions, no logic |
| `parsing_agent.py` | 75 | ParsingAgent abstract interface |
| `visualization_agent.py` | 145 | VisualizationAgent abstract interface |
| `explanation_agent.py` | 160 | ExplanationAgent abstract interface |
| `__init__.py` | 55 | Clean exports |

**Total:** ~615 lines of pure interface definitions

---

## 🧩 Agent 1: Parsing Agent

### Responsibility
Convert raw architecture specifications into `ArchitectureGraph` with semantic parameters.

### Interface
```python
class ParsingAgent(ABC):
    @abstractmethod
    def parse(
        self,
        source: ParsingSource,
        format_hint: str = "auto"
    ) -> ArchitectureGraph:
        """Parse architecture specification into graph structure."""
```

### Input Types
- `ConfigDict` — {name, layers, connections, metadata}
- `PaperExcerpt` — {type, content, source, metadata}
- `SymbolicDesc` — {type, spec, notation}
- `ParsingSource` — Union of above

### Contract

**MUST:**
- ✅ Parse well-formed specifications
- ✅ Attach semantic_params to every node
- ✅ Validate edges reference existing nodes
- ✅ Produce deterministic output
- ✅ Include descriptions for every node

**MUST NOT:**
- ❌ Create visualization objects
- ❌ Invent semantic params beyond type-based defaults
- ❌ Make architectural judgments
- ❌ Perform comparisons
- ❌ Have side effects

---

## 🧩 Agent 2: Visualization Agent

### Responsibility
Render `ArchitectureGraph` into visual representations with semantic-aware highlighting.

### Interface
```python
class VisualizationAgent(ABC):
    @abstractmethod
    def render(
        self,
        graph: ArchitectureGraph,
        mode: VisualizationMode = "single",
        comparison_ctx: Optional[ComparisonContext] = None,
        options: Optional[VisualizationOptions] = None,
    ) -> VisualRepresentation:
        """Render graph with semantic-aware visual cues."""

    @abstractmethod
    def get_visual_cues(self) -> Set[str]:
        """List supported visual cues."""
```

### Visualization Modes

**Single Mode:** Standard visualization with FLOPs-based coloring
**Compare Mode:** Side-by-side with comparison context highlighting

### Visual Cues (Priority Order)
1. 🔥 **Bottleneck badge** (#CC0000, penwidth 4.0)
2. 🔴 **Compute highlight** (#FF6666, penwidth 3.0)
3. 🟠 **Scaling highlight** (#FFA500, penwidth 3.0)
4. 🔵 **Spatial highlight** (#4169E1, penwidth 2.5)
5. ⚪ **Ghost overlay** (#CCCCCC, penwidth 1.0)

### Contract

**MUST:**
- ✅ Render valid graphs to visual output
- ✅ Apply visual cues in priority order
- ✅ Produce deterministic output
- ✅ Support both modes
- ✅ Consume comparison context without modifying it

**MUST NOT:**
- ❌ Construct new graphs
- ❌ Infer semantic parameters
- ❌ Generate explanations
- ❌ Make architectural judgments
- ❌ Modify input graphs

---

## 🧩 Agent 3: Explanation Agent

### Responsibility
Generate natural-language explanations that narrate existing reasoning.

### Interface
```python
class ExplanationAgent(ABC):
    @abstractmethod
    def explain_node(self, node: GraphNode) -> str:
        """Explain a single node."""

    @abstractmethod
    def explain_graph(self, graph: ArchitectureGraph) -> str:
        """Explain the overall architecture."""

    @abstractmethod
    def explain_comparison(
        self,
        graph_a: ArchitectureGraph,
        graph_b: ArchitectureGraph,
        comparison_result: ComparisonResult,
        visual_metadata: VisualMetadata,
    ) -> str:
        """Explain architectural differences."""

    @abstractmethod
    def get_explanation_templates(self) -> TemplateLibrary:
        """Expose templates for debugging."""
```

### Core Constraint
**Must not invent facts.** Only reorganize what is known.

### Contract

**MUST:**
- ✅ Narrate facts from semantic_params and comparison results
- ✅ Use pre-defined templates (no free-form generation)
- ✅ Reference visual cues accurately
- ✅ Produce deterministic output
- ✅ Handle ties/similarities explicitly
- ✅ Link to semantic reasoning

**MUST NOT:**
- ❌ Invent facts not in provided data
- ❌ Override comparison conclusions
- ❌ Generate novel insights
- ❌ Make recommendations
- ❌ Use LLMs
- ❌ Create visualizations

---

## 📋 Type System

### Parsing Types (17 types)
- `ConfigDict` — Explicit configuration
- `PaperExcerpt` — Raw paper text
- `SymbolicDesc` — Symbolic notation
- `ParsingSource` — Union type

### Visualization Types (6 types)
- `ComparisonContext` — Comparison metadata
- `VisualizationOptions` — Rendering configuration
- `VisualizationMode` — Literal["single", "compare"]
- `NodeVisuals` — Per-node styling
- `VisualRepresentation` — Complete output

### Explanation Types (7 types)
- `ComputeSummary` — From summarize_compute()
- `SpatialSummary` — From summarize_spatial_behavior()
- `ScalingSummary` — From summarize_scaling_behavior()
- `ComparisonResult` — All summaries combined
- `VisualMetadata` — Visual highlights metadata
- `ExplanationTemplate` — Single template definition
- `TemplateLibrary` — Template collection

**Total:** 30 type definitions

---

## ✅ Quality Assurance

### Verification Tests (8 tests, 100% pass rate)

| Test | Purpose | Status |
|------|---------|--------|
| `test_agent_interfaces_are_abstract` | All agents are properly abstract | ✅ |
| `test_parsing_agent_interface` | ParsingAgent has correct signature | ✅ |
| `test_visualization_agent_interface` | VisualizationAgent has correct signature | ✅ |
| `test_explanation_agent_interface` | ExplanationAgent has correct signature | ✅ |
| `test_type_definitions` | All types valid | ✅ |
| `test_no_implementation_logic` | No code inside agents | ✅ |
| `test_circular_dependencies` | No import cycles | ✅ |
| `test_determinism_docstrings` | All determinism guarantees documented | ✅ |

### Constraints Verified

- ✅ **No implementation logic** — All methods raise NotImplementedError
- ✅ **No Streamlit** — Zero imports of streamlit
- ✅ **No Graphviz** — Zero imports of graphviz
- ✅ **No circular dependencies** — Clean import hierarchy
- ✅ **Proper abstraction** — Cannot instantiate agents directly
- ✅ **Type safety** — All inputs/outputs explicitly typed
- ✅ **Documentation** — Every method documented with contract
- ✅ **Determinism** — All methods document determinism guarantee

---

## 🔗 Integration Architecture (Conceptual)

```
User Input
    │
    ├─→ [ParsingAgent.parse()] 
    │   └─→ ArchitectureGraph
    │
    ├─→ [VisualizationAgent.render()]
    │   ├─ Inputs: Graph + Mode + (optional) ComparisonContext
    │   └─→ VisualRepresentation
    │
    └─→ [ExplanationAgent.explain_*()]
        ├─ Inputs: Graph + ComparisonResult + VisualMetadata
        └─→ Markdown Explanation
```

**Composition Rules:**
- Parsing is **always required**
- Visualization and Explanation are **optional** depending on intent
- No agent depends on another (fully decoupled)
- Agents consume outputs from **existing utilities** (summarize_*, compare_*)

---

## 🚀 Why This Design Succeeds

### Debuggability
- **Single responsibility:** Each agent has one job
- **Deterministic I/O:** Same input → same output (no randomness)
- **Clear data flow:** No hidden state, no side effects
- **Isolation:** Test each agent independently

### Testability
- **Unit tests:** Test parse(), render(), explain_*() in isolation
- **Type safety:** TypedDict enforces correct input/output shapes
- **Golden tests:** Store expected outputs for regression testing
- **Contract validation:** Determinism guarantees are testable

### Extensibility
- **New architectures:** Add config → works automatically
- **New parameters:** Update type definitions → agents reuse
- **New metrics:** Extend summarize_* → explain templates handle it
- **Future LLM:** Insert at higher level without breaking agents
- **Batch processing:** Compose agents in pipelines

### Maintainability
- **Minimal code:** ~615 lines of pure interface
- **Clear contracts:** Every method documents its guarantee
- **No surprises:** Type system enforces requirements
- **Self-documenting:** Docstrings serve as specification

---

## 🧪 Test Coverage

```bash
python test_agent_interfaces.py
```

**Output:**
```
✅ ALL 8 TESTS PASSED

✓ Agent interfaces are properly abstract
✓ All required methods present
✓ All type definitions valid
✓ No implementation logic detected
✓ No circular dependencies
✓ Determinism guarantees documented

🎉 Phase 3.9.B.1 Agent Interfaces: COMPLETE
```

---

## 📖 Usage Example (Conceptual)

```python
# Phase 3.9.B.2+ will implement these

# Parsing
parsing_agent = ConcreteParsingAgent()
graph = parsing_agent.parse(config_dict)

# Visualization
vis_agent = ConcreteVisualizationAgent()
visuals = vis_agent.render(graph, mode="single")

# Explanation
exp_agent = ConcreteExplanationAgent()
explanation = exp_agent.explain_graph(graph)

# Comparison
comp_ctx = ComparisonContext(
    mode="compare",
    current_arch="A",
    dominant_compute="B",
    # ...
)
vis_compare = vis_agent.render(graph_a, mode="compare", comparison_ctx=comp_ctx)
exp_compare = exp_agent.explain_comparison(graph_a, graph_b, result, metadata)
```

---

## 🎯 Next Steps (Phase 3.9.B.2+)

### Implementation Priority
1. **ParsingAgent** — Concrete implementation (lowest risk)
2. **VisualizationAgent** — Refactor existing visualization logic
3. **ExplanationAgent** — Extract templates from existing explanations

### Testing
- Unit tests for each concrete agent
- Integration tests for full pipelines
- Golden test regression suite
- End-to-end Streamlit integration

### Documentation
- Implementation guides per agent
- Template library documentation
- Extension guides for custom agents
- Architecture decision record (ADR)

---

## 📊 Project Status

| Phase | Task | Status |
|-------|------|--------|
| 3.8 | Semantic Parameters | ✅ Complete |
| 3.8 | Semantic Reasoning | ✅ Complete |
| 3.9.A | Visual Comparison UI | ✅ Complete |
| **3.9.B.1** | **Agent Interfaces** | **✅ Complete** |
| 3.9.B.2 | Concrete Agents | ⏳ Planned |
| 3.9.B.3 | Streamlit Integration | ⏳ Planned |

---

## Summary

**Phase 3.9.B.1 delivers a clean, minimal, and enforceable agent interface system:**

- ✅ Three abstract agent classes with clear contracts
- ✅ 30 type definitions ensuring type safety
- ✅ Zero implementation logic (pure interfaces)
- ✅ 100% test verification
- ✅ Full determinism guarantees
- ✅ Future-proof extensibility

**The foundation is set. Ready for Phase 3.9.B.2 (Concrete Implementations).**

---

**Architecture Decision:** These thin interfaces make incorrect usage impossible at the type level while leaving implementation details for concrete classes. This maximizes debuggability, testability, and maintainability while keeping the system deterministic and composable.
