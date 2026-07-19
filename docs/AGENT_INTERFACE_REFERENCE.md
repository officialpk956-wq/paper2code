# Phase 3.9.B.1 — Agent Interfaces Reference
## Quick Interface Guide

---

## 📁 File Structure

```
src/agents/
├── __init__.py              # Exports: 19 items (3 agents + 16 types)
├── types.py                 # 30 type definitions (180 lines)
├── parsing_agent.py         # ParsingAgent (75 lines, 1 abstract method)
├── visualization_agent.py   # VisualizationAgent (145 lines, 2 abstract methods)
└── explanation_agent.py     # ExplanationAgent (160 lines, 4 abstract methods)
```

---

## 🧩 Agent Interfaces at a Glance

### ParsingAgent

**What it does:** Raw spec → ArchitectureGraph

**Method:**
```python
@abstractmethod
def parse(
    source: ParsingSource,
    format_hint: str = "auto"
) -> ArchitectureGraph:
    """Parse architecture specification into graph structure."""
```

**Accepted inputs:** ConfigDict | PaperExcerpt | SymbolicDesc

---

### VisualizationAgent

**What it does:** Graph → Visual representation

**Methods:**
```python
@abstractmethod
def render(
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

**Modes:** "single" | "compare"

**Visual cues (priority order):**
1. bottleneck_badge 🔥
2. compute_highlight 🔴
3. scaling_highlight 🟠
4. spatial_highlight 🔵
5. ghost_overlay ⚪

---

### ExplanationAgent

**What it does:** Facts → Natural language

**Methods:**
```python
@abstractmethod
def explain_node(self, node: GraphNode) -> str:
    """Explain a single node."""

@abstractmethod
def explain_graph(self, graph: ArchitectureGraph) -> str:
    """Explain the overall architecture."""

@abstractmethod
def explain_comparison(
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

---

## 📋 Type Categories

### Parsing Types (4)
- ConfigDict
- PaperExcerpt
- SymbolicDesc
- ParsingSource (Union)

### Visualization Types (6)
- VisualizationMode
- ComparisonContext
- VisualizationOptions
- NodeVisuals
- VisualRepresentation

### Explanation Types (7)
- ComputeSummary
- SpatialSummary
- ScalingSummary
- ComparisonResult
- VisualMetadata
- ExplanationTemplate
- TemplateLibrary

---

## ✅ Verification Checklist

- [x] All files compile without errors
- [x] All agents are abstract (cannot instantiate)
- [x] All required methods present
- [x] All types properly defined
- [x] Zero implementation logic
- [x] No Streamlit imports
- [x] No Graphviz imports
- [x] No circular dependencies
- [x] All determinism guarantees documented
- [x] Test coverage: 8/8 passing

---

## 🚀 Ready for Implementation

**Phase 3.9.B.2** will implement concrete agents:

```python
# Example (not yet implemented)
class ParsingAgentImpl(ParsingAgent):
    def parse(self, source, format_hint="auto"):
        # Implementation here
        pass

class VisualizationAgentImpl(VisualizationAgent):
    def render(self, graph, mode="single", comparison_ctx=None, options=None):
        # Implementation here
        pass
    
    def get_visual_cues(self):
        return {"bottleneck_badge", "compute_highlight", ...}

class ExplanationAgentImpl(ExplanationAgent):
    def explain_node(self, node):
        # Implementation here
        pass
    
    def explain_graph(self, graph):
        # Implementation here
        pass
    
    def explain_comparison(self, graph_a, graph_b, comparison_result, visual_metadata):
        # Implementation here
        pass
    
    def get_explanation_templates(self):
        # Implementation here
        pass
```

---

## 💡 Key Design Principles

1. **Single Responsibility** — Each agent does one thing
2. **Type Safety** — TypedDict enforces structure
3. **Deterministic** — Same input → same output
4. **Decoupled** — Agents don't depend on each other
5. **Composable** — Chain agents without side effects
6. **Testable** — Each agent tested in isolation
7. **Extensible** — New agents without modifying existing code
8. **No Invention** — Agents narrate, never invent

---

## 📚 Documentation

- **AGENT_SYSTEM_DESIGN.md** — Full system architecture
- **PHASE_3_9_B_1_COMPLETE.md** — Phase completion report
- **test_agent_interfaces.py** — Verification tests

---

**Status: Phase 3.9.B.1 ✅ COMPLETE**

Proceed to Phase 3.9.B.2 (Concrete Implementations) when ready.
