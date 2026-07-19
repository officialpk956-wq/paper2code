# Agent-Based System Design for paper2code
## Three-Agent Architecture for Parsing, Visualization, and Explanation

**Date:** February 4, 2026  
**Status:** Design Phase (No Implementation)  
**Purpose:** Define clean interfaces and responsibility contracts

---

## Executive Summary

The paper2code system will be decomposed into three deterministic agents:

1. **Parsing Agent** — Converts raw architecture descriptions → ArchitectureGraph
2. **Visualization Agent** — Renders ArchitectureGraph → visual representations with semantic highlighting
3. **Explanation Agent** — Narrates existing reasoning → natural-language markdown

**Key Principle:** Each agent is thin, deterministic, and operates on well-defined data structures. No agent invents reasoning or makes decisions outside its domain.

---

## 🧩 Agent 1: Parsing Agent

### Role
Convert raw architecture descriptions into a structured ArchitectureGraph with semantic parameters attached.

### Interface

```python
class ParsingAgent:
    """
    Responsibility: Transform architecture specifications into ArchitectureGraph
    
    Guarantee: Deterministic parsing (same input → same ArchitectureGraph)
    """
    
    def parse(
        self,
        source: ParsingSource,
        format_hint: str = "auto"
    ) -> ArchitectureGraph:
        """
        Parse architecture specification into graph structure.
        
        Args:
            source: One of:
                - ConfigDict: {"name": "ResNet-18", "layers": [...]}
                - PaperExcerpt: {"type": "markdown", "text": "..."}
                - SymbolicDesc: {"type": "symbolic", "spec": "Conv2D(64) → ..."}
            format_hint: Optional format specification ("yaml", "json", "symbolic", "auto")
        
        Returns:
            ArchitectureGraph with:
            - nodes: List[GraphNode] with id, label, type, params, description
            - edges: List[GraphEdge] with source, target, edge_type
            - semantic_params attached to each node
        
        Raises:
            ParsingError: If source is malformed or ambiguous
        """
        pass
```

### Input Formats (Initial Support)

```python
ParsingSource = Union[
    ConfigDict,          # {"name": "ResNet-18", "stem": {...}, "blocks": [...]}
    PaperExcerpt,        # {"type": "markdown", "text": "paper section..."}
    SymbolicDesc,        # {"type": "symbolic", "spec": "Conv2D(64)→..."}
    PythonDict           # Direct dict from visualizer builders
]

ConfigDict = TypedDict(
    'ConfigDict',
    {
        'name': str,
        'description': str,
        'layers': List[Dict[str, Any]],  # Layer specifications
        'connections': List[Tuple[str, str]],  # Edge definitions
    },
    total=False
)

PaperExcerpt = TypedDict(
    'PaperExcerpt',
    {
        'type': Literal['markdown', 'text', 'pdf_extract'],
        'content': str,
        'source': str,  # Paper ID, section name, etc.
    }
)

SymbolicDesc = TypedDict(
    'SymbolicDesc',
    {
        'type': Literal['symbolic'],
        'spec': str,  # e.g., "Conv2D(64,3,1)→ReLU→MaxPool→ResBlock(64)×3→..."
        'notation': str,  # Optional: "paper2code", "custom"
    }
)
```

### Composite Block Representation

**Rule:** Composite blocks are represented as `GraphNode` with `is_composite=True` and `internal_graph`.

```python
# Example: ResidualBlock as composite
node = GraphNode(
    id="res_block_1",
    label="ResidualBlock(64)",
    type="composite",
    is_composite=True,
    internal_graph=ArchitectureGraph(
        nodes=[
            GraphNode(id="rb1_conv1", label="Conv 3×3", ...),
            GraphNode(id="rb1_bn", label="BatchNorm", ...),
            GraphNode(id="rb1_relu", label="ReLU", ...),
        ],
        edges=[...]
    ),
    semantic_params={
        "skip_connection": "yes",
        "compute_role": "residual transformation"
    }
)
```

### Semantic Parameters Attachment

**Rule:** Semantic params are inferred from:
1. **Explicit annotations** in config (if provided)
2. **Type-based defaults** (Conv2D → flops="high")
3. **Structural rules** (skip edge → skip_connection="yes")

```python
SEMANTIC_PARAM_RULES = {
    # Type → default semantic_params
    "Conv2D": {"flops": "high", "compute_role": "feature extraction"},
    "MatMul": {"flops": "very high", "compute_role": "attention"},
    "MultiHeadAttention": {"attention": "quadratic", "flops": "very high"},
    "MaxPool": {"feature_map": "downsampling", "flops": "low"},
    "UpSample": {"feature_map": "upsampling", "flops": "low"},
    "ResidualBlock": {"skip_connection": "yes"},
    "Linear": {"flops": "high", "compute_role": "classification"},
}
```

### Semantic Params Output

**Contract:** Parsing agent attaches:
- `flops`: "low" | "medium" | "high" | "very high"
- `compute_role`: str (e.g., "feature extraction", "attention", "classification")
- `attention`: "none" | "linear" | "quadratic" (if applicable)
- `feature_map`: "constant" | "downsampling" | "upsampling"
- `skip_connection`: "yes" | "no"
- `tokens`: int | "constant" | "variable" (for Transformers)

### Out of Scope

❌ Visual representation  
❌ Comparison or ranking  
❌ Optimization suggestions  
❌ Code generation  
❌ Semantic inference beyond type-based defaults  

### Responsibility Contract

**Must:**
- ✅ Parse well-formed architecture specifications
- ✅ Attach semantic_params to every node
- ✅ Validate that edges reference existing nodes
- ✅ Produce deterministic output (same input → same graph)
- ✅ Include descriptions for every node

**Must NOT:**
- ❌ Create visualization objects
- ❌ Invent semantic params not in defaults or config
- ❌ Make architectural judgments ("this is inefficient")
- ❌ Reference existing architectures as examples
- ❌ Perform comparisons

### Design Questions & Answers

**Q1: What input formats are supported initially?**  
A: Config dict (from visualizer builders), symbolic notation, paper excerpts (stretch goal).

**Q2: How are composite blocks represented?**  
A: As GraphNode with `is_composite=True` and internal_graph field.

**Q3: How are semantic_params attached?**  
A: Via type-based defaults + explicit config overrides + structural rules.

**Q4: What is out of scope?**  
A: Visualization, comparison, optimization, code generation.

---

## 🧩 Agent 2: Visualization Agent

### Role
Render ArchitectureGraphs into visual representations with semantic-aware highlighting. No graph construction or reasoning — purely representational.

### Interface

```python
class VisualizationAgent:
    """
    Responsibility: Render ArchitectureGraph → visual representations
    
    Guarantee: Deterministic rendering (same graph + context → same visuals)
    """
    
    def render(
        self,
        graph: ArchitectureGraph,
        mode: VisualizationMode = "single",
        comparison_ctx: Optional[ComparisonContext] = None,
        options: VisualizationOptions = VisualizationOptions()
    ) -> VisualRepresentation:
        """
        Render graph with semantic-aware visual cues.
        
        Args:
            graph: ArchitectureGraph to render
            mode: "single" (standard) or "compare" (side-by-side)
            comparison_ctx: Required if mode="compare"
            options: Visual configuration (colors, highlights, etc.)
        
        Returns:
            VisualRepresentation containing:
            - graphviz_dot: String (Graphviz DOT language)
            - node_annotations: Dict[str, NodeVisuals]
            - visual_cues: Set[str] (list of applied highlights)
        
        Raises:
            VisualizationError: If graph is malformed
        """
        pass
    
    def get_visual_cues(self) -> Set[str]:
        """
        List all supported visual cues (for documentation).
        
        Returns:
            {"bottleneck_badge", "compute_highlight", "scaling_highlight",
             "spatial_highlight", "ghost_overlay", "tooltip"}
        """
        pass
```

### Input Data Structures

```python
VisualizationMode = Literal["single", "compare"]

ComparisonContext = TypedDict(
    'ComparisonContext',
    {
        'mode': Literal['compare'],
        'current_arch': Literal['A', 'B'],
        'dominant_compute': Optional[Literal['A', 'B']],
        'dominant_spatial': Optional[Literal['A', 'B']],
        'scaling_issue': Optional[Literal['A', 'B']],
        'bottleneck_node_id': Optional[str],
    }
)

VisualizationOptions = TypedDict(
    'VisualizationOptions',
    {
        'expand_composite': bool,  # Expand composite blocks
        'include_params': bool,     # Show params in node labels
        'theme': Literal['light', 'dark'],
        'rankdir': Literal['TB', 'LR'],  # Top-to-bottom or left-to-right
    },
    total=False
)

NodeVisuals = TypedDict(
    'NodeVisuals',
    {
        'color': str,          # Hex color
        'penwidth': float,     # Border thickness
        'fillcolor': str,      # Fill color
        'style': str,          # "rounded,filled" etc.
        'label_suffix': str,   # Badge text (e.g., "🔥 BOTTLENECK")
    },
    total=False
)

VisualRepresentation = TypedDict(
    'VisualRepresentation',
    {
        'graphviz_dot': str,
        'node_annotations': Dict[str, NodeVisuals],
        'visual_cues': List[str],
        'comparison_mode': bool,
    }
)
```

### Visual Cues (Allowed Signals)

**Single Architecture Mode:**
- 🟢 **FLOPs coloring:** Red/orange borders for high-FLOPs nodes (from semantic_params["flops"])
- 📌 **Tooltips:** Node descriptions (from semantic_params)

**Comparison Mode (Priority Order):**
1. 🔥 **Bottleneck badge** — "🔥 COMPUTE BOTTLENECK" label suffix
   - Applied to node matching `comparison_ctx.bottleneck_node_id`
   - Color: #CC0000 (dark red), penwidth: 4.0
   
2. 🔴 **Compute highlight** — Thick red borders
   - Applied when: `dominant_compute == current_arch` AND `flops in ["high", "very high"]`
   - Color: #FF6666, penwidth: 3.0
   
3. 🟠 **Scaling highlight** — Orange with warning badge
   - Applied when: `scaling_issue == current_arch` AND `attention == "quadratic"`
   - Color: #FFA500, penwidth: 3.0, label_suffix: "⚠ Quadratic Scaling"
   
4. 🔵 **Spatial highlight** — Blue borders
   - Applied when: `dominant_spatial == current_arch` AND `skip_connection == "yes"`
   - Color: #4169E1, penwidth: 2.5
   
5. ⚪ **Ghost overlay** — Greyed out
   - Applied to all remaining nodes
   - Color: #CCCCCC, penwidth: 1.0, fillcolor: #F8F8F8, style: "rounded,filled"

**Rule:** Higher priority cues **override** lower ones. A node receives only one cue.

### Mode Switching

```python
# Single mode: use node.semantic_params["flops"] for coloring
if mode == "single":
    for node in graph.nodes:
        if node.semantic_params["flops"] in FLOPS_COLORS:
            visuals = FLOPS_COLORS[node.semantic_params["flops"]]

# Compare mode: apply comparison context rules
elif mode == "compare":
    for node in graph.nodes:
        # Check priority 1: bottleneck
        if node.id == comparison_ctx["bottleneck_node_id"]:
            visuals = BOTTLENECK_STYLE
        # Check priority 2: compute
        elif comparison_ctx["dominant_compute"] == comparison_ctx["current_arch"]:
            if node.semantic_params["flops"] in ["high", "very high"]:
                visuals = COMPUTE_HIGHLIGHT_STYLE
        # ... (continue for priorities 3-5)
```

### Comparison Summary Consumption

**Rule:** Visualization agent consumes:
- `dominant_compute` (from summarize_compute)
- `dominant_spatial` (from summarize_spatial_behavior)
- `scaling_issue` (derived from scaling comparison)
- `bottleneck_node_id` (from primary_bottleneck)

**Does NOT:**
- ❌ Recompute summaries
- ❌ Make its own comparison decisions
- ❌ Modify the graph

### Out of Scope

❌ Graph construction  
❌ Semantic inference  
❌ Explanation text generation  
❌ Business logic (e.g., "which is better?")  

### Responsibility Contract

**Must:**
- ✅ Render valid ArchitectureGraph to visual output
- ✅ Apply visual cues in priority order
- ✅ Produce deterministic output (same input → same visuals)
- ✅ Support both single and comparison modes
- ✅ Consume comparison context without modifying it

**Must NOT:**
- ❌ Construct new graphs
- ❌ Infer semantic parameters
- ❌ Generate explanations
- ❌ Make architectural judgments
- ❌ Change graph structure

### Design Questions & Answers

**Q1: How does it switch between single and comparison mode?**  
A: Via `mode` parameter. Single mode uses semantic_params["flops"], compare mode uses comparison_ctx.

**Q2: How does it consume comparison summaries?**  
A: Via ComparisonContext dict with fields: dominant_compute, dominant_spatial, scaling_issue, bottleneck_node_id.

**Q3: What visual signals are allowed?**  
A: Only those derived from semantic_params or comparison results (colors, borders, badges, ghosting). No free-form styling.

**Q4: How are priorities enforced?**  
A: Nested if-elif checks. Priority 1 checked first; if true, applied. If not, check priority 2, etc.

---

## 🧩 Agent 3: Explanation Agent

### Role
Generate natural-language explanations that narrate existing reasoning. **No invention.** Only transforms structured facts into human-readable text.

### Interface

```python
class ExplanationAgent:
    """
    Responsibility: Narrate existing reasoning in natural language
    
    Guarantee: Deterministic narration (same input → same explanation)
    
    Core constraint: Must not invent facts. Only reorganize what is known.
    """
    
    def explain_node(
        self,
        node: GraphNode
    ) -> str:
        """
        Explain a single node in human-readable language.
        
        Uses:
        - node.description (from Parsing Agent)
        - node.semantic_params (from Parsing Agent)
        
        Returns:
            Markdown string explaining the node's role and properties
        """
        pass
    
    def explain_graph(
        self,
        graph: ArchitectureGraph
    ) -> str:
        """
        Explain the overall architecture.
        
        Uses:
        - graph.nodes and semantic_params
        - Node descriptions
        - Structural patterns (e.g., "encoder → decoder")
        
        Returns:
            Markdown string describing architecture purpose and design
        """
        pass
    
    def explain_comparison(
        self,
        graph_a: ArchitectureGraph,
        graph_b: ArchitectureGraph,
        comparison_result: ComparisonResult,
        visual_metadata: VisualMetadata
    ) -> str:
        """
        Explain architectural differences.
        
        Uses:
        - summarize_compute() results (compute differences)
        - summarize_spatial_behavior() results
        - summarize_scaling_behavior() results
        - compare_graphs() results
        - VisualMetadata (bottleneck labels, highlights)
        
        Returns:
            Markdown string narrating why one architecture differs
        """
        pass
    
    def get_explanation_templates(self) -> Dict[str, str]:
        """
        Expose explanation templates (for debugging/understanding).
        
        Returns:
            Dict mapping semantic facts to explanation phrases
        """
        pass
```

### Input Data Structures

```python
ComparisonResult = TypedDict(
    'ComparisonResult',
    {
        'compute_summary_a': ComputeSummary,
        'compute_summary_b': ComputeSummary,
        'spatial_summary_a': SpatialSummary,
        'spatial_summary_b': SpatialSummary,
        'scaling_summary_a': ScalingSummary,
        'scaling_summary_b': ScalingSummary,
    }
)

VisualMetadata = TypedDict(
    'VisualMetadata',
    {
        'bottleneck_a': Optional[str],
        'bottleneck_b': Optional[str],
        'highlighted_nodes_a': Set[str],
        'highlighted_nodes_b': Set[str],
        'visual_cues_applied': List[str],
    }
)
```

### Explanation Templates (Deterministic Mapping)

**Rule:** Explanation phrases are **pre-defined** for each semantic fact. Agent does not generate novel language.

```python
SEMANTIC_FACT_TEMPLATES = {
    # Compute intensity
    ("flops", "low"): "{node} has low computational cost",
    ("flops", "medium"): "{node} has moderate computational cost",
    ("flops", "high"): "{node} is computationally expensive",
    ("flops", "very high"): "{node} is extremely expensive and likely a bottleneck",
    
    # Attention complexity
    ("attention", "quadratic"): "{node} uses quadratic-complexity attention (O(n²))",
    ("attention", "linear"): "{node} uses linear-complexity attention (O(n))",
    
    # Feature map
    ("feature_map", "downsampling"): "{node} reduces spatial dimensions",
    ("feature_map", "upsampling"): "{node} increases spatial dimensions",
    
    # Skip connections
    ("skip_connection", "yes"): "{node} has skip connections for gradient flow",
    
    # Comparison templates
    ("compute_dominant", "A"): "**Architecture A** is more computationally efficient ({count_a} vs {count_b} high-FLOPs nodes)",
    ("compute_dominant", "B"): "**Architecture B** is more computationally efficient ({count_a} vs {count_b} high-FLOPs nodes)",
    ("scaling_issue", "A"): "**Architecture A** may struggle with large inputs due to {issue}",
    ("scaling_issue", "B"): "**Architecture B** may struggle with large inputs due to {issue}",
    ("spatial_preserved", "A"): "**Architecture A** preserves spatial structure better, suitable for dense prediction",
    ("spatial_preserved", "B"): "**Architecture B** preserves spatial structure better, suitable for dense prediction",
}

# Visual reference templates
VISUAL_REFERENCE_TEMPLATES = {
    "bottleneck": "the 🔥 **COMPUTE BOTTLENECK** node",
    "compute_highlight": "red-bordered nodes (high-FLOPs operations)",
    "scaling_highlight": "orange nodes with ⚠️ (quadratic complexity)",
    "spatial_highlight": "blue-bordered nodes (skip connections)",
    "ghost_overlay": "greyed-out nodes (shared structure)",
}
```

### Safe Comparison Explanation

**Rule:** Explanation agent **narrates** comparison results but **does not override** them.

```python
def explain_comparison(...):
    """
    Template-based comparison explanation:
    
    1. Identify primary difference drivers from comparison_result
    2. Reference visual metadata (bottleneck, highlights)
    3. Use pre-defined templates
    4. Link to semantic reasoning ("because {node} has {semantic_param}")
    """
    
    explanation = []
    
    # Section 1: Compute difference
    if comparison_result["compute_summary_a"]["total_high_flops"] > comparison_result["compute_summary_b"]["total_high_flops"]:
        explanation.append(
            SEMANTIC_FACT_TEMPLATES[("compute_dominant", "A")].format(
                count_a=comparison_result["compute_summary_a"]["total_high_flops"],
                count_b=comparison_result["compute_summary_b"]["total_high_flops"],
            )
        )
        explanation.append(
            f"See the {VISUAL_REFERENCE_TEMPLATES['compute_highlight']} in Architecture A."
        )
    
    # Section 2: Bottleneck callout
    if visual_metadata["bottleneck_a"]:
        explanation.append(
            f"The {VISUAL_REFERENCE_TEMPLATES['bottleneck']} is: {visual_metadata['bottleneck_a']}"
        )
    
    # ... (continue for spatial, scaling)
    
    return "\n\n".join(explanation)
```

### Tie Handling

**Rule:** When architectures are similar, agent reports similarity explicitly.

```python
def explain_comparison(...):
    # Detect ties
    if abs(
        comparison_result["compute_summary_a"]["total_high_flops"] -
        comparison_result["compute_summary_b"]["total_high_flops"]
    ) < 1:  # Threshold
        return "**Architectures are similar** in computational cost. Both use comparable high-FLOPs operations."
    
    # ... (continue with differences if any)
```

### Out of Scope

❌ Inventing facts not in comparison results  
❌ Making architectural recommendations  
❌ Suggesting optimizations  
❌ Generating code  
❌ Creating visualizations  

### Responsibility Contract

**Must:**
- ✅ Narrate existing facts from semantic_params and comparison results
- ✅ Use pre-defined explanation templates (no free-form generation)
- ✅ Reference visual cues accurately (bottleneck, highlights)
- ✅ Produce deterministic output (same input → same text)
- ✅ Clearly state when architectures are similar/tied
- ✅ Link explanations to semantic reasoning

**Must NOT:**
- ❌ Invent facts not in semantic_params or comparison results
- ❌ Override comparison conclusions
- ❌ Generate novel architectural insights
- ❌ Make recommendations beyond "suitable for X task"
- ❌ Create visualization objects

### Design Questions & Answers

**Q1: How does it reference visual elements safely?**  
A: Via pre-defined VISUAL_REFERENCE_TEMPLATES. Agent never invents visual descriptions.

**Q2: How does it stay aligned with rule-based logic?**  
A: Templates are derived directly from semantic_params values and comparison results. No inference beyond that.

**Q3: What tone is required?**  
A: Educational, analytical, deterministic. Neutral language (no "better/worse" without context).

**Q4: How does it handle ties or similar architectures?**  
A: Explicit tie detection and reporting. "Architectures are similar in X. Differences appear in Y."

---

## 🔗 Agent Orchestration (Conceptual)

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Input                             │
│  (config / paper excerpt / symbolic notation)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │  PARSING AGENT         │
        │  ─────────────────     │
        │  Input: Raw source     │
        │  Output: Graph + params│
        │  Deterministic         │
        └────────────┬───────────┘
                     │
        ┌────────────▼─────────────────────────────────┐
        │                                              │
        ▼                                              ▼
    ┌─────────────────┐                  ┌─────────────────┐
    │   SINGLE-GRAPH  │                  │  TWO-GRAPH      │
    │   WORKFLOW      │                  │  WORKFLOW       │
    │ ─────────────── │                  │ ─────────────── │
    │ Graph → Summar. │                  │ Graph A → Summ. │
    │                 │                  │ Graph B → Summ. │
    │ VIS AGENT       │                  │ Compare()       │
    │  (single mode)  │                  │                 │
    │                 │                  │ VIS AGENT       │
    │ EXP AGENT       │                  │  (compare mode) │
    │  (simple)       │                  │                 │
    │                 │                  │ EXP AGENT       │
    │ Output: MD      │                  │  (comparison)   │
    │         + Graph │                  │                 │
    │                 │                  │ Output: MD      │
    │                 │                  │         + Graph │
    └─────────────────┘                  └─────────────────┘
           │                                      │
           └──────────────┬───────────────────────┘
                          │
                          ▼
            ┌──────────────────────────┐
            │   Streamlit / Frontend   │
            │   Render all outputs     │
            └──────────────────────────┘
```

### Optional Agents per Context

| Context | Parsing | Visualization | Explanation |
|---------|---------|----------------|-------------|
| Load single architecture | ✅ | ✅ | ✅ (optional) |
| Inspect architecture | ✅ | ✅ | ✅ |
| Compare two architectures | ✅ | ✅ | ✅ (required) |
| Batch analysis | ✅ | ✅ (optional) | ✅ (optional) |
| Export graph | ✅ | ✅ | ❌ |

**Rule:** Parsing is always required. Visualization and Explanation depend on intent.

### Composition Without Dependencies

```python
# Agents are decoupled
parsing_agent = ParsingAgent()
vis_agent = VisualizationAgent()
exp_agent = ExplanationAgent()

# Single-architecture workflow
graph = parsing_agent.parse(config)
visuals = vis_agent.render(graph, mode="single")
explanation = exp_agent.explain_graph(graph)

# Comparison workflow
graph_a = parsing_agent.parse(config_a)
graph_b = parsing_agent.parse(config_b)

summ_a = summarize_compute(graph_a)  # Not an agent (existing util)
summ_b = summarize_compute(graph_b)

comparison = compare_graphs(graph_a, graph_b)  # Not an agent (existing util)

vis_a = vis_agent.render(graph_a, mode="compare", comparison_ctx=ctx_a)
vis_b = vis_agent.render(graph_b, mode="compare", comparison_ctx=ctx_b)

exp_comp = exp_agent.explain_comparison(graph_a, graph_b, comparison, visual_meta)
```

### Future RAG Integration (Conceptual)

**How to add RAG without breaking determinism:**

```
OPTION 1: RAG as Pre-Parsing Step
┌──────────────────────────────────┐
│  Paper → RAG Extraction → Config │  (Probabilistic)
└────────────────┬─────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  PARSING AGENT     │  (Deterministic)
        │  + Parsing Verify  │
        └────────────┬───────┘
                     │
                     ▼
            (Rest of pipeline)

Key: RAG produces multiple candidate configs, each fed to Parsing Agent.
Determinism maintained at Parsing → Explanation level.
```

**Key Invariant:** Parsing, Visualization, and Explanation agents remain deterministic. Any non-determinism is **upstream** (RAG extraction phase), not inside the agents themselves.

---

## 🚫 Explicit Non-Goals

❌ **No LLM autonomy** — Explanation Agent uses templates, not generative models  
❌ **No hidden reasoning** — All decisions trace to semantic_params or comparison results  
❌ **No probabilistic outputs** — All outputs are deterministic (same input → same output)  
❌ **No UI logic inside agents** — Rendering decisions are made by Visualization Agent, not inside other agents  
❌ **No circular dependencies** — Parsing doesn't depend on Visualization/Explanation, etc.  
❌ **No semantic invention** — Parsing Agent only uses type-based defaults or explicit config  
❌ **No override of determinism** — Explanation Agent narrates, never overrides comparison results  

---

## ✅ Why This Design Is Debuggable, Testable, Extensible

### Debuggability

**Single Responsibility:**
- If graph is wrong → debug Parsing Agent in isolation
- If visuals are wrong → debug Visualization Agent with fixed graph
- If explanation is wrong → debug Explanation Agent with fixed facts

**Deterministic I/O:**
- Same input always produces same output
- No randomness to hide bugs
- Reproducible issues

**Clear Data Flow:**
- Each agent accepts well-defined input, returns well-defined output
- No hidden state or side effects

### Testability

**Unit Testing:**
```python
# Test Parsing Agent
def test_parsing_agent():
    agent = ParsingAgent()
    graph = agent.parse(config)
    assert len(graph.nodes) == expected_count
    assert graph.nodes[0].semantic_params["flops"] in ["low", "medium", "high", "very high"]

# Test Visualization Agent
def test_visualization_agent():
    agent = VisualizationAgent()
    visuals = agent.render(graph, mode="single")
    assert "graphviz_dot" in visuals
    assert visuals["node_annotations"]["conv_1"]["color"] in HEX_COLORS

# Test Explanation Agent
def test_explanation_agent():
    agent = ExplanationAgent()
    exp = agent.explain_graph(graph)
    assert "Conv" in exp  # References node from graph
    assert len(exp) > 0
```

**Integration Testing:**
```python
# Full pipeline
graph = parsing_agent.parse(config)
visuals = vis_agent.render(graph)
explanation = exp_agent.explain_graph(graph)
# Assert all outputs are consistent
```

**Regression Testing:**
```python
# Store expected outputs for known inputs
GOLDEN_TESTS = {
    ("ResNet18_config", "single"): ("expected_graph", "expected_visuals", "expected_explanation")
}

# Run all golden tests
for input_config, mode in GOLDEN_TESTS:
    graph = parsing_agent.parse(input_config)
    visuals = vis_agent.render(graph, mode=mode)
    explanation = exp_agent.explain_graph(graph)
    assert (graph, visuals, explanation) == GOLDEN_TESTS[(input_config, mode)]
```

### Extensibility

**Adding new architectures:**
- Define config in supported format
- Pass to Parsing Agent (no code changes)
- Rest of pipeline works automatically

**Adding new semantic parameters:**
- Update SEMANTIC_PARAM_RULES in Parsing Agent
- Update VISUAL_CUE_* in Visualization Agent
- Update SEMANTIC_FACT_TEMPLATES in Explanation Agent
- No changes to agent interfaces

**Adding new comparison metrics:**
- Extend summarize_* functions (separate from agents)
- Pass results to agents
- Explanation Agent narrates new facts via new templates

**Adding new visualization modes:**
- Extend VisualizationMode type
- Add mode logic to Visualization Agent
- Other agents unaffected

**Adding RAG extraction:**
- Insert between user input and Parsing Agent
- Parsing Agent interface unchanged
- Everything downstream works with RAG output

---

## Summary: Contract per Agent

### Parsing Agent

**Input:** Raw architecture specification (config dict, paper excerpt, or symbolic notation)  
**Output:** ArchitectureGraph with semantic parameters  
**Guarantee:** Deterministic (same input → same graph)  
**Forbidden:** Visualization, comparison, explanation  

### Visualization Agent

**Input:** ArchitectureGraph + optional comparison context  
**Output:** Visual representation (Graphviz + annotations)  
**Guarantee:** Deterministic (same input → same visuals)  
**Forbidden:** Graph construction, semantic inference, explanation text  

### Explanation Agent

**Input:** ArchitectureGraph + comparison results + visual metadata  
**Output:** Natural-language markdown explanation  
**Guarantee:** Deterministic, template-based (no invention)  
**Forbidden:** Fact invention, optimization suggestions, recommendations  

---

## Next Steps

**When Ready to Implement:**

1. **Phase 1:** Formalize agent interfaces as Python protocols/ABCs
2. **Phase 2:** Implement Parsing Agent with config parsing
3. **Phase 3:** Implement Visualization Agent (reuse existing render logic)
4. **Phase 4:** Implement Explanation Agent (extract templates from existing explanations)
5. **Phase 5:** Wire agents into Streamlit app
6. **Phase 6:** Add RAG as pre-parsing step (optional, non-determinism upstream)

**Design Artifacts to Produce:**
- [ ] Agent protocol definitions (Python)
- [ ] Data structure TypedDicts (Python)
- [ ] Template library (YAML or Python dict)
- [ ] Unit test templates
- [ ] Integration test plan
- [ ] Deployment strategy (if agents run separately)

---

**Status:** Design complete, ready for implementation review.
