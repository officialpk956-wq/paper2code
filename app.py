import streamlit as st
import graphviz
import pandas as pd
import os

from core.visualizer_resnet import build_resnet18_graph
from core.visualizer_unet import build_unet_graph
from core.visualizer_vit import build_vit_graph
from core.explainers import explain_node, explain_graph
from core.comparators import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
    explain_architecture_comparison,
)
from core.orchestrator.pipeline import Paper2CodePipeline
from core.paper_to_code_generator import PaperToCodeGenerator


# Check GROQ availability for PDF extraction
_GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY"))


def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    import io
    try:
        import pdfplumber
    except ImportError:
        return ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages[:30]:  # Cap at 30 pages
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text)


# Helper function to determine comparison-based styling
def get_comparison_styling(node, comparison_ctx):
    """
    Return visual styling attributes for a node in comparison mode.
    
    Args:
        node: GraphNode to style
        comparison_ctx: dict with keys:
            - 'mode': 'single' or 'compare'
            - 'dominant_compute': 'A', 'B', or None
            - 'dominant_spatial': 'A', 'B', or None  
            - 'scaling_issue': 'A', 'B', or None
            - 'current_arch': 'A' or 'B'
            - 'bottleneck_node_id': str or None
    
    Returns:
        dict with optional keys: color, penwidth, label_suffix, fillcolor, style
    """
    if comparison_ctx.get('mode') != 'compare':
        return {}
    
    styling = {}
    current = comparison_ctx.get('current_arch')
    sp = node.semantic_params or {}
    
    # Task 2: Bottleneck badge
    if node.id == comparison_ctx.get('bottleneck_node_id'):
        styling['label_suffix'] = '\n🔥 COMPUTE BOTTLENECK'
        styling['color'] = '#CC0000'  # Darker red
        styling['penwidth'] = '4.0'
        return styling
    
    # Task 1: Highlight nodes driving compute difference
    if comparison_ctx.get('dominant_compute') == current:
        if sp.get('flops') in ['high', 'very high']:
            styling['color'] = '#FF6666'  # Lighter red highlight
            styling['penwidth'] = '3.0'
            return styling
    
    # Task 1: Highlight quadratic scaling issues
    if comparison_ctx.get('scaling_issue') == current:
        if sp.get('attention') == 'quadratic':
            styling['color'] = '#FFA500'  # Orange
            styling['penwidth'] = '3.0'
            styling['label_suffix'] = '\n⚠ Quadratic Scaling'
            return styling
    
    # Task 1: Highlight spatial structure (skip connections)
    if comparison_ctx.get('dominant_spatial') == current:
        if sp.get('skip_connection') == 'yes':
            styling['color'] = '#4169E1'  # Royal blue
            styling['penwidth'] = '2.5'
            return styling
    
    # Task 3: Ghost overlay for non-highlighted nodes in comparison
    if comparison_ctx.get('mode') == 'compare':
        styling['color'] = '#CCCCCC'  # Greyed out
        styling['penwidth'] = '1.0'
        styling['style'] = 'rounded,filled'
        styling['fillcolor'] = '#F8F8F8'
    
    return styling


def render_graph_with_comparison(graph, comparison_ctx=None, expand_blocks=False):
    """
    Render a graph with optional comparison styling.

    Args:
        graph: ArchitectureGraph to render
        comparison_ctx: Optional comparison context dict
        expand_blocks: Whether to expand composite blocks (currently for future use)

    Returns:
        graphviz.Digraph object
    """
    dot = graphviz.Digraph(
        comment=graph.name,
        graph_attr={"rankdir": "TB"}
    )
    
    for node in graph.nodes:
        # Build base label
        label = f"{node.label}\n{node.type}"
        for k, v in node.params.items():
            label += f"\n{k}: {v}"
        
        # Don't show semantic params in comparison mode to reduce clutter
        if not comparison_ctx or comparison_ctx.get('mode') != 'compare':
            if node.semantic_params:
                for k, v in node.semantic_params.items():
                    label += f"\n{k}: {v}"
        
        # Base styling
        node_attrs = {
            "label": label,
            "shape": "box",
            "style": "rounded",
            "tooltip": node.description or node.label
        }
        
        # Apply comparison styling if in comparison mode
        if comparison_ctx:
            comp_style = get_comparison_styling(node, comparison_ctx)
            
            # Add label suffix if present
            if 'label_suffix' in comp_style:
                node_attrs['label'] += comp_style['label_suffix']
            
            # Apply visual styling
            for key in ['color', 'penwidth', 'fillcolor', 'style']:
                if key in comp_style:
                    node_attrs[key] = comp_style[key]
        else:
            # Single-architecture mode: use FLOPs coloring
            if node.semantic_params and "flops" in node.semantic_params:
                flops_level = node.semantic_params["flops"]
                if flops_level in FLOPS_COLORS:
                    node_attrs["color"] = FLOPS_COLORS[flops_level]["color"]
                    node_attrs["penwidth"] = FLOPS_COLORS[flops_level]["penwidth"]
        
        dot.node(node.id, **node_attrs)
    
    # Edges
    for edge in graph.edges:
        style = EDGE_STYLES.get(edge.edge_type, EDGE_STYLES["flow"])
        dot.edge(edge.source, edge.target, **style)
    
    return dot


# Helper function for semantic reasoning
def generate_reasoning(node) -> str:
    """Generate 'Why This Matters' reasoning from semantic params."""
    reasoning = []
    
    if not node.semantic_params:
        return ""
    
    # FLOPs reasoning
    flops = node.semantic_params.get("flops")
    if flops == "high":
        reasoning.append("🔴 **High computational cost** — This block dominates the forward/backward pass")
    elif flops == "very high":
        reasoning.append("🔴🔴 **Very high computational cost** — Potential bottleneck for large inputs")
    elif flops == "medium":
        reasoning.append("🟡 **Moderate computational cost** — Reasonable overhead for the benefits")
    
    # Attention complexity
    attention = node.semantic_params.get("attention")
    if attention == "quadratic":
        reasoning.append("⚠️ **Quadratic complexity** — Scales poorly with sequence/spatial length (O(n²))")
    
    # Feature map
    feature_map = node.semantic_params.get("feature_map")
    if feature_map == "downsampling":
        reasoning.append("↓ **Reduces spatial dimensions** — Decreases memory and compute for downstream layers")
    elif feature_map == "upsampling":
        reasoning.append("↑ **Increases spatial dimensions** — Recovers fine-grained details for dense predictions")
    
    # Skip connections
    skip = node.semantic_params.get("skip_connection")
    if skip == "yes":
        reasoning.append("🔗 **Skip connections active** — Preserves low-level features and improves gradient flow")
    
    # Compute role
    role = node.semantic_params.get("compute_role")
    if role:
        reasoning.append(f"📌 **Purpose**: {role}")
    
    # Tokens (ViT)
    tokens = node.semantic_params.get("tokens")
    if tokens and tokens != "constant":
        reasoning.append(f"📍 **Token count**: {tokens} (affects attention complexity and memory)")
    elif tokens == "constant":
        reasoning.append("📍 **Constant token count** — Consistent complexity regardless of input resolution")
    
    return "\n".join(reasoning)


EDGE_STYLES = {
    "flow": {"style": "solid", "color": "black", "penwidth": "1.5"},
    "skip": {"style": "dashed", "color": "blue", "penwidth": "1.5"},
    "residual": {"style": "dashed", "color": "red", "penwidth": "1.5"},
}

# Color scheme for compute intensity
FLOPS_COLORS = {
    "high": {"color": "#FF4444", "penwidth": "2.5"},      # Red
    "very high": {"color": "#FF0000", "penwidth": "3"},    # Bright red
    "medium": {"color": "#FFA500", "penwidth": "2"},       # Orange
}


st.set_page_config(page_title="paper2code – Architecture Visualizer", layout="wide")
st.title("paper2code – Architecture Visualizer")
st.caption("Interactive visualization of deep learning architectures")


# --------------------------------------------------
# Session State Management
# --------------------------------------------------
def _init_session_state():
    """Initialize session state keys with safe defaults."""
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"name": str, "graph": ArchitectureGraph}


def _push_to_history(name: str, graph):
    """Push architecture to history, deduplicate by name, cap at 10."""
    st.session_state.history = [h for h in st.session_state.history if h["name"] != name]
    st.session_state.history.insert(0, {"name": name, "graph": graph})
    st.session_state.history = st.session_state.history[:10]


# Initialize session state immediately
_init_session_state()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Architecture Input")

# F10: Paper to Code (PDF/arXiv → PyTorch)
use_paper_to_code = st.sidebar.checkbox("📄 Paper to Code", value=False, key="use_paper_to_code")
if use_paper_to_code:
    st.sidebar.subheader("Paper to Code")
    st.sidebar.caption("Upload PDF or paste arXiv URL to generate working PyTorch code")

    # Option 1: Upload PDF
    uploaded_file = st.sidebar.file_uploader("Select PDF", type=["pdf"], key="p2c_pdf_uploader")

    # Option 2: arXiv URL
    arxiv_url = st.sidebar.text_input(
        "Or enter arXiv URL",
        placeholder="https://arxiv.org/abs/1512.03385",
        key="p2c_arxiv_input"
    )

    # Generate button
    if st.sidebar.button("🚀 Generate Code", key="p2c_generate_btn"):
        if uploaded_file is None and not arxiv_url:
            st.error("Please upload a PDF or enter an arXiv URL")
            st.stop()

        # Initialize generator
        generator = PaperToCodeGenerator()

        # Progress tracking
        progress_placeholder = st.empty()

        try:
            # Run generator (handles all 4 steps internally)
            if uploaded_file is not None:
                paper_bytes = uploaded_file.read()
                paper_name = uploaded_file.name.replace(".pdf", "")
                progress_placeholder.info("⏳ Step 1/4: Extracting text from PDF...")
                with st.spinner("Steps 2-4: Classifying sections, extracting architecture, generating code..."):
                    result = generator.from_pdf(paper_bytes, paper_name)
            elif arxiv_url:
                progress_placeholder.info("⏳ Step 1/4: Fetching PDF from arXiv...")
                with st.spinner("Steps 2-4: Classifying sections, extracting architecture, generating code..."):
                    result = generator.from_arxiv(arxiv_url)
            else:
                st.error("Invalid input")
                st.stop()

            progress_placeholder.success("✅ Code generation complete!")

            # Push to history
            _push_to_history(result["paper_name"], result["graph"])

            # Store result in session state for main display
            st.session_state.p2c_result = result
            st.stop()

        except Exception as e:
            progress_placeholder.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.stop()

# F9: Symbolic Notation Input
use_symbolic_input = st.sidebar.checkbox("Symbolic Notation", value=False, key="use_symbolic")
if use_symbolic_input:
    st.sidebar.caption("Format: Conv2D(64,3)→ReLU→ResBlock×3→Linear(10)")
    sym_spec = st.sidebar.text_input(
        "Architecture spec",
        placeholder="Conv2D(64,3)→ReLU→MaxPool→ResBlock×3→Linear(10)",
        key="symbolic_spec_input"
    )
    if sym_spec:
        from core.rag.symbolic_parser import parse_symbolic
        try:
            config = parse_symbolic(sym_spec)
            pipeline = Paper2CodePipeline()
            result = pipeline.run_single(config)
            graph = result["graph"]
            _push_to_history(graph.name, graph)
            st.subheader(f"Architecture: {graph.name}")
            dot_graph = render_graph_with_comparison(graph)
            st.graphviz_chart(dot_graph, use_container_width=True)
            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download .dot", str(dot_graph), f"{graph.name}.dot", "text/plain", key="dot_sym")
            with col2:
                svg_data = dot_graph.pipe(format='svg').decode('utf-8')
                st.download_button("⬇️ Download SVG", svg_data, f"{graph.name}.svg", "image/svg+xml", key="svg_sym")
            # Metrics and code gen
            from core.metrics_estimator import estimate_metrics_from_graph
            with st.expander("Compute Metrics", expanded=True):
                metrics = estimate_metrics_from_graph(graph)
                c1, c2, c3 = st.columns(3)
                c1.metric("Relative FLOPs Score", metrics["total_flops_score"])
                params = metrics["total_params_estimate"]
                c2.metric("Est. Parameters", f"~{params/1e6:.1f}M" if params > 1e6 else f"~{params/1e3:.0f}K")
                c3.metric("Depth", metrics["depth"])
                st.dataframe(pd.DataFrame(metrics["breakdown"]))
            # Code gen
            from core.codegen import get_pytorch_code
            if st.button("Generate PyTorch Code", key="btn_codegen_sym"):
                code = get_pytorch_code(graph.name, graph)
                st.code(code, language="python")
                st.download_button("Download .py", code, f"{graph.name}.py", "text/plain", key="dl_codegen_sym")
            st.markdown(result["explanation"])
        except Exception as e:
            st.error(f"Parse error: {e}")
        st.stop()
    else:
        st.sidebar.info("Enter a symbolic spec above")
        st.stop()

# RAG Mode: Text Input
use_text_input = st.sidebar.checkbox("Use Text Input (RAG Mode)", value=False)

if use_text_input:
    st.sidebar.subheader("Architecture Description")
    user_text = st.sidebar.text_area(
        "Enter architecture description",
        placeholder="e.g., ResNet with Conv layer, pooling, 3 residual blocks, and linear classifier",
        height=100
    )

    # Process text input through pipeline
    if user_text:
        pipeline = Paper2CodePipeline()
        result = pipeline.run_from_text(user_text)

        graph = result["graph"]
        visual = result["visual"]
        explanation = result["explanation"]

        # Push to history (F7)
        _push_to_history(graph.name, graph)

        # Show truncation warning if applicable
        if result.get("metadata", {}).get("truncated"):
            st.sidebar.warning(
                f"⚠️ Input too large — truncated from "
                f"{result['metadata']['original_layer_count']} to "
                f"{result['metadata']['layer_count']} layers for performance"
            )

        # Render graph and explanation
        st.subheader(f"Architecture: {graph.name}")
        dot_graph = render_graph_with_comparison(graph)
        st.graphviz_chart(dot_graph, use_container_width=True)

        # Export buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="⬇️ Download .dot",
                data=str(dot_graph),
                file_name=f"{graph.name}.dot",
                mime="text/plain"
            )
        with col2:
            svg_data = dot_graph.pipe(format='svg').decode('utf-8')
            st.download_button(
                label="⬇️ Download SVG",
                data=svg_data,
                file_name=f"{graph.name}.svg",
                mime="image/svg+xml"
            )

        # F3: Compute Metrics
        st.markdown("---")
        from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
        with st.expander("Compute Metrics", expanded=True):
            metrics = estimate_metrics_from_graph(graph)
            c1, c2, c3 = st.columns(3)
            c1.metric("Relative FLOPs Score", metrics["total_flops_score"])
            params = metrics["total_params_estimate"]
            c2.metric("Est. Parameters", f"~{params/1e6:.1f}M" if params > 1e6 else f"~{params/1e3:.0f}K")
            c3.metric("Depth", metrics["depth"])
            st.dataframe(pd.DataFrame(metrics["breakdown"]))

        # F6: Memory Estimation
        with st.expander("Estimated Activation Memory per Layer"):
            batch = st.slider("Batch size", 1, 64, 1, key="mem_batch_text")
            spatial = st.slider("Input spatial size", 32, 512, 224, 32, key="mem_spatial_text")
            mem_data = estimate_activation_memory(graph, batch_size=batch, input_spatial=spatial)
            df_mem = pd.DataFrame(mem_data)
            st.metric("Total Est. Activation Memory", f"{df_mem['mem_mb'].sum():.1f} MB")
            st.dataframe(df_mem)

        # F5: PyTorch Code Generation
        from core.codegen import get_pytorch_code
        if st.button("Generate PyTorch Code", key="btn_codegen_text"):
            code = get_pytorch_code(graph.name, graph)
            st.code(code, language="python")
            st.download_button("Download .py", code, f"{graph.name}.py", "text/plain", key="dl_codegen_text")

        st.markdown("---")
        st.subheader("Explanation")
        st.markdown(explanation)

        st.stop()  # Stop here if using text input
    else:
        st.sidebar.info("Enter an architecture description above")
        st.stop()

# Text-to-Text Comparison Mode
use_text_comparison = st.sidebar.checkbox("Compare Text Architectures", value=False)

if use_text_comparison:
    st.sidebar.subheader("Compare Two Architectures (Text)")

    text_a = st.sidebar.text_area(
        "First architecture description",
        placeholder="e.g., ResNet with Conv layers and residual blocks",
        height=80,
        key="text_a"
    )

    text_b = st.sidebar.text_area(
        "Second architecture description",
        placeholder="e.g., Vision Transformer with attention blocks",
        height=80,
        key="text_b"
    )

    if text_a and text_b:
        pipeline = Paper2CodePipeline()
        result = pipeline.run_comparison_from_text(text_a, text_b)

        graph_a = result["graph_a"]
        graph_b = result["graph_b"]
        explanation = result["explanation"]

        # Push to history (F7)
        _push_to_history(graph_a.name, graph_a)
        _push_to_history(graph_b.name, graph_b)

        st.subheader("📊 Text Architecture Comparison")

        # Show both architectures side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"#### Architecture A")
            dot_a = render_graph_with_comparison(graph_a)
            st.graphviz_chart(dot_a, use_container_width=True)
            col1a, col1b = st.columns(2)
            with col1a:
                st.download_button(
                    label="⬇️ .dot",
                    data=str(dot_a),
                    file_name=f"{graph_a.name}.dot",
                    mime="text/plain",
                    key="dot_a"
                )
            with col1b:
                svg_a = dot_a.pipe(format='svg').decode('utf-8')
                st.download_button(
                    label="⬇️ SVG",
                    data=svg_a,
                    file_name=f"{graph_a.name}.svg",
                    mime="image/svg+xml",
                    key="svg_a"
                )

        with col2:
            st.markdown(f"#### Architecture B")
            dot_b = render_graph_with_comparison(graph_b)
            st.graphviz_chart(dot_b, use_container_width=True)
            col2a, col2b = st.columns(2)
            with col2a:
                st.download_button(
                    label="⬇️ .dot",
                    data=str(dot_b),
                    file_name=f"{graph_b.name}.dot",
                    mime="text/plain",
                    key="dot_b"
                )
            with col2b:
                svg_b = dot_b.pipe(format='svg').decode('utf-8')
                st.download_button(
                    label="⬇️ SVG",
                    data=svg_b,
                    file_name=f"{graph_b.name}.svg",
                    mime="image/svg+xml",
                    key="svg_b"
                )

        st.markdown("---")
        st.subheader("Comparison Analysis")
        st.markdown(explanation)

        st.stop()  # Stop here if using text comparison
    else:
        st.sidebar.info("Enter both architecture descriptions above")
        st.stop()

# F4: PDF Upload + Auto-Extraction
if _GROQ_AVAILABLE:
    use_pdf = st.sidebar.checkbox("Upload PDF Paper", value=False, key="use_pdf")
    if use_pdf:
        st.sidebar.subheader("PDF Paper Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload research paper PDF",
            type=["pdf"],
            key="pdf_uploader"
        )

        if uploaded_file is not None:
            pdf_bytes = uploaded_file.read()

            with st.spinner("Extracting text from PDF..."):
                raw_text = _extract_text_from_pdf(pdf_bytes)

            if not raw_text.strip():
                st.error("Could not extract text from PDF. Ensure it is not a scanned/image-only PDF.")
                st.stop()

            with st.spinner("Classifying sections..."):
                from core.section_splitter import process_text
                try:
                    section_data = process_text(raw_text)
                except Exception as e:
                    st.error(f"Section classification failed: {e}")
                    st.stop()

            with st.spinner("Extracting architecture..."):
                from core.architecture_extractor import extract_architecture
                paper_name = uploaded_file.name.replace(".pdf", "")
                try:
                    schema = extract_architecture(section_data, paper_name)
                except Exception as e:
                    st.error(f"Architecture extraction failed: {e}")
                    st.stop()

            pipeline = Paper2CodePipeline()
            result = pipeline.run_single(schema)
            graph = result["graph"]
            _push_to_history(graph.name, graph)

            st.subheader(f"Extracted Architecture: {graph.name}")
            dot_graph = render_graph_with_comparison(graph)
            st.graphviz_chart(dot_graph, use_container_width=True)

            # Export buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download .dot", str(dot_graph), f"{graph.name}.dot", "text/plain", key="dot_pdf")
            with col2:
                svg_data = dot_graph.pipe(format='svg').decode('utf-8')
                st.download_button("⬇️ Download SVG", svg_data, f"{graph.name}.svg", "image/svg+xml", key="svg_pdf")

            # Metrics
            from core.metrics_estimator import estimate_metrics_from_graph
            with st.expander("Compute Metrics", expanded=True):
                metrics = estimate_metrics_from_graph(graph)
                c1, c2, c3 = st.columns(3)
                c1.metric("Relative FLOPs Score", metrics["total_flops_score"])
                params = metrics["total_params_estimate"]
                c2.metric("Est. Parameters", f"~{params/1e6:.1f}M" if params > 1e6 else f"~{params/1e3:.0f}K")
                c3.metric("Depth", metrics["depth"])

            st.markdown(result["explanation"])
            st.stop()
        else:
            st.sidebar.info("Upload a PDF to begin extraction")
            st.stop()

# F10: Display Paper to Code Results
if "p2c_result" in st.session_state:
    result = st.session_state.p2c_result
    graph = result["graph"]
    code = result["code"]
    code_source = result["code_source"]
    explanation = result["explanation"]
    spec = result["spec"]

    st.header(f"📄 Paper to Code: {result['paper_name']}")
    st.caption(f"Family: {result['family']} | Code source: {code_source}")

    # Two-column layout: Graph + Code
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Architecture Graph")
        dot_graph = render_graph_with_comparison(graph)
        st.graphviz_chart(dot_graph, use_container_width=True)

        # Export graph buttons
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "⬇️ .dot",
                str(dot_graph),
                f"{graph.name}.dot",
                "text/plain",
                key="p2c_dot_export"
            )
        with export_col2:
            svg_data = dot_graph.pipe(format='svg').decode('utf-8')
            st.download_button(
                "⬇️ SVG",
                svg_data,
                f"{graph.name}.svg",
                "image/svg+xml",
                key="p2c_svg_export"
            )

    with col2:
        st.subheader("Generated PyTorch Code")
        st.code(code, language="python")
        st.download_button(
            "⬇️ Download model.py",
            code,
            f"{result['paper_name']}_model.py",
            "text/plain",
            key="p2c_code_export"
        )

    # Metrics
    from core.metrics_estimator import estimate_metrics_from_graph
    st.subheader("Architecture Metrics")
    metrics = estimate_metrics_from_graph(graph)
    m1, m2, m3 = st.columns(3)
    m1.metric("Relative FLOPs", metrics["total_flops_score"])
    params = metrics["total_params_estimate"]
    m2.metric("Est. Parameters", f"~{params/1e6:.1f}M" if params > 1e6 else f"~{params/1e3:.0f}K")
    m3.metric("Depth", metrics["depth"])

    # Expandable sections
    with st.expander("Architecture Explanation", expanded=True):
        st.markdown(explanation)

    with st.expander("Raw Architecture Spec"):
        st.json(spec)

    st.stop()

# Standard Model Selection Mode
st.sidebar.title("Model Selector")

model_name = st.sidebar.selectbox(
    "Choose architecture",
    ["ResNet-18", "U-Net", "ViT"]
)

expand_blocks = st.sidebar.checkbox("Expand composite blocks", value=False)

# F7: History Panel
st.sidebar.markdown("---")
st.sidebar.subheader("History")
if not st.session_state.history:
    st.sidebar.info("No history yet")
else:
    for i, h in enumerate(st.session_state.history):
        st.sidebar.markdown(f"- {h['name']}")
    if len(st.session_state.history) >= 2:
        names = [h["name"] for h in st.session_state.history]
        ha = st.sidebar.selectbox("Compare A", names, key="hist_a_sel")
        hb = st.sidebar.selectbox("Compare B", names, key="hist_b_sel", index=min(1, len(names)-1))
        if st.sidebar.button("Compare from History", key="hist_compare_btn"):
            ga = next(h["graph"] for h in st.session_state.history if h["name"] == ha)
            gb = next(h["graph"] for h in st.session_state.history if h["name"] == hb)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {ha}")
                st.graphviz_chart(render_graph_with_comparison(ga), use_container_width=True)
            with col2:
                st.markdown(f"#### {hb}")
                st.graphviz_chart(render_graph_with_comparison(gb), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Node Inspector")

# Build graph first to get nodes
if model_name == "ResNet-18":
    graph = build_resnet18_graph()
elif model_name == "U-Net":
    graph = build_unet_graph()
elif model_name == "ViT":
    graph = build_vit_graph()
else:
    st.stop()

# Push to history (F7)
_push_to_history(model_name, graph)

# Node selection dropdown with id + label
node_display_options = [f"{n.id} – {n.label}" for n in graph.nodes]
selected_node_display = st.sidebar.selectbox("Select a node to inspect", node_display_options)

# Extract the selected node by matching the id
selected_node_id = selected_node_display.split(" – ")[0]
selected_node = next((n for n in graph.nodes if n.id == selected_node_id), None)

if selected_node:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {selected_node.label}")
    st.sidebar.info(explain_node(selected_node))


# --------------------------------------------------
# Build graph (continued from above)
# --------------------------------------------------


# --------------------------------------------------
# Graph rendering
# --------------------------------------------------
dot = render_graph_with_comparison(graph, expand_blocks=expand_blocks)

st.subheader(graph.name)
st.graphviz_chart(dot, use_container_width=True)

# Export buttons
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="⬇️ Download .dot",
        data=str(dot),
        file_name=f"{graph.name}.dot",
        mime="text/plain",
        key="dot_model"
    )
with col2:
    svg_data = dot.pipe(format='svg').decode('utf-8')
    st.download_button(
        label="⬇️ Download SVG",
        data=svg_data,
        file_name=f"{graph.name}.svg",
        mime="image/svg+xml",
        key="svg_model"
    )

# F3: Compute Metrics
st.markdown("---")
from core.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
from core.codegen import get_pytorch_code
with st.expander("Compute Metrics", expanded=True):
    metrics = estimate_metrics_from_graph(graph)
    c1, c2, c3 = st.columns(3)
    c1.metric("Relative FLOPs Score", metrics["total_flops_score"])
    params = metrics["total_params_estimate"]
    c2.metric("Est. Parameters", f"~{params/1e6:.1f}M" if params > 1e6 else f"~{params/1e3:.0f}K")
    c3.metric("Depth", metrics["depth"])
    st.dataframe(pd.DataFrame(metrics["breakdown"]))

# F6: Memory Estimation
with st.expander("Estimated Activation Memory per Layer"):
    batch = st.slider("Batch size", 1, 64, 1, key="mem_batch_model")
    spatial = st.slider("Input spatial size", 32, 512, 224, 32, key="mem_spatial_model")
    mem_data = estimate_activation_memory(graph, batch_size=batch, input_spatial=spatial)
    df_mem = pd.DataFrame(mem_data)
    st.metric("Total Est. Activation Memory", f"{df_mem['mem_mb'].sum():.1f} MB")
    st.dataframe(df_mem)

# F5: PyTorch Code Generation
if st.button("Generate PyTorch Code", key="btn_codegen_main"):
    code = get_pytorch_code(model_name, graph)
    st.code(code, language="python")
    st.download_button("Download .py", code, f"{model_name.replace(' ', '_').lower()}.py", "text/plain", key="dl_codegen_main")

# Node explanation panel (main content)
if selected_node:
    st.markdown("---")
    st.markdown(f"## Why This Block Matters: {selected_node.label}")
    
    # Explanation
    st.markdown(explain_node(selected_node))
    
    # Reasoning box
    reasoning = generate_reasoning(selected_node)
    if reasoning:
        st.markdown("### 💡 Design Implications")
        st.markdown(reasoning)

# Architecture explanation
with st.expander("Architecture Overview"):
    st.markdown(explain_graph(graph))

with st.expander("Edge Legend"):
    st.markdown("""
    **Edge Types**
    - **Solid black** → Forward data flow  
    - **Blue dashed** → Skip connection (U-Net)  
    - **Red dashed** → Residual connection (ResNet)
    
    **Node Border Colors (Compute Intensity)**
    - 🔴 **Red (thick)** → High or very high FLOPs
    - 🟡 **Orange** → Medium FLOPs
    - **Default** → Low/unknown compute cost
    """)


# --------------------------------------------------
# Architecture Comparison (Side-by-Side)
# --------------------------------------------------
st.markdown("---")
st.header("Architecture Comparison (Side-by-Side)")

col1, col2 = st.columns(2)

with col1:
    arch_a = st.selectbox(
        "Architecture A",
        ["ResNet-18", "U-Net", "Vision Transformer"],
        key="arch_a"
    )

with col2:
    arch_b = st.selectbox(
        "Architecture B",
        ["ResNet-18", "U-Net", "Vision Transformer"],
        key="arch_b",
        index=1
    )

# Build comparison graphs
if arch_a and arch_b:
    # Build graph A
    if arch_a == "ResNet-18":
        graph_a = build_resnet18_graph()
    elif arch_a == "U-Net":
        graph_a = build_unet_graph()
    elif arch_a == "Vision Transformer":
        graph_a = build_vit_graph()
    else:
        st.stop()
    
    # Build graph B
    if arch_b == "ResNet-18":
        graph_b = build_resnet18_graph()
    elif arch_b == "U-Net":
        graph_b = build_unet_graph()
    elif arch_b == "Vision Transformer":
        graph_b = build_vit_graph()
    else:
        st.stop()
    
    # Educational expander
    with st.expander("ℹ️ How to Read This Comparison"):
        st.markdown("""
        **Understanding the Metrics:**
        
        - **High-FLOPs Operations**: FLOPs (Floating Point Operations) measure computational cost. 
          "High-FLOPs" nodes like convolutions and matrix multiplications are expensive and dominate 
          runtime. Fewer high-FLOPs operations means faster inference.
        
        - **Spatial Preservation**: How well the architecture maintains spatial resolution (width × height). 
          - *High*: Minimal downsampling, good for dense prediction (segmentation, detection)
          - *Medium*: Moderate downsampling, balances efficiency and spatial detail
          - *Low*: Aggressive downsampling, optimized for classification tasks
        
        - **Quadratic Scaling**: Operations that scale as O(n²) with input size. 
          For example, self-attention in Transformers becomes *very expensive* with large images 
          (4× the pixels = 16× the computation). Linear or sub-quadratic scaling is preferred.
        
        - **Rule-Based Analysis**: All comparisons are deterministic and transparent—no LLMs or black boxes. 
          Every conclusion comes from explicit rules applied to semantic parameters (compute intensity, 
          spatial behavior, scaling properties).
        """)
    
    # Compute summaries for both architectures
    compute_summary_a = summarize_compute(graph_a)
    compute_summary_b = summarize_compute(graph_b)
    
    spatial_summary_a = summarize_spatial_behavior(graph_a)
    spatial_summary_b = summarize_spatial_behavior(graph_b)
    
    scaling_summary_a = summarize_scaling_behavior(graph_a)
    scaling_summary_b = summarize_scaling_behavior(graph_b)
    
    # Detect comparison context for visual highlighting
    comparison_mode = True
    
    # Determine compute dominance
    flops_a = compute_summary_a["total_high_flops"]
    flops_b = compute_summary_b["total_high_flops"]
    dominant_compute = 'A' if flops_a > flops_b else ('B' if flops_b > flops_a else None)
    
    # Determine spatial dominance  
    spatial_levels = {"high": 3, "medium": 2, "low": 1}
    spatial_a_val = spatial_levels.get(spatial_summary_a["spatial_preservation"], 2)
    spatial_b_val = spatial_levels.get(spatial_summary_b["spatial_preservation"], 2)
    dominant_spatial = 'A' if spatial_a_val > spatial_b_val else ('B' if spatial_b_val > spatial_a_val else None)
    
    # Determine scaling issues
    scaling_issue = None
    if scaling_summary_a["scaling"] == "poor" and scaling_summary_b["scaling"] != "poor":
        scaling_issue = 'A'
    elif scaling_summary_b["scaling"] == "poor" and scaling_summary_a["scaling"] != "poor":
        scaling_issue = 'B'
    
    # Get bottleneck node IDs
    bottleneck_a = compute_summary_a.get("primary_bottleneck")
    bottleneck_b = compute_summary_b.get("primary_bottleneck")
    
    # Task 4: Comparison Legend
    with st.expander("🎨 Visual Comparison Legend", expanded=True):
        st.markdown("""
        **Understanding the Visual Highlights:**
        
        - 🔥 **COMPUTE BOTTLENECK** — The single most expensive operation in this architecture
        - 🔴 **Thick red borders** — High-FLOPs nodes driving compute differences
        - 🟠 **Orange with ⚠️** — Quadratic scaling bottlenecks (attention blocks)
        - 🔵 **Blue borders** — Skip connections enabling better spatial preservation
        - ⚪ **Greyed/faded nodes** — Shared structure with no significant difference
        
        **How to Read the Comparison:**
        - Only nodes *responsible for the difference* are highlighted
        - Bottleneck badges appear on the primary compute hotspot only
        - When multiple highlights apply, bottleneck > compute > scaling > spatial > ghost
        - All highlights are deterministic and trace back to semantic parameters
        """)
    
    # Visual Graph Comparison
    st.markdown("---")
    st.subheader("📊 Visual Architecture Comparison")
    
    # Build comparison contexts
    ctx_a = {
        'mode': 'compare',
        'current_arch': 'A',
        'dominant_compute': dominant_compute,
        'dominant_spatial': dominant_spatial,
        'scaling_issue': scaling_issue,
        'bottleneck_node_id': bottleneck_a
    }
    
    ctx_b = {
        'mode': 'compare',
        'current_arch': 'B',
        'dominant_compute': dominant_compute,
        'dominant_spatial': dominant_spatial,
        'scaling_issue': scaling_issue,
        'bottleneck_node_id': bottleneck_b
    }
    
    # Render side-by-side graphs with comparison styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {arch_a}")
        dot_a = render_graph_with_comparison(graph_a, ctx_a, expand_blocks=expand_blocks)
        st.graphviz_chart(dot_a, use_container_width=True)
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.download_button(
                label="⬇️ .dot",
                data=str(dot_a),
                file_name=f"{arch_a.replace(' ', '_')}.dot",
                mime="text/plain",
                key="dot_comp_a"
            )
        with col_a2:
            svg_a = dot_a.pipe(format='svg').decode('utf-8')
            st.download_button(
                label="⬇️ SVG",
                data=svg_a,
                file_name=f"{arch_a.replace(' ', '_')}.svg",
                mime="image/svg+xml",
                key="svg_comp_a"
            )

    with col2:
        st.markdown(f"#### {arch_b}")
        dot_b = render_graph_with_comparison(graph_b, ctx_b, expand_blocks=expand_blocks)
        st.graphviz_chart(dot_b, use_container_width=True)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.download_button(
                label="⬇️ .dot",
                data=str(dot_b),
                file_name=f"{arch_b.replace(' ', '_')}.dot",
                mime="text/plain",
                key="dot_comp_b"
            )
        with col_b2:
            svg_b = dot_b.pipe(format='svg').decode('utf-8')
            st.download_button(
                label="⬇️ SVG",
                data=svg_b,
                file_name=f"{arch_b.replace(' ', '_')}.svg",
                mime="image/svg+xml",
                key="svg_comp_b"
            )
    
    # Display side-by-side comparison
    st.markdown("---")
    
    # Section 1: Computational Cost
    st.subheader("1. Computational Cost")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{arch_a}**")
        st.metric("High-FLOPs Nodes", compute_summary_a["total_high_flops"])
        if compute_summary_a["primary_bottleneck"]:
            st.markdown(f"**Primary Bottleneck:** `{compute_summary_a['primary_bottleneck']}`")
        else:
            st.markdown("**Primary Bottleneck:** None")
    
    with col2:
        st.markdown(f"**{arch_b}**")
        st.metric("High-FLOPs Nodes", compute_summary_b["total_high_flops"])
        if compute_summary_b["primary_bottleneck"]:
            st.markdown(f"**Primary Bottleneck:** `{compute_summary_b['primary_bottleneck']}`")
        else:
            st.markdown("**Primary Bottleneck:** None")
    
    # Section 2: Spatial Structure
    st.markdown("---")
    st.subheader("2. Spatial Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{arch_a}**")
        st.metric("Spatial Preservation", spatial_summary_a["spatial_preservation"].upper())
        st.markdown(f"*{spatial_summary_a['reason']}*")
    
    with col2:
        st.markdown(f"**{arch_b}**")
        st.metric("Spatial Preservation", spatial_summary_b["spatial_preservation"].upper())
        st.markdown(f"*{spatial_summary_b['reason']}*")
    
    # Section 3: Scaling Behavior
    st.markdown("---")
    st.subheader("3. Scaling Behavior")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{arch_a}**")
        st.metric("Input Size Scaling", scaling_summary_a["scaling"].upper())
        st.markdown(f"*{scaling_summary_a['reason']}*")
    
    with col2:\
        st.markdown(f"**{arch_b}**")
        st.metric("Input Size Scaling", scaling_summary_b["scaling"].upper())
        st.markdown(f"*{scaling_summary_b['reason']}*")
    
    # Comprehensive comparison explanation
    st.markdown("---")
    st.subheader("Why One Architecture May Be Better Than the Other")
    
    comparison_explanation = explain_architecture_comparison(graph_a, graph_b)
    st.markdown(comparison_explanation)
    
    # Key Takeaways
    st.markdown("---")
    st.subheader("🎯 Key Takeaways")
    
    takeaways = []
    
    # Efficiency based on compute
    flops_a = compute_summary_a["total_high_flops"]
    flops_b = compute_summary_b["total_high_flops"]
    if flops_a > flops_b:
        takeaways.append(f"**{arch_b}** is more computationally efficient (fewer high-cost operations)")
    elif flops_b > flops_a:
        takeaways.append(f"**{arch_a}** is more computationally efficient (fewer high-cost operations)")
    
    # Dense prediction based on spatial preservation
    spatial_levels = {"high": 3, "medium": 2, "low": 1}
    spatial_a_val = spatial_levels.get(spatial_summary_a["spatial_preservation"], 2)
    spatial_b_val = spatial_levels.get(spatial_summary_b["spatial_preservation"], 2)
    if spatial_a_val > spatial_b_val:
        takeaways.append(f"**{arch_a}** is better suited for dense prediction tasks (higher spatial preservation)")
    elif spatial_b_val > spatial_a_val:
        takeaways.append(f"**{arch_b}** is better suited for dense prediction tasks (higher spatial preservation)")
    
    # Scaling warnings
    if scaling_summary_a["scaling"] == "poor":
        takeaways.append(f"⚠️ **{arch_a}** may struggle with large input sizes (poor scaling)")
    if scaling_summary_b["scaling"] == "poor":
        takeaways.append(f"⚠️ **{arch_b}** may struggle with large input sizes (poor scaling)")
    
    # Render takeaways
    if takeaways:
        for takeaway in takeaways:
            st.markdown(f"- {takeaway}")
    else:
        st.markdown("- Both architectures have similar trade-offs")


# --------------------------------------------------
# F8: Multi-Architecture Radar Chart
# --------------------------------------------------
st.markdown("---")
st.header("Multi-Architecture Radar Chart")

from core.radar_chart import compute_radar_scores, build_radar_chart

# Build available architectures list (hardcoded + history)
available_archs = ["ResNet-18", "U-Net", "Vision Transformer"]
if st.session_state.get("history"):
    hist_names = [h["name"] for h in st.session_state.history]
    available_archs = list(dict.fromkeys(available_archs + hist_names))

selected_for_radar = st.multiselect(
    "Select architectures to compare (2–5)",
    options=available_archs,
    default=available_archs[:min(2, len(available_archs))],
    key="radar_select"
)

if len(selected_for_radar) >= 2 and len(selected_for_radar) <= 5:
    radar_graphs = []
    for arch_name in selected_for_radar:
        if arch_name == "ResNet-18":
            radar_graphs.append(build_resnet18_graph())
        elif arch_name == "U-Net":
            radar_graphs.append(build_unet_graph())
        elif arch_name in ("Vision Transformer", "ViT"):
            radar_graphs.append(build_vit_graph())
        else:
            # Try to find in history
            match = next((h["graph"] for h in st.session_state.get("history", []) if h["name"] == arch_name), None)
            if match:
                radar_graphs.append(match)

    if radar_graphs:
        records = [compute_radar_scores(g) for g in radar_graphs]
        st.altair_chart(build_radar_chart(records), use_container_width=True)
elif len(selected_for_radar) > 5:
    st.warning("Select at most 5 architectures for readability")
else:
    st.info("Select at least 2 architectures to display the radar chart")
