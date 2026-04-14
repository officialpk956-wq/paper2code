import streamlit as st
import graphviz

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_unet import build_unet_graph
from src.visualizer_vit import build_vit_graph
from src.explainers import explain_node, explain_graph
from src.comparators import (
    summarize_compute,
    summarize_spatial_behavior,
    summarize_scaling_behavior,
    explain_architecture_comparison,
)
from src.orchestrator.pipeline import Paper2CodePipeline


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


def render_graph_with_comparison(graph, comparison_ctx=None):
    """
    Render a graph with optional comparison styling.
    
    Args:
        graph: ArchitectureGraph to render
        comparison_ctx: Optional comparison context dict
    
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
# Sidebar
# --------------------------------------------------
st.sidebar.title("Architecture Input")

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

        # Show truncation warning if applicable
        if result.get("metadata", {}).get("truncated"):
            st.sidebar.warning(
                f"⚠️ Input too large — truncated from "
                f"{result['metadata']['original_layer_count']} to "
                f"{result['metadata']['layer_count']} layers for performance"
            )

        # Render graph and explanation
        st.subheader(f"Architecture: {graph.name}")
        st.graphviz_chart(render_graph_with_comparison(graph), use_container_width=True)

        st.markdown("---")
        st.subheader("Explanation")
        st.markdown(explanation)

        st.stop()  # Stop here if using text input
    else:
        st.sidebar.info("Enter an architecture description above")
        st.stop()

# Standard Model Selection Mode
st.sidebar.title("Model Selector")

model_name = st.sidebar.selectbox(
    "Choose architecture",
    ["ResNet-18", "U-Net", "ViT"]
)

expand_blocks = st.sidebar.checkbox("Expand composite blocks", value=False)

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
dot = render_graph_with_comparison(graph)

st.subheader(graph.name)
st.graphviz_chart(dot, use_container_width=True)

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
        dot_a = render_graph_with_comparison(graph_a, ctx_a)
        st.graphviz_chart(dot_a, use_container_width=True)
    
    with col2:
        st.markdown(f"#### {arch_b}")
        dot_b = render_graph_with_comparison(graph_b, ctx_b)
        st.graphviz_chart(dot_b, use_container_width=True)
    
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
    
    with col2:
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
