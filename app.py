import streamlit as st
import graphviz

from src.visualizer_resnet import build_resnet18_graph
from src.visualizer_unet import build_unet_graph
from src.visualizer_vit import build_vit_graph
from src.explainers import explain_node, explain_graph


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
dot = graphviz.Digraph(
    comment=graph.name,
    graph_attr={"rankdir": "TB"}
)

visible = set()

# Base nodes
for node in graph.nodes:
    visible.add(node.id)

    label = f"{node.label}\n{node.type}"
    for k, v in node.params.items():
        label += f"\n{k}: {v}"
    
    # Append semantic parameters if present
    if node.semantic_params:
        for k, v in node.semantic_params.items():
            label += f"\n{k}: {v}"

    # Determine node styling based on compute intensity (flops)
    node_attrs = {
        "label": label,
        "shape": "box",
        "style": "rounded",
        "tooltip": node.description or node.label
    }
    
    # Color code by FLOPs if present
    if "flops" in node.semantic_params:
        flops_level = node.semantic_params["flops"]
        if flops_level in FLOPS_COLORS:
            node_attrs["color"] = FLOPS_COLORS[flops_level]["color"]
            node_attrs["penwidth"] = FLOPS_COLORS[flops_level]["penwidth"]
    
    dot.node(node.id, **node_attrs)

    # 🔥 Expand composite blocks
    if expand_blocks and node.is_composite():
        sub = node.internal_graph

        for subnode in sub.nodes:
            dot.node(
                subnode.id,
                f"{subnode.label}\n{subnode.type}",
                shape="box",
                style="rounded,filled",
                fillcolor="#f0f0f0"
            )

        for e in sub.edges:
            style = EDGE_STYLES.get(e.edge_type, EDGE_STYLES["flow"])
            dot.edge(e.source, e.target, **style)


# Base edges
for edge in graph.edges:
    style = EDGE_STYLES.get(edge.edge_type, EDGE_STYLES["flow"])
    dot.edge(edge.source, edge.target, **style)


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
