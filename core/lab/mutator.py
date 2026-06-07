"""
core/lab/mutator.py — Architecture Mutation Operations (Phase 11)

Provides deterministic graph mutations on top of the existing visualizer graphs.
All mutations produce a NEW ArchitectureGraph (non-destructive, the original is preserved).

Supported mutations:
  - increase_depth(n)      : add N extra residual/conv blocks
  - decrease_depth(n)      : remove last N nodes (non-critical)
  - increase_width(factor) : scale channel sizes by factor
  - decrease_width(factor) : shrink channel sizes by factor
  - add_residual()         : inject skip edges between adjacent block nodes
  - remove_residual()      : remove all skip edges
  - add_attention()        : insert a MultiHeadAttention node after the deepest conv block
  - change_patch_size(ps)  : change patch_size param in patchembedding nodes
  - change_hidden_dim(dim) : change hidden_size/d_model param in transformer nodes
"""

import copy
from typing import Any, Dict, List, Optional
from core.architecture_graph import ArchitectureGraph, GraphNode, GraphEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_graph(graph: ArchitectureGraph) -> ArchitectureGraph:
    """Deep-copy a graph so mutations are non-destructive."""
    return copy.deepcopy(graph)


def _find_nodes_by_types(graph: ArchitectureGraph, types: List[str]) -> List[GraphNode]:
    """Return all nodes whose type (case-insensitive) is in the given list."""
    lower_types = [t.lower() for t in types]
    return [n for n in graph.nodes if (n.type or "").lower() in lower_types]


def _make_node_id(graph: ArchitectureGraph, prefix: str) -> str:
    """Generate a unique node ID."""
    existing = {n.id for n in graph.nodes}
    i = 1
    while f"{prefix}_{i}" in existing:
        i += 1
    return f"{prefix}_{i}"


# ---------------------------------------------------------------------------
# Individual mutation functions
# ---------------------------------------------------------------------------

def increase_depth(graph: ArchitectureGraph, n: int = 1) -> ArchitectureGraph:
    """
    Add N extra convolutional blocks at the end of the graph (before the final
    linear/pooling node if one exists).

    Strategy:
      1. Find the last "heavy" block node (residual, conv, dense, transformer).
      2. Clone it N times and insert after it.
      3. Update edges to wire new nodes in sequence.
    """
    g = _clone_graph(graph)

    heavy_types = [
        "residualblock", "block", "stage", "bottleneckblock",
        "denseblock", "conv2d", "transformerblock", "multiheadattention"
    ]
    heavy = _find_nodes_by_types(g, heavy_types)
    if not heavy:
        # Fallback: append generic conv nodes
        insertion_after_idx = len(g.nodes) - 2 if len(g.nodes) >= 2 else len(g.nodes) - 1
    else:
        last_heavy = heavy[-1]
        insertion_after_idx = next(
            (i for i, nd in enumerate(g.nodes) if nd.id == last_heavy.id), len(g.nodes) - 1
        )

    for i in range(n):
        # Clone the node at insertion_after_idx
        template = g.nodes[insertion_after_idx]
        new_id    = _make_node_id(g, template.type or "block")
        new_label = f"{template.label} ×{i+2}"
        new_node  = GraphNode(
            id=new_id,
            type=template.type,
            label=new_label,
            params=copy.deepcopy(template.params),
            block=template.block,
            description=template.description,
            semantic_params=copy.deepcopy(template.semantic_params),
        )
        # Insert right after insertion_after_idx
        g.nodes.insert(insertion_after_idx + 1, new_node)

        # Wire: prev_node → new_node → next_node
        prev_node = g.nodes[insertion_after_idx]
        if insertion_after_idx + 2 < len(g.nodes):
            next_node = g.nodes[insertion_after_idx + 2]
            # Remove direct edge prev → next if it exists
            g.edges = [
                e for e in g.edges
                if not (e.source == prev_node.id and e.target == next_node.id)
            ]
            g.edges.append(GraphEdge(source=prev_node.id, target=new_node.id, edge_type="flow"))
            g.edges.append(GraphEdge(source=new_node.id, target=next_node.id, edge_type="flow"))
        else:
            g.edges.append(GraphEdge(source=prev_node.id, target=new_node.id, edge_type="flow"))

        insertion_after_idx += 1

    return g


def decrease_depth(graph: ArchitectureGraph, n: int = 1) -> ArchitectureGraph:
    """
    Remove N non-critical nodes from the tail of the graph.

    Critical nodes that are never removed:
      - input, embedding, patchembedding, linear (last), output
    """
    g = _clone_graph(graph)
    protected_types = {"input", "embedding", "patchembedding", "output", "linear", "adaptiveavgpool2d"}

    removed = 0
    # Walk from second-to-last backwards
    for i in range(len(g.nodes) - 2, 0, -1):
        if removed >= n:
            break
        node = g.nodes[i]
        if (node.type or "").lower() in protected_types:
            continue
        # Remove node and its edges
        removed_id = node.id

        # Reconnect: find what was before and after this node
        sources = [e.source for e in g.edges if e.target == removed_id]
        targets = [e.target for e in g.edges if e.source == removed_id]
        g.edges = [e for e in g.edges if e.source != removed_id and e.target != removed_id]
        # Reconnect one-to-one if possible
        for src in sources:
            for tgt in targets:
                g.edges.append(GraphEdge(source=src, target=tgt, edge_type="flow"))

        g.nodes.pop(i)
        removed += 1

    return g


def increase_width(graph: ArchitectureGraph, factor: float = 1.5) -> ArchitectureGraph:
    """
    Scale all channel/hidden_size parameters by the given factor.
    Rounds to nearest even integer.
    """
    g = _clone_graph(graph)
    channel_keys = ["channels", "out_channels", "filters", "hidden_size", "d_model", "embed_dim", "embedding_dim"]

    for node in g.nodes:
        p = node.params or {}
        changed = False
        for key in channel_keys:
            if key in p and isinstance(p[key], (int, float)):
                new_val = max(1, round(p[key] * factor / 2) * 2)  # round to even
                p[key] = new_val
                changed = True
        if changed:
            node.params = p

    return g


def decrease_width(graph: ArchitectureGraph, factor: float = 0.5) -> ArchitectureGraph:
    """
    Shrink all channel/hidden_size parameters by the given factor (minimum 4).
    """
    g = _clone_graph(graph)
    channel_keys = ["channels", "out_channels", "filters", "hidden_size", "d_model", "embed_dim", "embedding_dim"]

    for node in g.nodes:
        p = node.params or {}
        changed = False
        for key in channel_keys:
            if key in p and isinstance(p[key], (int, float)):
                new_val = max(4, round(p[key] * factor / 2) * 2)
                p[key] = new_val
                changed = True
        if changed:
            node.params = p

    return g


def add_residual(graph: ArchitectureGraph) -> ArchitectureGraph:
    """
    Add skip (residual) connections between every other pair of heavy block nodes.
    Skip edges jump over one node: node[i] → node[i+2] with edge_type='skip'.
    Only adds a skip if neither endpoint is a pooling/norm/embedding node.
    """
    g = _clone_graph(graph)
    skip_types = {"residualblock", "block", "stage", "conv2d", "denseblock", "transformerblock"}
    light_types = {"maxpool2d", "avgpool2d", "adaptiveavgpool2d", "batchnorm2d", "layernorm",
                   "input", "output", "patchembedding", "embedding", "dropout"}

    # Collect indices of "heavy" nodes
    heavy_indices = [
        i for i, n in enumerate(g.nodes)
        if (n.type or "").lower() in skip_types
    ]

    # Existing skip edges (avoid duplicates)
    existing_skip = {(e.source, e.target) for e in g.edges if e.edge_type in ("skip", "residual")}

    added = 0
    for i in range(0, len(heavy_indices) - 1, 2):
        src_node = g.nodes[heavy_indices[i]]
        # Target is either the next heavy node or 2 positions ahead
        target_idx_candidate = heavy_indices[i] + 2
        if target_idx_candidate >= len(g.nodes):
            continue
        tgt_node = g.nodes[target_idx_candidate]
        if (tgt_node.type or "").lower() in light_types:
            continue
        key = (src_node.id, tgt_node.id)
        if key not in existing_skip:
            g.edges.append(GraphEdge(source=src_node.id, target=tgt_node.id, edge_type="skip"))
            existing_skip.add(key)
            added += 1

    return g


def remove_residual(graph: ArchitectureGraph) -> ArchitectureGraph:
    """Remove all skip/residual edges from the graph."""
    g = _clone_graph(graph)
    g.edges = [e for e in g.edges if e.edge_type not in ("skip", "residual")]
    return g


def add_attention(graph: ArchitectureGraph) -> ArchitectureGraph:
    """
    Insert a MultiHeadAttention node immediately after the deepest conv/residual block.
    This simulates hybrid CNN+Attention architectures (e.g., ViT late fusion).
    """
    g = _clone_graph(graph)
    heavy_types = ["residualblock", "block", "stage", "conv2d", "bottleneckblock"]
    heavy_nodes = _find_nodes_by_types(g, heavy_types)

    if not heavy_nodes:
        # No heavy nodes to insert after — append after second-to-last node
        insert_after_idx = max(0, len(g.nodes) - 2)
    else:
        last_heavy = heavy_nodes[-1]
        insert_after_idx = next(
            (i for i, n in enumerate(g.nodes) if n.id == last_heavy.id), len(g.nodes) - 2
        )

    # Determine a reasonable embed_dim from adjacent nodes
    adj_node = g.nodes[insert_after_idx]
    adj_ch = (adj_node.params or {}).get("channels", (adj_node.params or {}).get("filters", 64))

    new_id = _make_node_id(g, "multiheadattention")
    attn_node = GraphNode(
        id=new_id,
        type="multiheadattention",
        label="Multi-Head Attention",
        params={"heads": 8, "d_model": adj_ch, "channels": adj_ch},
        block="attention",
        description="Injected self-attention block for hybrid CNN+Attention architecture",
        semantic_params={"compute_role": "attention", "flops": "very high"},
    )

    g.nodes.insert(insert_after_idx + 1, attn_node)

    # Wire edges
    prev_node = g.nodes[insert_after_idx]
    if insert_after_idx + 2 < len(g.nodes):
        next_node = g.nodes[insert_after_idx + 2]
        g.edges = [
            e for e in g.edges
            if not (e.source == prev_node.id and e.target == next_node.id)
        ]
        g.edges.append(GraphEdge(source=prev_node.id, target=new_id, edge_type="flow"))
        g.edges.append(GraphEdge(source=new_id, target=next_node.id, edge_type="flow"))
    else:
        g.edges.append(GraphEdge(source=prev_node.id, target=new_id, edge_type="flow"))

    return g


def change_patch_size(graph: ArchitectureGraph, patch_size: int = 8) -> ArchitectureGraph:
    """
    Change the patch_size parameter in all patchembedding / patch_embedding nodes.
    Smaller patches → more tokens → higher compute but finer granularity.
    """
    g = _clone_graph(graph)
    patch_types = {"patchembedding", "patch_embedding", "embedding"}

    for node in g.nodes:
        if (node.type or "").lower() in patch_types:
            p = node.params or {}
            p["patch_size"] = patch_size
            # Recompute num_patches approximation (assumes 224×224 input)
            p["num_patches"] = (224 // patch_size) ** 2
            node.params = p
            sp = node.semantic_params or {}
            sp["patch_size"] = patch_size
            node.semantic_params = sp

    return g


def change_hidden_dim(graph: ArchitectureGraph, dim: int = 1024) -> ArchitectureGraph:
    """
    Change the hidden dimension (d_model / hidden_size / embed_dim) of all
    transformer / attention nodes.
    """
    g = _clone_graph(graph)
    transformer_types = {
        "transformerblock", "multiheadattention", "encoder_block",
        "decoder_block", "feedforward", "linear"
    }

    for node in g.nodes:
        if (node.type or "").lower() in transformer_types:
            p = node.params or {}
            for key in ("d_model", "hidden_size", "embed_dim", "embedding_dim"):
                if key in p:
                    p[key] = dim
            node.params = p

    return g


# ---------------------------------------------------------------------------
# Main mutation dispatch
# ---------------------------------------------------------------------------

MUTATION_REGISTRY = {
    "increase_depth":   increase_depth,
    "decrease_depth":   decrease_depth,
    "increase_width":   increase_width,
    "decrease_width":   decrease_width,
    "add_residual":     add_residual,
    "remove_residual":  remove_residual,
    "add_attention":    add_attention,
    "change_patch_size": change_patch_size,
    "change_hidden_dim": change_hidden_dim,
}


def apply_mutations(
    graph: ArchitectureGraph,
    mutations: List[Dict[str, Any]],
) -> ArchitectureGraph:
    """
    Apply a list of mutations sequentially to a graph.

    Each mutation dict has:
        { "type": "<mutation_name>", "params": { ... } }

    Returns the final mutated graph (original is not modified).
    """
    g = _clone_graph(graph)
    for m in mutations:
        mut_type = m.get("type", "")
        mut_params = m.get("params", {})
        fn = MUTATION_REGISTRY.get(mut_type)
        if fn is None:
            raise ValueError(f"Unknown mutation type: '{mut_type}'. Valid types: {list(MUTATION_REGISTRY.keys())}")
        g = fn(g, **mut_params)
    return g
