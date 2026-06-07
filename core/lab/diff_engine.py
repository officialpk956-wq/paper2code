"""
core/lab/diff_engine.py — Architecture Diff Engine (Phase 11)

Given two ArchitectureGraph objects (before and after a mutation), computes:
  - nodes_added     : list of node labels added
  - nodes_removed   : list of node labels removed
  - nodes_changed   : list of nodes whose params changed
  - edges_added     : list of (src, tgt) tuples added
  - edges_removed   : list of (src, tgt) tuples removed
  - param_delta     : absolute and % change in parameter count
  - flops_delta     : absolute and % change in FLOPs score
  - depth_delta     : change in graph depth (node count)
  - memory_delta_mb : approximate change in activation memory (MB)
  - summary_text    : human-readable summary of changes
"""

from typing import Any, Dict, List, Tuple
from core.architecture_graph import ArchitectureGraph
from core.metrics_estimator import (
    estimate_metrics_from_graph,
    estimate_activation_memory,
)


def compute_diff(
    before: ArchitectureGraph,
    after: ArchitectureGraph,
    batch_size: int = 1,
    input_spatial: int = 224,
) -> Dict[str, Any]:
    """
    Compute the structural and metric diff between two ArchitectureGraph objects.

    Args:
        before: The original ArchitectureGraph
        after: The mutated ArchitectureGraph
        batch_size: Batch size for activation memory estimates
        input_spatial: Input spatial resolution (H = W, square assumed)

    Returns:
        Dict with full diff breakdown.
    """
    # -----------------------------------------------------------------
    # 1. Node-level structural diff
    # -----------------------------------------------------------------
    before_nodes = {n.id: n for n in before.nodes}
    after_nodes  = {n.id: n for n in after.nodes}

    added_ids   = [nid for nid in after_nodes  if nid not in before_nodes]
    removed_ids = [nid for nid in before_nodes if nid not in after_nodes]

    # Changed = same id, different params
    changed_nodes = []
    for nid in before_nodes:
        if nid in after_nodes:
            b_params = before_nodes[nid].params or {}
            a_params = after_nodes[nid].params or {}
            if b_params != a_params:
                changed_nodes.append({
                    "id": nid,
                    "label": after_nodes[nid].label,
                    "before": b_params,
                    "after": a_params,
                })

    nodes_added   = [after_nodes[nid].label  for nid in added_ids]
    nodes_removed = [before_nodes[nid].label for nid in removed_ids]

    # -----------------------------------------------------------------
    # 2. Edge-level structural diff
    # -----------------------------------------------------------------
    before_edges = {(e.source, e.target) for e in before.edges}
    after_edges  = {(e.source, e.target) for e in after.edges}

    edges_added   = list(after_edges - before_edges)
    edges_removed = list(before_edges - after_edges)

    # Count skip/residual edges
    before_skip = sum(1 for e in before.edges if e.edge_type in ("skip", "residual"))
    after_skip  = sum(1 for e in after.edges  if e.edge_type in ("skip", "residual"))

    # -----------------------------------------------------------------
    # 3. Metrics diff
    # -----------------------------------------------------------------
    b_metrics = estimate_metrics_from_graph(before)
    a_metrics = estimate_metrics_from_graph(after)

    b_params = b_metrics["total_params_estimate"]
    a_params = a_metrics["total_params_estimate"]
    b_flops  = b_metrics["total_flops_score"]
    a_flops  = a_metrics["total_flops_score"]
    b_depth  = b_metrics["depth"]
    a_depth  = a_metrics["depth"]

    def _pct(new, old):
        if old == 0:
            return 0.0
        return round((new - old) / old * 100, 1)

    param_delta = {
        "before": b_params,
        "after": a_params,
        "absolute": a_params - b_params,
        "pct": _pct(a_params, b_params),
    }
    flops_delta = {
        "before": b_flops,
        "after": a_flops,
        "absolute": a_flops - b_flops,
        "pct": _pct(a_flops, b_flops),
    }
    depth_delta = {
        "before": b_depth,
        "after": a_depth,
        "absolute": a_depth - b_depth,
    }

    # -----------------------------------------------------------------
    # 4. Memory diff
    # -----------------------------------------------------------------
    b_mem_rows = estimate_activation_memory(before, batch_size, input_spatial)
    a_mem_rows = estimate_activation_memory(after,  batch_size, input_spatial)
    b_mem_mb   = round(sum(r["mem_mb"] for r in b_mem_rows), 2)
    a_mem_mb   = round(sum(r["mem_mb"] for r in a_mem_rows), 2)
    memory_delta = {
        "before_mb": b_mem_mb,
        "after_mb":  a_mem_mb,
        "absolute_mb": round(a_mem_mb - b_mem_mb, 2),
        "pct": _pct(a_mem_mb, b_mem_mb),
    }

    # -----------------------------------------------------------------
    # 5. Skip connection diff
    # -----------------------------------------------------------------
    skip_delta = {
        "before": before_skip,
        "after": after_skip,
        "absolute": after_skip - before_skip,
    }

    # -----------------------------------------------------------------
    # 6. Human-readable summary
    # -----------------------------------------------------------------
    summary_parts = []
    if nodes_added:
        summary_parts.append(f"+{len(nodes_added)} node(s) added ({', '.join(nodes_added[:3])}{'...' if len(nodes_added) > 3 else ''})")
    if nodes_removed:
        summary_parts.append(f"-{len(nodes_removed)} node(s) removed ({', '.join(nodes_removed[:3])}{'...' if len(nodes_removed) > 3 else ''})")
    if changed_nodes:
        summary_parts.append(f"{len(changed_nodes)} node(s) modified")
    if skip_delta["absolute"] > 0:
        summary_parts.append(f"+{skip_delta['absolute']} skip connection(s)")
    elif skip_delta["absolute"] < 0:
        summary_parts.append(f"{skip_delta['absolute']} skip connection(s) removed")
    if param_delta["pct"] != 0:
        direction = "+" if param_delta["pct"] > 0 else ""
        summary_parts.append(f"Parameters {direction}{param_delta['pct']}%")
    if flops_delta["absolute"] != 0:
        direction = "+" if flops_delta["absolute"] > 0 else ""
        summary_parts.append(f"FLOPs score {direction}{flops_delta['absolute']}")

    summary_text = " | ".join(summary_parts) if summary_parts else "No structural changes detected."

    return {
        "nodes_added":    nodes_added,
        "nodes_removed":  nodes_removed,
        "nodes_changed":  changed_nodes,
        "edges_added":    [{"source": s, "target": t} for s, t in edges_added],
        "edges_removed":  [{"source": s, "target": t} for s, t in edges_removed],
        "param_delta":    param_delta,
        "flops_delta":    flops_delta,
        "depth_delta":    depth_delta,
        "memory_delta":   memory_delta,
        "skip_delta":     skip_delta,
        "summary_text":   summary_text,
    }
