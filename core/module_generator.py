"""
core/module_generator.py

Module Generation Engine — Transforms a Paper2Code architecture pipeline result
into an ordered sequence of structured Learning Modules.

Input:
    - paper_name (str)
    - schema (dict)              ← code_ready JSON spec
    - pipeline_result (dict)     ← output of Paper2CodePipeline.run_single()

Output:
    List[LearningModule]

Design notes (matching module_generation_design.md):
    Step 1 — Extract macro-structures from the graph nodes + schema spec.
    Step 2 — Collapse trivial nodes (activations, norms, dropout) into parents.
    Step 3 — Detect repeated blocks and stage groupings.
    Step 4 — Topologically order modules (input → core → output).
    Step 5 — Assemble each module with all required context fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.architecture_graph import ArchitectureGraph, GraphNode
from core.explainers.graph_explainer import explain_node
from core.metrics_estimator import (
    FLOPS_SCORES,
    estimate_activation_memory,
    estimate_metrics_from_graph,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LearningModule:
    """
    A single step in the learning journey for a research paper.

    All fields map directly to the paper_modules DB table columns.
    """

    order_index: int
    layer_name: str  # Human-readable module title
    module_type: str  # stem | block | stage | head | encoder | bottleneck | decoder
    explanation: str  # Educational explanation (from graph_explainer)
    graph_nodes: list[dict[str, Any]]  # Source graph node ids + labels
    tensor_flow: list[dict[str, Any]]  # Per-node tensor flow / memory data
    flops_context: dict[str, Any]  # FLOPs scores + param estimates for nodes in this module
    confidence: float  # 0.0–1.0 — how well the module was auto-extracted


# ---------------------------------------------------------------------------
# Trivial node types — collapsed into parent modules, never standalone
# ---------------------------------------------------------------------------

_TRIVIAL_TYPES = frozenset(
    {
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "batchnorm2d",
        "batchnorm1d",
        "layernorm",
        "groupnorm",
        "instancenorm",
        "dropout",
        "identity",
        "flatten",
        "permute",
        "reshape",
        "softmax",
    }
)

# Activation types only — for special collapse check
_ACTIVATION_TYPES = frozenset({"relu", "gelu", "silu", "sigmoid", "tanh"})

# Types that mark spatial downsampling (and begin a new stage)
_DOWNSAMPLE_TYPES = frozenset({"maxpool2d", "avgpool2d", "adaptiveavgpool2d"})

# Types that mark spatial upsampling (decoder path)
_UPSAMPLE_TYPES = frozenset({"upsample", "convtranspose2d", "upconvolution"})


# ---------------------------------------------------------------------------
# Module type classifiers
# ---------------------------------------------------------------------------


def _classify_module_type(
    nodes: list[GraphNode], schema: dict[str, Any], index: int, total: int
) -> str:
    """Determine the module_type label for a group of nodes."""
    types = {n.type.lower() for n in nodes}
    labels = " ".join(n.label.lower() for n in nodes)

    # Schema-driven first
    family = schema.get("model_family", "").lower()

    if index == 0:
        return "stem"
    if index == total - 1:
        return "head"

    # UNet-specific phases
    if family == "unet":
        if any(t in _UPSAMPLE_TYPES for t in types) or "upsamp" in labels or "up" in labels:
            return "decoder"
        if "bottleneck" in labels or "bridge" in labels:
            return "bottleneck"
        if any(t in _DOWNSAMPLE_TYPES for t in types) or "down" in labels or "encoder" in labels:
            return "encoder"

    # Transformer-specific
    if any(t in ("multiheadattention", "transformerblock", "selfatt") for t in types):
        return "stage"

    # Residual / Conv stages
    if any("residual" in n.label.lower() or "resblock" in n.label.lower() for n in nodes):
        return "block"

    # Downsampling = stage boundary
    if any(t in _DOWNSAMPLE_TYPES for t in types):
        return "stage"

    return "block"


# ---------------------------------------------------------------------------
# Confidence scorer
# ---------------------------------------------------------------------------


def _score_confidence(module: dict) -> float:
    """
    Score how confidently we extracted a module.

    Rules:
    +0.30  has at least one graph node
    +0.20  has non-trivial explanation (> 50 chars)
    +0.20  has tensor_flow data
    +0.20  has flops_context with data
    +0.10  layer_name is not generic ("Layer N")
    """
    score = 0.0
    if module["graph_nodes"]:
        score += 0.30
    if len(module.get("explanation", "")) > 50:
        score += 0.20
    if module.get("tensor_flow"):
        score += 0.20
    if module.get("flops_context", {}).get("breakdown"):
        score += 0.20
    if not re.match(r"^Layer \d+$", module.get("layer_name", "")):
        score += 0.10
    # Guard: NaN or out-of-range values are clamped to [0.0, 1.0]
    if score != score:  # NaN check
        score = 0.0
    return round(min(max(score, 0.0), 1.0), 2)


# ---------------------------------------------------------------------------
# Step 1 + 2: Extract macro-structures + collapse trivial nodes
# ---------------------------------------------------------------------------


def _group_nodes_into_macro_structures(
    graph: ArchitectureGraph,
    schema: dict[str, Any],
) -> list[list[GraphNode]]:
    """
    Walk the graph nodes in order. Collapse trivial nodes (norms, activations,
    dropout) into the immediately preceding non-trivial node's group.
    Also groups upsampling layers with subsequent convolutions in decoder blocks.
    """
    groups: list[list[GraphNode]] = []
    current_group: list[GraphNode] = []
    nodes = list(graph.nodes)
    i = 0

    while i < len(nodes):
        node = nodes[i]
        node_type = (node.type or "").lower()

        # If it's an upsampling node, check if the next node is a Conv2d in the decoder path
        if node_type in _UPSAMPLE_TYPES and i + 1 < len(nodes):
            next_node = nodes[i + 1]
            if next_node.type.lower() == "conv2d":
                if current_group:
                    groups.append(current_group)
                # Group upsample + conv2d together
                current_group = [node, next_node]
                i += 2
                continue

        if node_type in _TRIVIAL_TYPES:
            # Collapse: attach to current group if one exists
            if current_group:
                current_group.append(node)
            # If no current group, start one (edge case: graph begins with norm)
            else:
                current_group.append(node)
        else:
            # Non-trivial node — flush current group, start a new one
            if current_group:
                groups.append(current_group)
            current_group = [node]
        i += 1

    # Flush final group
    if current_group:
        groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Step 3: Detect repeated blocks and stages to create stage-level groups
# ---------------------------------------------------------------------------


def _detect_and_merge_stages(
    groups: list[list[GraphNode]],
    schema: dict[str, Any],
) -> list[list[GraphNode]]:
    """
    Detect repeated block groups (e.g. 6 transformer layers, 3 ResNet blocks)
    and merge them into a single stage-level LearningModule.

    Strategy:
    - Groups with identical node-type fingerprints are "repeated blocks".
    - If a sequence of 2+ consecutive groups has the same fingerprint, they
      are merged into one representative group.
    - The merged group keeps all representative nodes plus a synthetic
      summary note in the first node's description.
    """
    if not groups:
        return groups

    def fingerprint(group: list[GraphNode]) -> str:
        # Include node type and key stage parameters (channels/hidden sizes)
        # to ensure stages of different sizes/capacities are not merged incorrectly.
        parts = []
        for n in group:
            t = n.type.lower()
            ch = (
                n.params.get("channels")
                or n.params.get("hidden_size")
                or n.params.get("embed_dim")
                or ""
            )
            parts.append(f"{t}:{ch}")
        return ",".join(sorted(parts))

    merged: list[list[GraphNode]] = []
    i = 0

    while i < len(groups):
        fp = fingerprint(groups[i])

        # Look ahead for repeats
        j = i + 1
        while j < len(groups) and fingerprint(groups[j]) == fp:
            j += 1

        repeat_count = j - i
        if repeat_count >= 2:
            # Merge all repeated groups into one representative group
            representative = list(groups[i])  # first block as representative
            # Annotate first node to convey repetition
            if representative:
                n = representative[0]
                n.description = ((n.description or "") + f" [Repeated ×{repeat_count}]").strip()
            merged.append(representative)
            i = j
        else:
            merged.append(groups[i])
            i += 1

    return merged


# ---------------------------------------------------------------------------
# Step 4 + 5: Build LearningModule objects
# ---------------------------------------------------------------------------


def _build_module(
    index: int,
    total: int,
    nodes: list[GraphNode],
    schema: dict[str, Any],
    flops_breakdown: list[dict[str, Any]],
    tensor_memory: list[dict[str, Any]],
    graph: ArchitectureGraph,
) -> LearningModule:
    """
    Assemble one LearningModule from a group of nodes and pre-computed metrics.
    """
    # Build node id → metric index (by label match)
    flops_by_label = {row["node"]: row for row in flops_breakdown}
    real_flops_events = {
        row.get("node_id", row.get("node_label")): row
        for row in graph.metadata.get("flops_events", [])
    }
    tensor_by_label = {row["node"]: row for row in tensor_memory}

    # Graph node context
    graph_node_refs = []
    module_flops_breakdown = []
    module_tensor_flow = []
    explanation_parts = []
    primary_label = None

    total_flops_score = 0
    total_params = 0
    total_mflops = 0.0

    for node in nodes:
        # FLOPs
        if node.label in flops_by_label:
            row = flops_by_label[node.label]
            row["score"] = FLOPS_SCORES.get(row.get("flops_level", "low"), 0)
            module_flops_breakdown.append(row)
            total_flops_score += row["score"]
            total_params += row.get("param_estimate", 0)

        if node.id in real_flops_events:
            total_mflops += real_flops_events[node.id].get("flops_mflops", 0.0)
        elif node.label in real_flops_events:
            total_mflops += real_flops_events[node.label].get("flops_mflops", 0.0)

    for node in nodes:
        # Only include non-trivial nodes in the node ref list for clarity
        if node.type.lower() not in _TRIVIAL_TYPES:
            graph_node_refs.append(
                {
                    "node_id": node.id,
                    "label": node.label,
                    "type": node.type,
                    "params": node.params,
                    "semantic_params": node.semantic_params,
                    "description": node.description,
                }
            )
            if primary_label is None:
                primary_label = node.label

        # Tensor flow
        if node.label in tensor_by_label:
            row = dict(tensor_by_label[node.label])
            if hasattr(node, "input_shape"):
                row["input_shape"] = node.input_shape
            if hasattr(node, "output_shape"):
                row["output_shape"] = node.output_shape
            module_tensor_flow.append(row)

        # Explanation — only explain non-trivial nodes
        if node.type.lower() not in _ACTIVATION_TYPES:
            try:
                # A group is "repeated" if any of its nodes carries the [Repeated] annotation
                is_repeated = any(
                    bool(re.search(r"\[Repeated", n.description or "")) for n in nodes
                )
                exp = explain_node(
                    node, context={"index": index, "total": total, "is_repeated": is_repeated}
                )
                if exp and len(exp) > 10:
                    explanation_parts.append(exp)
            except Exception:
                pass

    # Compose module name
    layer_name = _compose_module_name(index, nodes, schema, total)

    # FLOPs summary for this module
    total_flops_score = sum(
        r.get("flops_level", 0)
        if isinstance(r.get("flops_level"), int)
        else {"very high": 4, "high": 3, "medium": 2, "low": 1}.get(r.get("flops_level", "low"), 1)
        for r in module_flops_breakdown
    )
    total_params = sum(r.get("param_estimate", 0) for r in module_flops_breakdown)

    flops_context = {
        "total_flops_score": total_flops_score,
        "real_flops_mflops": total_mflops,
        "total_params_estimate": total_params,
        "breakdown": module_flops_breakdown,
    }

    # Module type
    module_type = _classify_module_type(nodes, schema, index, total)

    # Check for skip connection edges (e.g., U-Net skip connections)
    has_skip = False
    skip_sources = []
    for node in nodes:
        for edge in graph.edges:
            if edge.target == node.id and edge.edge_type == "skip":
                has_skip = True
                src_node = next((n for n in graph.nodes if n.id == edge.source), None)
                if src_node:
                    skip_sources.append(src_node.label)

    # Final explanation — join non-empty parts
    explanation = (
        "\n\n".join(explanation_parts) if explanation_parts else f"Structural module: {layer_name}"
    )

    # Append architecture context to ensure global uniqueness for fallback explanations
    explanation += f"\n\n*(Context: {graph.name} Architecture)*"

    if has_skip and skip_sources:
        layer_name += " (with Skip Connection)"
        skip_desc = (
            f"\n\n**Skip Connection / Feature Concatenation:**\n"
            f"This module receives high-resolution feature maps directly from **{', '.join(skip_sources)}** "
            f"in the contracting path. This skip connection preserves fine-grained spatial details, "
            f"enabling precise pixel-level localization in the output segmentation map."
        )
        explanation += skip_desc

    raw = {
        "order_index": index,
        "layer_name": layer_name,
        "module_type": module_type,
        "explanation": explanation,
        "graph_nodes": graph_node_refs,
        "tensor_flow": module_tensor_flow,
        "flops_context": flops_context,
        "confidence": 0.0,
    }

    raw["confidence"] = _score_confidence(raw)

    return LearningModule(**raw)


def _compose_module_name(
    index: int, nodes: list[GraphNode], schema: dict[str, Any], total: int = 1
) -> str:
    """Compose a human-readable module name from node group + schema context."""
    # Primary (non-trivial) node
    primary = next((n for n in nodes if n.type.lower() not in _TRIVIAL_TYPES), None)
    if primary is None:
        return f"Module {index + 1}"

    label = primary.label
    desc = primary.description or ""

    # Detect repetition annotation
    repeat_match = re.search(r"\[Repeated ×(\d+)\]", desc)
    repeat_suffix = f" (×{repeat_match.group(1)} Blocks)" if repeat_match else ""

    # Schema-driven naming for known positions
    if index == 0:
        return f"Stem & Input Processing{repeat_suffix}"

    # If it's transformer family, prioritize the node's descriptive label (e.g. MHSA, FFN)
    if schema.get("model_family", "").lower() == "transformer":
        return f"{label}{repeat_suffix}"

    # Detect common block types
    label_lower = label.lower()
    if "attention" in label_lower or "mha" in label_lower:
        return f"Multi-Head Self-Attention{repeat_suffix}"
    if "transformer" in label_lower:
        return f"Transformer Encoder Layer{repeat_suffix}"
    if "residual" in label_lower or "resblock" in label_lower:
        # Use more descriptive name if it has stage name
        if "conv" in label_lower and "_" in label_lower:
            return f"{label}{repeat_suffix}"
        return f"Residual Block{repeat_suffix}"
    if "bottleneck" in label_lower or "bridge" in label_lower:
        return f"Bottleneck / Bridge{repeat_suffix}"
    if "decoder" in label_lower or "expanding" in label_lower or "upsamp" in label_lower:
        return f"Decoder (Expanding Path){repeat_suffix}"
    if "encoder" in label_lower or "contracting" in label_lower:
        return f"Encoder (Contracting Path){repeat_suffix}"
    if "patch" in label_lower or "embed" in label_lower:
        return f"Patch Embedding & Positional Encoding{repeat_suffix}"
    if "pool" in label_lower:
        if index < total / 3:
            return f"Initial Spatial Downsampling{repeat_suffix}"
        return f"Global Pooling & Classification Head{repeat_suffix}"
    if "linear" in label_lower or "fc" in label_lower or "dense" in label_lower:
        return f"Classification Head{repeat_suffix}"
    if "conv" in label_lower and index > 0:
        return f"{label}{repeat_suffix}"

    return f"{label}{repeat_suffix}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_modules(
    paper_name: str,
    schema: dict[str, Any],
    pipeline_result: dict[str, Any],
) -> tuple[dict[str, Any], list[LearningModule]]:
    """
    Transform a Paper2Code pipeline result into a list of LearningModules.

    Args:
        paper_name:       Human-readable paper name.
        schema:           Code-ready JSON schema dict.
        pipeline_result:  Output of Paper2CodePipeline.run_single().

    Returns:
        (paper_meta, modules)
        paper_meta:  dict with title, authors, abstract, architecture_graph, flops_analysis
        modules:     ordered List[LearningModule]
    """
    graph: ArchitectureGraph = pipeline_result["graph"]
    explanation: str = pipeline_result.get("explanation", "")
    kag_motifs: list = pipeline_result.get("kag_motifs", [])

    # --- Pre-compute metrics for all nodes ---
    metrics = estimate_metrics_from_graph(graph)
    tensor_memory = estimate_activation_memory(graph, batch_size=1, input_spatial=224)

    # --- Step 1+2: Extract macro-structures and collapse trivial nodes ---
    groups = _group_nodes_into_macro_structures(graph, schema)

    # --- Step 3: Detect and merge repeated stages ---
    groups = _detect_and_merge_stages(groups, schema)

    # --- Step 4+5: Build one LearningModule per group ---
    total = len(groups)
    modules: list[LearningModule] = []
    for i, node_group in enumerate(groups):
        module = _build_module(
            index=i,
            total=total,
            nodes=node_group,
            schema=schema,
            flops_breakdown=metrics["breakdown"],
            tensor_memory=tensor_memory,
            graph=graph,
        )
        modules.append(module)

    # --- Paper metadata ---
    paper_meta = {
        "title": paper_name,
        "authors": schema.get("authors", None),
        "abstract": explanation,
        "architecture_graph": {
            "name": graph.name,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "label": n.label,
                    "params": n.params,
                }
                for n in graph.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "type": e.edge_type} for e in graph.edges
            ],
            "kag_motifs": kag_motifs,
        },
        "flops_analysis": {
            "total_flops_score": metrics["total_flops_score"],
            "total_params_estimate": metrics["total_params_estimate"],
            "depth": metrics["depth"],
            "breakdown": metrics["breakdown"],
        },
    }

    return paper_meta, modules
