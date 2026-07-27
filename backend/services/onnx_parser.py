import io
import logging
from typing import Any

from backend.services.onnx_flops import estimate_node_cost
from backend.services.onnx_motifs import detect_motifs, motif_summary

logger = logging.getLogger(__name__)


def _shape_from_type_proto(type_proto) -> list[int]:
    try:
        return [d.dim_value if d.dim_value > 0 else -1 for d in type_proto.tensor_type.shape.dim]
    except Exception:
        return []


def _parse_attr(attr) -> Any:
    """Convert ONNX AttributeProto to a plain Python value."""
    t = attr.type
    if t == 1:  # FLOAT
        return round(float(attr.f), 6)
    if t == 2:  # INT
        return int(attr.i)
    if t == 3:  # STRING
        return attr.s.decode("utf-8", errors="replace")
    if t == 6:  # FLOATS
        return [round(float(v), 6) for v in attr.floats]
    if t == 7:  # INTS
        return [int(v) for v in attr.ints]
    if t == 8:  # STRINGS
        return [s.decode("utf-8", errors="replace") for s in attr.strings]
    # t == 4 (TENSOR) or t == 5 (GRAPH) — too large to serialize inline
    return None


def parse_onnx(file_bytes: bytes) -> dict:
    """
    Parse an ONNX model from raw bytes.
    Returns a graph dict with nodes, edges, and meta suitable for React Flow rendering.
    """
    import onnx
    from onnx import shape_inference

    model = onnx.load(io.BytesIO(file_bytes))

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("Shape inference failed (partial shapes only): %s", e)

    graph = model.graph

    # ── shape map: tensor_name → [dim, dim, ...] ──────────────────────────────
    shape_map: dict[str, list[int]] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        shape_map[vi.name] = _shape_from_type_proto(vi.type)

    # ── initializers (weights): name → dims ───────────────────────────────────
    init_names: set[str] = set()
    init_dims: dict[str, list[int]] = {}
    for init in graph.initializer:
        init_names.add(init.name)
        init_dims[init.name] = list(init.dims)

    # ── map output tensor → node_id for edge construction ─────────────────────
    # Only operator nodes, not initializers or graph inputs
    tensor_to_node: dict[str, str] = {}
    nodes_list = list(graph.node)
    for i, node in enumerate(nodes_list):
        for out in node.output:
            if out:
                tensor_to_node[out] = f"node_{i}"

    # ── parse operator nodes ──────────────────────────────────────────────────
    parsed_nodes: list[dict] = []
    for i, node in enumerate(nodes_list):
        node_id = f"node_{i}"

        # Parameter count: sum over weight initializer inputs
        total_params = 0
        for inp in node.input:
            if inp in init_dims:
                dims = init_dims[inp]
                if dims:
                    p = 1
                    for d in dims:
                        p *= d
                    total_params += p

        # Attributes
        attrs: dict[str, Any] = {}
        for attr in node.attribute:
            val = _parse_attr(attr)
            if val is not None:
                attrs[attr.name] = val

        # Input shapes (data inputs only, not weights)
        input_shapes: dict[str, list[int]] = {}
        for inp in node.input:
            if inp and inp not in init_names and inp in shape_map:
                input_shapes[inp] = shape_map[inp]

        # Output shapes
        output_shapes: dict[str, list[int]] = {}
        for out in node.output:
            if out and out in shape_map:
                output_shapes[out] = shape_map[out]

        # Primary output shape for node card display (first non-empty output)
        primary_out_shape: list[int] = []
        for out in node.output:
            if out and out in shape_map and shape_map[out]:
                primary_out_shape = shape_map[out]
                break

        # Per-node compute cost (FLOPs) + activation memory, from the shapes above.
        weight_dims = [init_dims[inp] for inp in node.input if inp in init_dims]
        cost = estimate_node_cost(
            node.op_type, list(input_shapes.values()), primary_out_shape, weight_dims, attrs
        )

        parsed_nodes.append(
            {
                "id": node_id,
                "op_type": node.op_type,
                "label": node.name or node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "primary_out_shape": primary_out_shape,
                "params": total_params,
                "attrs": attrs,
                "flops": cost["flops"],
                "memory_mb": cost["memory_mb"],
                "severity": cost["severity"],
            }
        )

    # ── build edges from tensor name connections ──────────────────────────────
    # Key includes tensor name: a multi-output op can feed the same downstream
    # node via two different output tensors (e.g. LSTM → Add via both h and c).
    edges: list[dict] = []
    edge_id = 0
    seen: set[tuple[str, str, str]] = set()

    for i, node in enumerate(nodes_list):
        node_id = f"node_{i}"
        for inp in node.input:
            if not inp or inp in init_names:
                continue
            src = tensor_to_node.get(inp)
            if src and src != node_id:
                key = (src, node_id, inp)
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        {
                            "id": f"e_{edge_id}",
                            "source": src,
                            "target": node_id,
                            "shape": shape_map.get(inp, []),
                            "tensor": inp,
                        }
                    )
                    edge_id += 1

    # ── meta ──────────────────────────────────────────────────────────────────
    total_params = sum(n["params"] for n in parsed_nodes)
    total_flops = sum(n["flops"] for n in parsed_nodes)
    graph_inputs = {
        inp.name: _shape_from_type_proto(inp.type)
        for inp in graph.input
        if inp.name not in init_names
    }
    graph_outputs = {out.name: _shape_from_type_proto(out.type) for out in graph.output}

    opset = model.opset_import[0].version if model.opset_import else 0

    # Repeated-block motifs so the client can collapse a big model into a handful
    # of expandable super-nodes.
    groups = detect_motifs(parsed_nodes)
    summary = motif_summary(groups)

    return {
        "nodes": parsed_nodes,
        "edges": edges,
        "groups": groups,
        "meta": {
            "total_nodes": len(parsed_nodes),
            "total_params": total_params,
            "total_flops": total_flops,
            "total_edges": len(edges),
            "motif_count": summary["motif_count"],
            "grouped_nodes": summary["grouped_nodes"],
            "motifs": summary["motifs"],
            "graph_inputs": graph_inputs,
            "graph_outputs": graph_outputs,
            "ir_version": model.ir_version,
            "opset_version": opset,
        },
    }
