"""Repeated-block ("motif") detection for ONNX graphs.

Big models are unreadable as a flat wall of nodes (audit #2). Almost all of that
size is *repetition* — a ResNet stage or a Transformer block copied N times. This
finds those repeats so the viewer can collapse each occurrence into one expandable
super-node.

ONNX stores nodes in topological order, so a repeated block shows up as a repeated
contiguous run of op-types. We scan for the run that covers the most nodes at each
position (greedy, deterministic) and emit one group per occurrence.

Pure — no `onnx` import — so it is unit-testable on plain node dicts.
"""

from __future__ import annotations

from typing import Any

_SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_SEV_NAME = {v: k for k, v in _SEV_RANK.items()}


def _compact_label(ops: list[str]) -> str:
    if len(ops) <= 3:
        return " → ".join(ops)
    return f"{ops[0]} → … → {ops[-1]}  ({len(ops)} ops)"


def _max_sev(members: list[dict]) -> str:
    rank = max((_SEV_RANK.get(m.get("severity", "low"), 0) for m in members), default=0)
    return _SEV_NAME[rank]


def detect_motifs(
    nodes: list[dict[str, Any]],
    *,
    min_len: int = 2,
    max_len: int = 16,
    min_repeat: int = 2,
) -> list[dict]:
    """Return a list of groups, one per occurrence of a detected repeated block.

    Each group: {id, signature, label, node_ids, member_ops, repeat_index,
    repeat_count, flops, params, memory_mb, severity}. Nodes not part of any
    repeated block are simply absent from the result (rendered individually).
    """
    seq = [n.get("op_type", "") for n in nodes]
    n = len(seq)
    groups: list[dict] = []
    gid = 0
    i = 0
    # ponytail: O(n · max_len · run) greedy scan — fine for real models (≤ a few
    # thousand nodes); revisit with suffix automata only if that ceiling is hit.
    while i < n:
        best_len = 0
        best_r = 0
        best_cov = 0
        for length in range(min_len, min(max_len, n - i) + 1):
            block = seq[i : i + length]
            r = 1
            while i + (r + 1) * length <= n and seq[i + r * length : i + (r + 1) * length] == block:
                r += 1
            if r >= min_repeat:
                cov = r * length
                # Most coverage wins; on a tie the SMALLEST period wins (the true
                # motif — [block]×4, not [block×2]×2). Lengths ascend, so a strict
                # '>' keeps the first (smallest) length that reaches max coverage.
                if cov > best_cov:
                    best_cov, best_len, best_r = cov, length, r
        if best_len:
            ops = seq[i : i + best_len]
            sig = " → ".join(ops)
            label = _compact_label(ops)
            for j in range(best_r):
                start = i + j * best_len
                members = nodes[start : start + best_len]
                groups.append(
                    {
                        "id": f"grp_{gid}",
                        "signature": sig,
                        "label": label,
                        "node_ids": [m["id"] for m in members],
                        "member_ops": ops,
                        "repeat_index": j,
                        "repeat_count": best_r,
                        "flops": sum(m.get("flops", 0) or 0 for m in members),
                        "params": sum(m.get("params", 0) or 0 for m in members),
                        "memory_mb": round(sum(m.get("memory_mb", 0) or 0 for m in members), 4),
                        "severity": _max_sev(members),
                    }
                )
                gid += 1
            i += best_cov
        else:
            i += 1
    return groups


def motif_summary(groups: list[dict]) -> dict:
    """Aggregate counts for a legend: distinct motifs and how many nodes collapse."""
    by_sig: dict[str, int] = {}
    grouped_nodes = 0
    for g in groups:
        by_sig[g["signature"]] = by_sig.get(g["signature"], 0) + 1
        grouped_nodes += len(g["node_ids"])
    return {
        "motif_count": len(by_sig),
        "grouped_nodes": grouped_nodes,
        "motifs": [
            {"signature": s, "count": c} for s, c in sorted(by_sig.items(), key=lambda kv: -kv[1])
        ],
    }


def _demo():
    def node(i, op):
        return {
            "id": f"node_{i}",
            "op_type": op,
            "flops": 10,
            "params": 5,
            "memory_mb": 1.0,
            "severity": "low",
        }

    # 3 identical [Conv, BN, Relu] blocks, wrapped by a unique input and output op
    ops = ["Gemm"] + ["Conv", "BatchNormalization", "Relu"] * 3 + ["Softmax"]
    ns = [node(i, op) for i, op in enumerate(ops)]
    gs = detect_motifs(ns)
    assert len(gs) == 3, gs
    assert all(g["signature"] == "Conv → BatchNormalization → Relu" for g in gs), gs
    assert [g["repeat_index"] for g in gs] == [0, 1, 2], gs
    assert gs[0]["node_ids"] == ["node_1", "node_2", "node_3"], gs
    assert gs[0]["flops"] == 30 and gs[0]["params"] == 15, gs
    s = motif_summary(gs)
    assert s["motif_count"] == 1 and s["grouped_nodes"] == 9, s
    # no repetition -> no groups
    assert detect_motifs([node(i, op) for i, op in enumerate(["A", "B", "C", "D"])]) == []
    print("onnx_motifs demo OK")


if __name__ == "__main__":
    _demo()
