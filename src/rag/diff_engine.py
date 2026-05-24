"""
Graph Diff Engine for Neural Architectures.

Compares two architecture graphs and identifies topological and semantic differences.
"""

from typing import Dict, Any, List, Set
from src.architecture_graph import ArchitectureGraph, GraphNode

class GraphDiffEngine:
    """
    Engine to compute differences between two architecture graphs.
    """

    def compare(self, graph_a: ArchitectureGraph, graph_b: ArchitectureGraph) -> Dict[str, Any]:
        """
        Compare two graphs and return a structured diff report.
        """
        # 1. Metric Deltas
        metrics_a = self._get_metrics(graph_a)
        metrics_b = self._get_metrics(graph_b)
        
        deltas = {
            "flops": metrics_b["flops"] - metrics_a["flops"],
            "params": metrics_b["params"] - metrics_a["params"],
            "depth": metrics_b["depth"] - metrics_a["depth"],
            "memory": metrics_b["memory"] - metrics_a["memory"]
        }

        # 2. Topological Diff
        nodes_a = {n.id: n for n in graph_a.nodes}
        nodes_b = {n.id: n for n in graph_b.nodes}
        
        ids_a = set(nodes_a.keys())
        ids_b = set(nodes_b.keys())
        
        added_nodes = ids_b - ids_a
        removed_nodes = ids_a - ids_b
        common_nodes = ids_a & ids_b
        
        changed_params = []
        for nid in common_nodes:
            node_a = nodes_a[nid]
            node_b = nodes_b[nid]
            if node_a.type != node_b.type or node_a.params != node_b.params:
                changed_params.append({
                    "id": nid,
                    "label": node_b.label,
                    "type_changed": node_a.type != node_b.type,
                    "params_changed": node_a.params != node_b.params,
                    "from": {"type": node_a.type, "params": node_a.params},
                    "to": {"type": node_b.type, "params": node_b.params}
                })

        # 3. Semantic Summary
        summary = self._generate_semantic_summary(graph_a, graph_b, deltas)

        return {
            "deltas": deltas,
            "added_nodes": [nodes_b[nid].label for nid in added_nodes],
            "removed_nodes": [nodes_a[nid].label for nid in removed_nodes],
            "changed_params": changed_params,
            "summary": summary
        }

    def _get_metrics(self, graph: ArchitectureGraph) -> Dict[str, float]:
        from src.metrics_estimator import estimate_metrics_from_graph, estimate_activation_memory
        metrics = estimate_metrics_from_graph(graph)
        mem_data = estimate_activation_memory(graph, batch_size=1, input_spatial=224)
        total_mem = sum(row['mem_mb'] for row in mem_data) if mem_data else 0
        
        return {
            "flops": metrics["total_flops_score"],
            "params": metrics["total_params_estimate"],
            "depth": metrics["depth"],
            "memory": total_mem
        }

    def _generate_semantic_summary(self, graph_a: ArchitectureGraph, graph_b: ArchitectureGraph, deltas: Dict[str, float]) -> str:
        """
        Uses rule-based logic to describe the evolution between architectures.
        """
        reasons = []
        
        # Motif Evolution
        motifs_a = set(graph_a.metadata.get("kag_motifs", []))
        motifs_b = set(graph_b.metadata.get("kag_motifs", []))
        
        added_motifs = motifs_b - motifs_a
        removed_motifs = motifs_a - motifs_b
        
        if "CNN" in motifs_a and "Transformer" in motifs_b:
            reasons.append("Replacing convolutional feature extraction with global self-attention.")
        
        if "BERT" in motifs_a and "GPT" in motifs_b:
            reasons.append("Transitioning from bidirectional encoding to autoregressive decoding.")
            
        if added_motifs:
            for m in added_motifs:
                reasons.append(f"Introduces **{m}** logic for improved architectural reasoning.")
        
        # Scaling
        if deltas["flops"] > 1000:
            reasons.append("Significant increase in computational complexity (FLOPs).")
        elif deltas["flops"] < -500:
            reasons.append("Optimized for efficiency with reduced total FLOPs.")
            
        if deltas["params"] > 1000000:
            reasons.append("Increased model capacity via larger parameter count.")

        # Specific component changes
        types_a = {n.type for n in graph_a.nodes}
        types_b = {n.type for n in graph_b.nodes}
        
        if "cross_attention" in types_b and "cross_attention" not in types_a:
            reasons.append("Added **Cross-Attention** fusion points for encoder-decoder communication.")

        if not reasons:
            return f"Incremental evolution from {graph_a.name} to {graph_b.name}."
            
        return " ".join(reasons)
