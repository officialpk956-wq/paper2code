"""
Knowledge Graph Engine for KAG (Knowledge-Augmented Generation).

This module defines an in-memory ontology of Deep Learning architectures
using NetworkX. It is used to enforce architectural rules and prevent
LLM hallucinations during RAG extraction.
"""

import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_ontology()

    def _build_ontology(self):
        """Build the hardcoded knowledge graph of DL concepts."""
        # 1. Base Layer Families
        self.graph.add_node("conv2d", type="layer", dimensionality="4D")
        self.graph.add_node("linear", type="layer", dimensionality="2D")
        self.graph.add_node("multiheadattention", type="layer", dimensionality="3D")
        self.graph.add_node("batchnorm2d", type="norm", dimensionality="4D")
        self.graph.add_node("layernorm", type="norm", dimensionality="any")

        # New base layers
        self.graph.add_node("cross_attention", type="layer", dimensionality="3D")
        self.graph.add_node("causal_attention", type="layer", dimensionality="3D")
        self.graph.add_node("residual_add", type="layer", dimensionality="any")

        # 2. Advanced Blocks
        self.graph.add_node(
            "residualblock", type="block", contains=["conv2d", "batchnorm2d", "relu"]
        )
        self.graph.add_node(
            "transformerblock", type="block", contains=["multiheadattention", "layernorm", "linear"]
        )
        self.graph.add_node(
            "transformer_encoder",
            type="block",
            contains=["multiheadattention", "layernorm", "linear"],
        )
        self.graph.add_node(
            "transformer_decoder",
            type="block",
            contains=["causal_attention", "cross_attention", "layernorm", "linear"],
        )
        self.graph.add_node("patchembedding", type="block", outputs="3D")
        self.graph.add_node("sequence_pooling", type="block", outputs="2D")

        # 3. Rules & Edges (Architecture Constraints)
        self.graph.add_edge(
            "conv2d",
            "linear",
            relation="REQUIRES_FLATTEN",
            reason="Transition from 4D spatial to 2D flat requires flattening",
        )
        self.graph.add_edge(
            "batchnorm2d",
            "linear",
            relation="INCOMPATIBLE",
            reason="BatchNorm2D expects 4D input, Linear expects 2D",
        )
        self.graph.add_edge(
            "transformerblock",
            "conv2d",
            relation="INCOMPATIBLE",
            reason="Standard transformers operate on 3D token sequences, not 4D image grids",
        )
        self.graph.add_edge(
            "patchembedding",
            "transformerblock",
            relation="COMPATIBLE",
            reason="Patch embedding provides the correct 3D token sequence for transformers",
        )

        # Sequence Propagation Rules
        self.graph.add_edge(
            "transformer_encoder",
            "transformer_encoder",
            relation="SEQUENCE_PRESERVED",
            reason="Encoder stacks preserve sequence lengths",
        )
        self.graph.add_edge(
            "causal_attention",
            "causal_attention",
            relation="CAUSAL_PRESERVED",
            reason="Autoregressive generation mask must be maintained",
        )

        # Residual Compatibility Rules
        self.graph.add_edge(
            "residual_add",
            "residual_add",
            relation="REQUIRES_EQUAL_DIMS",
            reason="Residual connections strictly require identically sized tensors",
        )

        # Encoder-Decoder Constraints
        self.graph.add_edge(
            "transformer_encoder",
            "cross_attention",
            relation="PROVIDES_MEMORY",
            reason="Encoder provides Key/Value memory for Decoder cross-attention",
        )
        self.graph.add_edge(
            "causal_attention",
            "cross_attention",
            relation="REQUIRES_PRIOR",
            reason="Decoder cross-attention typically follows causal self-attention",
        )

        # 4. Semantic Role Mappings (KAG)
        for role in [
            "patch_embedding",
            "token_mixer",
            "sequence_encoder",
            "feature_aggregator",
            "classifier_head",
            "encoder",
            "decoder",
        ]:
            self.graph.add_node(role, type="semantic_role")

        self.graph.add_edge("patchembedding", "patch_embedding", relation="implements")
        self.graph.add_edge("multiheadattention", "token_mixer", relation="performs")
        self.graph.add_edge("cross_attention", "token_mixer", relation="performs")
        self.graph.add_edge("causal_attention", "token_mixer", relation="performs")
        self.graph.add_edge("mhsa", "token_mixer", relation="implements")
        self.graph.add_edge("transformerblock", "sequence_encoder", relation="implements")
        self.graph.add_edge("transformer_encoder", "encoder", relation="implements")
        self.graph.add_edge("transformer_decoder", "decoder", relation="implements")
        self.graph.add_edge("globalavgpool", "feature_aggregator", relation="performs")
        self.graph.add_edge("sequence_pooling", "feature_aggregator", relation="performs")
        self.graph.add_edge("linear", "classifier_head", relation="implements")

    def get_semantic_role(self, node_type: str) -> str | None:
        """Infer the semantic role of a node type from the Knowledge Graph."""
        node_type = node_type.lower()
        if node_type not in self.graph:
            return None

        for neighbor in self.graph.neighbors(node_type):
            edge_data = self.graph.get_edge_data(node_type, neighbor)
            if edge_data.get("relation") in ["implements", "performs"]:
                return neighbor
        return None

    def get_context_for_terms(self, terms: list[str]) -> str:
        """
        Given a list of architectural terms extracted from text,
        query the Knowledge Graph for explicit rules and constraints.
        """
        rules = []
        term_set = {t.lower() for t in terms}

        nodes = list(self.graph.nodes())
        found_nodes = [n for n in nodes if any(t in n or n in t for t in term_set)]

        for u in found_nodes:
            props = self.graph.nodes[u]
            if "contains" in props:
                rules.append(f"- Rule: A '{u}' MUST contain {props['contains']}.")

            for v in found_nodes:
                if self.graph.has_edge(u, v):
                    edge_data = self.graph.get_edge_data(u, v)
                    relation = edge_data.get("relation")
                    reason = edge_data.get("reason")

                    if relation == "REQUIRES_FLATTEN":
                        rules.append(
                            f"- Rule: Connecting '{u}' to '{v}' REQUIRES a Flatten or GlobalAveragePooling operation ({reason})."
                        )
                    elif relation == "INCOMPATIBLE":
                        rules.append(
                            f"- Rule: Connecting '{u}' to '{v}' is logically INCOMPATIBLE ({reason})."
                        )
                    elif relation == "REQUIRES_EQUAL_DIMS":
                        rules.append(f"- Rule: {reason}.")

        if not rules:
            return ""

        return "CRITICAL KNOWLEDGE GRAPH CONSTRAINTS:\n" + "\n".join(set(rules))

    def detect_motifs(self, arch_graph) -> list[str]:
        """Detect architectural motifs like BERT stacks or UNet skips in an ArchitectureGraph."""
        motifs = []
        node_types = [n.type.lower() for n in arch_graph.nodes]

        # BERT-style encoder stack
        encoder_count = sum(1 for t in node_types if "encoder" in t or "transformerblock" in t)
        if encoder_count >= 6 and "decoder" not in "".join(node_types):
            motifs.append("BERT-style Encoder Stack")

        # GPT-style decoder stack
        causal_count = sum(1 for t in node_types if "causal" in t or "decoder" in t)
        if (
            causal_count >= 6
            and "encoder" not in "".join(node_types)
            and "cross_attention" not in "".join(node_types)
        ):
            motifs.append("GPT-style Autoregressive Decoder Stack")

        # ViT token pipeline
        if "patchembedding" in "".join(node_types) and (
            "transformerblock" in "".join(node_types) or "encoder" in "".join(node_types)
        ):
            if "sequence_pooling" in "".join(node_types) or "clstoken" in "".join(node_types):
                motifs.append("ViT Token Pipeline")

        # UNet skip structures
        has_downsample = any(t in ["maxpool2d", "conv2d_stride2"] for t in node_types)
        has_upsample = any(t in ["convtranspose2d", "upsample"] for t in node_types)
        has_concat = "concat" in node_types
        if has_downsample and has_upsample and has_concat:
            motifs.append("UNet-style Skip Structure")

        return motifs

    def verify_topology(self, arch_graph) -> list[str]:
        """Graph reasoning utility: detect anomalies and incompatible layer transitions."""
        anomalies = []
        node_types = [n.type.lower() for n in arch_graph.nodes]

        # 1. Topology Anomaly Detection
        if "cross_attention" in node_types and not any("encoder" in t for t in node_types):
            anomalies.append(
                "Anomaly: Cross-Attention detected without an explicit Encoder memory source."
            )

        if (
            "causal_attention" in node_types
            and "cross_attention" not in node_types
            and "encoder" in node_types
        ):
            anomalies.append(
                "Anomaly: Causal Attention mixed with Encoder blocks but missing Cross-Attention bridge."
            )

        # 2. Incompatible Layer Detection
        for edge in arch_graph.edges:
            src_node = next((n for n in arch_graph.nodes if n.id == edge.source), None)
            tgt_node = next((n for n in arch_graph.nodes if n.id == edge.target), None)

            if src_node and tgt_node:
                s_type, t_type = src_node.type.lower(), tgt_node.type.lower()

                # Check KAG for INCOMPATIBLE
                if self.graph.has_edge(s_type, t_type):
                    edge_data = self.graph.get_edge_data(s_type, t_type)
                    if edge_data.get("relation") == "INCOMPATIBLE":
                        anomalies.append(
                            f"Incompatible Transition: '{s_type}' to '{t_type}'. Reason: {edge_data.get('reason')}"
                        )

                # Tensor Semantic Verification based on rules
                if s_type == "conv2d" and t_type == "linear":
                    anomalies.append(
                        f"Missing Operation: Connecting '{s_type}' directly to '{t_type}' requires flattening."
                    )

        return list(set(anomalies))

    def identify_terms(self, text: str) -> list[str]:
        """Simple keyword matching to identify graph entities in text."""
        text_lower = text.lower()
        found = []
        for node in self.graph.nodes():
            if node in text_lower:
                found.append(node)
        return found
