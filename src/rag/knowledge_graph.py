"""
Knowledge Graph Engine for KAG (Knowledge-Augmented Generation).

This module defines an in-memory ontology of Deep Learning architectures
using NetworkX. It is used to enforce architectural rules and prevent
LLM hallucinations during RAG extraction.
"""

import networkx as nx
from typing import List, Dict, Any, Tuple

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
        
        # 2. Advanced Blocks
        self.graph.add_node("residualblock", type="block", contains=["conv2d", "batchnorm2d", "relu"])
        self.graph.add_node("transformerblock", type="block", contains=["multiheadattention", "layernorm", "linear"])
        self.graph.add_node("patchembedding", type="block", outputs="3D")
        
        # 3. Rules & Edges
        # Convolutional rules
        self.graph.add_edge("conv2d", "linear", relation="REQUIRES_FLATTEN", reason="Transition from 4D spatial to 2D flat requires flattening")
        self.graph.add_edge("batchnorm2d", "linear", relation="INCOMPATIBLE", reason="BatchNorm2D expects 4D input, Linear expects 2D")
        
        # Transformer rules
        self.graph.add_edge("transformerblock", "conv2d", relation="INCOMPATIBLE", reason="Standard transformers operate on 3D token sequences, not 4D image grids")
        self.graph.add_edge("patchembedding", "transformerblock", relation="COMPATIBLE", reason="Patch embedding provides the correct 3D token sequence for transformers")

        # 4. Semantic Role Mappings (KAG)
        self.graph.add_node("patch_embedding", type="semantic_role")
        self.graph.add_node("token_mixer", type="semantic_role")
        self.graph.add_node("sequence_encoder", type="semantic_role")
        self.graph.add_node("feature_aggregator", type="semantic_role")
        self.graph.add_node("classifier_head", type="semantic_role")
        
        self.graph.add_edge("patchembedding", "patch_embedding", relation="implements")
        self.graph.add_edge("multiheadattention", "token_mixer", relation="performs")
        self.graph.add_edge("mhsa", "token_mixer", relation="implements")
        self.graph.add_edge("transformerblock", "sequence_encoder", relation="implements")
        self.graph.add_edge("globalavgpool", "feature_aggregator", relation="performs")
        self.graph.add_edge("linear", "classifier_head", relation="implements")

    def get_semantic_role(self, node_type: str) -> Optional[str]:
        """Infer the semantic role of a node type from the Knowledge Graph."""
        node_type = node_type.lower()
        if node_type not in self.graph:
            return None
            
        for neighbor in self.graph.neighbors(node_type):
            edge_data = self.graph.get_edge_data(node_type, neighbor)
            if edge_data.get("relation") in ["implements", "performs"]:
                return neighbor
        return None

    def get_context_for_terms(self, terms: List[str]) -> str:
        """
        Given a list of architectural terms extracted from text,
        query the Knowledge Graph for explicit rules and constraints.
        """
        rules = []
        term_set = {t.lower() for t in terms}
        
        # Check all possible pairs in the extracted terms for known edge constraints
        nodes = list(self.graph.nodes())
        found_nodes = [n for n in nodes if any(t in n or n in t for t in term_set)]
        
        for u in found_nodes:
            # Add node properties
            props = self.graph.nodes[u]
            if "contains" in props:
                rules.append(f"- Rule: A '{u}' MUST contain {props['contains']}.")
                
            for v in found_nodes:
                if self.graph.has_edge(u, v):
                    edge_data = self.graph.get_edge_data(u, v)
                    relation = edge_data.get("relation")
                    reason = edge_data.get("reason")
                    
                    if relation == "REQUIRES_FLATTEN":
                        rules.append(f"- Rule: Connecting '{u}' to '{v}' REQUIRES a Flatten or GlobalAveragePooling operation ({reason}).")
                    elif relation == "INCOMPATIBLE":
                        rules.append(f"- Rule: Connecting '{u}' to '{v}' is logically INCOMPATIBLE ({reason}).")
                        
        if not rules:
            return ""
            
        return "CRITICAL KNOWLEDGE GRAPH CONSTRAINTS:\n" + "\n".join(set(rules))

    def identify_terms(self, text: str) -> List[str]:
        """Simple keyword matching to identify graph entities in text."""
        text_lower = text.lower()
        found = []
        for node in self.graph.nodes():
            if node in text_lower:
                found.append(node)
        return found
