"""
Transformer Builder Module.

Provides a reusable abstraction layer for building Transformer backbones,
supporting encoder-only (ViT, BERT), decoder-only (GPT, Llama), and
encoder-decoder (T5) architectures.
"""

from typing import Dict, Any, List, Optional
from src.architecture_graph import ArchitectureGraph, GraphNode

class TransformerBuilder:
    def __init__(self, name: str = "TransformerBackbone"):
        self.graph = ArchitectureGraph(name=name)
        self.node_count = 0
        self.metadata = {
            "architecture_family": "transformer",
            "attention_type": "standard",
            "sequence_type": "generic",
            "embedding_dim": 768,
            "num_heads": 12
        }
        self.graph.metadata.update(self.metadata)
        self.last_node_id = None

    def _next_id(self, prefix: str) -> str:
        self.node_count += 1
        return f"{prefix}_{self.node_count}"

    def _add_node(self, node: GraphNode) -> str:
        self.graph.add_node(node)
        if self.last_node_id:
            self.graph.add_edge(self.last_node_id, node.id)
        self.last_node_id = node.id
        return node.id

    def set_metadata(self, key: str, value: Any):
        self.metadata[key] = value
        self.graph.metadata[key] = value

    # --- Core Abstractions ---
    def add_token_embedding(self, vocab_size: int, embed_dim: int):
        node_id = self._next_id("token_embedding")
        self.set_metadata("embedding_dim", embed_dim)
        node = GraphNode(
            id=node_id,
            type="token_embedding",
            label="Token Embed",
            params={"vocab_size": vocab_size, "embed_dim": embed_dim},
            semantic_params={"semantic_role": "token_embedding"}
        )
        return self._add_node(node)

    def add_positional_embedding(self, max_seq_len: int, embed_dim: int):
        node_id = self._next_id("positional_embedding")
        node = GraphNode(
            id=node_id,
            type="positionalembedding",
            label="Pos Embed",
            params={"max_seq_len": max_seq_len, "embed_dim": embed_dim},
            semantic_params={"semantic_role": "positional_embedding"}
        )
        return self._add_node(node)

    def add_segment_embedding(self, type_vocab_size: int, embed_dim: int):
        node_id = self._next_id("segment_embedding")
        node = GraphNode(
            id=node_id,
            type="segment_embedding",
            label="Segment Embed",
            params={"type_vocab_size": type_vocab_size, "embed_dim": embed_dim},
            semantic_params={"semantic_role": "token_embedding"}
        )
        return self._add_node(node)

    def add_elementwise_add(self, sources: List[str]):
        node_id = self._next_id("elementwise_add")
        node = GraphNode(
            id=node_id,
            type="elementwise_add",
            label="Add",
            params={},
            semantic_params={"semantic_role": "residual"}
        )
        self.graph.add_node(node)
        for src in sources:
            self.graph.add_edge(src, node_id)
        self.last_node_id = node_id
        return node_id

    def add_normalization(self, norm_type: str = "layernorm", dim: Optional[int] = None):
        node_id = self._next_id("normalization")
        dim = dim or self.metadata["embedding_dim"]
        node = GraphNode(
            id=node_id,
            type=norm_type,
            label="Norm",
            params={"dim": dim},
            semantic_params={"semantic_role": "normalization"}
        )
        return self._add_node(node)

    def add_self_attention(self, num_heads: int, embed_dim: Optional[int] = None, causal: bool = False):
        node_id = self._next_id("self_attention")
        embed_dim = embed_dim or self.metadata["embedding_dim"]
        self.set_metadata("num_heads", num_heads)
        if causal:
            self.set_metadata("attention_type", "causal")
        node = GraphNode(
            id=node_id,
            type="multiheadattention",
            label="Self-Attention",
            params={"num_heads": num_heads, "embed_dim": embed_dim, "causal": causal},
            semantic_params={"semantic_role": "token_mixer"}
        )
        return self._add_node(node)

    def add_feedforward(self, hidden_dim: int, embed_dim: Optional[int] = None):
        node_id = self._next_id("feedforward")
        embed_dim = embed_dim or self.metadata["embedding_dim"]
        node = GraphNode(
            id=node_id,
            type="feedforward",
            label="FFN",
            params={"hidden_size": hidden_dim, "embed_dim": embed_dim},
            semantic_params={"semantic_role": "feature_transformation"}
        )
        return self._add_node(node)

    def add_residual_add(self, source_id: str, target_id: str):
        # We don't change last_node_id, just inject the residual connection.
        node_id = self._next_id("residual_add")
        node = GraphNode(
            id=node_id,
            type="residual_add",
            label="Add",
            params={},
            semantic_params={"semantic_role": "residual"}
        )
        self.graph.add_node(node)
        self.graph.add_edge(target_id, node_id)
        self.graph.add_edge(source_id, node_id, edge_type="skip")
        self.last_node_id = node_id
        return node_id

    def add_sequence_pooling(self, pool_type: str = "mean"):
        node_id = self._next_id("sequence_pooling")
        node = GraphNode(
            id=node_id,
            type="sequence_pooling",
            label="Seq Pool",
            params={"pool_type": pool_type},
            semantic_params={"semantic_role": "feature_aggregator"}
        )
        return self._add_node(node)

    def add_classifier_head(self, num_classes: int, in_features: Optional[int] = None):
        node_id = self._next_id("classifier_head")
        in_features = in_features or self.metadata["embedding_dim"]
        node = GraphNode(
            id=node_id,
            type="linear",
            label="Classifier",
            params={"hidden_size": num_classes, "in_features": in_features},
            semantic_params={"semantic_role": "classifier_head"}
        )
        return self._add_node(node)

    # --- High-Level Architecture Utilities ---

    def add_encoder_block(self, embed_dim: int, num_heads: int, ffn_dim: int, pre_norm: bool = True):
        """Standard Transformer Encoder Block."""
        block_input = self.last_node_id

        if pre_norm:
            norm1 = self.add_normalization("layernorm", embed_dim)
            attn = self.add_self_attention(num_heads, embed_dim, causal=False)
            res1 = self.add_residual_add(block_input, attn)
            
            norm2 = self.add_normalization("layernorm", embed_dim)
            ffn = self.add_feedforward(ffn_dim, embed_dim)
            res2 = self.add_residual_add(res1, ffn)
        else:
            attn = self.add_self_attention(num_heads, embed_dim, causal=False)
            res1 = self.add_residual_add(block_input, attn)
            norm1 = self.add_normalization("layernorm", embed_dim)
            
            ffn = self.add_feedforward(ffn_dim, embed_dim)
            res2 = self.add_residual_add(norm1, ffn)
            norm2 = self.add_normalization("layernorm", embed_dim)

        return self.last_node_id

    def add_decoder_block(self, embed_dim: int, num_heads: int, ffn_dim: int, cross_attention: bool = False):
        """Standard Transformer Decoder Block (Causal)."""
        block_input = self.last_node_id
        
        norm1 = self.add_normalization("layernorm", embed_dim)
        attn = self.add_self_attention(num_heads, embed_dim, causal=True)
        res1 = self.add_residual_add(block_input, attn)
        
        if cross_attention:
            norm_cross = self.add_normalization("layernorm", embed_dim)
            # Typically cross-attention has causal=False since it attends to the full encoder output
            cross_attn = self.add_self_attention(num_heads, embed_dim, causal=False)
            res1 = self.add_residual_add(res1, cross_attn)
            
        norm2 = self.add_normalization("layernorm", embed_dim)
        ffn = self.add_feedforward(ffn_dim, embed_dim)
        res2 = self.add_residual_add(res1, ffn)
        
        return self.last_node_id

    def get_graph(self) -> ArchitectureGraph:
        return self.graph
