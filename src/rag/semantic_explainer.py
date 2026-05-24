"""
Semantic Explainer Module.

Provides deterministic, educational explanations for architectural components
based on their semantic roles and parameters.
"""

from typing import Dict, Any

class SemanticExplainer:
    """
    KAG-driven explainer that provides deterministic explanations for layers.
    Focuses on the "Why" (educational) rather than just the "What" (structural).
    """
    
    _EXPLANATIONS = {
        "patch_embedding": "This layer converts spatial image patches into learnable token embeddings by projecting pixel grids into a sequence. It is the bridge between pixels and tokens.",
        "token_mixer": "Enables communication between different tokens in the sequence. It allows the model to capture global relationships and dependencies regardless of distance.",
        "sequence_encoder": "Refines token representations through hierarchical processing. In Transformers, this typically alternates between attention and feedforward blocks.",
        "feature_aggregator": "Aggregates information from all tokens or spatial locations into a single global representation suitable for classification or regression.",
        "classifier_head": "The final decision-making layer that maps abstract neural features to specific target categories or probabilities.",
        "encoder": "A bidirectional encoder stack that builds deep contextual representations by attending to all positions simultaneously, forming the backbone of models like BERT.",
        "decoder": "An autoregressive decoder stack that generates sequences token-by-token, using causal self-attention and cross-attention to incorporate encoder context.",
        "residual": "A skip connection that allows gradients and information to bypass certain layers. This prevents signal degradation and allows the training of very deep networks.",
        "normalization": "Stabilizes the internal dynamics of the network by scaling features. It ensures that no single feature dominates the learning process, leading to faster convergence.",
        "activation": "Introduces non-linearity into the system, allowing the network to learn complex, non-linear mappings between inputs and outputs."
    }

    @classmethod
    def explain(cls, node_type: str, semantic_role: str, params: Dict[str, Any]) -> str:
        """
        Generate an educational explanation for a node.
        """
        # Specific overrides for high-fidelity ViT components
        if node_type == "patchembedding":
            ps = params.get("patch_size", 16)
            return f"Transforms the image into {ps}x{ps} patches. Each patch is projected into a vector (token), turning a 2D image into a 1D sequence for the Transformer."
        
        if node_type == "clstoken":
            return "Appends a special [CLS] token to the sequence. This token 'listens' to all other patches through attention and eventually represents the entire image's state."
            
        if node_type == "positionalembedding":
            return "Injects spatial awareness into the sequence. Since Transformers process tokens in parallel, this tells the model exactly where each patch or token belongs in the original input."

        if node_type == "token_embedding":
            return "Maps discrete vocabulary IDs into dense continuous vector representations, enabling the model to learn semantic relationships between words."
            
        if node_type == "segment_embedding":
            return "Distinguishes between different sequences (e.g., Sentence A vs Sentence B) to facilitate tasks like Natural Language Inference or Question Answering."

        if node_type == "causal_mask":
            return "Applies an autoregressive mask to block future tokens from being attended to. This ensures that token prediction depends only on past and present context."

        if semantic_role == "classifier_head" and params.get("is_next_token", False):
            return "A next-token prediction head that maps the contextualized decoder sequence representations back to the vocabulary space for autoregressive generation."

        if node_type in ["mhsa", "multiheadattention"]:
            causal = params.get("causal", False)
            if not causal:
                return "Bidirectional attention mechanisms allow the model to build deep contextual encoding by looking at both the left and right context simultaneously for sequence representation learning."
            else:
                return "Autoregressive causal attention prevents the model from looking ahead, forcing it to predict the next token based only on the past sequence, simulating human-like forward generation."

        if node_type == "feedforward":
            return "A point-wise MLP that processes each token independently. It helps in projecting features into a higher-dimensional space for more complex feature extraction."

        if node_type in ["cross_attention", "cross_attn"]:
            return (
                "Cross-attention routes Decoder queries to attend over the Encoder's full memory "
                "(Key/Value pairs). This is the mechanism by which the Decoder conditions its "
                "generation on the source sequence, forming the core of Encoder-Decoder architectures like T5."
            )

        if node_type in ["causal_attention", "causal_mask"]:
            return (
                "A causally-masked self-attention block that enforces the autoregressive constraint: "
                "each token can only attend to itself and tokens that came before it. "
                "This simulates left-to-right generation as used in GPT-style models."
            )

        if node_type in ["transformer_encoder", "encoder_block"]:
            num_heads = params.get("num_heads", "?")
            embed_dim = params.get("embed_dim", "?")
            return (
                f"A bidirectional Transformer encoder block with {num_heads} attention heads "
                f"and embedding dimension {embed_dim}. It encodes contextual relationships by "
                "attending to all positions simultaneously, building rich token representations "
                "for downstream tasks."
            )

        if node_type in ["transformer_decoder", "decoder_block"]:
            num_heads = params.get("num_heads", "?")
            return (
                f"A Transformer decoder block with {num_heads} attention heads. It combines "
                "causal self-attention (for left-to-right generation) with cross-attention "
                "(to attend to encoder memory), enabling sequence-to-sequence generation."
            )

        if node_type == "sequence_pooling":
            return (
                "Collapses the sequence dimension by aggregating all token representations into "
                "a single vector. Commonly used as a final step before classification, replacing "
                "the CLS-token approach in some architectures."
            )

        if node_type == "residual_add":
            return (
                "A residual addition (skip connection) that adds the block's input directly to "
                "its output. This creates an identity shortcut that ensures gradient flow and "
                "prevents representation collapse in very deep networks."
            )

        # Fallback to semantic role explanations
        return cls._EXPLANATIONS.get(semantic_role, f"A {node_type} component optimized for {semantic_role or 'architectural consistency'}.")
