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
            return "Injects spatial awareness into the sequence. Since Transformers process tokens in parallel, this tells the model exactly where each patch belongs in the original grid."

        if node_type == "mhsa":
            return "Multi-Head Self-Attention allows each patch to attend to every other patch simultaneously, enabling the model to understand global context and object relationships."

        if node_type == "feedforward":
            return "A point-wise MLP that processes each token independently. It helps in projecting features into a higher-dimensional space for more complex feature extraction."

        # Fallback to semantic role explanations
        return cls._EXPLANATIONS.get(semantic_role, f"A {node_type} component optimized for {semantic_role or 'architectural consistency'}.")
