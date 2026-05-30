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
    def explain(cls, node_type: str, semantic_role: str, params: Dict[str, Any], context: Dict[str, Any] = None) -> str:
        """
        Generate an educational explanation for a node.
        """
        context = context or {}
        index = context.get("index", 0)
        total = context.get("total", 1)
        is_repeated = context.get("is_repeated", False)
        
        # Determine depth characteristic
        if index < total / 3:
            depth_desc = f" Operating at step {index} of {total} (an early stage), this module focuses on capturing low-level, local features."
        elif index > 2 * total / 3:
            depth_desc = f" Operating at step {index} of {total} (a deep stage), this module focuses on extracting high-level, global semantics."
        else:
            depth_desc = f" Operating at step {index} of {total} (an intermediate stage), this module iteratively refines representations."

        exp = None
        
        # Specific overrides for high-fidelity ViT components
        if node_type == "patchembedding":
            ps = params.get("patch_size", 16)
            exp = f"Transforms the image into {ps}x{ps} patches. Each patch is projected into a vector (token), turning a 2D image into a 1D sequence for the Transformer."
        elif node_type == "clstoken":
            exp = "Appends a special [CLS] token to the sequence. This token 'listens' to all other patches through attention and eventually represents the entire image's state."
        elif node_type == "positionalembedding":
            exp = "Injects spatial awareness into the sequence. Since Transformers process tokens in parallel, this tells the model exactly where each patch or token belongs in the original input."
        elif node_type == "token_embedding":
            exp = "Maps discrete vocabulary IDs into dense continuous vector representations, enabling the model to learn semantic relationships between words."
        elif node_type == "segment_embedding":
            exp = "Distinguishes between different sequences (e.g., Sentence A vs Sentence B) to facilitate tasks like Natural Language Inference or Question Answering."
        elif node_type == "causal_mask":
            exp = "Applies an autoregressive mask to block future tokens from being attended to. This ensures that token prediction depends only on past and present context."
        elif semantic_role == "classifier_head" and params.get("is_next_token", False):
            exp = "A next-token prediction head that maps the contextualized decoder sequence representations back to the vocabulary space for autoregressive generation."
        elif node_type in ["mhsa", "multiheadattention"]:
            causal = params.get("causal", False)
            if not causal:
                exp = "Bidirectional attention mechanisms allow the model to build deep contextual encoding by looking at both the left and right context simultaneously for sequence representation learning."
            else:
                exp = "Autoregressive causal attention prevents the model from looking ahead, forcing it to predict the next token based only on the past sequence, simulating human-like forward generation."

        elif node_type == "feedforward":
            exp = "A point-wise MLP that processes each token independently. It helps in projecting features into a higher-dimensional space for more complex feature extraction."
        elif node_type in ["cross_attention", "cross_attn"]:
            exp = (
                "Cross-attention routes Decoder queries to attend over the Encoder's full memory "
                "(Key/Value pairs). This is the mechanism by which the Decoder conditions its "
                "generation on the source sequence, forming the core of Encoder-Decoder architectures like T5."
            )
        elif node_type in ["causal_attention", "causal_mask"]:
            exp = (
                "A causally-masked self-attention block that enforces the autoregressive constraint: "
                "each token can only attend to itself and tokens that came before it. "
                "This simulates left-to-right generation as used in GPT-style models."
            )
        elif node_type in ["transformer_encoder", "encoder_block"]:
            num_heads = params.get("num_heads", "?")
            embed_dim = params.get("embed_dim", "?")
            exp = (
                f"A bidirectional Transformer encoder block with {num_heads} attention heads "
                f"and embedding dimension {embed_dim}. It encodes contextual relationships by "
                "attending to all positions simultaneously, building rich token representations "
                "for downstream tasks."
            )
        elif node_type in ["transformer_decoder", "decoder_block"]:
            num_heads = params.get("num_heads", "?")
            exp = (
                f"A Transformer decoder block with {num_heads} attention heads. It combines "
                "causal self-attention (for left-to-right generation) with cross-attention "
                "(to attend to encoder memory), enabling sequence-to-sequence generation."
            )
        elif node_type == "sequence_pooling":
            exp = (
                "Collapses the sequence dimension by aggregating all token representations into "
                "a single vector. Commonly used as a final step before classification, replacing "
                "the CLS-token approach in some architectures."
            )
        elif node_type == "residual_add":
            exp = (
                "A residual addition (skip connection) that adds the block's input directly to "
                "its output. This creates an identity shortcut that ensures gradient flow and "
                "prevents representation collapse in very deep networks."
            )
        elif node_type in ["residualblock", "block", "stage"]:
            ch = params.get("channels", "?")
            stride = params.get("stride", 1)
            exp = (
                f"A residual block operating on {ch}-channel feature maps with stride {stride}. "
                "It applies two or three convolutions and adds the input directly to the output "
                "via a shortcut connection. This design allows gradients to flow backward through "
                "the identity path, solving the vanishing gradient problem and enabling training "
                "of networks with hundreds of layers."
            )
        elif node_type in ("conv2d", "conv1d", "conv3d"):
            ch = params.get("channels", params.get("out_channels", params.get("filters", "?")))
            k = params.get("kernel_size", 3)
            exp = (
                f"A {k}×{k} convolutional layer producing {ch}-channel output feature maps. "
                "It slides a learned filter across the spatial dimensions, detecting local patterns "
                "such as edges, textures, or semantic regions. Stacking many filters builds a rich "
                "hierarchy of visual representations progressing from simple to complex."
            )
        elif node_type == "maxpool2d":
            exp = (
                "A max-pooling layer that halves the spatial resolution by retaining only the "
                "maximum activation in each pooling window. This induces spatial invariance to "
                "small translations and reduces the memory and compute requirements for all "
                "subsequent layers by compressing the feature map footprint."
            )
        elif node_type in ["avgpool2d", "adaptiveavgpool2d"]:
            exp = (
                "A global average pooling layer that collapses each feature map to a single scalar "
                "by averaging all spatial positions. This converts spatial feature maps of any size "
                "into a fixed-length vector, eliminating the need for fully-connected flattening "
                "and dramatically reducing the number of parameters in the classifier head."
            )
        elif node_type == "linear":
            hidden = params.get("hidden_size", params.get("out_features", "?"))
            exp = (
                f"A fully-connected linear layer projecting to {hidden} output dimensions. "
                "It applies a learned affine transformation (W·x + b) to the input vector. "
                "As the classification head, it maps the abstract feature representation to "
                "class logits from which probabilities are derived via softmax."
            )
        elif node_type in ["upsample", "convtranspose2d", "upconvolution"]:
            exp = (
                "An upsampling layer that doubles the spatial resolution of the feature map, "
                "restoring detail lost during the contracting (encoding) path. In U-Net, this is "
                "the first step of each decoder block; the expanded map is then concatenated with "
                "the corresponding encoder skip connection, fusing coarse semantics with "
                "fine-grained spatial detail for precise pixel-level prediction."
            )
        elif node_type == "patchembedding":
            ps = params.get("patch_size", 16)
            d = params.get("embed_dim", params.get("channels", "?"))
            exp = (
                f"Divides the input image into non-overlapping {ps}×{ps} pixel patches and "
                f"linearly projects each patch into a {d}-dimensional embedding vector. "
                "This converts the 2D image into a 1D sequence of tokens, the required input "
                "format for the Transformer encoder, analogous to word tokens in NLP models."
            )
            
        if exp:
            return depth_desc + " " + exp

        # Fallback to semantic role explanations
        base_exp = cls._EXPLANATIONS.get(semantic_role)
        if base_exp:
            return depth_desc + " " + base_exp
            
        node_type_clean = (node_type or "Unknown").replace("_", " ").title()
        role_clean = (semantic_role or "feature processing").replace("_", " ")
        
        if role_clean == "unknown":
            role_clean = "feature transformation and aggregation"
            
        return depth_desc + f" A **{node_type_clean}** component optimized for {role_clean}. It processes the incoming tensor to modify its dimensions or extract relevant patterns for downstream use."
