"""
Tensor Flow Tracking Module.

Implements compiler-like validation by tracking abstract tensor shapes
(Batch, Channels, Height, Width) or (Batch, SeqLen, Dim) through the
ArchitectureGraph to ensure mathematical correctness and compatibility.
"""

from typing import Dict, Tuple, Optional, List, Any
from core.architecture_graph import ArchitectureGraph, GraphNode
from core.rag.flops_engine import FLOPsEngine
import logging

class TensorMismatchError(Exception):
    """Raised when tensor shapes are mathematically incompatible."""
    def __init__(self, message: str, node_id: str, upstream_path: List[str] = None):
        super().__init__(message)
        self.node_id = node_id
        self.upstream_path = upstream_path or []

# Configure logging for tensor flow tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorTracker")

class TensorTracker:
    def __init__(self):
        # Default starting shapes
        self.default_vision_shape = ("B", 3, 224, 224)
        self.default_sequence_shape = ("B", 196, 768)
        self.trace: List[str] = []
        # Structured cross-attention events for the visualizer
        self.cross_attention_events: List[Dict[str, Any]] = []
        # Per-layer FLOPs / memory results
        self.flops_events: List[Dict[str, Any]] = []
        self._flops_engine = FLOPsEngine()
        self.dependency_chains: Dict[str, List[str]] = {}

    # --- Utility Methods for Reusability ---
    
    def _validate_head_divisibility(self, node_id: str, dim: int, num_heads: int):
        """Ensure embedding dimension is divisible by the number of heads."""
        if num_heads <= 0:
            raise TensorMismatchError(f"Multi-head Error at {node_id}: num_heads must be > 0, got {num_heads}")
        if dim % num_heads != 0:
            raise TensorMismatchError(
                f"Multi-head Error at {node_id}: Dimension {dim} is not divisible by {num_heads} heads. "
                f"Each head would have a non-integer size. Transformer heads must divide the d_model dimension evenly.",
                node_id=node_id,
                upstream_path=self.dependency_chains.get(node_id, [])
            )

    def _resolve_reshape(self, node_id: str, in_shape: tuple, target_shape: list) -> tuple:
        """
        Validate and resolve a reshape operation.
        Supports '-1' as a wildcard for one dimension.
        """
        # Calculate total elements (ignoring Batch 'B')
        in_elements = 1
        for d in in_shape[1:]:
            if isinstance(d, int): in_elements *= d
            else: return tuple(target_shape) # Dynamic/Symbolic bypass
            
        out_elements = 1
        wildcard_idx = -1
        
        for i, d in enumerate(target_shape):
            if d == -1:
                if wildcard_idx != -1:
                    raise TensorMismatchError(f"Reshape Error at {node_id}: Multiple -1 wildcards in {target_shape}", node_id)
                wildcard_idx = i
            elif isinstance(d, int):
                out_elements *= d
                
        if wildcard_idx != -1:
            if in_elements % out_elements != 0:
                raise TensorMismatchError(f"Reshape Error at {node_id}: Cannot reshape {in_shape} to {target_shape}", node_id)
            target_shape[wildcard_idx] = in_elements // out_elements
        elif in_elements != out_elements:
             raise TensorMismatchError(f"Reshape Error at {node_id}: Total elements mismatch. In: {in_elements}, Out: {out_elements}", node_id)
             
        return tuple(target_shape)

    def log_step(self, node_id: str, node_type: str, in_shape: tuple, out_shape: tuple,
                 extra: Optional[Dict[str, Any]] = None):
        msg = f"[Trace] {node_id} ({node_type}): {in_shape} -> {out_shape}"
        self.trace.append(msg)
        logger.info(msg)

    def _log_flops(self, node: GraphNode, in_shape: tuple, out_shape: tuple):
        """Compute and store FLOPs/memory for a node via FLOPsEngine."""
        try:
            result = self._flops_engine.estimate(
                node_id=node.id,
                node_type=node.type,
                in_shape=in_shape,
                out_shape=out_shape,
                params=node.params or {},
                label=node.label,
            )
            self.flops_events.append(result.to_dict())
        except Exception:
            pass   # FLOPs engine is best-effort

    def propagate_shapes(self, graph: ArchitectureGraph, initial_shape: Optional[tuple] = None) -> None:
        """
        Perform a forward pass simulation through the graph to track tensor shapes.
        """
        self.trace = []
        self.cross_attention_events = []
        self.flops_events = []
        self.dependency_chains = {}
        
        if not graph.nodes:
            return

        if not initial_shape:
            if any(t in graph.nodes[0].type for t in ["attention", "transformer", "bert", "gpt"]):
                initial_shape = self.default_sequence_shape
            else:
                initial_shape = self.default_vision_shape

        shape_memory: Dict[str, tuple] = {}
        dependencies = {n.id: [] for n in graph.nodes}
        for edge in graph.edges:
            if edge.target in dependencies:
                dependencies[edge.target].append(edge.source)

        try:
            for node in graph.nodes:
                # 1. Gather Input Shapes and Build Dependency Chains
                input_shapes = []
                chain = []
                if not dependencies[node.id]:
                    input_shapes.append(initial_shape)
                else:
                    for src in dependencies[node.id]:
                        if src in shape_memory:
                            input_shapes.append(shape_memory[src])
                            chain.append(src)
                            chain.extend(self.dependency_chains.get(src, []))
                
                # Deduplicate and store chain
                self.dependency_chains[node.id] = list(set(chain))

                if not input_shapes:
                    continue

                # 2. Validation & Processing
                if len(input_shapes) > 1:
                    if node.type == "cross_attention":
                        if len(input_shapes) != 2:
                             raise TensorMismatchError(f"Cross-Attention Error: Expected 2 inputs (Q, KV), got {len(input_shapes)}", node.id, self.dependency_chains[node.id])
                        q_shape, kv_shape = input_shapes[0], input_shapes[1]
                        if q_shape[-1] != kv_shape[-1]:
                             raise TensorMismatchError(f"Cross-Attention Error: Embed dim mismatch. Q: {q_shape[-1]}, KV: {kv_shape[-1]}", node.id, self.dependency_chains[node.id])
                        node.input_shape = q_shape
                    elif node.type in ["mhsa", "multiheadattention", "attention_merge"]:
                        main_shape = input_shapes[0]
                        for s in input_shapes[1:]:
                            if s[-1] != main_shape[-1]:
                                raise TensorMismatchError(
                                    f"Cannot merge tensors: Dimension mismatch. Branch A: {main_shape[-1]}, Branch B: {s[-1]}",
                                    node.id, self.dependency_chains[node.id]
                                )
                        node.input_shape = main_shape
                    elif node.type == "attention_scores":
                        if len(input_shapes) != 2:
                             raise TensorMismatchError(f"Attention Scores Error: Expected 2 inputs (Q, K), got {len(input_shapes)}", node.id, self.dependency_chains[node.id])
                        node.input_shape = (input_shapes[0], input_shapes[1])
                    else:
                        main_shape = input_shapes[0]
                        for s in input_shapes[1:]:
                            if node.type == "residual_add" and s[1:] != main_shape[1:]:
                                raise TensorMismatchError(
                                    f"Residual connection failed because branch A outputs {main_shape} while branch B outputs {s}. Dimensions must match exactly for addition.",
                                    node.id, self.dependency_chains[node.id]
                                )
                        node.input_shape = main_shape
                else:
                    node.input_shape = input_shapes[0]

                # 3. Compute Output Shape
                node.output_shape = self._compute_output_shape(node, node.input_shape)
                shape_memory[node.id] = node.output_shape
                self.log_step(node.id, node.type, node.input_shape, node.output_shape)
                self._log_flops(node, node.input_shape, node.output_shape)

        except TensorMismatchError as e:
            # Catch and store failure info in metadata
            graph.metadata["failure"] = {
                "node_id": e.node_id,
                "message": str(e),
                "upstream_path": e.upstream_path
            }
            logger.warning(f"Propagation failed at {e.node_id}: {str(e)}")
            raise

        # Attach events to metadata
        graph.metadata["tensor_trace"] = self.trace
        graph.metadata["cross_attention_events"] = self.cross_attention_events
        graph.metadata["flops_events"] = self.flops_events

    def _compute_output_shape(self, node: GraphNode, in_shape: Any) -> tuple:
        """Apply mathematical rules for specific layers to determine output shape."""
        
        if node.type == "attention_scores":
            # in_shape is (q_shape, k_shape)
            # Example: q_shape = (B, H, N_q, D_H), k_shape = (B, H, N_k, D_H)
            q_shape, k_shape = in_shape
            if len(q_shape) != 4 or len(k_shape) != 4:
                raise TensorMismatchError(f"Attention Scores Error at {node.id}: Expected 4D inputs, got {q_shape} and {k_shape}", node.id)
            B, H, N_q, D_H = q_shape
            _, _, N_k, _ = k_shape
            return (B, H, N_q, N_k)

        # Standard processing for single input shapes
        if not isinstance(in_shape, tuple) or (len(in_shape) > 0 and isinstance(in_shape[0], tuple)):
            # Fallback if somehow we got here with weird types
            return in_shape

        # --- Strict Dimensionality Checks (ViT specific) ---
        if node.type == "clstoken":
            if len(in_shape) != 3:
                raise TensorMismatchError(f"CLS Token Error at {node.id}: Expected 3D input (B, N, D), got {in_shape}", node.id)
            return (in_shape[0], in_shape[1] + 1, in_shape[2])

        # Sequence Embeddings (B, N) -> (B, N, D) or (B, N, D) -> (B, N, D)
        if node.type in ["token_embedding", "positionalembedding", "segment_embedding", "positional_embedding"]:
            expected_dim = node.params.get("embed_dim") or node.params.get("embedding_dim", 768)
            if len(in_shape) == 2:
                # E.g. (B, SeqLen) input -> (B, SeqLen, D)
                return (in_shape[0], in_shape[1], expected_dim)
            elif len(in_shape) == 3:
                if in_shape[2] != expected_dim:
                    raise TensorMismatchError(f"Embedding Error at {node.id}: Dimension mismatch. Expected dim {expected_dim}, got {in_shape[2]}", node.id)
                return in_shape
            else:
                 raise TensorMismatchError(f"Embedding Error at {node.id}: Expected 2D or 3D input, got {in_shape}", node.id)

        # Handle 4D Vision Tensors: (B, C, H, W)
        if len(in_shape) == 4:
            B, C, H, W = in_shape
            
            if node.type == "patchembedding":
                patch_size = node.params.get("patch_size", 16)
                embed_dim = node.params.get("embed_dim", 768)
                num_patches = (H // patch_size) * (W // patch_size)
                return (B, num_patches, embed_dim)
            
            if node.type in ["conv2d", "conv1d", "conv"]:
                out_c = node.params.get("channels", node.params.get("out_channels", node.params.get("filters", C)))
                stride = node.params.get("stride", 1)
                # Approximate spatial reduction
                out_h = H // stride
                out_w = W // stride
                return (B, out_c, out_h, out_w)
            
            if node.type.lower() in ["residualblock", "block", "stage"]:
                # Residual blocks may downsample; read stride + out_channels if available
                out_c = node.params.get("channels", node.params.get("out_channels", node.params.get("filters", C)))
                stride = node.params.get("stride", 1)
                return (B, out_c, max(1, H // stride), max(1, W // stride))
                
            if node.type in ["maxpool2d", "avgpool2d"]:
                stride = node.params.get("stride", 2)
                return (B, C, H // stride, W // stride)
                
            if node.type == "globalavgpool2d":
                return (B, C, 1, 1)
                
            if node.type == "linear" or node.type == "dense":
                out_features = node.params.get("hidden_size", node.params.get("channels", 1000))
                # Auto-flatten rule: If 4D tensor hits Linear, it flattens.
                return (B, out_features)
                
            if node.type == "flatten":
                return (B, H * W, C)

            if node.type == "reshape":
                target = node.params.get("shape")
                if not target: return in_shape
                return self._resolve_reshape(node.id, in_shape, list(target))
                
            if node.type == "causal_mask":
                # For 4D attention scores (B, H, N_q, N_k)
                if in_shape[2] != in_shape[3] and node.params.get("strict_square", True):
                    raise TensorMismatchError(f"Causal Mask Error at {node.id}: Expected square attention matrix for masking, got {in_shape[2]}x{in_shape[3]}", node.id)
                return in_shape

        # Handle 2D/3D/4D/5D generic Transformer Tensors
        B = in_shape[0]
        
        # --- Multi-Head Splitting/Merging ---
        if node.type == "split_heads":
            # (B, N, D) -> (B, H, N, D/H)
            num_heads = node.params.get("num_heads", 8)
            dim = in_shape[-1]
            self._validate_head_divisibility(node.id, dim, num_heads)
            return (B, num_heads, in_shape[1], dim // num_heads)

        if node.type == "merge_heads":
            # (B, H, N, D_H) -> (B, N, H * D_H)
            if len(in_shape) != 4:
                raise TensorMismatchError(f"Merge Heads Error at {node.id}: Expected 4D input (B, H, N, D_H), got {in_shape}", node.id)
            return (B, in_shape[2], in_shape[1] * in_shape[3])

        # --- Dimension Manipulation ---
        if node.type == "transpose":
            dims = node.params.get("dims", (1, 2))
            if len(dims) != 2: raise TensorMismatchError(f"Transpose Error at {node.id}: Expected 2 dimensions to swap.", node.id)
            res = list(in_shape)
            res[dims[0]], res[dims[1]] = res[dims[1]], res[dims[0]]
            return tuple(res)

        if node.type == "reshape":
            target = node.params.get("shape")
            if not target: return in_shape
            return self._resolve_reshape(node.id, in_shape, list(target))

        # --- Sequence / Feature Operations ---
        if node.type in ["linear", "query_projection", "key_projection", "value_projection", "attention_merge"]:
            out_features = node.params.get("hidden_size", node.params.get("channels", in_shape[-1]))
            return (*in_shape[:-1], out_features)
            
        if node.type == "feedforward":
            out_features = node.params.get("embed_dim", node.params.get("hidden_size", in_shape[-1]))
            return (*in_shape[:-1], out_features)
            
        if node.type in ["multiheadattention", "transformerblock", "mhsa", "cross_attention"]:
            num_heads = node.params.get("num_heads", 8)
            dim = in_shape[-1]
            self._validate_head_divisibility(node.id, dim, num_heads)
            
            causal = node.params.get("causal", False)
            if causal and node.type == "cross_attention":
                raise TensorMismatchError(f"Cross-Attention Error at {node.id}: Cross-attention typically shouldn't be causal autoregressive.", node.id)
            return in_shape

        if node.type in ["residual_add", "layernorm"]:
            return in_shape


        if node.type in ["sequence_pooling", "globalavgpool", "global_pool"]:
            if len(in_shape) == 3:
                # (B, N, D) -> (B, D)
                return (in_shape[0], in_shape[2])
            elif len(in_shape) == 4:
                # (B, C, H, W) -> (B, C)
                return (in_shape[0], in_shape[1])

        if node.type == "concat":
            # (B, N1, D) + (B, N2, D) -> (B, N1+N2, D)
            # This logic assumes we handle multi-input elsewhere or via list of shapes
            # For simplicity, we assume sequence concatenation unless dim specified
            dim = node.params.get("dim", 1)
            # Validation logic would normally be in propagate_shapes merge block
            # but we return a predicted shape here
            return in_shape 

        # --- Safety Check ---
        if len(in_shape) in [2, 3]:
            if node.type == "conv2d":
                raise TensorMismatchError(f"Tensor Shape Error: Cannot pass flat tensor {in_shape} into spatial Conv2D layer ({node.id}).", node.id)

        # Fallback (Pass-through)
        return in_shape
