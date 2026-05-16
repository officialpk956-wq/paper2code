"""
Tensor Flow Tracking Module.

Implements compiler-like validation by tracking abstract tensor shapes
(Batch, Channels, Height, Width) or (Batch, SeqLen, Dim) through the
ArchitectureGraph to ensure mathematical correctness and compatibility.
"""

from typing import Dict, Tuple, Optional, List, Any
from src.architecture_graph import ArchitectureGraph, GraphNode
import logging

class TensorMismatchError(Exception):
    """Raised when tensor shapes are mathematically incompatible."""
    pass

# Configure logging for tensor flow tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TensorTracker")

class TensorTracker:
    def __init__(self):
        # Default starting shape if unknown (B, C, H, W)
        self.default_vision_shape = ("B", 3, 224, 224)
        self.default_sequence_shape = ("B", 196, 768)
        self.trace: List[str] = []

    # --- Utility Methods for Reusability ---
    
    def _validate_head_divisibility(self, node_id: str, dim: int, num_heads: int):
        """Ensure embedding dimension is divisible by the number of heads."""
        if num_heads <= 0:
            raise TensorMismatchError(f"Multi-head Error at {node_id}: num_heads must be > 0, got {num_heads}")
        if dim % num_heads != 0:
            raise TensorMismatchError(
                f"Multi-head Error at {node_id}: Dimension {dim} is not divisible by {num_heads} heads. "
                f"Each head would have a non-integer size."
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
                    raise TensorMismatchError(f"Reshape Error at {node_id}: Multiple -1 wildcards in {target_shape}")
                wildcard_idx = i
            elif isinstance(d, int):
                out_elements *= d
                
        if wildcard_idx != -1:
            if in_elements % out_elements != 0:
                raise TensorMismatchError(f"Reshape Error at {node_id}: Cannot reshape {in_shape} to {target_shape}")
            target_shape[wildcard_idx] = in_elements // out_elements
        elif in_elements != out_elements:
             raise TensorMismatchError(f"Reshape Error at {node_id}: Total elements mismatch. In: {in_elements}, Out: {out_elements}")
             
        return tuple(target_shape)

    def log_step(self, node_id: str, node_type: str, in_shape: tuple, out_shape: tuple):
        msg = f"[Trace] {node_id} ({node_type}): {in_shape} -> {out_shape}"
        self.trace.append(msg)
        logger.info(msg)

    def propagate_shapes(self, graph: ArchitectureGraph, initial_shape: Optional[tuple] = None) -> None:
        """
        Perform a forward pass simulation through the graph to track tensor shapes.
        Validates connections and prevents impossible topologies.
        """
        if not graph.nodes:
            return

        # Infer initial shape from first node if not provided
        if not initial_shape:
            if "attention" in graph.nodes[0].type or "transformer" in graph.nodes[0].type:
                initial_shape = self.default_sequence_shape
            else:
                initial_shape = self.default_vision_shape

        # A mapping of node ID to its resolved output shape
        shape_memory: Dict[str, tuple] = {}
        
        # Build dependency graph
        dependencies = {n.id: [] for n in graph.nodes}
        for edge in graph.edges:
            if edge.target in dependencies:
                dependencies[edge.target].append(edge.source)

        for node in graph.nodes:
            # 1. Gather Input Shapes
            input_shapes = []
            if not dependencies[node.id]:
                input_shapes.append(initial_shape)
            else:
                for src in dependencies[node.id]:
                    if src in shape_memory:
                        input_shapes.append(shape_memory[src])

            if not input_shapes:
                continue

            # If node receives multiple inputs (e.g. Skip Connection / Concat)
            if len(input_shapes) > 1:
                # Merge shapes (for now assume add/residual merges to same shape)
                main_shape = input_shapes[0]
                for s in input_shapes[1:]:
                    if s != main_shape:
                        # Stricter check: All dimensions except Batch must match
                        if s[1:] != main_shape[1:]:
                            raise TensorMismatchError(
                                f"Topology Error at {node.id} ({node.type}): "
                                f"Cannot merge tensors with different spatial dimensions: {main_shape} vs {s}"
                            )
                node.input_shape = main_shape

            else:
                node.input_shape = input_shapes[0]

            # 2. Compute Output Shape
            node.output_shape = self._compute_output_shape(node, node.input_shape)
            shape_memory[node.id] = node.output_shape
            self.log_step(node.id, node.type, node.input_shape, node.output_shape)
            
        # Attach trace to graph metadata for inspection
        graph.metadata["tensor_trace"] = self.trace

    def _compute_output_shape(self, node: GraphNode, in_shape: tuple) -> tuple:
        """Apply mathematical rules for specific layers to determine output shape."""
        
        # --- Strict Dimensionality Checks (ViT specific) ---
        if node.type == "clstoken":
            if len(in_shape) != 3:
                raise TensorMismatchError(f"CLS Token Error at {node.id}: Expected 3D input (B, N, D), got {in_shape}")
            return (in_shape[0], in_shape[1] + 1, in_shape[2])

        if node.type == "positionalembedding":
            if len(in_shape) != 3:
                raise TensorMismatchError(f"Positional Embedding Error at {node.id}: Expected 3D input (B, N, D), got {in_shape}")
            # Validate consistency if parameters are provided
            expected_dim = node.params.get("embed_dim") or node.params.get("embedding_dim")
            if expected_dim and expected_dim != in_shape[2]:
                raise TensorMismatchError(f"Positional Embedding Error at {node.id}: Dimension mismatch. Expected dim {expected_dim}, got {in_shape[2]}")
            return in_shape

        # Handle 4D Vision Tensors: (B, C, H, W)
        if len(in_shape) == 4:
            B, C, H, W = in_shape
            
            if node.type == "patchembedding":
                patch_size = node.params.get("patch_size", 16)
                embed_dim = node.params.get("embed_dim", 768)
                num_patches = (H // patch_size) * (W // patch_size)
                return (B, num_patches, embed_dim)
            
            if node.type in ["conv2d", "conv1d"]:
                out_c = node.params.get("channels", C)
                stride = node.params.get("stride", 1)
                # Approximate spatial reduction
                out_h = H // stride
                out_w = W // stride
                return (B, out_c, out_h, out_w)
                
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
                raise TensorMismatchError(f"Merge Heads Error at {node.id}: Expected 4D input (B, H, N, D_H), got {in_shape}")
            return (B, in_shape[2], in_shape[1] * in_shape[3])

        # --- Dimension Manipulation ---
        if node.type == "transpose":
            dims = node.params.get("dims", (1, 2))
            if len(dims) != 2: raise TensorMismatchError(f"Transpose Error at {node.id}: Expected 2 dimensions to swap.")
            res = list(in_shape)
            res[dims[0]], res[dims[1]] = res[dims[1]], res[dims[0]]
            return tuple(res)

        if node.type == "reshape":
            target = node.params.get("shape")
            if not target: return in_shape
            return self._resolve_reshape(node.id, in_shape, list(target))

        # --- Sequence / Feature Operations ---
        if node.type in ["linear", "query_projection", "key_projection", "value_projection", "attention_merge", "feedforward"]:
            out_features = node.params.get("hidden_size", node.params.get("channels", in_shape[-1]))
            return (*in_shape[:-1], out_features)
            
        if node.type in ["multiheadattention", "transformerblock", "mhsa"]:
            num_heads = node.params.get("num_heads", 8)
            dim = in_shape[-1]
            self._validate_head_divisibility(node.id, dim, num_heads)
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
                raise TensorMismatchError(f"Tensor Shape Error: Cannot pass flat tensor {in_shape} into spatial Conv2D layer ({node.id}).")


        # Fallback (Pass-through)
        return in_shape
