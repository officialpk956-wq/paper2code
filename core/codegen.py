"""
PyTorch code generation from architectures.

F5: Generates runnable PyTorch nn.Module code from:
- Hardcoded models (loads builder source)
- Dynamic architectures (generates skeleton)
"""

import importlib
import inspect
import re

from core.architecture_graph import ArchitectureGraph

_BUILDER_MAP = {
    "ResNet-18": ("core.model_builder", "ResNetBuilder"),
    "U-Net": ("core.unet_builder", "UNetBuilder"),
    "Vision Transformer": ("core.vit_builder", "ViTBuilder"),
    "ViT": ("core.vit_builder", "ViTBuilder"),
}


def get_pytorch_code(model_name: str, graph: ArchitectureGraph) -> str:
    """
    Generate PyTorch code for an architecture.

    For hardcoded models, loads the actual builder source code.
    For dynamic models, generates a skeleton nn.Module.

    Args:
        model_name: Name of the model (e.g., "ResNet-18", or custom name from graph)
        graph: ArchitectureGraph to generate code for

    Returns:
        Python code as string (ready to copy/execute)
    """
    # Try to load builder source for hardcoded models
    if model_name in _BUILDER_MAP:
        try:
            mod_path, cls_name = _BUILDER_MAP[model_name]
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            source = inspect.getsource(cls)
            return f"# Builder for: {model_name}\n\n{source}"
        except Exception:
            pass

    # Fall back to skeleton generation
    return _generate_skeleton(graph)


def _generate_skeleton(graph: ArchitectureGraph) -> str:
    """
    Generate a sequential PyTorch nn.Module skeleton from graph nodes.
    """
    # Sanitize model name for Python class
    name = re.sub(r"[^A-Za-z0-9]", "", graph.name) or "CustomModel"
    if name[0].isdigit():
        name = "_" + name

    lines = [
        "# NOTE: Skip connections not represented in this sequential skeleton",
        "import torch",
        "import torch.nn as nn",
        "",
        "class ViTPatchEmbed(nn.Module):",
        "    def __init__(self, in_channels, embed_dim, patch_size):",
        "        super().__init__()",
        "        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)",
        "    def forward(self, x):",
        "        return self.proj(x).flatten(2).transpose(1, 2)",
        "",
        f"class {name}(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]

    # Build __init__ layer definitions
    forward_calls = []

    for node in graph.nodes:
        layer_str = _node_to_layer(node)
        if layer_str:
            lines.append(f"        self.{node.id} = {layer_str}")
        forward_calls.append(f"        x = self.{node.id}(x)  # {node.label}")

    # Build forward method
    forward = ["", "    def forward(self, x):"]
    forward.extend(forward_calls)
    forward.append("        return x")

    return "\n".join(lines + forward)


def _node_to_layer(node) -> str:
    """
    Convert a GraphNode to its nn.* constructor call.

    Maps canonical types to PyTorch layers with estimated parameters.
    """
    p = node.params or {}
    ch = p.get("channels", p.get("filters", 64))
    k = p.get("kernel_size", 3)
    in_hs = (
        node.input_shape[-1]
        if hasattr(node, "input_shape") and isinstance(node.input_shape[-1], int)
        else 512
    )
    out_hs = p.get("hidden_size", 512)
    heads = p.get("num_heads", p.get("heads", 8))

    # Map types to nn.* constructors
    MAP = {
        "conv2d": f"nn.Conv2d({ch}, {ch}, kernel_size={k}, padding={k // 2})",
        "conv1d": f"nn.Conv1d({ch}, {ch}, kernel_size={k}, padding={k // 2})",
        "linear": f"nn.Linear({in_hs}, {out_hs})",
        "relu": "nn.ReLU(inplace=True)",
        "gelu": "nn.GELU()",
        "sigmoid": "nn.Sigmoid()",
        "maxpool2d": "nn.MaxPool2d(kernel_size=2, stride=2)",
        "avgpool2d": "nn.AvgPool2d(kernel_size=2, stride=2)",
        "batchnorm2d": f"nn.BatchNorm2d({ch})",
        "batchnorm1d": f"nn.BatchNorm1d({ch})",
        "layernorm": f"nn.LayerNorm({in_hs})",
        "dropout": "nn.Dropout(p=0.1)",
        "transpose": "lambda x: x.transpose(1, 2)",
        "upsample": "nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)",
        "multiheadattention": f"nn.MultiheadAttention(embed_dim={in_hs}, num_heads={heads}, batch_first=True)",
        "transformerblock": f"nn.TransformerEncoderLayer(d_model={in_hs}, nhead={heads}, batch_first=True)",
        "patchembedding": f"ViTPatchEmbed(3, {p.get('embed_dim', 768)}, {p.get('patch_size', 16)})",
        "sequence_pooling": "lambda x: x.mean(dim=1)",
    }

    return MAP.get(node.type, None)
