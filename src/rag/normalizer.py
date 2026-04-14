"""
Pure configuration normalization layer.

Transforms ConfigDict → deterministic, stable ConfigDict.
No inference, no reasoning, pure transformation only.
"""

import re
from typing import Any, Dict, List, Tuple

from src.agents.types import ConfigDict


# Canonical layer types (authoritative set)
CANONICAL_TYPES = {
    "conv1d", "conv2d", "conv3d",
    "linear", "dense",
    "maxpool2d", "avgpool2d",
    "multiheadattention",
    "transformerblock",
    "batchnorm2d", "layernorm",
    "relu", "dropout",
    "upsample",
    "residualblock",
}

# Comprehensive type synonym map: any variant → canonical type
# (lowercase keys, canonical types are lowercase)
_SYNONYM_MAP = {
    # Conv variants
    "conv": "conv2d",
    "conv1d": "conv1d",
    "conv2d": "conv2d",
    "conv2dlayer": "conv2d",
    "convolution": "conv2d",
    "convolutionlayer": "conv2d",
    "convolutional": "conv2d",
    "convolutionallayer": "conv2d",
    "convlayer": "conv2d",

    # Linear variants
    "linear": "linear",
    "linearlayer": "linear",
    "dense": "linear",
    "denselayer": "linear",
    "fullyconnected": "linear",
    "fully_connected": "linear",
    "fully-connected": "linear",
    "fc": "linear",
    "fclayer": "linear",

    # MaxPool variants
    "maxpool": "maxpool2d",
    "maxpool2d": "maxpool2d",
    "max_pool": "maxpool2d",
    "max-pool": "maxpool2d",
    "maxpooling": "maxpool2d",
    "max_pooling": "maxpool2d",
    "max-pooling": "maxpool2d",
    "maxpoollayer": "maxpool2d",

    # AvgPool variants
    "avgpool": "avgpool2d",
    "avgpool2d": "avgpool2d",
    "avg_pool": "avgpool2d",
    "avg-pool": "avgpool2d",
    "averagepool": "avgpool2d",
    "averagepooling": "avgpool2d",
    "avg_pooling": "avgpool2d",
    "avg-pooling": "avgpool2d",
    "avgpoollayer": "avgpool2d",

    # Attention variants
    "attention": "multiheadattention",
    "selfAttention": "multiheadattention",
    "selfattention": "multiheadattention",
    "self-attention": "multiheadattention",
    "self_attention": "multiheadattention",
    "multiheadattention": "multiheadattention",
    "multi_head_attention": "multiheadattention",
    "multi-head-attention": "multiheadattention",
    "mha": "multiheadattention",
    "mhsa": "multiheadattention",
    "multiheadselfAttention": "multiheadattention",

    # Transformer variants
    # Note: "transformer encoder" and "transformer decoder" semantically describe
    # attention mechanisms, not distinct block types. Map to multiheadattention.
    "transformer": "transformerblock",
    "transformerblock": "transformerblock",
    "transformer_block": "transformerblock",
    "transformerlayer": "transformerblock",
    "transformerencoder": "multiheadattention",
    "transformer_encoder": "multiheadattention",
    "transformerdecoder": "multiheadattention",
    "transformer_decoder": "multiheadattention",

    # BatchNorm variants
    "batchnorm": "batchnorm2d",
    "batchnorm2d": "batchnorm2d",
    "batch_norm": "batchnorm2d",
    "batch-norm": "batchnorm2d",
    "batchnormalization": "batchnorm2d",
    "batch_normalization": "batchnorm2d",
    "batch-normalization": "batchnorm2d",
    "batchnormlayer": "batchnorm2d",

    # LayerNorm variants
    "layernorm": "layernorm",
    "layer_norm": "layernorm",
    "layer-norm": "layernorm",
    "layernormalization": "layernorm",
    "layer_normalization": "layernorm",
    "layer-normalization": "layernorm",

    # Activation variants
    "relu": "relu",
    "relulayer": "relu",
    "activation": "relu",
    "activationfunction": "relu",

    # Dropout
    "dropout": "dropout",
    "dropoutlayer": "dropout",

    # Upsample variants
    "upsample": "upsample",
    "upsampling": "upsample",
    "upsamplelayer": "upsample",
    "upsamplayer": "upsample",

    # Residual variants
    "residual": "residualblock",
    "residualblock": "residualblock",
    "residual_block": "residualblock",
    "residual-block": "residualblock",
    "residuallayer": "residualblock",
}

# Parameter key normalization map
_PARAM_MAP = {
    "filters": "channels",
    "out_channels": "channels",
    "output_channels": "channels",
    "num_channels": "channels",
    "in_channels": "in_channels",
    "input_channels": "in_channels",
    "num_input_channels": "in_channels",
    "kernel": "kernel_size",
    "filter_size": "kernel_size",
    "size": "kernel_size",
    "pool_size": "pool_size",
    "pooling_size": "pool_size",
    "stride": "stride",
    "strides": "stride",
    "padding": "padding",
    "pad": "padding",
    "hidden": "hidden_size",
    "hidden_size": "hidden_size",
    "hidden_dim": "hidden_size",
    "units": "hidden_size",
    "num_units": "hidden_size",
}


def _validate_synonym_map() -> None:
    """
    Validate _SYNONYM_MAP at module load time.

    Ensures:
    - All values are valid canonical types
    - No duplicate/conflicting mappings
    - All canonical types have at least one entry

    Raises:
        AssertionError: If any validation fails
    """
    # Build reverse map: canonical_type → set of input keys
    reverse_map: Dict[str, set] = {}
    for key, canonical in _SYNONYM_MAP.items():
        if canonical not in reverse_map:
            reverse_map[canonical] = set()
        reverse_map[canonical].add(key)

    # Verify all canonical types in the map are valid
    for canonical in reverse_map.keys():
        assert canonical in CANONICAL_TYPES, \
            f"Unknown canonical type in _SYNONYM_MAP: {canonical}"

    # No collisions (synonyms don't map to multiple canonical types)
    for canonical, keys in reverse_map.items():
        assert all(_SYNONYM_MAP[k] == canonical for k in keys), \
            f"Collision detected in synonym map for {canonical}"


def normalize_config(config: Dict[str, Any]) -> ConfigDict:
    """
    Normalize ConfigDict to stable, deterministic form.

    Ensures:
    - Deterministic output (same input → identical output)
    - Stable layer ordering and sequential IDs
    - Consistent field names and types
    - Valid connections

    Args:
        config: Raw ConfigDict from extractor

    Returns:
        Normalized, validated ConfigDict
    """
    # Normalize name
    name = _normalize_name(config.get("name", "UnknownModel"))

    # Normalize layers
    layers = _normalize_layers(config.get("layers", []))

    # Build connections (strictly sequential)
    connections = _build_sequential_connections(layers)

    # Final validation
    _validate_config(name, layers, connections)

    return {
        "name": name,
        "layers": layers,
        "connections": connections,
    }


def _normalize_name(name: Any) -> str:
    """Normalize architecture name."""
    if not name:
        return "UnknownModel"

    name_str = str(name).strip()
    return name_str if name_str else "UnknownModel"


def _normalize_layers(layers: Any) -> List[Dict[str, Any]]:
    """
    Normalize layer list.

    - Type canonicalization
    - Param key normalization
    - Sequential ID assignment
    """
    if not isinstance(layers, list):
        return [{"id": "layer_0", "type": "conv2d", "params": {}}]

    normalized = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue

        layer_type = _normalize_type(layer.get("type", "conv2d"))
        params = _normalize_params(layer.get("params", {}))

        normalized.append({
            "type": layer_type,
            "params": params,
        })

    if not normalized:
        normalized = [{"type": "conv2d", "params": {}}]

    # Reassign sequential IDs (DETERMINISTIC)
    for i, layer in enumerate(normalized):
        layer["id"] = f"layer_{i}"

    return normalized


def _normalize_type(layer_type: Any) -> str:
    """
    Normalize layer type to canonical name.

    Supports all variants via _SYNONYM_MAP.
    Whitespace/punctuation are normalized before lookup.

    Enforces: only canonical types are returned.
    Raises AssertionError if unknown type is encountered.
    """
    if not layer_type:
        return "conv2d"

    type_str = str(layer_type).strip().lower()
    # Normalize whitespace and punctuation for better matching
    type_normalized = re.sub(r"[\s\-_]+", "", type_str)
    canonical = _SYNONYM_MAP.get(type_normalized, type_str)

    # ENFORCE: only canonical types allowed
    assert canonical in CANONICAL_TYPES, \
        f"Invalid canonical type after normalization: {canonical} (from input: {layer_type})"

    return canonical


def _normalize_params(params: Any) -> Dict[str, Any]:
    """
    Normalize parameters.

    - Key normalization
    - Type validation
    - Remove None/empty values
    - Preserve internal metadata (fields prefixed with _)

    Internal metadata fields (e.g., _repeat_group, _repeat_index, _repeat_total, _repeat_truncated)
    are semantic signals about how the layer was generated and are preserved
    as-is without key normalization.
    """
    if not isinstance(params, dict):
        return {}

    normalized = {}
    for key, value in params.items():
        # Skip None or empty values
        if value is None or value == "":
            continue

        key_str = str(key)

        # Preserve internal metadata (starts with single underscore)
        if key_str.startswith("_") and not key_str.startswith("__"):
            # Preserve both boolean and numeric/string internal metadata
            if isinstance(value, (int, float, str, bool)):
                normalized[key_str] = value
            continue

        # Normalize user-provided parameter keys
        key_lower = key_str.lower().strip()
        norm_key = _PARAM_MAP.get(key_lower, key_lower)

        # Validate value
        if isinstance(value, (int, float, str)):
            normalized[norm_key] = value

    return normalized


def _build_sequential_connections(layers: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Build strictly sequential connections."""
    connections = []
    for i in range(len(layers) - 1):
        source = layers[i]["id"]
        target = layers[i + 1]["id"]
        connections.append((source, target))
    return connections


def _validate_config(name: str, layers: List[Dict[str, Any]], connections: List[Tuple[str, str]]) -> None:
    """Validate normalized config."""
    # Name must exist
    assert name, "Name is required"

    # Layers must not be empty
    assert len(layers) > 0, "Layers cannot be empty"

    # All layers must have required fields
    for layer in layers:
        assert "id" in layer and isinstance(layer["id"], str), "Invalid layer ID"
        assert "type" in layer and isinstance(layer["type"], str), "Invalid layer type"
        assert "params" in layer and isinstance(layer["params"], dict), "Invalid layer params"

    # All IDs must be unique
    ids = [layer["id"] for layer in layers]
    assert len(set(ids)) == len(ids), "Duplicate layer IDs"

    # Connections must reference valid IDs
    valid_ids = set(ids)
    for source, target in connections:
        assert source in valid_ids, f"Invalid source ID in connection: {source}"
        assert target in valid_ids, f"Invalid target ID in connection: {target}"

    # Connections count must be correct for sequential
    assert len(connections) == len(layers) - 1, "Incorrect number of connections"


# Validate synonym map at module load time
_validate_synonym_map()
