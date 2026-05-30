"""
Symbolic notation parser for architecture specifications.

Converts compact notation like "Conv2D(64,3)→ReLU→ResBlock×3→Linear(10)"
into ConfigDict format compatible with the pipeline.

F9: Symbolic Notation Input
"""

import re
from typing import Dict, Any
from core.rag.normalizer import _SYNONYM_MAP, normalize_config


# Token grammar: LayerType(params)×count
# Examples: Conv2D(64,3), ReLU, ResBlock×3, Linear(10)
_TOKEN_RE = re.compile(
    r"^(?P<type>[A-Za-z][A-Za-z0-9_]*)"          # layer type
    r"(?:\((?P<params>[^)]*)\))?"                  # optional (params)
    r"(?:[xX×*](?P<repeat>\d+))?$"                 # optional ×count
)


def parse_symbolic(spec: str) -> Dict[str, Any]:
    """
    Parse symbolic notation into a ConfigDict.

    Args:
        spec: Architecture specification string
              Format: "Conv2D(64,3)→ReLU→ResBlock×3→Linear(10)"
              Supports arrows: →, ->, =>, -->

    Returns:
        ConfigDict ready for pipeline.run_single()

    Raises:
        ValueError: If specification is malformed
    """
    # Normalize arrows to → for consistent splitting
    spec = re.sub(r"\s*(?:->|=>|-->|[→⟶])\s*", "→", spec)

    # Split tokens
    tokens = [t.strip() for t in spec.split("→") if t.strip()]
    if not tokens:
        raise ValueError("Empty specification")

    layers = []

    for token in tokens:
        m = _TOKEN_RE.match(token.strip())
        if not m:
            raise ValueError(f"Invalid token format: {token}")

        raw_type = m.group("type")
        raw_params = m.group("params") or ""
        repeat = max(1, int(m.group("repeat") or 1))

        # Normalize type via synonym map
        type_lower = raw_type.lower()
        canonical = _SYNONYM_MAP.get(type_lower, type_lower)

        # Parse parameters
        params = _parse_params(raw_params)

        # Expand repeats
        for i in range(repeat):
            layer = {"type": canonical, "params": dict(params)}
            if repeat > 1:
                layer["params"]["_repeat_index"] = i
                layer["params"]["_repeat_total"] = repeat
            layers.append(layer)

    if not layers:
        raise ValueError("No valid layers parsed")

    # Return normalized config
    raw_config = {"name": "SymbolicModel", "layers": layers}
    return normalize_config(raw_config)


def _parse_params(raw: str) -> Dict[str, Any]:
    """
    Parse parameter string into dict.

    Supports:
    - Empty string → {}
    - Positional: "64,3" → {"channels": 64, "kernel_size": 3}
    - Key=value: "d_model=512,heads=8"
    - Mixed: positional first, then key=value
    """
    if not raw.strip():
        return {}

    result = {}
    parts = [p.strip() for p in raw.split(",")]
    
    # Context-aware positional keys
    type_lower = "" # We don't have type here easily, so we check canonical list
    POSITIONAL_KEYS = ["channels", "kernel_size", "stride", "padding"]
    
    # If the first part looks like a large number (e.g. 512) and it's followed by a small number (e.g. 8),
    # it's likely a Transformer (embed_dim, num_heads).
    if len(parts) >= 2:
        try:
            val1 = int(parts[0])
            val2 = int(parts[1])
            if val1 >= 32 and val2 <= 64: # Heuristic for (dim, heads)
                POSITIONAL_KEYS = ["embed_dim", "num_heads"]
        except ValueError:
            pass

    pos_idx = 0
    for part in parts:
        if "=" in part:
            # Key=value format
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                result[k] = int(v)
            except ValueError:
                result[k] = v
        else:
            # Positional format
            try:
                if pos_idx < len(POSITIONAL_KEYS):
                    result[POSITIONAL_KEYS[pos_idx]] = int(part)
                    pos_idx += 1
            except ValueError:
                pass

    return result
