"""Audit cached extraction parameters without calling an LLM.

Usage:
    python -m benchmarks.audit_params [CACHE_DIR]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_OUTPUT_KEYS = {"channels", "num_classes"}


def audit_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Summarize explicit parameter evidence and suspicious repeated values."""
    layers = spec.get("layers") if isinstance(spec, dict) else []
    layers = layers if isinstance(layers, list) else []
    typed_layers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parameter_counts: Counter[tuple[str, str, str]] = Counter()
    incomplete_layers: list[dict[str, Any]] = []

    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        layer_type = str(layer.get("type") or "unknown").lower()
        params = layer.get("params")
        params = params if isinstance(params, dict) else {}
        typed_layers[layer_type].append(params)
        for key, value in params.items():
            parameter_counts[(layer_type, str(key), repr(value))] += 1
        if layer_type in {"conv2d", "linear"} and not (_OUTPUT_KEYS & params.keys()):
            incomplete_layers.append({"index": index, "type": layer_type, "params": params})

    repeated = []
    for (layer_type, key, value), count in sorted(parameter_counts.items()):
        same_type_layers = len(typed_layers[layer_type])
        repeated.append(
            {
                "type": layer_type,
                "key": key,
                "value": value,
                "count": count,
                "same_type_layers": same_type_layers,
                "fabrication_flag": count > same_type_layers / 2,
            }
        )

    return {
        "total_layers": len(layers),
        "layers_with_params": sum(
            1
            for layer in layers
            if isinstance(layer, dict) and isinstance(layer.get("params"), dict) and layer["params"]
        ),
        "parameter_counts": repeated,
        "fabrication_flags": [entry for entry in repeated if entry["fabrication_flag"]],
        "incomplete_layers": incomplete_layers,
    }


def audit_cache(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[dict[str, Any]]:
    """Audit every non-diagnostic cached extraction JSON file in ``cache_dir``."""
    results = []
    for path in sorted(Path(cache_dir).glob("*.json")):
        if path.name.endswith(".diag.json"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = payload.get("spec") if isinstance(payload, dict) else None
        if not isinstance(spec, dict):
            continue
        results.append({"path": str(path), "paper_id": path.stem, **audit_spec(spec)})
    return results


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        raise ValueError("Usage: python -m benchmarks.audit_params [CACHE_DIR]")
    print(json.dumps(audit_cache(args[0] if args else DEFAULT_CACHE_DIR), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
