"""Static architecture-fidelity checks for generated PyTorch source.

The score is independent of execution validation: it compares what the
extracted specification says with what the generated source visibly contains.
Only applicable checks contribute to the denominator; informational
``not_stated`` hyperparameter checks do not.
"""

import ast
from typing import Any

from core.architecture_graph import GraphNode
from core.codegen import _node_to_layer
from core.knowledge.operations import OPERATIONS

_HYPERPARAMETERS = (
    "num_heads",
    "hidden_size",
    "channels",
    "num_layers",
    "kernel_size",
)


def _call_name(node: ast.AST) -> str | None:
    """Return a dotted call name, such as ``nn.Conv2d``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _nn_constructor_calls(tree: ast.AST) -> list[str]:
    return [
        name
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and (name := _call_name(call.func)) is not None
        and name.startswith("nn.")
    ]


def _expected_constructor(layer_type: str, params: dict[str, Any]) -> str | None:
    """Resolve a canonical type through the existing codegen/operation tables."""
    operation = OPERATIONS.get(layer_type)
    syntax = operation.get("syntax") if operation else None
    if syntax:
        try:
            expression = ast.parse(syntax, mode="eval").body
            if isinstance(expression, ast.Call):
                return _call_name(expression.func)
        except SyntaxError:
            pass

    # _node_to_layer consults its parameterized MAP before operation fallback.
    # Calling it here keeps this checker tied to the production mapping instead
    # of introducing a third, drift-prone type table.
    try:
        syntax = _node_to_layer(
            GraphNode(id="fidelity", type=layer_type, label=layer_type, params=params)
        )
        expression = ast.parse(syntax, mode="eval").body if syntax else None
        if isinstance(expression, ast.Call):
            return _call_name(expression.func)
    except (SyntaxError, TypeError, ValueError):
        return None
    return None


def _self_attributes(function: ast.FunctionDef) -> set[str]:
    return {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Load)
    }


def _declared_modules(function: ast.FunctionDef) -> set[str]:
    declared: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not (_call_name(value.func) or "").startswith("nn."):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                declared.add(target.attr)
    return declared


def _literal_call_arguments(tree: ast.AST) -> list[Any]:
    values: list[Any] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
            if isinstance(argument, ast.Constant):
                values.append(argument.value)
    return values


def _specified_hyperparameters(
    value: Any, found: dict[str, list[Any]] | None = None
) -> dict[str, list[Any]]:
    """Collect explicitly stated supported hyperparameters from nested spec data."""
    found = found if found is not None else {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _HYPERPARAMETERS and isinstance(child, (int, float, str, bool)):
                found.setdefault(key, []).append(child)
            _specified_hyperparameters(child, found)
    elif isinstance(value, list):
        for child in value:
            _specified_hyperparameters(child, found)
    return found


def _requires_residual(spec: dict) -> bool:
    values = spec.get("connection_types", [])
    if isinstance(values, str):
        values = [values]
    return any("skip" in str(value).lower() or "residual" in str(value).lower() for value in values)


def _forward_has_residual(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return True
        if isinstance(node, ast.Call) and _call_name(node.func) in {"torch.cat", "torch.add"}:
            return True
    return False


def score_fidelity(spec: dict, code: str, graph=None) -> dict:
    """Compare an extracted spec against generated code using static AST analysis.

    Returns a score from 0.0 to 1.0, individual checks, and mismatch details.
    Only applicable checks are counted in the score denominator; for example,
    hyperparameters not stated by a paper and residual checks for a plain CNN
    are excluded. This function never executes generated code and never raises.
    """
    try:
        if (
            not isinstance(spec, dict)
            or not isinstance(spec.get("layers"), list)
            or not spec["layers"]
        ):
            detail = "spec.layers must be a non-empty list for fidelity scoring"
            return {
                "score": 0.0,
                "checks": [{"name": "input", "passed": False, "detail": detail}],
                "mismatches": [detail],
            }
        if not isinstance(code, str) or not code.strip():
            detail = "generated code is empty"
            return {
                "score": 0.0,
                "checks": [{"name": "input", "passed": False, "detail": detail}],
                "mismatches": [detail],
            }

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            detail = f"code could not be parsed: {exc.msg} (line {exc.lineno})"
            return {
                "score": 0.0,
                "checks": [{"name": "syntax", "passed": False, "detail": detail}],
                "mismatches": [detail],
            }

        checks: list[dict] = []
        applicable: list[bool] = []
        mismatches: list[str] = []

        def add_check(name: str, passed: bool, detail: str, *, counts: bool = True) -> None:
            checks.append({"name": name, "passed": passed, "detail": detail})
            if counts:
                applicable.append(passed)
            if counts and not passed:
                mismatches.append(detail)

        constructors = _nn_constructor_calls(tree)
        expected_count = len(spec["layers"])
        actual_count = len(constructors)
        tolerance = expected_count * 0.2
        count_passed = abs(actual_count - expected_count) <= tolerance
        add_check(
            "layer_count",
            count_passed,
            f"layer_count: spec={expected_count}, code={actual_count}, tolerance=±{tolerance:g}",
        )

        expected_types: dict[str, str | None] = {}
        for layer in spec["layers"]:
            if not isinstance(layer, dict) or not isinstance(layer.get("type"), str):
                continue
            layer_type = layer["type"]
            expected_types.setdefault(
                layer_type, _expected_constructor(layer_type, layer.get("params") or {})
            )
        missing_types = [
            layer_type
            for layer_type, constructor in expected_types.items()
            if constructor is None or constructor not in constructors
        ]
        type_passed = bool(expected_types) and not missing_types
        add_check(
            "layer_types_present",
            type_passed,
            "all expected constructors are present"
            if type_passed
            else f"missing plausible constructors for: {', '.join(missing_types) or 'no typed layers'}",
        )

        declaration_problems: list[str] = []
        for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            methods = {
                method.name: method
                for method in class_node.body
                if isinstance(method, ast.FunctionDef)
            }
            init_method = methods.get("__init__")
            forward_method = methods.get("forward")
            if init_method is None and forward_method is None:
                continue
            declared = _declared_modules(init_method) if init_method else set()
            used = _self_attributes(forward_method) if forward_method else set()
            unused = sorted(declared - used)
            unassigned = sorted(used - declared)
            if unused:
                declaration_problems.append(
                    f"{class_node.name} declared but unused: {', '.join(unused)}"
                )
            if unassigned:
                declaration_problems.append(
                    f"{class_node.name} used but unassigned: {', '.join(unassigned)}"
                )
        declared_passed = not declaration_problems
        add_check(
            "declared_vs_used",
            declared_passed,
            "all self.nn modules declared in __init__ are used by forward"
            if declared_passed
            else "; ".join(declaration_problems),
        )

        literal_arguments = _literal_call_arguments(tree)
        for name in _HYPERPARAMETERS:
            values = _specified_hyperparameters(spec).get(name, [])
            if not values:
                add_check(f"key_hyperparams.{name}", True, "not_stated", counts=False)
                continue
            missing_values = [value for value in values if value not in literal_arguments]
            passed = not missing_values
            found = sorted({repr(value) for value in literal_arguments})
            add_check(
                f"key_hyperparams.{name}",
                passed,
                f"{name}: expected {values!r}, found literal arguments [{', '.join(found)}]",
            )

        if _requires_residual(spec):
            forward_methods = [
                method
                for class_node in tree.body
                if isinstance(class_node, ast.ClassDef)
                for method in class_node.body
                if isinstance(method, ast.FunctionDef) and method.name == "forward"
            ]
            residual_passed = any(_forward_has_residual(method) for method in forward_methods)
            add_check(
                "residual_present",
                residual_passed,
                "forward contains a residual addition or concatenation"
                if residual_passed
                else "spec declares skip/residual connections, but forward has no addition or torch.cat",
            )

        score = sum(applicable) / len(applicable) if applicable else 0.0
        return {"score": score, "checks": checks, "mismatches": mismatches}
    except Exception as exc:  # Fidelity must never destabilize generation.
        detail = f"fidelity analysis failed: {type(exc).__name__}: {exc}"
        return {
            "score": 0.0,
            "checks": [{"name": "analysis", "passed": False, "detail": detail}],
            "mismatches": [detail],
        }
