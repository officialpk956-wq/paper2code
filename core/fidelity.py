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


# Canonical types with a real nn.* form that neither OPERATIONS nor
# _node_to_layer supplies. Without these, a spec naming any of them failed
# layer_types_present no matter what the generated code contained.
_DIRECT_CONSTRUCTORS = {
    "conv3d": "nn.Conv3d",
    "dense": "nn.Linear",
    "embedding": "nn.Embedding",
    "clstoken": "nn.Parameter",
    "positionalembedding": "nn.Parameter",
    "query_projection": "nn.Linear",
    "key_projection": "nn.Linear",
    "value_projection": "nn.Linear",
    "attention_merge": "nn.Linear",
    "causal_attention": "nn.MultiheadAttention",
    "cross_attention": "nn.MultiheadAttention",
}

# Canonical types that are structural, not constructible: no single nn.* call
# can represent them, so their absence from the constructor list says nothing
# about fidelity. They are reported informationally and excluded from the
# score rather than counted as failures. residualblock/residual_add are
# already covered by the separate residual_present check; concat and crop are
# forward()-level operations; mbconv and ssm are composite blocks.
_STRUCTURAL_TYPES = frozenset(
    {
        "residualblock",
        "residual_add",
        "concat",
        "crop",
        "mbconv",
        "ssm",
        "sequence_pooling",
    }
)


def _expected_constructor(layer_type: str, params: dict[str, Any]) -> str | None:
    """Resolve a canonical type through the existing codegen/operation tables."""
    direct = _DIRECT_CONSTRUCTORS.get(layer_type)
    if direct:
        return direct
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


def _assigned_attributes(function: ast.FunctionDef) -> set[str]:
    """Every ``self.x = ...`` in the function, whatever the value.

    Used for the "used but never assigned" direction. Restricting this to
    ``nn.*`` calls reported every custom submodule and every plain attribute as
    unassigned -- ``self.patch_embed = PatchEmbedding(...)`` and
    ``self.d_model = int(...)`` are both perfectly ordinary.
    """
    assigned: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.Assign):
                targets = node.targets
            else:
                targets = [node.target]
        elif isinstance(node, ast.For):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for item in target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]:
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "self"
                ):
                    assigned.add(item.attr)
    return assigned


def _declared_modules(function: ast.FunctionDef) -> set[str]:
    """Only ``self.x = nn.Something(...)``.

    Used for the "declared but never used" direction, where the point is a
    dead nn module. An unused plain attribute is not a fidelity problem.
    """
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


def _bound_literals(tree: ast.AST) -> list[Any]:
    """Every literal bound to a name, key, or argument position.

    Four sources, all of them ways a generator parameterises a model:
      nn.Linear(512, 1000)            -- call arguments, positional and keyword
      {"d_model": 512}                -- dict literals (how builder schemas work)
      self.d_model = 512              -- assignment right-hand sides
      def __init__(self, dim=512)     -- parameter defaults

    Only call arguments used to be collected, which missed the schema dicts
    that carry the hyperparameters for every builder-path model.
    """
    values: list[Any] = []

    def take(node: ast.AST | None) -> None:
        if isinstance(node, ast.Constant):
            values.append(node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                take(argument)
        elif isinstance(node, ast.Dict):
            for item in node.values:
                take(item)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            take(node.value)
        elif isinstance(node, ast.arguments):
            for default in [*node.defaults, *(d for d in node.kw_defaults if d is not None)]:
                take(default)
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


def score_fidelity(spec: dict, code: str) -> dict:
    """Compare an extracted spec against generated code using static AST analysis.

    Returns ``score`` (0.0-1.0), ``status``, ``checks`` and ``mismatches``.
    Only applicable checks count toward the denominator; hyperparameters a
    paper never stated, and residual checks for a plain CNN, are excluded.
    Never executes generated code and never raises.

    ``status`` is the part callers must branch on:

      "scored"     -- the code was analysed; ``score`` is a measurement.
      "unscorable" -- nothing could be analysed (no spec, no code, code does
                      not parse). ``score`` is 0.0 but means "unknown".
      "error"      -- the scorer itself failed. ``score`` is 0.0 and means
                      "the instrument broke", which is not a fact about the
                      code under test.

    All three used to return a bare 0.0, indistinguishable from a genuine
    zero, and the benchmark averaged them together -- a broken scorer and
    broken code produced the same aggregate. Aggregate only over "scored".

    A ``graph`` parameter used to be accepted here and never read; callers
    passing it got a silent no-op. Removed rather than left as a lie.
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
                "status": "unscorable",
                "checks": [{"name": "input", "passed": False, "detail": detail}],
                "mismatches": [detail],
            }
        if not isinstance(code, str) or not code.strip():
            detail = "generated code is empty"
            return {
                "score": 0.0,
                "status": "unscorable",
                "checks": [{"name": "input", "passed": False, "detail": detail}],
                "mismatches": [detail],
            }

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            detail = f"code could not be parsed: {exc.msg} (line {exc.lineno})"
            return {
                "score": 0.0,
                "status": "unscorable",
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

        expected_types: dict[str, str | None] = {}
        for layer in spec["layers"]:
            if not isinstance(layer, dict) or not isinstance(layer.get("type"), str):
                continue
            layer_type = layer["type"]
            expected_types.setdefault(
                layer_type, _expected_constructor(layer_type, layer.get("params") or {})
            )
        # A type with no constructor representation cannot be looked for in the
        # constructor list; counting it as missing measured the completeness of
        # this module's own mapping table, not the code under test. ViT scored
        # 0.00 on a file that contained every constructor its spec named.
        structural = sorted(t for t in expected_types if t in _STRUCTURAL_TYPES)
        checkable = {
            layer_type: constructor
            for layer_type, constructor in expected_types.items()
            if layer_type not in _STRUCTURAL_TYPES
        }
        unmapped = sorted(t for t, c in checkable.items() if c is None)
        missing_types = sorted(
            layer_type
            for layer_type, constructor in checkable.items()
            if constructor is not None and constructor not in constructors
        )
        # An unmapped type is this module's gap, not the code's: report it, do
        # not fail the code for it.
        type_passed = bool(checkable) and not missing_types
        detail_parts = []
        if missing_types:
            detail_parts.append(f"missing constructors for: {', '.join(missing_types)}")
        if unmapped:
            detail_parts.append(f"no constructor mapping (not scored): {', '.join(unmapped)}")
        if structural:
            detail_parts.append(f"structural, not constructible: {', '.join(structural)}")
        if not checkable:
            detail_parts.append("no constructor-checkable layer types in spec")
        add_check(
            "layer_types_present",
            type_passed,
            "; ".join(detail_parts) if detail_parts else "all expected constructors are present",
            counts=bool(checkable),
        )

        # Breadth, compared in matching units. The spec's LOGICAL layer count is
        # not comparable to the number of nn.* CALL SITES in the source: a
        # generator that builds 12 transformer blocks in a loop writes one call
        # site, and a helper class hides several. U-Net scored spec=49 vs
        # code=12 on correct code. Distinct expected constructors vs distinct
        # constructors present are the same unit, and under-building still
        # fails. Extra constructors are not a fidelity fault -- a helper class
        # may legitimately add Dropout the spec never named.
        expected_distinct = {c for c in checkable.values() if c is not None}
        present_distinct = expected_distinct & set(constructors)
        breadth_passed = len(present_distinct) >= len(expected_distinct)
        add_check(
            "layer_count",
            breadth_passed,
            f"distinct expected constructors: {len(expected_distinct)}, present: "
            f"{len(present_distinct)} (spec lists {len(spec['layers'])} logical layers; "
            f"code has {len(constructors)} constructor call sites -- not comparable directly)",
            counts=bool(expected_distinct),
        )

        declaration_problems: list[str] = []
        declaration_checked = False
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
            assigned = _assigned_attributes(init_method) if init_method else set()
            used = _self_attributes(forward_method) if forward_method else set()
            if declared or used:
                declaration_checked = True
            unused = sorted(declared - used)
            # Compare against EVERY assignment, not just nn.* ones.
            unassigned = sorted(used - assigned)
            if unused:
                declaration_problems.append(
                    f"{class_node.name} declared but unused: {', '.join(unused)}"
                )
            if unassigned:
                declaration_problems.append(
                    f"{class_node.name} used but unassigned: {', '.join(unassigned)}"
                )
        # Vacuous truth is not a pass: a class with no modules at all trivially
        # satisfies "everything declared is used", which scored empty code 0.25
        # instead of 0.0. Only count this check when there was something to check.
        declared_passed = not declaration_problems
        add_check(
            "declared_vs_used",
            declared_passed,
            (
                "all self.nn modules declared in __init__ are used by forward"
                if declared_passed
                else "; ".join(declaration_problems)
            )
            if declaration_checked
            else "no declared or used modules to compare",
            counts=declaration_checked,
        )

        literal_arguments = _bound_literals(tree)
        for name in _HYPERPARAMETERS:
            values = _specified_hyperparameters(spec).get(name, [])
            if not values:
                add_check(f"key_hyperparams.{name}", True, "not_stated", counts=False)
                continue
            missing_values = [value for value in values if value not in literal_arguments]
            passed = not missing_values
            numeric = sorted(
                (
                    v
                    for v in set(literal_arguments)
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ),
                key=float,
            )
            add_check(
                f"key_hyperparams.{name}",
                passed,
                f"{name}: expected {values!r}"
                + ("" if passed else f", missing {missing_values!r}")
                + f"; bound numeric literals in code: {numeric}",
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

        if not applicable:
            # Every check was skipped as not-applicable, so there is nothing to
            # report. That is "unknown", not "zero fidelity".
            detail = "no applicable checks for this spec"
            return {
                "score": 0.0,
                "status": "unscorable",
                "checks": checks,
                "mismatches": [detail],
            }
        score = sum(applicable) / len(applicable)
        return {"score": score, "status": "scored", "checks": checks, "mismatches": mismatches}
    except Exception as exc:  # Fidelity must never destabilize generation.
        detail = f"fidelity analysis failed: {type(exc).__name__}: {exc}"
        return {
            "score": 0.0,
            "status": "error",
            "checks": [{"name": "analysis", "passed": False, "detail": detail}],
            "mismatches": [detail],
        }
