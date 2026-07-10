import ast

_BANNED_MODULES = {
    "os",
    "subprocess",
    "sys",
    "importlib",
    "shutil",
    "pathlib",
    "socket",
    "ctypes",
    "multiprocessing",
    "pty",
    "signal",
}


def _check_code_safety(code: str) -> None:
    """Raise ValueError if generated code imports dangerous modules."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return  # lint step already handles syntax errors
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module.split(".")[0]] if node.module else [])
            )
            for name in names:
                if name in _BANNED_MODULES:
                    raise ValueError(
                        f"Generated code imports banned module '{name}'. "
                        "Rejecting to prevent arbitrary code execution."
                    )
