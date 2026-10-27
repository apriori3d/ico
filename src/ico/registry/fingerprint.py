"""Code fingerprints: which code an operator runs.

A fingerprint is the sha256 of the sources of a module and of every module
it imports, transitively, except the standard library, installed
distributions and ``ico`` itself: their versions are pinned by the caller's
lock file. Editable checkouts and the domain's own packages are followed,
so a change in a helper module that an operator imports changes the
fingerprint. See ``docs/design/registry.md``, "Code fingerprints and
non-deterministic operators".
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import sysconfig
from collections import deque
from pathlib import Path

# Imported by every operator and pinned as a distribution of its own.
FRAMEWORK = "ico"


def module_fingerprint(module: str) -> str | None:
    """``"sha256:<hex>"`` of the files of :func:`module_sources`, or ``None``
    when ``module`` cannot be located."""
    sources = module_sources(module)
    if sources is None:
        return None
    digests = sorted(
        (name, hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in sources.items()
    )
    return "sha256:" + hashlib.sha256(json.dumps(digests).encode()).hexdigest()


def module_sources(module: str) -> dict[str, Path] | None:
    """The files of ``module`` and of the modules it imports, transitively,
    by module name; ``None`` when ``module`` cannot be located.

    Imports are read from the source, in module bodies and inside functions,
    so lazy imports count. The standard library, installed distributions
    and ``ico`` are not followed; neither are imports by name at run time
    (``importlib.import_module(name)``).
    """
    root = _locate(module)
    if root is None:
        return None
    installed = _installed_roots()
    files: dict[str, Path] = {module: root}
    skipped: set[str] = set()
    queue = deque(_imports(module, root))
    while queue:
        name = queue.popleft()
        if name in files or _under(name, skipped):
            continue
        path = _locate(name)
        if path is None or _excluded(name, path, installed):
            skipped.add(name)
            continue
        files[name] = path
        queue.extend(_imports(name, path))
    return files


def _locate(name: str) -> Path | None:
    """The file of a module, without importing it when it is not imported
    yet (a dotted name imports its parent packages, which are followed
    before their submodules)."""
    module = sys.modules.get(name)
    if module is not None:
        file = getattr(module, "__file__", None)
        return Path(file) if isinstance(file, str) else None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.has_location or spec.origin is None:
        return None
    return Path(spec.origin)


def _imports(name: str, path: Path) -> list[str]:
    """Modules that ``name`` imports (and their parent packages), in source
    order; for ``from base import x``, ``base.x`` too, in case it is a
    submodule."""
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except SyntaxError:
        return []
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.extend(_with_parents(alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(node, package)
            if base is None:
                continue
            found.extend(_with_parents(base))
            found.extend(f"{base}.{alias.name}" for alias in node.names)
    return found


def _resolve(node: ast.ImportFrom, package: str) -> str | None:
    if node.level == 0:
        return node.module
    relative = "." * node.level + (node.module or "")
    try:
        return importlib.util.resolve_name(relative, package)
    except ImportError:  # beyond the top-level package
        return None


def _with_parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def _under(name: str, skipped: set[str]) -> bool:
    """Whether ``name`` or one of its parent packages was skipped."""
    return any(parent in skipped for parent in _with_parents(name))


def _excluded(name: str, path: Path, installed: tuple[Path, ...]) -> bool:
    if name == FRAMEWORK or name.startswith(FRAMEWORK + "."):
        return True
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in installed)


def _installed_roots() -> tuple[Path, ...]:
    """The standard library and the site-packages directories."""
    paths = sysconfig.get_paths()
    keys = ("stdlib", "platstdlib", "purelib", "platlib")
    return tuple({Path(paths[key]).resolve() for key in keys if key in paths})
