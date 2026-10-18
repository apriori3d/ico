"""Load operator modules from a directory into the default registry.

A server process starts with an empty registry, while operators (including
ones proposed by an agent) are Python files in a directory. `load_operators`
imports them so that their `@register` decorators fill `default_registry`.

Security:
    Importing an operator module executes its code with the full permissions
    of this process. Nothing is sandboxed. Never load untrusted code into a
    process that holds secrets or data it must protect: isolation (a separate
    process or container with only the capabilities declared in `caps`) is
    the caller's responsibility, e.g. `ico-mcp`.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

from ico.registry.registry import STATUSES, RegistryEntry, Status, default_registry
from ico.spec.model import JsonValue


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Result of :func:`load_operators`.

    Attributes:
        loaded: ``name@version`` refs registered by this call.
        skipped: Modules that were already imported and were not reloaded.
        errors: ``(file, "ErrorType: message")`` per module that failed.
    """

    loaded: tuple[str, ...]
    skipped: tuple[str, ...]
    errors: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, JsonValue]:
        """JSON-compatible form of the report."""
        loaded: list[JsonValue] = []
        loaded.extend(self.loaded)
        skipped: list[JsonValue] = []
        skipped.extend(self.skipped)
        errors: list[JsonValue] = []
        errors.extend({"file": file, "error": error} for file, error in self.errors)
        return {"loaded": loaded, "skipped": skipped, "errors": errors}


def load_operators(
    path: str | Path,
    *,
    strict: bool = False,
    replace: bool = False,
    status: Status | None = None,
    modules: Collection[str] | None = None,
) -> LoadReport:
    """Import every operator module of a package directory.

    The directory must be a package (contain ``__init__.py``). Its parent is
    appended to ``sys.path`` if needed, and modules are imported by name,
    ``<package>.<module>``, so that ``module:qualname`` references are
    reproducible and spawned ``MPWorker`` runner processes can import them too. Only
    the top-level ``*.py`` files are loaded, in name order.

    Args:
        path: The operator package directory.
        strict: Raise the first error instead of collecting it.
        replace: Reload modules that were already imported. Allowed only if
            all their registered entries are ``draft``: a ``verified`` or
            ``stable`` operator can only be superseded by a new version.
            If a reload fails, the previous entries are restored.
        status: Status of every entry registered by this package, overriding
            what the modules passed to ``@register``: status comes from the
            package location (e.g. ``draft`` for a drafts package, ``stable``
            for the reviewed one). ``None`` keeps the modules' own status.
            ``replace`` checks the overridden status.
        modules: Names of the package's modules to import (``"frames"`` for
            ``<package>/frames.py``); ``None`` imports them all. Modules a
            selected one imports (e.g. relatively) are imported by Python as
            usual, so a process can load only the operators it needs. An
            unknown name is reported in ``errors``.

    Returns:
        A :class:`LoadReport`. One broken file does not prevent the others
        from loading (unless ``strict``).

    Raises:
        ValueError: If ``status`` is unknown, ``path`` is not a package
            directory, or its package name already refers to another directory.

    Warning:
        Importing executes the modules' code with this process's permissions;
        see the module docstring.
    """
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    directory = Path(path).resolve()
    if not (directory / "__init__.py").is_file():
        raise ValueError(f"{directory} is not a package: it has no __init__.py")
    package = directory.name
    if not package.isidentifier():
        raise ValueError(f"Package name {package!r} is not a valid identifier")

    parent = str(directory.parent)
    if parent not in sys.path:
        sys.path.append(parent)
    importlib.invalidate_caches()
    imported = importlib.import_module(package)
    imported_dir = Path(cast(str, imported.__file__)).parent.resolve()
    if imported_dir != directory:
        raise ValueError(
            f"Package {package!r} already refers to {imported_dir}, not {directory}"
        )

    loaded: list[str] = []
    skipped: list[str] = []
    errors: list[tuple[str, str]] = []

    files = sorted(f for f in directory.glob("*.py") if f.name != "__init__.py")
    if modules is not None:
        present = {f.stem for f in files}
        errors.extend(
            (
                str(directory / f"{name}.py"),
                f"ModuleNotFoundError: no module {name!r} in package {package!r}",
            )
            for name in sorted(set(modules) - present)
        )
        files = [f for f in files if f.stem in modules]

    for file in files:
        module_name = f"{package}.{file.stem}"
        existing = sys.modules.get(module_name)
        if existing is not None and not replace:
            skipped.append(module_name)
            continue
        try:
            if existing is None:
                refs = _import_new(module_name)
            else:
                refs = _reload_drafts(existing)
            if status is not None:
                _apply_status(refs, status)
            loaded.extend(refs)
        except Exception as e:
            if strict:
                raise
            errors.append((str(file), f"{type(e).__name__}: {e}"))

    return LoadReport(tuple(loaded), tuple(skipped), tuple(errors))


def _import_new(module_name: str) -> list[str]:
    before = set(_refs())
    try:
        importlib.import_module(module_name)
    except BaseException:
        # Leave no half-imported module or entries: a fixed file can be
        # loaded again by the next call.
        sys.modules.pop(module_name, None)
        for ref in set(_refs()) - before:
            default_registry.unregister(ref)
        raise
    return sorted(set(_refs()) - before)


def _reload_drafts(module: ModuleType) -> list[str]:
    entries = [e for e in default_registry if e.module == module.__name__]
    fixed = [e.ref for e in entries if e.meta.status != "draft"]
    if fixed:
        raise ValueError(
            f"cannot replace non-draft operators {fixed}; register a new version"
        )

    for entry in entries:
        default_registry.unregister(entry.ref)
    _drop_bytecode(module)
    before = set(_refs())
    try:
        importlib.reload(module)
    except BaseException:
        for ref in set(_refs()) - before:
            default_registry.unregister(ref)
        for entry in entries:
            _restore(entry)
        raise
    return sorted(set(_refs()) - before)


def _apply_status(refs: list[str], status: Status) -> None:
    """Re-register entries with the given status (entries are immutable)."""
    for ref in refs:
        entry = default_registry.get(ref)
        if entry.meta.status == status:
            continue
        default_registry.unregister(ref)
        meta = dataclasses.replace(entry.meta, status=status)
        _restore(dataclasses.replace(entry, meta=meta))


def _drop_bytecode(module: ModuleType) -> None:
    """Remove cached bytecode so a reload reads the edited source.

    The bytecode cache is validated by mtime in whole seconds and size, so a
    file rewritten within the same second with the same size (an agent
    iterating on a draft) would otherwise reload stale code.
    """
    source = cast(object, getattr(module, "__file__", None))
    if isinstance(source, str):
        Path(importlib.util.cache_from_source(source)).unlink(missing_ok=True)


def _restore(entry: RegistryEntry) -> None:
    if entry.cls is not None:
        default_registry.add_class(entry.cls, entry.meta, description=entry.description)
    elif entry.node is not None:
        default_registry.add(entry.node, entry.meta, description=entry.description)


def _refs() -> list[str]:
    return [entry.ref for entry in default_registry]
