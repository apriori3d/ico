"""Pipeline parameters: per-run values of node params.

A param value of any node may refer to a pipeline parameter instead of a
fixed value::

    {"kind": "op", "op": "cut_clip@1",
     "params": {"out_dir": {"$param": "out_dir"},
                "ffmpeg": {"$param": "ffmpeg", "default": "ffmpeg"}}}

:func:`bind` substitutes the values of one run before the spec is built, so
an operator receives an ordinary constructor argument. A parameter is
required unless it has a ``default`` (which may be ``null``); the same
parameter may be used by several nodes if they declare it identically.
Only whole param values are parameters: no substitution inside strings or
nested structures.

Example:
    >>> spec_params(spec)       # what a saved pipeline accepts
    >>> run(spec, "a.mp4", params={"out_dir": "/tmp/a"})
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, overload

from ico.core.tree_utils import TreePathIndex
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError

if TYPE_CHECKING:
    from ico.registry.registry import IcoRegistry

PARAM_KEY = "$param"
DEFAULT_KEY = "default"


@dataclass(frozen=True, slots=True)
class SpecParam:
    """A pipeline parameter declared in a spec.

    Attributes:
        name: Parameter name.
        type: Formatted annotation of the operator param it sets, from the
            registry, or ``None`` when unknown (e.g. a structural param).
        required: Whether a value must be given (no ``default``).
        default: Default value (meaningless when ``required``).
        paths: Nodes that use it, in spec-tree coordinates.
    """

    name: str
    type: str | None
    required: bool
    default: JsonValue = None
    paths: tuple[TreePathIndex, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        out: dict[str, JsonValue] = {"name": self.name, "required": self.required}
        if self.type is not None:
            out["type"] = self.type
        if not self.required:
            out["default"] = self.default
        out["paths"] = [str(p) for p in self.paths]
        return out


@dataclass(frozen=True, slots=True)
class _Use:
    name: str
    required: bool
    default: JsonValue
    type: str | None
    path: TreePathIndex


def spec_params(
    spec: PipelineSpec | NodeSpec, registry: IcoRegistry | None = None
) -> list[SpecParam]:
    """The parameters a spec declares, in order of first use.

    Args:
        spec: A spec document or a single node spec.
        registry: Registry to look up the param types of operator classes;
            defaults to the default registry.

    Raises:
        SpecError: On a malformed reference, or a parameter declared
            differently by two nodes (default or type).
    """
    from ico.registry.registry import default_registry

    target = registry if registry is not None else default_registry
    uses: list[_Use] = []
    _collect(_root(spec), TreePathIndex(), target, uses)

    params: dict[str, SpecParam] = {}
    for use in uses:
        known = params.get(use.name)
        if known is None:
            params[use.name] = SpecParam(
                use.name, use.type, use.required, use.default, (use.path,)
            )
            continue
        if (known.required, known.default) != (use.required, use.default):
            raise SpecError(
                f"parameter {use.name!r} is declared with different defaults",
                path=use.path,
            )
        if known.type is not None and use.type is not None and known.type != use.type:
            raise SpecError(
                f"parameter {use.name!r} sets params of different types: "
                f"{known.type} and {use.type}",
                path=use.path,
                expected=known.type,
                got=use.type,
            )
        params[use.name] = replace(
            known, type=known.type or use.type, paths=(*known.paths, use.path)
        )
    return list(params.values())


@overload
def bind(
    spec: PipelineSpec, values: Mapping[str, JsonValue] | None = None
) -> PipelineSpec: ...
@overload
def bind(spec: NodeSpec, values: Mapping[str, JsonValue] | None = None) -> NodeSpec: ...
@overload
def bind(
    spec: PipelineSpec | NodeSpec, values: Mapping[str, JsonValue] | None = None
) -> PipelineSpec | NodeSpec: ...
def bind(
    spec: PipelineSpec | NodeSpec, values: Mapping[str, JsonValue] | None = None
) -> PipelineSpec | NodeSpec:
    """Substitute parameter values (and defaults) into a spec.

    Returns:
        The spec without parameter references; a spec without parameters
        is returned as is.

    Raises:
        SpecError: On an unknown parameter name, a required parameter without
            a value (all are listed), or a malformed reference.
    """
    given = dict(values or {})
    uses: list[_Use] = []
    root = _root(spec)
    _collect(root, TreePathIndex(), None, uses)
    declared = {use.name for use in uses}

    unknown = sorted(set(given) - declared)
    if unknown:
        listed = ", ".join(sorted(declared)) or "none"
        raise SpecError(
            f"unknown parameter(s) {', '.join(unknown)}; declared: {listed}"
        )
    missing = sorted({u.name for u in uses if u.required and u.name not in given})
    if missing:
        raise SpecError(f"missing value for parameter(s) {', '.join(missing)}")
    if not uses:
        return spec

    defaults = {u.name: u.default for u in uses if not u.required}
    bound = _substitute(root, {**defaults, **given})
    if isinstance(spec, PipelineSpec):
        return replace(spec, root=bound)
    return bound


def _root(spec: PipelineSpec | NodeSpec) -> NodeSpec:
    return spec.root if isinstance(spec, PipelineSpec) else spec


def _reference(
    value: JsonValue, path: TreePathIndex
) -> tuple[str, bool, JsonValue] | None:
    """``(name, required, default)`` of a parameter reference, else ``None``."""
    if not isinstance(value, dict) or PARAM_KEY not in value:
        return None
    extra = sorted(set(value) - {PARAM_KEY, DEFAULT_KEY})
    name = value[PARAM_KEY]
    if extra or not isinstance(name, str) or not name:
        raise SpecError(
            f"a parameter reference is {{{PARAM_KEY!r}: name, {DEFAULT_KEY!r}: value}}, "
            f"got {value!r}",
            path=path,
        )
    return name, DEFAULT_KEY not in value, value.get(DEFAULT_KEY)


def _collect(
    node: NodeSpec, path: TreePathIndex, registry: IcoRegistry | None, uses: list[_Use]
) -> None:
    for key, value in node.params.items():
        found = _reference(value, path)
        if found is not None:
            name, required, default = found
            uses.append(
                _Use(name, required, default, _param_type(node, key, registry), path)
            )
    for index, child in enumerate(node.children):
        _collect(child, path.add_child(index), registry, uses)


def _param_type(node: NodeSpec, key: str, registry: IcoRegistry | None) -> str | None:
    """The annotation of the operator class param ``key``, if known."""
    if registry is None or node.op is None:
        return None
    try:
        entry = registry.get(node.op)
    except KeyError:
        return None
    return next((p.type for p in entry.params if p.name == key), None)


def _substitute(node: NodeSpec, values: Mapping[str, JsonValue]) -> NodeSpec:
    params = {
        key: values[found[0]]
        if (found := _reference(value, TreePathIndex())) is not None
        else value
        for key, value in node.params.items()
    }
    children = tuple(_substitute(child, values) for child in node.children)
    return replace(node, params=params, children=children)
