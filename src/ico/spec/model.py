"""Serializable pipeline specification model.

A spec is a plain, dependency-free tree of :class:`NodeSpec` objects.
It is the machine-readable counterpart of ``describe()``: agents and tools
exchange it as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, cast

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

SPEC_VERSION = 1


def is_json_value(value: object) -> bool:
    """Whether ``value`` is a JSON-compatible :data:`JsonValue` (recursively)."""
    if value is None or isinstance(value, str | int | float | bool):
        return True
    if isinstance(value, list):
        return all(is_json_value(v) for v in cast(list[object], value))
    if isinstance(value, dict):
        items = cast(dict[object, object], value).items()
        return all(isinstance(k, str) and is_json_value(v) for k, v in items)
    return False


class SpecError(ValueError):
    """Raised when a flow cannot be represented as a spec."""


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Specification of a single flow node.

    Attributes:
        kind: How the node combines its children (e.g. ``"op"``, ``"seq"``,
            ``"async_stream"``). Determines how ``children`` are interpreted.
        name: Optional human-readable node name.
        sig: Formatted ICO signature, e.g. ``"int → str"``.
        ref: Importable reference ``"module:qualname"`` for leaf callables,
            or ``None`` for structural nodes and anonymous callables.
        op: Registry reference ``"name@version"`` for registered operators.
            Such a node is opaque: its children are not exported and
            :func:`ico.spec.build` resolves it through the registry.
        params: Kind-specific parameters (JSON-compatible values only).
        children: Child specs, ordered as defined by ``kind``.
    """

    kind: str
    name: str | None = None
    sig: str | None = None
    ref: str | None = None
    op: str | None = None
    params: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    children: tuple[NodeSpec, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        """Convert to a compact JSON-compatible dict (empty fields omitted)."""
        out: dict[str, JsonValue] = {"kind": self.kind}
        if self.name is not None:
            out["name"] = self.name
        if self.sig is not None:
            out["sig"] = self.sig
        if self.ref is not None:
            out["ref"] = self.ref
        if self.op is not None:
            out["op"] = self.op
        if self.params:
            out["params"] = dict(self.params)
        if self.children:
            out["children"] = [c.to_dict() for c in self.children]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> NodeSpec:
        """Build a spec from a dict produced by :meth:`to_dict`."""
        kind = data.get("kind")
        if not isinstance(kind, str):
            raise SpecError(f"Spec node requires a string 'kind', got {kind!r}")

        params = data.get("params", {})
        if not isinstance(params, dict):
            raise SpecError(f"'params' of '{kind}' must be a mapping")

        raw_children = data.get("children", [])
        if not isinstance(raw_children, list):
            raise SpecError(f"'children' of '{kind}' must be a list")
        children: list[NodeSpec] = []
        for child in raw_children:
            if not isinstance(child, dict):
                raise SpecError(f"Child of '{kind}' must be a mapping")
            children.append(cls.from_dict(child))

        return cls(
            kind=kind,
            name=_opt_str(data, "name"),
            sig=_opt_str(data, "sig"),
            ref=_opt_str(data, "ref"),
            op=_opt_str(data, "op"),
            params=dict(params),
            children=tuple(children),
        )


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Top-level spec document: a versioned root node."""

    root: NodeSpec
    version: int = SPEC_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        return {"version": self.version, "root": self.root.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> PipelineSpec:
        version = data.get("version")
        if version != SPEC_VERSION:
            raise SpecError(
                f"Unsupported spec version {version!r}, expected {SPEC_VERSION}"
            )
        root = data.get("root")
        if not isinstance(root, dict):
            raise SpecError("Spec document requires a 'root' mapping")
        return cls(root=NodeSpec.from_dict(root), version=SPEC_VERSION)


def _opt_str(data: dict[str, JsonValue], key: str) -> str | None:
    value = data.get(key)
    if value is None or isinstance(value, str):
        return value
    raise SpecError(f"'{key}' must be a string, got {value!r}")
