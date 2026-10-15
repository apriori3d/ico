"""Serializer registry and flow → spec conversion."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar, cast

from ico.core.node import IcoNodeProtocol
from ico.spec.model import (
    JsonValue,
    NodeSpec,
    PipelineSpec,
    SpecError,
    is_json_value,
)

N = TypeVar("N", bound=IcoNodeProtocol)

Serializer = Callable[[N, "SpecContext"], NodeSpec]

_SERIALIZERS: dict[type, Callable[[IcoNodeProtocol, SpecContext], NodeSpec]] = {}


def spec_serializer(
    node_type: type[N],
) -> Callable[[Serializer[N]], Serializer[N]]:
    """Register a spec serializer for a node type (and its subclasses).

    The most specific registered type in the node's MRO wins, so a
    serializer for a subclass overrides the one for its base class.

    Example:
        >>> @spec_serializer(MyNode)
        ... def _my_node(node: MyNode, ctx: SpecContext) -> NodeSpec:
        ...     return NodeSpec(kind="my_node", children=(ctx.walk(node.body),))
    """

    def decorator(fn: Serializer[N]) -> Serializer[N]:
        _SERIALIZERS[node_type] = cast(
            Callable[[IcoNodeProtocol, SpecContext], NodeSpec], fn
        )
        return fn

    return decorator


def find_serializer(
    node: IcoNodeProtocol,
) -> Callable[[IcoNodeProtocol, SpecContext], NodeSpec]:
    """Return the serializer registered for the closest type in node's MRO."""
    for cls in type(node).__mro__:
        serializer = _SERIALIZERS.get(cls)
        if serializer is not None:
            return serializer
    raise SpecError(f"No spec serializer registered for {type(node).__name__}")


class SpecContext:
    """Traversal state passed to serializers.

    Attributes:
        strict: When True, anonymous callables (lambdas, local functions)
            raise :class:`SpecError` because they cannot be restored from
            a spec. When False they are emitted with ``ref=None``.
    """

    __slots__ = ("strict",)

    def __init__(self, *, strict: bool) -> None:
        self.strict = strict

    def walk(self, node: IcoNodeProtocol) -> NodeSpec:
        """Serialize a node: registered operators by reference, others by type."""
        registered = registered_spec(node)
        if registered is not None:
            return registered
        return find_serializer(node)(node, self)


def registered_spec(node: IcoNodeProtocol) -> NodeSpec | None:
    """Opaque spec of an operator registered in an operator registry.

    Registered nodes, including whole flows registered as one operator, are
    exported as ``kind="op"`` with an ``op="name@version"`` reference and no
    children: the registry entry is the source of truth for their contents.
    Instances of registered operator classes also export their constructor
    kwargs, read from same-named attributes, as ``params``.

    Raises:
        SpecError: If a constructor parameter has no same-named attribute or
            its value is not JSON-compatible.
    """
    from ico.registry.registry import get_meta, init_params, registered_class
    from ico.spec.serializers import node_signature

    meta = get_meta(node)
    if meta is None:
        return None

    params: dict[str, JsonValue] = {}
    cls = registered_class(node)
    if cls is not None:
        for param in init_params(cls):
            if not hasattr(node, param.name):
                raise SpecError(
                    f"Cannot export '{meta.ref}': constructor parameter "
                    f"'{param.name}' must be stored in an attribute of the same name."
                )
            value = cast(object, getattr(node, param.name))
            if not is_json_value(value):
                raise SpecError(
                    f"Cannot export '{meta.ref}': parameter '{param.name}' = "
                    f"{value!r} is not JSON-compatible."
                )
            params[param.name] = cast(JsonValue, value)

    return NodeSpec(
        kind="op",
        name=node.name,
        sig=node_signature(node),
        op=meta.ref,
        params=params,
    )


def to_spec(node: IcoNodeProtocol, *, strict: bool = False) -> PipelineSpec:
    """Convert an ICO flow into a serializable :class:`PipelineSpec`.

    Args:
        node: Root node of the flow.
        strict: Raise :class:`SpecError` on anonymous callables.

    Example:
        >>> spec = to_spec(load | parse | save)
        >>> print(to_json(spec))
    """
    # Import built-in serializers lazily to avoid import cycles with core.
    import ico.spec.serializers  # noqa: F401  # pyright: ignore[reportUnusedImport]

    return PipelineSpec(root=SpecContext(strict=strict).walk(node))


def to_json(spec: PipelineSpec, *, indent: int | None = 2) -> str:
    """Serialize a spec to a JSON string."""
    return json.dumps(spec.to_dict(), indent=indent, ensure_ascii=False)


def from_json(text: str) -> PipelineSpec:
    """Parse a spec from a JSON string produced by :func:`to_json`."""
    data = cast(JsonValue, json.loads(text))
    if not isinstance(data, dict):
        raise SpecError("Spec JSON must be an object")
    return PipelineSpec.from_dict(data)
