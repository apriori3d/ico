"""Spec → flow reconstruction through an operator registry.

Leaf operators are resolved by their ``op="name@version"`` reference; the
structural nodes around them are rebuilt from ``kind``, ``params`` and
``children``. Importable ``ref="module:qualname"`` references are never
imported: a spec may come from an agent, so only registered operators can
be instantiated.

Round trip ``to_spec(build(spec)) == spec`` holds for every buildable kind.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from ico.core.batcher import IcoBatcher
from ico.core.chain import chain
from ico.core.filter import IcoFilter
from ico.core.fork import IcoFork
from ico.core.identity import IcoIdentity
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperatorProtocol
from ico.core.partition import IcoPartition
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.route import IcoRoute
from ico.core.stream import IcoStream
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError

AnyOp: TypeAlias = IcoOperatorProtocol[Any, Any]

Builder: TypeAlias = Callable[[NodeSpec, "BuildContext"], IcoNodeProtocol]

_BUILDERS: dict[str, Builder] = {}


def spec_builder(kind: str) -> Callable[[Builder], Builder]:
    """Register a builder that turns a spec node of ``kind`` into a flow node.

    Example:
        >>> @spec_builder("my_node")
        ... def _my_node(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
        ...     return MyNode(ctx.op(spec.children[0]), name=spec.name)
    """

    def decorator(fn: Builder) -> Builder:
        _BUILDERS[kind] = fn
        return fn

    return decorator


if TYPE_CHECKING:
    from ico.registry.registry import IcoRegistry


class BuildContext:
    """Traversal state passed to builders.

    Attributes:
        registry: Registry used to resolve ``op`` references.
    """

    __slots__ = ("registry",)

    def __init__(self, registry: IcoRegistry) -> None:
        self.registry = registry

    def build(self, spec: NodeSpec) -> IcoNodeProtocol:
        """Build a node: registered operators by reference, others by kind."""
        if spec.op is not None:
            return self._resolve(spec)

        builder = _BUILDERS.get(spec.kind)
        if builder is None:
            where = f" '{spec.name}'" if spec.name else ""
            if spec.kind == "op" or spec.ref is not None:
                raise SpecError(
                    f"Cannot build leaf{where} ({spec.ref or 'anonymous'}): only "
                    "registered operators (op='name@version') can be built."
                )
            raise SpecError(f"Cannot build node{where} of kind '{spec.kind}' yet.")
        return builder(spec, self)

    def op(self, spec: NodeSpec) -> AnyOp:
        """Build a child that is used as an operator."""
        return cast(AnyOp, self.build(spec))

    def _resolve(self, spec: NodeSpec) -> IcoNodeProtocol:
        from ico.registry.registry import parse_ref

        ref = cast(str, spec.op)
        try:
            parse_ref(ref)
            entry = self.registry.get(ref)
        except (ValueError, KeyError) as e:
            raise SpecError(f"Cannot resolve operator '{ref}': {e.args[0]}") from None

        actual = entry.signature.format()
        if spec.sig is not None and spec.sig != actual:
            raise SpecError(
                f"Signature of '{ref}' changed: spec has {spec.sig!r}, "
                f"registry has {actual!r}."
            )
        return entry.node


def build(
    spec: PipelineSpec | NodeSpec, registry: IcoRegistry | None = None
) -> IcoNodeProtocol:
    """Rebuild a flow from a spec, resolving operators through a registry.

    Args:
        spec: A spec document or a single node spec.
        registry: Registry for ``op`` references; defaults to
            :data:`ico.registry.default_registry`.

    Returns:
        The rebuilt flow. Registered operators are returned as the registered
        instances (not copies).

    Raises:
        SpecError: On unknown operators, signature drift between the spec
            and the registry, unregistered leaves, invalid params or kinds
            that cannot be built yet (async_stream, epoch, agent, context
            operators, runtime wrappers, monitors, progress).

    Example:
        >>> flow = build(from_json(text))
    """
    from ico.registry.registry import default_registry

    root = spec.root if isinstance(spec, PipelineSpec) else spec
    target = registry if registry is not None else default_registry
    return BuildContext(target).build(root)


# ────────────────────────────────────────────────
# Param helpers
# ────────────────────────────────────────────────


def _param(spec: NodeSpec, key: str, kind: type, *, optional: bool = False) -> Any:
    if key not in spec.params:
        if optional:
            return None
        raise SpecError(f"'{spec.kind}' requires param '{key}'.")
    value = spec.params[key]
    # bool is an int subclass; require exact types for numeric params.
    if type(value) is not kind:
        raise SpecError(
            f"Param '{key}' of '{spec.kind}' must be {kind.__name__}, got {value!r}."
        )
    return value


def _children(spec: NodeSpec, count: int | None = None, *, at_least: int = 0) -> None:
    n = len(spec.children)
    if (count is not None and n != count) or n < at_least:
        expected = count if count is not None else f"at least {at_least}"
        raise SpecError(f"'{spec.kind}' expects {expected} children, got {n}.")


# ────────────────────────────────────────────────
# Builders
# ────────────────────────────────────────────────


@spec_builder("seq")
def build_seq(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, at_least=2)
    steps = [ctx.op(child) for child in spec.children]
    return cast(IcoNodeProtocol, reduce(chain, steps))


@spec_builder("stream")
def build_stream(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoStream(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("pipeline")
def build_pipeline(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, at_least=1)
    return IcoPipeline(*(ctx.op(c) for c in spec.children), name=spec.name)


@spec_builder("process")
def build_process(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    if "num_iterations" in spec.params:
        _children(spec, 1)
        return IcoProcess(
            ctx.op(spec.children[0]),
            num_iterations=_param(spec, "num_iterations", int),
            name=spec.name,
        )

    _children(spec, 2)
    on_limit = _param(spec, "on_limit", str)
    if on_limit not in ("return", "raise"):
        raise SpecError(f"Invalid on_limit {on_limit!r} in 'process'.")
    return IcoProcess(
        ctx.op(spec.children[0]),
        until=ctx.op(spec.children[1]),
        max_iterations=_param(spec, "max_iterations", int),
        on_limit=on_limit,
        name=spec.name,
    )


@spec_builder("fork")
def build_fork(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, at_least=2)
    node: IcoFork[Any, Any] = IcoFork(
        *(ctx.op(c) for c in spec.children), name=spec.name
    )
    return node


@spec_builder("route")
def build_route(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    keys = cast(list[JsonValue], _param(spec, "keys", list))
    has_default = _param(spec, "default", bool)
    _children(spec, 1 + len(keys) + int(has_default))

    key_op, *branches = (ctx.op(c) for c in spec.children)
    default = branches.pop() if has_default else None
    for key in keys:
        if isinstance(key, list | dict):
            raise SpecError(f"Route key {key!r} is not a JSON scalar.")
    return IcoRoute(
        key_op,
        dict(zip(keys, branches, strict=True)),
        default=default,
        name=spec.name,
    )


@spec_builder("filter")
def build_filter(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoFilter(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("partition")
def build_partition(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 1)
    return IcoPartition(ctx.op(spec.children[0]), name=spec.name)


@spec_builder("batcher")
def build_batcher(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 0)
    # Not parametrized (IcoBatcher[T]): a spec carries no Python types, and an
    # explicit parameter would change the inferred signature.
    node: IcoBatcher[Any] = IcoBatcher(
        batch_size=_param(spec, "batch_size", int),
        drop_last=_param(spec, "drop_last", bool),
        name=spec.name,
    )
    return node


@spec_builder("identity")
def build_identity(spec: NodeSpec, ctx: BuildContext) -> IcoNodeProtocol:
    _children(spec, 0)
    node: IcoIdentity[Any] = IcoIdentity(name=spec.name)
    return node
