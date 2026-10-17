"""Built-in spec serializers for ICO core node types.

Each serializer emits a ``kind`` describing how children are combined,
kind-specific ``params`` and the serialized children. New node types
register their own serializer with :func:`ico.spec.spec_serializer`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import IcoAsyncStream
from ico.core.batcher import IcoBatcher
from ico.core.chain import IcoChain
from ico.core.context_operator import IcoContextOperator
from ico.core.context_pipeline import IcoContextPipeline
from ico.core.epoch import IcoEpoch
from ico.core.filter import IcoFilter
from ico.core.fork import IcoFork
from ico.core.identity import IcoIdentity
from ico.core.node import IcoNode, IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.partition import IcoPartition
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.route import IcoRoute
from ico.core.runtime.monitor import IcoMonitor
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from ico.core.runtime.worker import IcoWorker
from ico.core.signature import IcoSignature
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.spec.model import JsonValue, NodeSpec, SpecError
from ico.spec.registry import SpecContext, spec_serializer

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────


def node_signature(node: IcoNodeProtocol) -> str | None:
    """Return the formatted ICO signature of a node, if it has one."""
    try:
        signature = cast(object, getattr(node, "signature", None))
    except Exception:  # signature inference must never break export
        return None
    if isinstance(signature, IcoSignature):
        return signature.format()
    return None


def callable_ref(fn: object, ctx: SpecContext, *, owner: str) -> str | None:
    """Return an importable ``module:qualname`` reference for a callable.

    Lambdas, local functions and callables without a qualified name are
    anonymous: they raise :class:`SpecError` in strict mode and yield
    ``None`` otherwise.
    """
    module = cast(object, getattr(fn, "__module__", None))
    qualname = cast(object, getattr(fn, "__qualname__", None))
    if (
        isinstance(module, str)
        and isinstance(qualname, str)
        and "<lambda>" not in qualname
        and "<locals>" not in qualname
    ):
        return f"{module}:{qualname}"
    if ctx.strict:
        raise SpecError(
            f"Anonymous callable {fn!r} in {owner} cannot be referenced by a spec. "
            "Use a module-level function instead of a lambda or local function."
        )
    return None


def _type_ref(node: object) -> str:
    cls = type(node)
    return f"{cls.__module__}:{cls.__qualname__}"


def _is_own_method(fn: object, node: object) -> bool:
    """True when fn is a bound method of node (structural operator)."""
    return cast(object, getattr(fn, "__self__", None)) is node


def _flatten_chain(node: IcoChain[object, object, object]) -> list[IcoNodeProtocol]:
    steps: list[IcoNodeProtocol] = []
    for child in node.children:
        sub = _unregistered_chain(child)
        if sub is not None:
            steps.extend(_flatten_chain(sub))
        else:
            steps.append(child)
    return steps


def _unregistered_chain(
    node: IcoNodeProtocol,
) -> IcoChain[object, object, object] | None:
    """The node as a chain to flatten; a registered chain is one opaque step."""
    from ico.registry.registry import get_meta

    if not isinstance(node, IcoChain):
        return None
    sub = cast(IcoChain[object, object, object], node)
    return sub if get_meta(sub) is None else None


# ────────────────────────────────────────────────
# Generic fallback
# ────────────────────────────────────────────────


@spec_serializer(IcoNode)
def serialize_node(node: IcoNode, ctx: SpecContext) -> NodeSpec:
    """Fallback for node types without a dedicated serializer."""
    return NodeSpec(
        kind="node",
        name=node.name,
        sig=node_signature(node),
        params={"type": _type_ref(node)},
        children=tuple(ctx.walk(c) for c in node.children),
    )


# ────────────────────────────────────────────────
# Leaf operators
# ────────────────────────────────────────────────


@spec_serializer(IcoOperator)
def serialize_operator(node: IcoOperator[object, object], ctx: SpecContext) -> NodeSpec:
    if _is_own_method(node.fn, node):
        # Subclass implemented via its own method: not a leaf callable.
        return serialize_node(node, ctx)
    return NodeSpec(
        kind="op",
        name=node.name,
        sig=node_signature(node),
        ref=callable_ref(node.fn, ctx, owner="operator"),
    )


@spec_serializer(IcoAsyncOperator)
def serialize_async_operator(
    node: IcoAsyncOperator[object, object], ctx: SpecContext
) -> NodeSpec:
    return NodeSpec(
        kind="async_op",
        name=node.name,
        sig=node_signature(node),
        ref=callable_ref(node.fn, ctx, owner="async operator"),
    )


@spec_serializer(IcoContextOperator)
def serialize_context_operator(
    node: IcoContextOperator[object, object, object], ctx: SpecContext
) -> NodeSpec:
    return NodeSpec(
        kind="context_op",
        name=node.name,
        sig=node_signature(node),
        ref=callable_ref(node.fn, ctx, owner="context operator"),
    )


@spec_serializer(IcoSource)
def serialize_source(node: IcoSource[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="source",
        name=node.name,
        sig=node_signature(node),
        ref=callable_ref(node.provider, ctx, owner="source"),
    )


@spec_serializer(IcoSink)
def serialize_sink(node: IcoSink[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="sink",
        name=node.name,
        sig=node_signature(node),
        ref=callable_ref(node.consumer, ctx, owner="sink"),
    )


@spec_serializer(IcoIdentity)
def serialize_identity(node: IcoIdentity[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(kind="identity", name=node.name, sig=node_signature(node))


@spec_serializer(IcoMonitor)
def serialize_monitor(node: IcoMonitor[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="monitor",
        name=node.name,
        sig=node_signature(node),
        params={"type": _type_ref(node)},
    )


@spec_serializer(IcoProgress)
def serialize_progress(node: IcoProgress[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="progress",
        name=node.name,
        sig=node_signature(node),
        params={"total": node.total} if node.total is not None else {},
    )


@spec_serializer(IcoBatcher)
def serialize_batcher(node: IcoBatcher[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="batcher",
        name=node.name,
        sig=node_signature(node),
        params={"batch_size": node.batch_size, "drop_last": node.drop_last},
    )


# ────────────────────────────────────────────────
# Composite nodes
# ────────────────────────────────────────────────


@spec_serializer(IcoChain)
def serialize_chain(
    node: IcoChain[object, object, object], ctx: SpecContext
) -> NodeSpec:
    # Nested binary chains (a | b) | c are flattened into one sequence; the
    # names of flattened inner chains are dropped, the outer name is kept.
    left, right = node.children
    auto_name = f"{left} | {right}"  # IcoChain's default name
    return NodeSpec(
        kind="seq",
        name=node.name if node.name != auto_name else None,
        sig=node_signature(node),
        children=tuple(ctx.walk(step) for step in _flatten_chain(node)),
    )


@spec_serializer(IcoStream)
def serialize_stream(node: IcoStream[object, object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="stream",
        name=node.name,
        sig=node_signature(node),
        children=(ctx.walk(node.body),),
    )


@spec_serializer(IcoProcess)
def serialize_process(node: IcoProcess[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="process",
        name=node.name,
        sig=node_signature(node),
        params=(
            {"num_iterations": node.max_iterations}
            if node.until is None
            else {"max_iterations": node.max_iterations, "on_limit": node.on_limit}
        ),
        children=tuple(ctx.walk(child) for child in node.children),
    )


@spec_serializer(IcoEpoch)
def serialize_epoch(node: IcoEpoch[object, object], ctx: SpecContext) -> NodeSpec:
    # children: [source, context_operator]
    return NodeSpec(
        kind="epoch",
        name=node.name,
        sig=node_signature(node),
        children=(ctx.walk(node.source), ctx.walk(node.context_operator)),
    )


@spec_serializer(IcoPipeline)
def serialize_pipeline(node: IcoPipeline[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="pipeline",
        name=node.name,
        sig=node_signature(node),
        children=tuple(ctx.walk(step) for step in node.body),
    )


@spec_serializer(IcoFork)
def serialize_fork(node: IcoFork[object, object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="fork",
        name=node.name,
        sig=node_signature(node),
        children=tuple(ctx.walk(branch) for branch in node.branches),
    )


@spec_serializer(IcoFilter)
def serialize_filter(node: IcoFilter[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="filter",
        name=node.name,
        sig=node_signature(node),
        children=(ctx.walk(node.predicate),),
    )


@spec_serializer(IcoPartition)
def serialize_partition(node: IcoPartition[object], ctx: SpecContext) -> NodeSpec:
    return NodeSpec(
        kind="partition",
        name=node.name,
        sig=node_signature(node),
        children=(ctx.walk(node.predicate),),
    )


@spec_serializer(IcoRoute)
def serialize_route(
    node: IcoRoute[object, object, object], ctx: SpecContext
) -> NodeSpec:
    # children: [key, *branches (in key order), default?]
    keys = [_route_key(key, node, ctx) for key in node.routes]
    return NodeSpec(
        kind="route",
        name=node.name,
        sig=node_signature(node),
        params={"keys": keys, "default": node.default is not None},
        children=tuple(ctx.walk(child) for child in node.children),
    )


def _route_key(
    key: object, node: IcoRoute[object, object, object], ctx: SpecContext
) -> JsonValue:
    if key is None or isinstance(key, str | int | float | bool):
        return key
    if ctx.strict:
        raise SpecError(
            f"Route key {key!r} in '{node.name}' is not a JSON scalar "
            "(str, int, float, bool or None) and cannot be exported."
        )
    return repr(key)


@spec_serializer(IcoContextPipeline)
def serialize_context_pipeline(
    node: IcoContextPipeline[object, object, object], ctx: SpecContext
) -> NodeSpec:
    # children: [apply, *body]
    return NodeSpec(
        kind="context_pipeline",
        name=node.name,
        sig=node_signature(node),
        children=(ctx.walk(node.apply), *(ctx.walk(step) for step in node.body)),
    )


@spec_serializer(IcoRuntimeWrapper)
def serialize_runtime_wrapper(
    node: IcoRuntimeWrapper[object, object], ctx: SpecContext
) -> NodeSpec:
    return NodeSpec(
        kind="runtime_wrapper",
        name=node.name,
        sig=node_signature(node),
        children=(ctx.walk(node.operator),),
    )


@spec_serializer(IcoAsyncStream)
def serialize_async_stream(
    node: IcoAsyncStream[object, object], ctx: SpecContext
) -> NodeSpec:
    workers: tuple[NodeSpec, ...]
    if node.pool_from_factory:
        # All workers are identical copies: emit one worker plus pool size.
        workers = (ctx.walk(node.pool[0]),)
    else:
        # Explicit pool: workers share one input queue (load balancing,
        # not broadcast), so every worker is listed.
        workers = tuple(ctx.walk(w) for w in node.pool)
    params: dict[str, JsonValue] = {
        "pool_size": node.pool_size,
        "ordered": node.ordered,
        "factory": node.pool_from_factory,
    }
    # Error handling is only exported when it differs from the defaults,
    # so specs of existing flows are unchanged.
    if node.on_error != "raise":
        params["on_error"] = node.on_error
    retry = node.retry
    if retry.max_retries > 0:
        params["retry"] = {
            "max_retries": retry.max_retries,
            "delay": retry.delay,
            "backoff": retry.backoff,
            "retry_on": [f"{t.__module__}:{t.__qualname__}" for t in retry.retry_on],
        }
    return NodeSpec(
        kind="async_stream",
        name=node.name,
        sig=node_signature(node),
        params=params,
        children=workers,
    )


# IcoWorker is abstract; registering a base type is the point of the registry.
@spec_serializer(IcoWorker)  # type: ignore[type-abstract]
def serialize_worker(node: IcoWorker[object, object], ctx: SpecContext) -> NodeSpec:
    # The worker subtree fully describes the factory: build() recreates it
    # with a SpecFactory, so the original factory reference is not exported.
    factory = cast(Callable[[], IcoNodeProtocol], node.get_remote_flow_factory())
    return NodeSpec(
        kind="worker",
        name=node.name,
        sig=node_signature(node),
        params={"type": _type_ref(node)},
        children=(ctx.walk(factory()),),
    )
