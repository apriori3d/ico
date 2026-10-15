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
from ico.core.identity import IcoIdentity
from ico.core.node import IcoNode, IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.runtime.agent import IcoAgent
from ico.core.runtime.monitor import IcoMonitor
from ico.core.runtime.progress import IcoProgress
from ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from ico.core.signature import IcoSignature
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.spec.model import NodeSpec, SpecError
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
        if isinstance(child, IcoChain):
            steps.extend(_flatten_chain(cast(IcoChain[object, object, object], child)))
        else:
            steps.append(child)
    return steps


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
        params={"total": node.total},
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
def serialize_chain(node: IcoChain[object, object, object], ctx: SpecContext) -> NodeSpec:
    # Nested binary chains (a | b) | c are flattened into one sequence.
    return NodeSpec(
        kind="seq",
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
        params={"num_iterations": node.num_iterations},
        children=(ctx.walk(node.body),),
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
    if node.pool_from_factory:
        # All workers are identical copies: emit one worker plus pool size.
        workers = (ctx.walk(node.pool[0]),)
    else:
        # Explicit pool: workers share one input queue (load balancing,
        # not broadcast), so every worker is listed.
        workers = tuple(ctx.walk(w) for w in node.pool)
    return NodeSpec(
        kind="async_stream",
        name=node.name,
        sig=node_signature(node),
        params={
            "pool_size": node.pool_size,
            "ordered": node.ordered,
            "factory": node.pool_from_factory,
        },
        children=workers,
    )


@spec_serializer(IcoAgent)
def serialize_agent(node: IcoAgent[object, object], ctx: SpecContext) -> NodeSpec:
    factory = cast(Callable[[], IcoNodeProtocol], node.get_remote_flow_factory())
    return NodeSpec(
        kind="agent",
        name=node.name,
        sig=node_signature(node),
        ref=callable_ref(factory, ctx, owner="agent flow factory"),
        params={"type": _type_ref(node)},
        children=(ctx.walk(factory()),),
    )
