"""Instrument a flow in place: a probe on every operator of the flow."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import IcoAsyncStream
from ico.core.context_operator import IcoContextOperator
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.probe import get_probe, set_probe
from ico.tools.profiler.probe import NodeProbe

if TYPE_CHECKING:
    from ico.tools.profiler.tool import ProfilerTool

F = TypeVar("F", bound=IcoNodeProtocol)

ROOT_KEY = "0"


def instrument(flow: F, profiler: ProfilerTool) -> F:
    """Set a probe on every operator of a built, not yet activated flow.

    The flow keeps its nodes, names, signatures and plan; only its calls are
    measured, into ``profiler``. Call it before creating the runtime::

        profiler = ProfilerTool(on_profile=print)
        flow = instrument(flow, profiler)
        IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()

    Keys are the paths of the nodes in the plan: the root is ``0`` and the
    ``i``-th child of ``k`` is ``k.i``. The members of an async stream pool
    built from a factory share the key of the first member, as the plan
    shows one of them, so their measurements merge.

    An operator that already has a probe is left as is (instrumenting twice
    changes nothing), and a node used at several places of the flow is
    measured under its first key. The flow stays instrumented: build a flow
    for a profiled run (as :func:`ico.spec.run` does).

    Not measured yet: the flows of workers in other processes (``MPWorker``
    runners); their portal calls are.

    Returns:
        ``flow`` itself.
    """
    table = profiler.table
    seen: set[int] = set()

    def visit(node: IcoNodeProtocol, key: str) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, IcoOperator | IcoContextOperator | IcoAsyncOperator):
            operator = cast(
                IcoOperator[Any, Any]
                | IcoContextOperator[Any, Any, Any]
                | IcoAsyncOperator[Any, Any],
                node,
            )
            if get_probe(operator) is None:
                set_probe(operator, NodeProbe(key, table))
                table.add_node(key, label(operator), type(operator).__name__)
        if (
            isinstance(node, IcoAsyncStream)
            and cast(IcoAsyncStream[Any, Any], node).pool_from_factory
        ):
            for member in node.children:
                visit(member, f"{key}.0")
            return
        for i, child in enumerate(node.children):
            visit(child, f"{key}.{i}")

    visit(flow, ROOT_KEY)
    return flow


def label(node: IcoNodeProtocol) -> str:
    """Registry reference, else name, else function name, else class."""
    from ico.registry.registry import get_meta

    meta = get_meta(node)
    if meta is not None:
        return meta.ref
    if node.name:
        return node.name
    fn: object = getattr(node, "fn", None)
    if isinstance(fn, types.MethodType) and fn.__self__ is node:
        return type(node).__name__  # a structural node (`_stream_fn`, ...)
    name: object = getattr(fn, "__name__", None)
    if isinstance(name, str) and name != "<lambda>":
        return name
    return type(node).__name__
