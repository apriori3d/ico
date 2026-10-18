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
from ico.core.runtime.worker import IcoWorker
from ico.tools.profiler.probe import NodeProbe
from ico.tools.profiler.remote import ProfiledFlowFactory
from ico.tools.profiler.table import ProfileTable

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

    The flows of workers (``MPWorker``) are instrumented in their runner
    processes when the workers activate: their keys continue below the
    worker's key, as in the plan with remote flows.

    Returns:
        ``flow`` itself.
    """
    instrument_table(flow, profiler.table, ROOT_KEY, profiler.interval)
    return flow


def instrument_table(
    flow: IcoNodeProtocol, table: ProfileTable, root_key: str, interval: float | None
) -> None:
    """Instrument ``flow`` into ``table``, its root keyed ``root_key``.

    Workers get a runner flow factory that instruments their flow in the
    runner process (keys continue below the worker's key) and reports it
    with ``IcoProfileEvent``.
    """
    seen: set[int] = set()

    def visit(node: IcoNodeProtocol, key: str, member: bool) -> None:
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
                timed = member or isinstance(node, IcoAsyncStream)
                set_probe(operator, NodeProbe(key, table, timed=timed))
                table.add_node(key, label(operator), type(operator).__name__)
        if isinstance(node, IcoWorker):
            _instrument_worker(cast(IcoWorker[Any, Any], node), key, table, interval)
        if isinstance(node, IcoAsyncStream):
            stream = cast(IcoAsyncStream[Any, Any], node)
            members = [
                (child, f"{key}.0" if stream.pool_from_factory else f"{key}.{i}")
                for i, child in enumerate(node.children)
            ]
            keys = tuple(dict.fromkeys(k for _, k in members))
            table.layout.pools[key] = (keys, stream.pool_size)
            for child, child_key in members:
                visit(child, child_key, True)
            return
        for i, child in enumerate(node.children):
            visit(child, f"{key}.{i}", False)

    visit(flow, root_key, False)


def _instrument_worker(
    worker: IcoWorker[Any, Any], key: str, table: ProfileTable, interval: float | None
) -> None:
    # The runner's flow is the worker's last child in the plan.
    remote = f"{key}.{len(worker.children)}"
    table.layout.workers[key] = remote
    if isinstance(worker.runner_flow_factory, ProfiledFlowFactory):
        return
    inner = worker.runner_flow_factory
    if inner is None:
        inner = worker.flow_factory
    worker.runner_flow_factory = ProfiledFlowFactory(
        inner, remote, table.metrics, interval
    )


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
