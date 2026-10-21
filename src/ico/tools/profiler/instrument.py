"""Instrument a flow in place: a probe on every operator of the flow."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ico.core.async_operator import IcoAsyncOperator
from ico.core.async_stream import IcoAsyncStream
from ico.core.context_operator import IcoContextOperator
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.plan_keys import ROOT_KEY, child_key, plan_nodes, remote_flow_key
from ico.core.runtime.worker import IcoWorker
from ico.tools.profiler.probe import NodeProbe
from ico.tools.profiler.remote import RunnerProfiler
from ico.tools.profiler.table import ProfileTable

if TYPE_CHECKING:
    from ico.tools.profiler.tool import ProfilerTool

F = TypeVar("F", bound=IcoNodeProtocol)


def instrument(flow: F, profiler: ProfilerTool) -> F:
    """Set a probe on every operator of a built, not yet activated flow.

    The flow keeps its nodes, names, signatures and plan; only its calls are
    measured, into ``profiler``. Call it before creating the runtime::

        profiler = ProfilerTool(on_profile=print)
        flow = instrument(flow, profiler)
        IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()

    Keys are the plan keys of the nodes (:mod:`ico.core.plan_keys`,
    the ``keys`` of :func:`~ico.plan_view`): the root is ``0`` and the
    ``i``-th child of ``k`` is ``k.i``. The members of an async stream pool
    built from a factory share the key of the first member, as the plan
    shows one of them, so their measurements merge.

    Instrumenting twice into one profiler changes nothing; another
    profiler takes the nodes over; probes of other tools are kept. A node
    used at several places of the flow is measured under its first key. The
    flow stays instrumented, so build a flow of its own for a profiled run,
    without shared nodes such as registered functions (as
    :func:`ico.spec.run` does with a fresh build).

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

    Workers get a runner probe that instruments their flow in the runner
    process (keys continue below the worker's key) and reports it
    with ``IcoProfileEvent``.
    """
    seen: set[int] = set()
    for planned in plan_nodes(flow, root_key=root_key):
        node, key = planned.node, planned.key
        if id(node) in seen:
            continue  # a node used at several places: its first key
        seen.add(id(node))
        member = planned.parent is not None and isinstance(
            planned.parent.node, IcoAsyncStream
        )
        if isinstance(node, IcoOperator | IcoContextOperator | IcoAsyncOperator):
            operator = cast(
                IcoOperator[Any, Any]
                | IcoContextOperator[Any, Any, Any]
                | IcoAsyncOperator[Any, Any],
                node,
            )
            probe = operator.probe
            # Replace a probe of an earlier profiler, keep other tools' probes.
            if probe is None or (
                isinstance(probe, NodeProbe) and probe.table is not table
            ):
                timed = member or isinstance(node, IcoAsyncStream)
                operator.probe = NodeProbe(key, table, timed=timed)
                table.add_node(key, label(operator), type(operator).__name__)
        if isinstance(node, IcoWorker):
            _instrument_worker(cast(IcoWorker[Any, Any], node), key, table, interval)
        if isinstance(node, IcoAsyncStream):
            stream = cast(IcoAsyncStream[Any, Any], node)
            members = range(1 if stream.pool_from_factory else len(node.children))
            keys = tuple(child_key(key, i) for i in members)
            table.layout.pools[key] = (keys, stream.pool_size)


def _instrument_worker(
    worker: IcoWorker[Any, Any], key: str, table: ProfileTable, interval: float | None
) -> None:
    remote = remote_flow_key(key, worker)
    table.layout.workers[key] = remote
    probe = RunnerProfiler(remote, table.metrics, interval, label(worker))
    current = worker.runner_probe
    if current is None or (isinstance(current, RunnerProfiler) and current != probe):
        worker.runner_probe = probe


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
