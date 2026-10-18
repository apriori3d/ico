"""Per-node aggregates of one process, kept in shards per thread."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypedDict, final

from ico.tools.profiler.metrics import IcoMetric
from ico.tools.profiler.stats import Stats, StatsView

# Kinds of probe events.
VALUE = 0
"""A call that returned a value: a call, in the distribution."""
OPEN = 1
"""A call that returned an iterator: a call, in the totals only."""
CALL_ERROR = 2
"""A call that raised."""
START = 3
"""The first ``next()`` of an iterator that gave an item: an item, in
``startup``."""
ITEM = 4
"""A later ``next()`` that gave an item: an item, in the distribution."""
END = 5
"""The ``next()`` that raised ``StopIteration``: in ``end``."""
NEXT_ERROR = 6
"""A ``next()`` that raised another exception."""
FIRST = 7
"""The first call of a node instance that returned a value (lazy setup,
e.g. opening a dataset): a call, in ``startup``."""


class MetricView(StatsView):
    """A metric of a node: totals of all its events, and the distribution of
    its calls (nodes returning values) or items (after the first)."""

    total: float
    self: float
    """``total`` minus the time of the probed calls nested in it; for a
    worker portal, also minus its runner's flow (the transfer is left)."""


class NodeProfile(TypedDict):
    """The profile of one node."""

    key: str
    """Path of the node in the plan (``0.1.0``)."""
    label: str
    type: str
    calls: int
    items: int
    errors: int
    metrics: dict[str, MetricView]
    startup: dict[str, MetricView] | None
    """Setup: the first call of each instance of a node returning values
    (lazy initialization, e.g. one per worker process), and the first
    ``next()`` of each iterator a node returned (setup before the first
    item); ``None`` without either. Kept out of the distribution, so that
    ``p95`` describes the steady state. Its ``self`` excludes the startup
    of the upstream an iterator pulled."""
    end: dict[str, MetricView] | None
    """The final ``next()`` of each iterator (cleanup)."""
    utilization: float | None
    """Async streams: the busy time of the pool members divided by the time
    the stream was active times the pool size."""
    transfer: float | None
    """Workers: seconds of portal calls (wall) not spent in the runner's
    flow: pickling, queues, waiting for the runner."""


@dataclass(frozen=True, slots=True)
class NodeInfo:
    label: str
    type: str


@dataclass(slots=True)
class Layout:
    """What the aggregates of a table describe; sent with remote snapshots."""

    nodes: dict[str, NodeInfo] = field(default_factory=dict[str, NodeInfo])
    """Instrumented nodes by key."""
    pools: dict[str, tuple[tuple[str, ...], int]] = field(
        default_factory=dict[str, tuple[tuple[str, ...], int]]
    )
    """Async stream key → the keys of its pool members and the pool size."""
    workers: dict[str, str] = field(default_factory=dict[str, str])
    """Worker key → the key of the root of its runner's flow."""

    def update(self, other: Layout) -> None:
        for key, info in other.nodes.items():
            self.nodes.setdefault(key, info)
        self.pools.update(other.pools)
        self.workers.update(other.workers)


@final
class NodeAggregate:
    """Aggregates of one node in one shard."""

    __slots__ = (
        "calls",
        "items",
        "errors",
        "frames",
        "total",
        "own",
        "events",
        "startup",
        "end",
        "startup_own",
        "end_own",
        "busy",
        "active",
    )

    def __init__(self, metrics: int) -> None:
        self.calls = 0
        self.items = 0
        self.errors = 0
        self.frames = 0
        self.total = [0] * metrics
        self.own = [0] * metrics
        self.events = [Stats() for _ in range(metrics)]
        self.startup = [Stats() for _ in range(metrics)]
        self.end = [Stats() for _ in range(metrics)]
        self.startup_own = [0] * metrics
        self.end_own = [0] * metrics
        self.busy = 0
        """Nanoseconds in calls of a pool member (timed probes only)."""
        self.active = 0
        """Nanoseconds from the first to the last ``next()`` of each iterator
        (timed probes only)."""

    def record(
        self,
        kind: int,
        elapsed: Sequence[int | None],
        own: Sequence[int | None],
        busy: int,
    ) -> None:
        self.frames += 1
        self.busy += busy
        if kind in (VALUE, OPEN, CALL_ERROR, FIRST):
            self.calls += 1
        elif kind in (START, ITEM):
            self.items += 1
        if kind in (CALL_ERROR, NEXT_ERROR):
            self.errors += 1
        target = (
            self.events
            if kind in (VALUE, ITEM)
            else self.startup
            if kind in (START, FIRST)
            else self.end
            if kind == END
            else None
        )
        for i, value in enumerate(elapsed):
            if value is None:
                continue
            self.total[i] += value
            self.own[i] += own[i] or 0
            if target is not None:
                target[i].add(value)
            if kind in (START, FIRST):
                self.startup_own[i] += own[i] or 0
            elif kind == END:
                self.end_own[i] += own[i] or 0

    def merge(self, other: NodeAggregate) -> None:
        self.calls += other.calls
        self.items += other.items
        self.errors += other.errors
        self.frames += other.frames
        self.busy += other.busy
        self.active += other.active
        for i in range(len(self.total)):
            self.total[i] += other.total[i]
            self.own[i] += other.own[i]
            self.events[i].merge(other.events[i])
            self.startup[i].merge(other.startup[i])
            self.end[i].merge(other.end[i])
            self.startup_own[i] += other.startup_own[i]
            self.end_own[i] += other.end_own[i]


@final
class Shard:
    """The aggregates written by one thread."""

    __slots__ = ("lock", "nodes", "metrics")

    def __init__(self, metrics: int) -> None:
        self.lock = threading.Lock()
        """Taken by the owning thread and by snapshots only: uncontended."""
        self.nodes: dict[str, NodeAggregate] = {}
        self.metrics = metrics

    def record(
        self,
        key: str,
        kind: int,
        elapsed: Sequence[int | None],
        own: Sequence[int | None],
        busy: int = 0,
    ) -> None:
        with self.lock:
            node = self._node(key)
            node.record(kind, elapsed, own, busy)

    def record_active(self, key: str, active: int) -> None:
        with self.lock:
            self._node(key).active += active

    def _node(self, key: str) -> NodeAggregate:
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = NodeAggregate(self.metrics)
        return node


@final
class ProfileTable:
    """The profiled nodes of a flow and their aggregates in this process."""

    def __init__(self, metrics: Sequence[IcoMetric]) -> None:
        names = [m.name for m in metrics]
        if not names:
            raise ValueError("at least one metric is needed")
        if len(set(names)) != len(names):
            raise ValueError(f"metric names must be unique, got {names}")
        self.metrics: tuple[IcoMetric, ...] = tuple(metrics)
        self.layout = Layout()
        self._shards: list[Shard] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def add_node(self, key: str, label: str, type_name: str) -> None:
        self.layout.nodes.setdefault(key, NodeInfo(label, type_name))

    def shard(self) -> Shard:
        """The shard of the calling thread."""
        shard: Shard | None = getattr(self._local, "shard", None)
        if shard is None:
            shard = Shard(len(self.metrics))
            self._local.shard = shard
            with self._lock:
                self._shards.append(shard)
        return shard

    def reset(self) -> None:
        """Forget all measurements (the nodes stay)."""
        with self._lock:
            for shard in self._shards:
                with shard.lock:
                    shard.nodes.clear()

    def merged(self) -> dict[str, NodeAggregate]:
        """The aggregates of all threads, merged per node."""
        out: dict[str, NodeAggregate] = {}
        with self._lock:
            shards = list(self._shards)
        for shard in shards:
            with shard.lock:
                for key, node in shard.nodes.items():
                    target = out.get(key)
                    if target is None:
                        target = out[key] = NodeAggregate(len(self.metrics))
                    target.merge(node)
        return out


def merge_into(
    target: dict[str, NodeAggregate], source: dict[str, NodeAggregate]
) -> None:
    """Add the aggregates of ``source`` (e.g. of another process)."""
    for key, node in source.items():
        existing = target.get(key)
        if existing is None:
            existing = target[key] = NodeAggregate(len(node.total))
        existing.merge(node)


def profiles(
    metrics: Sequence[IcoMetric],
    layout: Layout,
    merged: dict[str, NodeAggregate],
) -> list[NodeProfile]:
    """Profiles of all nodes of ``layout``, in plan order."""
    empty = NodeAggregate(len(metrics))
    wall = next((i for i, m in enumerate(metrics) if m.name == "wall"), None)
    out: list[NodeProfile] = []
    for key in sorted(layout.nodes, key=_path):
        info = layout.nodes[key]
        node = merged.get(key, empty)
        own = _own(metrics, layout, merged, key, node)
        views: dict[str, MetricView] = {}
        for i, metric in enumerate(metrics):
            view = node.events[i].view(metric.scale)
            views[metric.name] = {
                **view,
                "total": node.total[i] * metric.scale,
                "self": own[i] * metric.scale,
            }
        out.append(
            {
                "key": key,
                "label": info.label,
                "type": info.type,
                "calls": node.calls,
                "items": node.items,
                "errors": node.errors,
                "metrics": views,
                "startup": _phase(metrics, node.startup, node.startup_own),
                "end": _phase(metrics, node.end, node.end_own),
                "utilization": _utilization(layout, merged, key),
                "transfer": _transfer(metrics, layout, merged, key, wall),
            }
        )
    return out


def _own(
    metrics: Sequence[IcoMetric],
    layout: Layout,
    merged: dict[str, NodeAggregate],
    key: str,
    node: NodeAggregate,
) -> list[int]:
    """Self values of a node. A worker portal waits while its runner works
    in another process, where the runner's frames cannot nest in the
    portal's: its self excludes the time of the runner's flow (for metrics
    that are not thread-bound; its CPU time is its own anyway), leaving the
    transfer."""
    own = list(node.own)
    remote = merged.get(layout.workers.get(key, ""))
    if remote is not None:
        for i, metric in enumerate(metrics):
            if not metric.thread_bound:
                own[i] = max(own[i] - remote.total[i], 0)
    return own


def _path(key: str) -> tuple[int, ...]:
    return tuple(int(part) for part in key.split("."))


def _phase(
    metrics: Sequence[IcoMetric], stats: list[Stats], own: list[int]
) -> dict[str, MetricView] | None:
    if all(s.count == 0 for s in stats):
        return None
    return {
        metric.name: {
            **stats[i].view(metric.scale),
            "total": stats[i].sum * metric.scale,
            "self": own[i] * metric.scale,
        }
        for i, metric in enumerate(metrics)
    }


def _utilization(
    layout: Layout, merged: dict[str, NodeAggregate], key: str
) -> float | None:
    pool = layout.pools.get(key)
    stream = merged.get(key)
    if pool is None or stream is None or stream.active <= 0:
        return None
    members, size = pool
    busy = sum(merged[m].busy for m in members if m in merged)
    return busy / (stream.active * size)


def _transfer(
    metrics: Sequence[IcoMetric],
    layout: Layout,
    merged: dict[str, NodeAggregate],
    key: str,
    wall: int | None,
) -> float | None:
    remote = layout.workers.get(key)
    if remote is None or wall is None or key not in merged or remote not in merged:
        return None
    portal, runner = merged[key].total[wall], merged[remote].total[wall]
    return max(portal - runner, 0) * metrics[wall].scale
