"""Per-node aggregates of one process, kept in shards per thread."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
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


class MetricView(StatsView):
    """A metric of a node: totals of all its events, and the distribution of
    its calls (nodes returning values) or items (after the first)."""

    total: float
    self: float
    """``total`` minus the time of the probed calls nested in it."""


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
    startup: dict[str, StatsView] | None
    """The first ``next()`` of each iterator the node returned (setup before
    the first item); ``None`` for a node that returned no iterator."""
    end: dict[str, StatsView] | None
    """The final ``next()`` of each iterator (cleanup)."""


@dataclass(frozen=True, slots=True)
class NodeInfo:
    label: str
    type: str


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

    def record(
        self, kind: int, elapsed: Sequence[int | None], own: Sequence[int | None]
    ) -> None:
        self.frames += 1
        if kind in (VALUE, OPEN, CALL_ERROR):
            self.calls += 1
        elif kind in (START, ITEM):
            self.items += 1
        if kind in (CALL_ERROR, NEXT_ERROR):
            self.errors += 1
        target = (
            self.events
            if kind in (VALUE, ITEM)
            else self.startup
            if kind == START
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

    def merge(self, other: NodeAggregate) -> None:
        self.calls += other.calls
        self.items += other.items
        self.errors += other.errors
        self.frames += other.frames
        for i in range(len(self.total)):
            self.total[i] += other.total[i]
            self.own[i] += other.own[i]
            self.events[i].merge(other.events[i])
            self.startup[i].merge(other.startup[i])
            self.end[i].merge(other.end[i])


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
    ) -> None:
        with self.lock:
            node = self.nodes.get(key)
            if node is None:
                node = self.nodes[key] = NodeAggregate(self.metrics)
            node.record(kind, elapsed, own)


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
        self.nodes: dict[str, NodeInfo] = {}
        """Registered nodes by key, in plan order."""
        self._shards: list[Shard] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def add_node(self, key: str, label: str, type_name: str) -> None:
        self.nodes.setdefault(key, NodeInfo(label, type_name))

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

    def profiles(self, merged: dict[str, NodeAggregate]) -> list[NodeProfile]:
        """Profiles of all registered nodes, in plan order."""
        empty = NodeAggregate(len(self.metrics))
        out: list[NodeProfile] = []
        for key, info in self.nodes.items():
            node = merged.get(key, empty)
            metrics: dict[str, MetricView] = {}
            for i, metric in enumerate(self.metrics):
                view = node.events[i].view(metric.scale)
                metrics[metric.name] = {
                    **view,
                    "total": node.total[i] * metric.scale,
                    "self": node.own[i] * metric.scale,
                }
            out.append(
                {
                    "key": key,
                    "label": info.label,
                    "type": info.type,
                    "calls": node.calls,
                    "items": node.items,
                    "errors": node.errors,
                    "metrics": metrics,
                    "startup": self._phase(node.startup),
                    "end": self._phase(node.end),
                }
            )
        return out

    def _phase(self, stats: list[Stats]) -> dict[str, StatsView] | None:
        if all(s.count == 0 for s in stats):
            return None
        return {
            metric.name: stats[i].view(metric.scale)
            for i, metric in enumerate(self.metrics)
        }
