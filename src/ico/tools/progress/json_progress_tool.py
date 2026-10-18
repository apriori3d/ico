"""Runtime tool that reports progress as JSON snapshots to a callback."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import final

from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.progress import (
    IcoProgress,
    IcoProgressEvent,
    IcoProgressTotalEvent,
)
from ico.core.runtime.tool import IcoRuntimeTool
from ico.core.tree_utils import TreePathIndex

ProgressSnapshot = dict[str, str | float | None]
"""``{"key", "label", "node", "done", "total", "rate", "eta"}``;
JSON-compatible."""


@dataclass(slots=True)
class _NodeProgress:
    key: str
    label: str
    node: str | None
    """Plan key of the progress node (``None`` outside the plan)."""
    total: float | None
    fixed: bool = True
    """Whether ``total`` is the node's own (else it came from its stream)."""
    done: float = 0.0
    started: float | None = None
    last_emit: float | None = None
    dirty: bool = False


@final
class JsonProgressTool(IcoRuntimeTool):
    """Report the progress of every ``IcoProgress`` node as JSON snapshots.

    Renders nothing: each update of a progress node is passed to a callback
    as a JSON-compatible snapshot, for external consumers such as
    ``ico-mcp`` (which decides how to turn them into MCP progress
    notifications) or a dashboard::

        {"key": "1.0.2", "label": "Train Epoch Progress", "node": "0.2",
         "done": 120, "total": 2000, "rate": 85.3, "eta": 22.1}

    - ``key``: path of the node in the runtime tree (``TreePathIndex``), a
      stable key for the node; progress nodes inside ``MPWorker`` runner
      processes get their own keys.
    - ``label``: the node name, else ``"Progress <n>"`` by registration
      order, as in ``RichProgressTool``.
    - ``node``: the plan key of the progress node (``plan_view(...)["keys"]``,
      also inside ``MPWorker`` runners), to show the progress on its plan
      row; ``None`` for a node outside the runtime's ``plan_root``.
    - ``done`` / ``total``: as in ``RichProgressTool``, a node that reached
      its total starts again from zero on the next update (next iteration).
      ``total`` is ``None`` for a node without a known count: it only
      counts up, and ``eta`` stays ``None``.
    - ``rate``: items per second since the node (re)started; ``eta``:
      seconds left at that rate. Both are ``None`` while unknown.

    On activation every registered node is reported once with ``done = 0``,
    in registration order (the order of the nodes in the flow), so a
    consumer knows all nodes, e.g. the first one, before any progress. Progress nodes inside ``MPWorker`` runner processes
    that register later are reported from their first update. Snapshots are
    throttled per node to at most ``max_rate`` per second; the final state
    of every node with pending updates is always emitted on deactivation. The callback may be called from different threads (async
    streams, worker channels), but never concurrently.

    Example:
        >>> snapshots = []
        >>> runtime = IcoRuntime(flow, tools=[JsonProgressTool(snapshots.append)])
        >>> runtime.activate().run().deactivate()
    """

    __slots__ = ("on_progress", "max_rate", "clock", "_nodes", "_lock")

    on_progress: Callable[[ProgressSnapshot], None]
    max_rate: float
    clock: Callable[[], float]

    def __init__(
        self,
        on_progress: Callable[[ProgressSnapshot], None],
        *,
        max_rate: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the tool.

        Args:
            on_progress: Receives every emitted snapshot.
            max_rate: Maximum snapshots per second for each progress node.
            clock: Time source in seconds (replaceable in tests).

        Raises:
            ValueError: If ``max_rate`` is not positive.
        """
        if max_rate <= 0:
            raise ValueError(f"max_rate must be positive, got {max_rate}")
        super().__init__()
        self.on_progress = on_progress
        self.max_rate = max_rate
        self.clock = clock
        self._nodes: dict[TreePathIndex, _NodeProgress] = {}
        self._lock = threading.Lock()

    # ─── Registration (IcoRuntimeToolRegistrationProtocol) ───

    def register_node(self, node: IcoRuntimeNode, path: TreePathIndex) -> None:
        """Track ``IcoProgress`` nodes found in the runtime tree."""
        if isinstance(node, IcoProgress):
            label = node.name or f"Progress {len(self._nodes)}"
            self._nodes[path] = _NodeProgress(
                key=str(path),
                label=label,
                node=node.plan_key,
                total=node.total,
                fixed=node.total is not None,
            )

    # ─── Lifecycle ───

    def on_activate(self) -> None:
        super().on_activate()
        with self._lock:
            for state in self._nodes.values():
                state.done = 0.0
                state.started = state.last_emit = None
                state.dirty = False
                # Initial state: consumers learn every node before any
                # progress. Not throttled.
                self.on_progress(_snapshot(state, None, None))

    def on_deactivate(self) -> None:
        super().on_deactivate()
        with self._lock:
            now = self.clock()
            for state in self._nodes.values():
                if state.dirty:
                    self._emit(state, now)

    # ─── Events ───

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        super().on_forward_event(event)
        if isinstance(event, IcoProgressTotalEvent):
            self._start(event)
            return
        if not isinstance(event, IcoProgressEvent):
            return

        path = event.trace.reverse()
        state = self._nodes.get(path)
        if state is None:
            raise RuntimeError(
                f"Received progress event for unregistered progress node at path {path}"
            )

        with self._lock:
            now = self.clock()
            finished = state.total is not None and state.done >= state.total
            if state.started is None or finished:
                # First update, or the next iteration of a finished node.
                state.done = 0.0
                state.started = now
                if finished and not state.fixed:
                    state.total = None  # a stream total held for one iteration
            state.done += event.advance
            state.dirty = True
            if state.last_emit is None or now - state.last_emit >= 1.0 / self.max_rate:
                self._emit(state, now)

    def _start(self, event: IcoProgressTotalEvent) -> None:
        """A node without a fixed total starts an iteration of known size."""
        state = self._nodes.get(event.trace.reverse())
        if state is None:
            return
        with self._lock:
            state.total = event.total
            state.done = 0.0
            state.started = self.clock()
            self._emit(state, state.started)

    def _emit(self, state: _NodeProgress, now: float) -> None:
        elapsed = now - state.started if state.started is not None else 0.0
        rate = state.done / elapsed if elapsed > 0 else None
        eta = (
            max(state.total - state.done, 0.0) / rate
            if rate and state.total is not None
            else None
        )
        state.last_emit = now
        state.dirty = False
        self.on_progress(_snapshot(state, rate, eta))


def _snapshot(
    state: _NodeProgress, rate: float | None, eta: float | None
) -> ProgressSnapshot:
    return {
        "key": state.key,
        "label": state.label,
        "node": state.node,
        "done": state.done,
        "total": state.total,
        "rate": rate,
        "eta": eta,
    }
