"""Runs shown by the Live Board: progress, outcome and plan of each run.

:class:`BoardSource` is what the board reads. :class:`BoardRuns` is the
default source: active runs and the most recent finished ones, fed by
progress snapshots (``JsonProgressTool``). Another process can host the
board with its own source or feed a :class:`BoardRuns` from messages, e.g.
``ico-mcp`` from its kernel's notifications.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Literal, Protocol, TypedDict

from ico.describe.plan.json_renderer import PlanView
from ico.tools.profiler.report import (
    ProfileCells,
    ResourceCells,
    profile_cells,
    resource_cells,
)
from ico.tools.profiler.tool import ProfileSnapshot

RunState = Literal["running", "done", "failed"]

ProgressSnapshot = Mapping[str, str | float | None]
"""A ``JsonProgressTool`` snapshot: ``key``, ``label``, ``done``, ``total``,
``rate``, ``eta``."""


class NodeProgress(TypedDict):
    key: str
    label: str
    node: str | None
    """Plan key of the progress node (see ``plan["keys"]``), if known."""
    done: float
    total: float | None
    rate: float | None
    eta: float | None


class RunView(TypedDict):
    """One run as the board shows it; JSON-compatible."""

    id: str
    title: str
    state: RunState
    started: str
    elapsed: float
    progress: list[NodeProgress]
    error: str | None
    summary: str | None
    has_plan: bool
    has_profile: bool


class BoardSource(Protocol):
    """What the board reads; called from the board's server threads."""

    def runs(self) -> list[RunView]:
        """Runs, newest first."""
        ...

    def plan(self, run_id: str) -> PlanView | None:
        """The plan of a run, if known."""
        ...

    def profile(self, run_id: str) -> dict[str, ProfileCells] | None:
        """The profile of a run by plan key (see ``plan["keys"]``), if
        profiled."""
        ...

    def resources(self, run_id: str) -> list[ResourceCells] | None:
        """The resource rows of a profiled run (uses, loads, reloads,
        size), if profiled."""
        ...


class BoardRun:
    """One run of a :class:`BoardRuns`; its methods are thread-safe."""

    def __init__(self, title: str, plan: PlanView | None, lock: threading.Lock) -> None:
        self.id = secrets.token_hex(6)
        self.title = title
        self.plan = plan
        self.state: RunState = "running"
        self.error: str | None = None
        self.summary: str | None = None
        self._started = datetime.now(timezone.utc)
        self._clock = time.monotonic()
        self._duration: float | None = None
        self._progress: dict[str, NodeProgress] = {}
        self._profile: dict[str, ProfileCells] | None = None
        self._resources: list[ResourceCells] | None = None
        self._lock = lock

    def progress(self, snapshot: ProgressSnapshot) -> None:
        """Keep the last snapshot of a progress node (flow order is kept)."""
        node = _node(snapshot)
        with self._lock:
            self._progress[node["key"]] = node

    def profile(self, snapshot: ProfileSnapshot) -> None:
        """Keep the latest profile of the run (a ``ProfilerTool`` snapshot)."""
        cells = profile_cells(snapshot)
        rows = resource_cells(snapshot)
        with self._lock:
            self._profile = cells
            self._resources = rows

    def profile_cells(self) -> dict[str, ProfileCells] | None:
        with self._lock:
            return self._profile

    def resource_rows(self) -> list[ResourceCells] | None:
        with self._lock:
            return self._resources

    def finish(self, error: str | None = None, *, summary: str | None = None) -> None:
        """Mark the run ``done``, or ``failed`` with an error message.

        ``summary`` is a one-line outcome shown under the progress, e.g.
        ``"40 outputs · 0 item errors"``.
        """
        with self._lock:
            self.summary = summary
            self._duration = time.monotonic() - self._clock
            self.state = "failed" if error is not None else "done"
            self.error = error

    def view(self) -> RunView:
        with self._lock:
            elapsed = (
                self._duration
                if self._duration is not None
                else time.monotonic() - self._clock
            )
            return {
                "id": self.id,
                "title": self.title,
                "state": self.state,
                "started": self._started.isoformat().replace("+00:00", "Z"),
                "elapsed": round(elapsed, 3),
                "progress": list(self._progress.values()),
                "error": self.error,
                "summary": self.summary,
                "has_plan": self.plan is not None,
                "has_profile": self._profile is not None,
            }


class BoardRuns:
    """Active runs and the ``max_finished`` most recent finished ones."""

    def __init__(self, *, max_finished: int = 20) -> None:
        if max_finished < 1:
            raise ValueError(f"max_finished must be positive, got {max_finished}")
        self.max_finished = max_finished
        self._runs: dict[str, BoardRun] = {}
        self._lock = threading.Lock()

    def start(self, title: str, plan: PlanView | None = None) -> BoardRun:
        """Add a running run; drops the oldest finished runs over the limit."""
        run = BoardRun(title, plan, self._lock)
        with self._lock:
            self._runs[run.id] = run
            finished = [r for r in self._runs.values() if r.state != "running"]
            for old in finished[: max(0, len(finished) - self.max_finished)]:
                del self._runs[old.id]
        return run

    def runs(self) -> list[RunView]:
        with self._lock:
            runs = list(reversed(self._runs.values()))
        return [run.view() for run in runs]

    def plan(self, run_id: str) -> PlanView | None:
        with self._lock:
            run = self._runs.get(run_id)
        return run.plan if run is not None else None

    def profile(self, run_id: str) -> dict[str, ProfileCells] | None:
        with self._lock:
            run = self._runs.get(run_id)
        return run.profile_cells() if run is not None else None

    def resources(self, run_id: str) -> list[ResourceCells] | None:
        with self._lock:
            run = self._runs.get(run_id)
        return run.resource_rows() if run is not None else None


def _node(snapshot: ProgressSnapshot) -> NodeProgress:
    def number(key: str) -> float | None:
        value = snapshot.get(key)
        return float(value) if isinstance(value, int | float) else None

    return {
        "key": str(snapshot.get("key")),
        "label": str(snapshot.get("label")),
        "node": node if isinstance(node := snapshot.get("node"), str) else None,
        "done": number("done") or 0.0,
        "total": number("total"),
        "rate": number("rate"),
        "eta": number("eta"),
    }
