"""Runtime tool that shows a run on the Live Board in the browser."""

from __future__ import annotations

import sys
import threading
from typing import final

from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.tool import IcoRuntimeTool
from ico.core.tree_utils import TreePathIndex
from ico.describe.plan.json_renderer import plan_view
from ico.tools.board.runs import BoardRun, BoardRuns
from ico.tools.board.server import LiveBoard
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot

_default: tuple[BoardRuns, LiveBoard] | None = None
_default_lock = threading.Lock()


def default_board() -> tuple[BoardRuns, LiveBoard]:
    """The process-wide board, started on first use; its URL is printed to
    stderr once. Runs of all ``LiveBoardTool()`` instances show up there."""
    global _default
    with _default_lock:
        if _default is None:
            runs = BoardRuns()
            board = LiveBoard(runs)
            print(f"ICO Live Board: {board.start()}", file=sys.stderr, flush=True)
            _default = (runs, board)
        return _default


@final
class LiveBoardTool(IcoRuntimeTool):
    """Show the progress and the plan of a runtime on the Live Board.

    The browser counterpart of ``RichProgressTool``: one progress bar per
    ``IcoProgress`` node (also inside ``MPWorker`` runner processes) and the
    pipeline graph (``plan_view``)::

        runtime = IcoRuntime(flow, tools=[LiveBoardTool()])
        runtime.activate().run().deactivate()

    Each activation adds a run to the board; deactivation marks it ``done``,
    or ``failed`` if the runtime is in the fault state. A run whose runtime
    is never deactivated stays ``running``.

    Args:
        runs: Where to add runs; by default the process-wide board
            (:func:`default_board`). Pass your own to host the board yourself.
        title: Run title; by default the flow name.
        max_rate: Progress snapshots per second per node.
    """

    __slots__ = ("runs", "title", "_progress", "_runtime", "_run")

    def __init__(
        self,
        runs: BoardRuns | None = None,
        *,
        title: str | None = None,
        max_rate: float = 2.0,
    ) -> None:
        super().__init__()
        self.runs = runs
        self.title = title
        self._progress = JsonProgressTool(self._on_progress, max_rate=max_rate)
        self._runtime: IcoRuntime | None = None
        self._run: BoardRun | None = None

    # ─── Registration (IcoRuntimeToolRegistrationProtocol) ───

    def register_node(self, node: IcoRuntimeNode, path: TreePathIndex) -> None:
        if self._runtime is None and isinstance(node, IcoRuntime):
            self._runtime = node
        self._progress.register_node(node, path)

    # ─── Lifecycle ───

    def on_activate(self) -> None:
        super().on_activate()
        runs = self.runs if self.runs is not None else default_board()[0]
        flow = self._runtime.closure if self._runtime is not None else None
        title = self.title or (flow.name if flow is not None else None) or "run"
        self._run = runs.start(title, plan_view(flow) if flow is not None else None)
        self._progress.on_activate()

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self._progress.on_deactivate()
        if self._run is not None:
            fault = self._runtime is not None and self._runtime.state.is_fault
            self._run.finish("the runtime is in the fault state" if fault else None)
            self._run = None

    # ─── Events ───

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        super().on_forward_event(event)
        self._progress.on_forward_event(event)

    def _on_progress(self, snapshot: ProgressSnapshot) -> None:
        if self._run is not None:
            self._run.progress(snapshot)
