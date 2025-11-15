# tests/core/runtime/test_runtime_contour.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Generic

import pytest

from apriori.flow.progress.noop import NoOpProgress
from apriori.ico.core import (
    IcoOperator,
    IcoRuntimeContour,
    IcoSink,
    IcoSource,
)
from apriori.ico.core.runtime.progress import ProgressMixin
from apriori.ico.core.runtime.types import IcoRuntimeCommandType
from apriori.ico.core.types import I
from apriori.ico.tests.core.runtime.test_utils import (
    LifecycleEventsRecordingOperator,
)


def drain_all(xs: Iterable[I]) -> None:
    """Helper sink function that drains all items."""
    _ = list(xs)
    return None


# ─── Basic Flow Construction ───


def test_runtime_contour_executes_source_to_sink() -> None:
    """Contour should execute full Source → Operator → Sink chain."""
    # Source: () → Iterable[int]
    source = IcoSource[int](lambda: range(3), name="dataset")

    # Operator: int → int
    op = IcoOperator[int, int](lambda x: x * 2, name="double")

    output = []

    def capture_output(xs: Iterable[int]) -> None:
        nonlocal output
        output += list(xs)

    # Sink: Iterable[int] → None (prints result)
    sink = IcoSink[int](capture_output, name="capture")

    # Compose flow and wrap in contour
    flow = source | op.map() | sink
    contour = IcoRuntimeContour(flow)

    contour.ready().run().idle()

    # Capture printed output
    assert output == [0, 2, 4]


# ─── Lifecycle Event Propagation ───


def test_runtime_contour_lifecycle_event_propagation() -> None:
    """
    Verify that lifecycle events are broadcast to all nested operators.
    """
    # ─── Build a small flow with recording operators ───
    source = IcoSource[int](lambda: range(1), name="src")
    recorder1 = LifecycleEventsRecordingOperator()
    recorder2 = LifecycleEventsRecordingOperator()
    sink = IcoSink[int](drain_all, name="sink")

    flow = source | recorder1 | recorder2 | sink
    contour = IcoRuntimeContour(flow)

    # ─── Trigger events ───
    contour.ready().reset().idle()

    # ─── Verify recorded events ───
    recorded1 = recorder1.received_events
    recorded2 = recorder2.received_events

    # Both should have received the same sequence of lifecycle events
    expected = [
        IcoRuntimeCommandType.activate,
        IcoRuntimeCommandType.reset,
        IcoRuntimeCommandType.deavtivate,
    ]

    assert recorded1 == expected, f"Recorder1 got {recorded1}, expected {expected}"
    assert recorded2 == expected, f"Recorder2 got {recorded2}, expected {expected}"


# ─── Progress Binding ───


class ProgressRecorder(NoOpProgress):
    """Simple stub progress relay that records assigned operators."""

    messages: list[str]

    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    def print(self, message: str) -> None:
        self.messages.append(message)


class PrintOperator(IcoOperator[I, I], Generic[I], ProgressMixin):
    """An operator that prints progress messages."""

    def __init__(self) -> None:
        super().__init__(self._print_fn)

    def _print_fn(self, item: I) -> I:
        self.progress.print(f"Processed {item} in {self.name}")
        return item


def test_runtime_contour_bind_progress_and_collect_messages() -> None:
    """
    Progress relay should propagate to all HasProgress nodes,
    and each PrintOperator should emit messages through the shared recorder.
    """
    # Construct minimal flow with two print operators
    source = IcoSource[int](lambda: range(3), name="dataset")
    printer1 = PrintOperator[int]()
    printer1.name = "printer1"
    printer2 = PrintOperator[int]()
    printer2.name = "printer2"

    sink = IcoSink[int](drain_all, name="sink")

    # Build full flow and contour
    contour = IcoRuntimeContour(source | printer1.map() | printer2.map() | sink)

    # Bind shared recorder
    progress = ProgressRecorder()
    contour.bind_progress(progress)

    # Execute full flow
    contour.ready().run().idle()

    # ─── Assertions ───

    # 1. Recorder bound to contour
    assert contour.progress is progress
    assert isinstance(contour.progress, ProgressRecorder)

    # 2. All emitted messages collected
    # Each source produces 3 items, so both printers should emit 3 messages
    expected_msgs = [
        "Processed 0 in printer1",
        "Processed 0 in printer2",
        "Processed 1 in printer1",
        "Processed 1 in printer2",
        "Processed 2 in printer1",
        "Processed 2 in printer2",
    ]

    assert progress.messages == expected_msgs, (
        f"Expected {expected_msgs}, got {progress.messages}"
    )


def test_runtime_contour_raises_on_missing_source() -> None:
    sink = IcoSink[int](drain_all)
    op = IcoOperator[int, int](lambda x: x)
    with pytest.raises(ValueError, match="Invalid flow form"):
        IcoRuntimeContour(op | sink)  # type: ignore[arg-type]


def test_runtime_contour_raises_on_missing_sink() -> None:
    src = IcoSource[int](lambda: range(3))
    op = IcoOperator[int, int](lambda x: x)
    with pytest.raises(ValueError, match="Invalid flow form"):
        IcoRuntimeContour(src | op)  # type: ignore[arg-type]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
