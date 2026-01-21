# tests/core/runtime/test_runtime_contour.py
from __future__ import annotations

from collections.abc import Iterator

from apriori.ico.core.operator import I, IcoOperator
from apriori.ico.core.runtime.shell import IcoShell
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource


def drain_all(xs: Iterator[I]) -> None:
    """Helper sink function that drains all items."""
    list(xs)
    return None


# ─── Basic Flow Construction ───


def test_runtime_contour_executes_source_to_sink() -> None:
    """Contour should execute full Source → Operator → Sink chain."""
    # Source: () → Iterable[int]
    source = IcoSource[int](lambda: iter(range(3)), name="dataset")

    # Operator: int → int
    op = IcoOperator[int, int](lambda x: x * 2, name="double")

    output: list[int] = []

    def capture_output(xs: Iterator[int]) -> None:
        nonlocal output
        output += list(xs)

    # Sink: Iterable[int] → None (prints result)
    sink = IcoSink[int](capture_output, name="capture")

    # Compose flow and wrap in contour
    flow = source | op.stream() | sink
    contour = IcoShell(flow)
    contour.activate().run().deactivate()

    # Capture printed output
    assert output == [0, 2, 4]


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
