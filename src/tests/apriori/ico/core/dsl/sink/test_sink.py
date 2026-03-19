from collections.abc import Iterator

from ico.core.operator import IcoOperator
from ico.core.sink import IcoSink
from ico.core.source import IcoSource

IntOperator = IcoOperator[int, int]


def sink_fn(data: int) -> None:
    pass  # Simulate consuming the data


def test_data_consumes_iterable() -> None:
    """
    Test that IcoSink consume an iterable when called.
    """

    read_all = False

    def source_fn() -> Iterator[int]:
        nonlocal read_all
        yield from [1, 2, 3]
        read_all = True

    source = IcoSource[int](source_fn, name="dataset")
    sink = IcoSink(sink_fn, name="sink")
    flow = source | sink
    flow(None)
    assert read_all


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
