from collections.abc import Iterator

from ico.core.operator import IcoOperator
from ico.core.stream import IcoStream

IntOperator = IcoOperator[int, int]


def sum(iterable: Iterator[int]) -> int:
    """Wrapper for mypy Any issues."""
    import builtins

    return builtins.sum(iterable)


def test_stream_maps_operator_over_iterable() -> None:
    """
    Test that IcoStream applies its body operator to each item in an iterable.
    """

    # Simple transformation: multiply by 2
    scale = IntOperator(lambda x: x * 2, name="scale")

    # Wrap in stream
    stream = IcoStream(scale)

    # Process iterable
    result = list(stream(iter([1, 2, 3])))

    assert result == [2, 4, 6]


def test_stream_composed_with_another_operator() -> None:
    """
    Test that IcoStream can be composed with another operator.
    """

    scale = IntOperator(lambda x: x * 2, name="scale")
    stream = IcoStream(scale)

    # Aggregate sum after streaming
    total = IcoOperator[Iterator[int], int](sum, name="sum")

    flow = stream | total

    result = flow(iter([1, 2, 3]))
    assert result == 12  # (1,2,3) *2 = (2,4,6) → sum = 12


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
