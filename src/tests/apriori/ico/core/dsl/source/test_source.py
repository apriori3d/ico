from collections.abc import Iterator

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream

IntOperator = IcoOperator[int, int]


def sum(iterable: Iterator[int]) -> int:
    """Wrapper for mypy Any issues."""
    import builtins

    return builtins.sum(iterable)


def test_data_produces_iterable() -> None:
    """
    Test that IcoData produces an iterable when called.
    """

    dataset = IcoSource[int](lambda: iter([1, 2, 3]), name="dataset")
    result = list(dataset(None))

    assert result == [1, 2, 3]


def test_data_stream_composition() -> None:
    """
    Test that IcoData can be composed with IcoStream and downstream operators.
    """

    dataset = IcoSource[int](lambda: iter([1, 2, 3]), name="dataset")
    scale = IntOperator(lambda x: x * 2, name="scale")
    total = IcoOperator[Iterator[int], int](sum, name="sum")

    stream = IcoStream(scale)

    # Compose: Data → Stream → Sum
    flow = dataset | stream | total

    result = flow(None)

    assert result == 12  # (1,2,3) *2 = (2,4,6) → sum = 12


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
