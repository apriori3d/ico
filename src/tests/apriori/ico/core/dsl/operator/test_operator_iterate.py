from collections.abc import Iterator

from apriori.ico.core.operator import IcoOperator

IntOperator = IcoOperator[int, int]


def sum(iterable: Iterator[int]) -> int:
    """Wrapper for mypy Any issues."""
    import builtins

    return builtins.sum(iterable)


def test_map_applies_elementwise() -> None:
    double = IntOperator(lambda x: x * 2)
    mapped = double.stream()

    result = list(mapped(iter([1, 2, 3])))
    assert result == [2, 4, 6]


def test_map_and_compose_chain() -> None:
    scale = IntOperator(lambda x: x * 2)
    total = IcoOperator[Iterator[int], int](sum)

    flow = scale.stream() | total
    assert flow(iter([1, 2, 3])) == 12


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
