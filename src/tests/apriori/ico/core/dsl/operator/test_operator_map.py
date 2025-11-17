from collections.abc import Iterable

from apriori.ico.core.dsl.operator import IcoOperator

IntOperator = IcoOperator[int, int]


def test_map_applies_elementwise() -> None:
    double = IntOperator(lambda x: x * 2)
    mapped = double.map()

    result = list(mapped(iter([1, 2, 3])))
    assert result == [2, 4, 6]


def test_map_and_compose_chain() -> None:
    scale = IntOperator(lambda x: x * 2)
    total = IcoOperator[Iterable[int], int](sum)

    flow = scale.map() | total
    assert flow(iter([1, 2, 3])) == 12


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
