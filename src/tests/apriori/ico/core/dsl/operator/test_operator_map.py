from collections.abc import Iterable

from apriori.ico.core.dsl.operator import IcoOperator


def test_map_applies_elementwise() -> None:
    double = IcoOperator[float, float](lambda x: x * 2)
    mapped = double.map()

    result = list(mapped([1, 2, 3]))
    assert result == [2, 4, 6]


def test_map_and_compose_chain() -> None:
    scale = IcoOperator[float, float](lambda x: x * 2)
    total = IcoOperator[Iterable[float], float](lambda xs: sum(xs))

    pipeline = scale.map() | total
    assert pipeline([1, 2, 3]) == 12
