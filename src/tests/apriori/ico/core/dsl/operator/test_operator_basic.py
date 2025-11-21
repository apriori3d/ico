import pytest

from apriori.ico.core.operator import IcoOperator


def test_basic_composition() -> None:
    double = IcoOperator[float, float](lambda x: x * 2)
    square = IcoOperator[float, float](lambda x: x**2)

    composed = double | square
    assert composed(3) == 36


def test_then_aliases_are_equivalent() -> None:
    inc = IcoOperator[float, float](lambda x: x + 1)
    double = IcoOperator[float, float](lambda x: x * 2)

    assert (inc | double)(3) == 8
    assert inc.chain(double)(3) == 8


def test_operator_is_callable() -> None:
    negate = IcoOperator[float, float](lambda x: -x)
    assert negate(5) == -5


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
