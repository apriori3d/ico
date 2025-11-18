from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.pipeline import IcoPipeline

IntPipeline = IcoPipeline[int, int, int]
IntOperator = IcoOperator[int, int]


def test_pipeline_execution_order() -> None:
    p = IntPipeline(
        context=IntOperator(lambda x: x + 1),
        body=[
            IntOperator(lambda x: x * 2),
            IntOperator(lambda x: x - 3),
        ],
        output=IntOperator(lambda x: x * 10),
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_callable_init() -> None:
    p = IntPipeline(
        context=lambda x: x + 1,
        body=[
            lambda x: x * 2,
            lambda x: x - 3,
        ],
        output=lambda x: x * 10,
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_len_and_iter() -> None:
    p = IntPipeline(
        context=IntOperator(lambda x: x + 1),
        body=[
            IntOperator(lambda x: x + 1),
            IntOperator(lambda x: x + 2),
        ],
        output=IntOperator(lambda x: x),
    )
    assert len(p) == 2


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
