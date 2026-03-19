from ico.core.operator import IcoOperator
from ico.core.pipeline import IcoPipeline

IntPipeline = IcoPipeline[int]
IntOperator = IcoOperator[int, int]


def test_pipeline_execution_order() -> None:
    p = IntPipeline(
        IntOperator(lambda x: x + 1),
        IntOperator(lambda x: x * 2),
        IntOperator(lambda x: x - 3),
        IntOperator(lambda x: x * 10),
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_callable_init() -> None:
    p = IntPipeline(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x - 3,
        lambda x: x * 10,
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
