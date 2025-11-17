from collections.abc import Iterable

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.pipeline import IcoPipeline

FloatPipeline = IcoPipeline[float, float, float]
FloatOperator = IcoOperator[float, float]


def test_operator_wraps_pipeline() -> None:
    # Basic pipeline: float → float
    p = FloatPipeline(
        context=lambda x: x + 1,
        body=[lambda x: x * 2, lambda x: x + 3],
        output=lambda x: round(x, 2),
    )

    op = IcoOperator(p)

    # Direct call
    assert op(1.0) == (1 + 1) * 2 + 3  # 7

    # Composition with another operator
    normalize = FloatOperator(lambda x: x / 10)
    composed = op | normalize
    assert composed(1.0) == 0.7


def test_pipeline_inside_map_operator() -> None:
    # Define a small pipeline that squares a number
    square_pipeline = FloatPipeline(
        context=lambda x: x,
        body=[lambda x: x * x],
        output=lambda x: x,
    )
    square_op = IcoOperator(square_pipeline)
    total_op = IcoOperator[Iterable[float], float](sum)

    # Apply map() and reduce-like composition
    flow = square_op.map() | total_op
    result = flow(iter([1.0, 2.0, 3.0]))
    assert result == 14  # 1² + 2² + 3²


def test_nested_pipeline_composition() -> None:
    # First pipeline: scale and shift
    p1 = FloatPipeline(
        context=lambda x: x + 1,
        body=[lambda x: x * 3],
        output=lambda x: x,
    )

    # Second pipeline: convert to string
    p2 = IcoPipeline[float, str, str](
        context=lambda x: f"[{x}]",
        body=[lambda s: s + "!"],
        output=lambda s: s,
    )

    composed = IcoOperator(p1) | IcoOperator(p2)
    assert composed(4) == "[15]!"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
