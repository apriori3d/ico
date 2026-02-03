from collections.abc import Iterator

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline

FloatPipeline = IcoPipeline[float]
FloatOperator = IcoOperator[float, float]


def sum(iterable: Iterator[float]) -> float:
    """Wrapper for mypy Any issues."""
    import builtins

    return builtins.sum(iterable)


def test_operator_wraps_pipeline() -> None:
    # Basic pipeline: float → float
    p = FloatPipeline(
        lambda x: x + 1,
        lambda x: x * 2,
        lambda x: x + 3,
        lambda x: round(x, 2),
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
        lambda x: x,
        lambda x: x * x,
        lambda x: x,
    )
    square_op = IcoOperator(square_pipeline)
    total_op = IcoOperator[Iterator[float], float](sum)

    # Apply iterate() and reduce-like composition
    flow = square_op.stream() | total_op
    result = flow(iter([1.0, 2.0, 3.0]))
    assert result == 14  # 1² + 2² + 3²


def test_nested_pipeline_composition() -> None:
    # First pipeline: scale and shift
    p1 = FloatPipeline(
        lambda x: x + 1,
        lambda x: x * 3,
        lambda x: x,
    )

    to_str = IcoOperator[float, str](lambda x: str(x))
    # Second pipeline: convert to string
    p2 = IcoPipeline[str](
        lambda x: f"[{x}]",
        lambda s: s + "!",
        lambda s: s,
    )

    composed = IcoOperator(p1) | to_str | IcoOperator(p2)
    assert composed(4) == "[15]!"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
