from collections.abc import Iterable

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.stream import IcoStream

IntOperator = IcoOperator[int, int]


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
    total = IcoOperator[Iterable[int], int](sum, name="sum")

    flow = stream | total

    result = flow(iter([1, 2, 3]))
    assert result == 12  # (1,2,3) *2 = (2,4,6) → sum = 12


# TODO: Re-enable when IcoFlowMeta is available

# def test_stream_structure_representation() -> None:
#     """
#     Test that IcoStream exposes correct flow structure.
#     """

#     scale = IcoOperator[int, int](lambda x: x * 2, name="scale")
#     stream = IcoStream[int, int](body=scale)

#     flow = IcoFlowMeta.from_operator(stream)

#     # Root node should be a stream
#     assert flow.node_type == NodeType.stream
#     assert len(flow.children) == 1
#     assert flow.children[0].name == "scale"
#     assert flow.children[0].node_type == NodeType.operator


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
