from collections.abc import Iterable

from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream
from apriori.ico.core.types import IcoNodeType

IntOperator = IcoOperator[int, int]


def test_data_produces_iterable() -> None:
    """
    Test that IcoData produces an iterable when called.
    """

    dataset = IcoSource[int](lambda _: iter([1, 2, 3]), name="dataset")
    result = list(dataset(None))

    assert result == [1, 2, 3]


def test_data_stream_composition() -> None:
    """
    Test that IcoData can be composed with IcoStream and downstream operators.
    """

    dataset = IcoSource[int](lambda _: iter([1, 2, 3]), name="dataset")
    scale = IntOperator(lambda x: x * 2, name="scale")
    total = IcoOperator[Iterable[int], int](sum, name="sum")

    stream = IcoStream(scale)

    # Compose: Data → Stream → Sum
    flow = dataset | stream | total

    result = flow(None)

    assert result == 12  # (1,2,3) *2 = (2,4,6) → sum = 12


def test_data_structure_representation() -> None:
    """
    Test that IcoData exposes correct flow structure.
    """

    dataset = IcoSource[int](lambda _: iter(range(5)), name="dataset")
    scale = IntOperator(lambda x: x * 2, name="scale")
    stream = IcoStream(scale)
    flow = dataset | stream

    structure = IcoFlowMeta.from_operator(flow)

    # Root node should be composition
    assert structure.node_type == IcoNodeType.chain

    # Check child order
    data_node, stream_node = structure.children
    assert data_node.node_type == IcoNodeType.source
    assert data_node.name == "dataset"

    assert stream_node.node_type == IcoNodeType.stream
    assert stream_node.children[0].name == "scale"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
