from collections.abc import Iterable

from apriori.ico.core import IcoFlowMeta, IcoOperator, IcoSource, IcoStream, NodeType


def test_data_produces_iterable() -> None:
    """
    Test that IcoData produces an iterable when called.
    """

    dataset = IcoSource(lambda: [1, 2, 3], name="dataset")
    result = list(dataset(None))

    assert result == [1, 2, 3]


def test_data_stream_composition() -> None:
    """
    Test that IcoData can be composed with IcoStream and downstream operators.
    """

    dataset = IcoSource(lambda: [1, 2, 3], name="dataset")
    scale = IcoOperator[int, int](lambda x: x * 2, name="scale")
    total = IcoOperator[Iterable[int], int](sum, name="sum")

    stream = IcoStream[int, int](body=scale)

    # Compose: Data → Stream → Sum
    flow = dataset | stream | total

    result = flow()

    assert result == 12  # (1,2,3) *2 = (2,4,6) → sum = 12


def test_data_structure_representation() -> None:
    """
    Test that IcoData exposes correct flow structure.
    """

    dataset = IcoSource(lambda: range(5), name="dataset")
    scale = IcoOperator[int, int](lambda x: x * 2, name="scale")
    stream = IcoStream[int, int](body=scale)
    flow = dataset | stream

    structure = IcoFlowMeta.from_operator(flow)

    # Root node should be composition
    assert structure.node_type == NodeType.chain

    # Check child order
    data_node, stream_node = structure.children
    assert data_node.node_type == NodeType.source
    assert data_node.name == "dataset"

    assert stream_node.node_type == NodeType.stream
    assert stream_node.children[0].name == "scale"
