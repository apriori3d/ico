from collections.abc import Iterable, Iterator

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.sink import IcoSink
from apriori.ico.core.dsl.source import IcoSource
from apriori.ico.core.dsl.stream import IcoStream
from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.types import IcoNodeType

IntOperator = IcoOperator[int, int]


def sink_fn(data: Iterable[int]) -> None:
    for _ in data:
        pass  # Simulate consuming the data


def test_data_consumes_iterable() -> None:
    """
    Test that IcoSink consume an iterable when called.
    """

    read_all = False

    def source_fn(_: None) -> Iterator[int]:
        nonlocal read_all
        yield from [1, 2, 3]
        read_all = True

    source = IcoSource[int](source_fn, name="dataset")
    sink = IcoSink(sink_fn, name="sink")
    flow = source | sink
    flow(None)
    assert read_all


def test_data_structure_representation() -> None:
    """
    Test that IcoData exposes correct flow structure.
    """

    dataset = IcoSource[int](lambda _: iter(range(5)), name="dataset")
    scale = IntOperator(lambda x: x * 2, name="scale")
    stream = IcoStream(scale)
    sink = IcoSink(sink_fn, name="sink")
    flow = dataset | stream | sink

    structure = IcoFlowMeta.from_operator(flow)

    # Root node should be composition
    assert structure.node_type == IcoNodeType.chain

    # Check child order
    chain2, sink = structure.children
    assert chain2.node_type == IcoNodeType.chain
    assert sink.node_type == IcoNodeType.sink

    # Check child order
    data_node, stream_node = chain2.children
    assert data_node.node_type == IcoNodeType.source
    assert data_node.name == "dataset"

    assert stream_node.node_type == IcoNodeType.stream
    assert stream_node.children[0].name == "scale"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
