from collections.abc import Iterator

from apriori.ico.core.meta.flow_meta import IcoFlowMeta, IcoNodeType
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream

Batch = Iterator[int]
DataStream = Iterator[Batch]


def test_ico_integration_data_runner_pipeline() -> None:
    """
    Integration test for full ICO dataflow:
        IcoSource → IcoStream(IcoPipeline) → IcoSink

    Verifies correct data transformation and structural hierarchy.
    """

    # ─────────────────────────────
    # 1. Define data source
    # ─────────────────────────────
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    def data_generator(_: None) -> DataStream:
        for batch in data:
            yield iter(batch)

    dataset = IcoSource[Batch](data_generator)

    # ─────────────────────────────
    # 2. Define atomic operators
    # ─────────────────────────────
    to_context = IcoOperator[int, int](lambda x: x)
    scale = IcoOperator[int, int](lambda x: x * 2)
    to_output = IcoOperator[int, int](lambda x: x)

    # ─────────────────────────────
    # 3. Define pipelines
    # ─────────────────────────────
    augment = IcoPipeline[int, int, int](
        context=to_context,
        body=[scale],
        output=to_output,
    )

    identity = IcoOperator[list[int], list[int]](lambda x: x)

    collate = IcoPipeline[Batch, list[int], int](
        context=list,
        body=[identity],
        output=max,
    )

    pipeline = augment.iterate() | collate

    # ─────────────────────────────
    # 4. Wrap into runner and connect with data
    # ─────────────────────────────
    data_stream = IcoStream[Batch, int](pipeline)
    data_flow = dataset | data_stream

    # ─────────────────────────────
    # 5. Execute flow
    # ─────────────────────────────
    result = list(data_flow(None))
    assert result == [6, 12, 18]  # Max of each batch after doubling

    # ─────────────────────────────
    # 6. Validate flow structure
    # ─────────────────────────────
    flow = IcoFlowMeta.from_node(data_flow)

    # Root should be compose node
    assert flow.node_type == IcoNodeType.chain

    # Compose should have two children: Data and Runner
    assert len(flow.children) == 2
    assert flow.children[0].node_type == IcoNodeType.source
    assert flow.children[1].node_type == IcoNodeType.stream

    # Runner should contain one child, the compose of map for augmentation over batch and collation
    runner_node = flow.children[1]
    assert len(runner_node.children) == 1
    assert runner_node.children[0].node_type == IcoNodeType.chain

    # Compose should have two children: the iterate for augmentation + collation
    compose_node = runner_node.children[0]
    assert len(compose_node.children) == 2

    # Map should have one children: the pipeline for augmentation
    map_node = compose_node.children[0]
    assert map_node.node_type == IcoNodeType.iterate
    assert len(map_node.children) == 1

    # Augmentation pipeline should contain exactly 3 children (context, flow step, output)
    pipeline_node = map_node.children[0]
    assert pipeline_node.node_type == IcoNodeType.pipeline
    assert len(pipeline_node.children) == 3
    for node in pipeline_node.children:
        assert node.node_type == IcoNodeType.operator

    # Collation pipeline should contain exactly zero children, because it use functions
    collate_node = compose_node.children[1]
    assert collate_node.node_type == IcoNodeType.pipeline
    assert len(collate_node.children) == 3


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(sys.argv))
