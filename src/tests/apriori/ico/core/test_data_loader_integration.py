from collections.abc import Iterable

from apriori.ico.core import (
    IcoFlowMeta,
    IcoOperator,
    IcoPipeline,
    IcoSource,
    IcoStream,
    NodeType,
)


def test_ico_integration_data_runner_pipeline() -> None:
    """
    Integration test for full ICO dataflow:
        IcoData → IcoRunner → IcoPipeline → IcoOperators

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

    def data_generator() -> Iterable[Iterable[float]]:
        yield from data

    dataset = IcoSource[Iterable[float]](data_generator)

    # ─────────────────────────────
    # 2. Define atomic operators
    # ─────────────────────────────
    to_context = IcoOperator[float, float](lambda x: x)
    scale = IcoOperator[float, float](lambda x: x * 2)
    to_output = IcoOperator[float, float](lambda x: x)

    # ─────────────────────────────
    # 3. Define pipelines
    # ─────────────────────────────
    augment = IcoPipeline[float, float, float](
        context=to_context,
        body=[scale],
        output=to_output,
    )

    collate = IcoPipeline[Iterable[float], Iterable[float], float](
        context=list,
        body=[],
        output=max,
    )

    pipeline = augment.map() | collate

    # ─────────────────────────────
    # 4. Wrap into runner and connect with data
    # ─────────────────────────────
    runner = IcoStream[Iterable[float], float](pipeline)
    data_flow = dataset | runner

    # ─────────────────────────────
    # 5. Execute flow
    # ─────────────────────────────
    result = list(data_flow(None))
    assert result == [6, 12, 18]  # Max of each batch after doubling

    # ─────────────────────────────
    # 6. Validate flow structure
    # ─────────────────────────────
    flow = IcoFlowMeta.from_operator(data_flow)

    # Root should be compose node
    assert flow.node_type == NodeType.chain

    # Compose should have two children: Data and Runner
    assert len(flow.children) == 2
    assert flow.children[0].node_type == NodeType.source
    assert flow.children[1].node_type == NodeType.stream

    # Runner should contain one child, the compose of map for augmentation over batch and collation
    runner_node = flow.children[1]
    assert len(runner_node.children) == 1
    assert runner_node.children[0].node_type == NodeType.chain

    # Compose should have two children: the map for augmentation + collation
    compose_node = runner_node.children[0]
    assert len(compose_node.children) == 2

    # Map should have one children: the pipeline for augmentation
    map_node = compose_node.children[0]
    assert map_node.node_type == NodeType.map
    assert len(map_node.children) == 1

    # Augmentation pipeline should contain exactly 3 children (context, flow step, output)
    pipeline_node = map_node.children[0]
    assert pipeline_node.node_type == NodeType.pipeline
    assert len(pipeline_node.children) == 3
    for node in pipeline_node.children:
        assert node.node_type == NodeType.operator

    # Collation pipeline should contain exactly 2 children (context, output)
    collate_node = compose_node.children[1]
    assert collate_node.node_type == NodeType.pipeline
    assert len(collate_node.children) == 2
    for node in collate_node.children:
        assert node.node_type == NodeType.operator
