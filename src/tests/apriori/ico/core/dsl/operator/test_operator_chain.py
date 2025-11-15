from apriori.ico.core.dsl.operator import IcoOperator, iterate_nodes, iterate_parents
from apriori.ico.core.types import NodeType


def test_pipeline_execution_order() -> None:
    pipeline = (
        IcoOperator[int, int](lambda x: x + 1)
        | IcoOperator[int, int](lambda x: x * 2)
        | IcoOperator[int, int](lambda x: x - 3)
        | IcoOperator[int, int](lambda x: x * 10)
    )
    assert pipeline(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_callable_init() -> None:
    pipeline = (
        IcoOperator[int, int](lambda x: x + 1)
        | IcoOperator[int, int](lambda x: x * 2)
        | IcoOperator[int, int](lambda x: x - 3)
        | IcoOperator[int, int](lambda x: x * 10)
    )
    assert pipeline(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_iter_order() -> None:
    a = IcoOperator[int, int](lambda x: x + 1, name="a")
    b = IcoOperator[int, int](lambda x: x + 2, name="b")
    c = IcoOperator[int, int](lambda x: x, name="c")
    pipeline = a | b | c

    all_children = list(iterate_nodes(pipeline))
    assert len(all_children) == 5
    assert all_children[0].node_type == NodeType.chain
    assert all_children[1].node_type == NodeType.chain
    assert all_children[2] == a
    assert all_children[3] == b
    assert all_children[4] == c

    all_parents = list(iterate_parents(a))
    assert len(all_parents) == 2
    assert all_parents[0].node_type == NodeType.chain
    assert all_parents[1].node_type == NodeType.chain

    # Wrap pipeline in a contour
    contour = IcoOperator[None, None](lambda x: x, name="contour")
    contour.children.append(pipeline)
    pipeline.parent = contour

    all_parents = list(iterate_parents(c))
    assert len(all_parents) == 2
    assert all_parents[0].node_type == NodeType.chain
    assert all_parents[1] == contour


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
