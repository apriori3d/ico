from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.tree import iterate_nodes, iterate_parents
from apriori.ico.core.types import NodeType

IntOperator = IcoOperator[int, int]


def test_pipeline_execution_order() -> None:
    pipeline = (
        IntOperator(lambda x: x + 1)
        | IntOperator(lambda x: x * 2)
        | IntOperator(lambda x: x - 3)
        | IntOperator(lambda x: x * 10)
    )
    assert pipeline(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_callable_init() -> None:
    pipeline = (
        IntOperator(lambda x: x + 1)
        | IntOperator(lambda x: x * 2)
        | IntOperator(lambda x: x - 3)
        | IntOperator(lambda x: x * 10)
    )
    assert pipeline(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_iter_order() -> None:
    a = IntOperator(lambda x: x + 1, name="a")
    b = IntOperator(lambda x: x + 2, name="b")
    c = IntOperator(lambda x: x, name="c")
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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
