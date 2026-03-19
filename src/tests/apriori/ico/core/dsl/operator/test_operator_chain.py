from ico.core.node import iterate_nodes, iterate_parents
from ico.core.operator import IcoOperator

IntOperator = IcoOperator[int, int]


def test_chain_execution_order() -> None:
    chain = (
        IntOperator(lambda x: x + 1)
        | IntOperator(lambda x: x * 2)
        | IntOperator(lambda x: x - 3)
        | IntOperator(lambda x: x * 10)
    )
    assert chain(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_chain_callable_init() -> None:
    chain = (
        IntOperator(lambda x: x + 1)
        | IntOperator(lambda x: x * 2)
        | IntOperator(lambda x: x - 3)
        | IntOperator(lambda x: x * 10)
    )
    assert chain(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_chain_iter_order() -> None:
    a = IntOperator(lambda x: x + 1, name="a")
    b = IntOperator(lambda x: x + 2, name="b")
    c = IntOperator(lambda x: x, name="c")
    chain = a | b | c
    all_children = list(iterate_nodes(chain))

    assert len(all_children) == 5
    assert all_children[2] == a
    assert all_children[3] == b
    assert all_children[4] == c

    all_parents = list(iterate_parents(a))
    assert len(all_parents) == 2


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
