from collections.abc import Iterator
from typing import TypeVar

import pytest

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.partition import IcoPartition, partition

T = TypeVar("T")


def is_valid(x: int) -> bool:
    return x >= 0


def identity(x: T) -> T:
    return x


def test_partition_splits_and_keeps_order() -> None:
    assert partition(is_valid)(iter([3, -1, 1, -2, 2])) == ([3, 1, 2], [-1, -2])


def test_partition_empty_stream() -> None:
    assert partition(is_valid)(iter([])) == ([], [])


def test_partition_consumes_stream() -> None:
    items = iter([1, -1])
    partition(is_valid)(items)
    assert list(items) == []


def test_partition_wraps_predicate_as_child() -> None:
    p = IcoPartition(is_valid)
    assert isinstance(p.predicate, IcoOperator)
    assert list(p.children) == [p.predicate]
    assert p.name == "partition"


def test_partition_signature() -> None:
    assert partition(is_valid).signature.format() == (
        "Iterator[int] → tuple[list[int], list[int]]"
    )


def test_partition_signature_uninferred_for_untyped_predicate() -> None:
    assert not partition(identity).signature.infered


def test_partition_static_types() -> None:
    # Checked by pyright and mypy in `make ci`.
    p: IcoOperatorProtocol[Iterator[int], tuple[list[int], list[int]]] = partition(
        is_valid
    )
    assert p(iter([1])) == ([1], [])


def test_describe_partition(capsys: pytest.CaptureFixture[str]) -> None:
    IcoPartition(is_valid, name="split").describe()
    assert "split" in capsys.readouterr().out
