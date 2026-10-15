from collections.abc import Iterator
from typing import TypeVar

import pytest

from ico.core.filter import IcoFilter, keep
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.sink import IcoSink
from ico.core.source import IcoSource

T = TypeVar("T")


def is_even(x: int) -> bool:
    return x % 2 == 0


def first_char(x: str) -> str:
    return x[:1]


def identity(x: T) -> T:
    return x


def numbers() -> Iterator[int]:
    return iter([1, 2, 3, 4, 5, 6])


def test_keep_filters_items() -> None:
    assert list(keep(is_even)(iter([1, 2, 3, 4]))) == [2, 4]


def test_keep_uses_truthiness() -> None:
    assert list(keep(first_char)(iter(["a", "", "b"]))) == ["a", "b"]


def test_keep_is_lazy() -> None:
    seen: list[int] = []

    def track(x: int) -> bool:
        seen.append(x)
        return True

    out = keep(track)(iter([1, 2, 3]))
    assert seen == []
    assert next(out) == 1
    assert seen == [1]


def test_keep_wraps_predicate_as_child() -> None:
    f = IcoFilter(is_even)
    assert isinstance(f.predicate, IcoOperator)
    assert list(f.children) == [f.predicate]
    assert f.name == "keep"
    assert IcoFilter(is_even, name="evens").name == "evens"


def test_keep_in_flow() -> None:
    collected: list[int] = []

    def collect(x: int) -> None:
        collected.append(x)

    flow = IcoSource(numbers) | keep(is_even) | IcoSink(collect)
    flow(None)
    assert collected == [2, 4, 6]


def test_keep_signature() -> None:
    assert keep(is_even).signature.format() == "Iterator[int] → Iterator[int]"


def test_keep_signature_from_explicit_generic() -> None:
    f = IcoFilter[str](lambda x: bool(x))
    assert f.signature.format() == "Iterator[str] → Iterator[str]"


def test_keep_signature_uninferred_for_untyped_predicate() -> None:
    assert not keep(identity).signature.infered


def test_keep_static_types() -> None:
    # Checked by pyright and mypy in `make ci`.
    f: IcoOperatorProtocol[Iterator[int], Iterator[int]] = keep(is_even)
    assert list(f(iter([2]))) == [2]


def test_describe_keep(capsys: pytest.CaptureFixture[str]) -> None:
    IcoFilter(is_even, name="evens").describe()
    out = capsys.readouterr().out
    assert "evens" in out
    assert "Iterator[int]" in out
